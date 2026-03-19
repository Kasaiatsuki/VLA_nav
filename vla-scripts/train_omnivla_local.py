import os
import sys
import torch
import draccus
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch.distributed as dist
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AdamW, get_cosine_schedule_with_warmup

# Add project root to sys path
sys.path.append(os.getcwd())

from prismatic.models.load import load_vla
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.local_dataset import Local_Dataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_MMN

@dataclass
class LocalTrainingConfig:
    # Model & Dataset Paths
    vla_path: str = "openvla/openvla-7b"
    data_dir: str = "path/to/your/dataset"
    run_id: str = "local-vla-finetune"
    
    # Training Parameters
    batch_size: int = 2
    grad_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_steps: int = 1000
    save_steps: int = 250
    
    # LoRA Parameters
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Optimization
    warmup_steps: int = 50
    weight_decay: float = 0.0
    
    # Hardware
    mixed_precision: str = "bf16" # or "fp16"
    
    # Output
    output_dir: str = "checkpoints"

def train():
    # 0. Load Config
    parser = draccus.ArgumentParser(config_class=LocalTrainingConfig)
    cfg = parser.parse_args()
    
    # 1. Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
    )
    device = accelerator.device
    
    # 2. Load Model & Processor
    model, processor = load_vla(cfg.vla_path)
    
    # Apply LoRA if configured
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 3. Create Dataset & DataLoader
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = Local_Dataset(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        data_folder=cfg.data_dir,
    )
    
    collator = PaddedCollatorForActionPrediction_Nav_MMN(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    # 5. Prepare for distributed training
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # 6. Training Loop
    model.train()
    print(f"Starting training for {cfg.max_steps} steps...")
    
    step = 0
    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps: break
            
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.is_main_process:
                    if step % 10 == 0:
                        print(f"Step {step}: Loss = {loss.item():.4f}")
                        
                    # Save Checkpoint
                    if step > 0 and step % cfg.save_steps == 0:
                        save_path = Path(cfg.output_dir) / cfg.run_id / f"checkpoint-{step}"
                        accelerator.save_state(save_path)
                        print(f"Saved checkpoint to {save_path}")

            step += 1

    print("Training complete!")

if __name__ == "__main__":
    train()
