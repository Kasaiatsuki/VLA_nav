import torch
from transformers import AutoProcessor
import os
import sys

# VLA_nav root
sys.path.append("/home/orne/kasai_ws/src/VLA_nav")
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.local_dataset import Local_Dataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

data_dir = "/home/orne/kasai_ws/src/VLA_nav/data_sample" 
vla_path = "openvla/openvla-7b"

processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)

dataset = Local_Dataset(
    action_tokenizer=action_tokenizer,
    base_tokenizer=processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
    data_folder=data_dir,
    horizon=8
)

sample = dataset[0]
input_ids = sample['input_ids']

print("Token ID list and decoded strings:")
for tid in input_ids:
    print(f"{tid.item()}: {repr(processor.tokenizer.decode([tid.item()]))}")

print(f"Total tokens: {len(input_ids)}")
