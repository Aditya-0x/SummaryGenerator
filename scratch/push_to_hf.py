import os
from transformers import BartForConditionalGeneration, BartTokenizer

# Load local checkpoint
checkpoint_path = r"c:\Users\adive\OneDrive\Desktop\summary\mlplo\checkpoints\bart-large-xsum\checkpoint-2970"
print(f"Loading model from {checkpoint_path}...")
model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
tokenizer = BartTokenizer.from_pretrained(checkpoint_path)

# Push to Hugging Face
repo_name = "Adive01/bart-large-xsum-finetuned"
print(f"Pushing model to {repo_name}...")
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
print("Push complete!")
