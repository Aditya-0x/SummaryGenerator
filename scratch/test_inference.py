import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os

# Path to the latest checkpoint
checkpoint_path = r"c:\Users\adive\OneDrive\Desktop\summary\mlplo\checkpoints\bart-large-xsum\checkpoint-2970"

print(f"Loading model from {checkpoint_path}...")
tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
model = BartForConditionalGeneration.from_pretrained(checkpoint_path)

if torch.cuda.is_available():
    model = model.to("cuda")

text = """The BART-large model has been successfully trained on the XSum dataset. It achieved a ROUGE-1 score of 42.16% and a ROUGE-L score of 33.76%. This model is much more powerful than the previous BART-base version and produces higher quality summaries. The training process took approximately 13 hours on an NVIDIA RTX 4050 GPU."""

inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("Generating...")
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=1024,
    min_length=100,
    length_penalty=2.0,
    early_stopping=True,
    no_repeat_ngram_size=3
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"\nRESULT: {summary}")
