import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os

text = """The BART-large model has been successfully trained on the XSum dataset. It achieved a ROUGE-1 score of 42.16% and a ROUGE-L score of 33.76%. This model is much more powerful than the previous BART-base version and produces higher quality summaries. The training process took approximately 13 hours on an NVIDIA RTX 4050 GPU."""

def run_test(path, name):
    print(f"\n--- Testing {name} ---")
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
        
    tokenizer = BartTokenizer.from_pretrained(path, local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained(path, local_files_only=True)
    if torch.cuda.is_available():
        model = model.to("cuda")

    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=64,
        min_length=10,
        length_penalty=0.6,
        no_repeat_ngram_size=3
    )
    print(f"RESULT: {tokenizer.decode(summary_ids[0], skip_special_tokens=True)}")

base_dir = "mlplo/checkpoints/bart-large-xsum"
run_test(os.path.join(base_dir, "checkpoint-594"), "Epoch 1 (Checkpoint-594)")
run_test(os.path.join(base_dir, "checkpoint-2970"), "Epoch 5 (Checkpoint-2970)")
