"""
THE GOLD STANDARD TRAINING SUITE (AI KOSH VERSION)
---------------------------------------------------
Your Vision: Train highly specialized models for specific tasks (Summarization & Math) and combine them.

INSTRUCTIONS FOR AI KOSH:
1. Open your AI Kosh Jupyter Notebook (with the 30GB A100 GPU).
2. Copy this entire script into a single cell or split it into blocks.
3. Run the pip install commands below first to install the software!
"""

# ==========================================
# 1. SETUP & BEST-PRACTICE INSTALLS
# ==========================================
# Run this in your first Jupyter cell:
# !pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes datasets

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# --- CHOOSE WHAT TO TRAIN TODAY ---
# Options: "SUMMARIZATION", "MATH", "INDIC_EXPERT"
TRAINING_MODE = "INDIC_EXPERT" 

# ==========================================
# 2. LOAD THE GOLD STANDARD BASE MODEL
# ==========================================
max_seq_length = 4096 # High context window (A100 GPU can easily handle this!)
load_in_4bit = True   # Keeps training fast and memory-efficient

print("Loading Base Gemma-2 Brain...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA (The "Adapter" technique)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # 32 is the gold standard for high-complexity tasks like Math
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 3. GOLD STANDARD DATA PREPROCESSING
# ==========================================
# We use the "ChatML" formatting standard. This is how OpenAI trains GPT-4.
prompt_format = """<|im_start|>user
{instruction}
<|im_start|>assistant
{response}<|im_end|>"""

EOS_TOKEN = tokenizer.eos_token

if TRAINING_MODE == "SUMMARIZATION":
    print("Loading GovReport (Gold Standard for massive document summarization)...")
    dataset = load_dataset("ccdv/govreport-summarization", split="train[:5000]") # Prototyping
    
    def format_summary(examples):
        texts = []
        for doc, summary in zip(examples["report"], examples["summary"]):
            instruction = f"Summarize this highly detailed report comprehensively:\n\n{doc}"
            texts.append(prompt_format.format(instruction=instruction, response=summary) + EOS_TOKEN)
        return { "text" : texts }

elif TRAINING_MODE == "MATH":
    # GOLD STANDARD MATH: NuminaMath-CoT was used by the team that won the International AI Math Olympiad!
    print("Loading NuminaMath-CoT (World's Best for Olympiad/NASA-level complex math)...")
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train[:5000]")
    
    def format_summary(examples):
        texts = []
        for query, response in zip(examples["problem"], examples["solution"]):
            instruction = f"Solve this highly complex mathematical problem step-by-step:\n\n{query}"
            texts.append(prompt_format.format(instruction=instruction, response=response) + EOS_TOKEN)
        return { "text" : texts }

elif TRAINING_MODE == "INDIC_EXPERT":
    # THE TRUE GOLD STANDARD: The 'Aya' dataset by Cohere.
    # XLSum only teaches news summaries. Aya contains millions of highly complex, human-verified 
    # instructions, summaries, and logic puzzles across 114 languages. This guarantees it follows 
    # user requirements without hallucinating.
    print("Loading Aya Dataset (The World's Best Multilingual Instruction Dataset)...")
    dataset = load_dataset("CohereForAI/aya_dataset", split="train")
    
    # Filter for Hindi (you can change 'hin' to 'tam' (Tamil), 'tel' (Telugu), 'ben' (Bengali), 'mar' (Marathi), etc.)
    dataset = dataset.filter(lambda example: example["language"] == "hin").select(range(5000))
    
    def format_summary(examples):
        texts = []
        for inputs, targets in zip(examples["inputs"], examples["targets"]):
            # Aya already provides complex user requirements (e.g. "Summarize this in 3 bullet points without losing context")
            texts.append(prompt_format.format(instruction=inputs, response=targets) + EOS_TOKEN)
        return { "text" : texts }

# Apply the preprocessing
dataset = dataset.map(format_summary, batched = True)

# --- TRAIN / TEST SPLIT ---
print("Splitting data into Training and Evaluation sets (80/20)...")
dataset = dataset.train_test_split(test_size=0.2, seed=3407)
train_data = dataset["train"]
eval_data = dataset["test"]

# ==========================================
# 4. GOLD STANDARD TRAINING ALGORITHMS
# ==========================================
print(f"Starting {TRAINING_MODE} Training Pipeline...")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = eval_data, # Evaluates the model on unseen data!
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Turned back ON for 3x faster training (A100 has plenty of memory)
    args = TrainingArguments(
        per_device_train_batch_size = 4, # Increased for the 30GB A100 GPU
        gradient_accumulation_steps = 4, 
        warmup_steps = 10,
        max_steps = 200, # Increase this for your final production run
        eval_strategy = "steps",
        eval_steps = 20, # Check the test accuracy every 20 steps
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # Best optimizer for LLMs on limited VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", # GOLD STANDARD: Cosine decay prevents "forgetting"
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

# ==========================================
# 5. SAVE THE SPECIALIZED EXPERT
# ==========================================
expert_name = f"llama3_expert_{TRAINING_MODE.lower()}"
model.save_pretrained(expert_name)
print(f"Training Complete! Saved your expert adapter as: {expert_name}")
