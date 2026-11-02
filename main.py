import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. CONFIGURATION (The "Control Center")
# ==========================================
class Config:
    """Central configuration to manage hyperparameters and paths."""
    MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
    DATASET_ID = "ed-donner/pricer-data"
    OUTPUT_DIR = "my_fine_tuned_price_model"

    # QLoRA Parameters
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    LORA_RANK = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Training Parameters
    BATCH_SIZE = 2
    GRAD_ACCUMULATION = 4
    MAX_STEPS = 60  # Set to 300+ for production
    LEARNING_RATE = 2e-4
    SEED = 3407

# ==========================================
# 2. PROMPT TEMPLATE
# ==========================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert price predictor. Estimate the price of the following item.

### Input:
{}

### Response:
{}"""

# ==========================================
# 3. MODULAR FUNCTIONS
# ==========================================

def load_model_and_tokenizer():
    """Loads the base model and attaches the QLoRA adapter."""
    print(f"⬇️ Loading Model: {Config.MODEL_ID}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_ID,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=Config.LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.LORA_RANK,
        target_modules=Config.TARGET_MODULES,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=Config.SEED,
    )
    return model, tokenizer

def format_data_for_training(tokenizer):
    """Loads and formats the dataset for SFT."""
    print(f"📊 Loading Dataset: {Config.DATASET_ID}...")
    dataset = load_dataset(Config.DATASET_ID, split="train")

    def formatting_prompts_func(examples):
        inputs = examples["text"]
        outputs = examples["price"]
        texts = []
        for input_text, output_text in zip(inputs, outputs):
            # Must add EOS token so the model knows when to stop
            text = alpaca_prompt.format(input_text, output_text) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True)

def train_model(model, tokenizer, dataset):
    """Configures and executes the training loop."""
    print("🚀 Starting Training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRAD_ACCUMULATION,
            warmup_steps=5,
            max_steps=Config.MAX_STEPS,
            learning_rate=Config.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=Config.SEED,
            output_dir="outputs",
        ),
    )
    trainer.train()
    return trainer

def save_artifacts(model, tokenizer):
    """Saves the fine-tuned adapters."""
    print(f"💾 Saving model to {Config.OUTPUT_DIR}...")
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

def run_inference_test(model, tokenizer, item_description):
    """Runs a quick sanity check inference."""
    print("\n🧪 Running Inference Test...")
    FastLanguageModel.for_inference(model)

    prompt = alpaca_prompt.format(item_description, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
    result = tokenizer.batch_decode(outputs)

    # Extract just the price
    prediction = result[0].split("### Response:")[-1].replace("<|end_of_text|>", "").strip()
    return prediction

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    model, tokenizer = load_model_and_tokenizer()

    # 2. Data Prep
    train_dataset = format_data_for_training(tokenizer)

    # 3. Train
    train_model(model, tokenizer, train_dataset)

    # 4. Save
    save_artifacts(model, tokenizer)

    # 5. Verify
    test_item = "Sony WH-1000XM5 Wireless Noise Canceling Headphones, Black"
    price = run_inference_test(model, tokenizer, test_item)

    print(f"\n✅ Pipeline Complete.")
    print(f"Test Item: {test_item}")
    print(f"Predicted Price: {price}")