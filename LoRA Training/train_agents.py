import os
import gc
import glob
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def format_instruction(example):
    """
    Wraps the raw data into the exact Prompt format expected by Llama 3.1
    """
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{example['system']}<|eot_id|>\n"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{example['user']}<|eot_id|>\n"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{example['assistant']}<|eot_id|>"
    return {"text": prompt}

def main():
    # Define relative paths for the project structure
    input_dir = "LoRA inputs"
    output_base_dir = "LoRA Weights"

    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Automatically discover all JSONL files in the input directory
    dataset_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    if not dataset_files:
        print(f"No .jsonl files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(dataset_files)} dataset(s) to train.")

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Configure QLoRA for 4-bit memory optimization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Define LoRA hyperparameters targeting all linear layers for complex reasoning
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Loop through each dataset and train a corresponding LoRA agent
    for dataset_path in dataset_files:
        file_name = os.path.basename(dataset_path)
        agent_name = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_base_dir, f"lora_{agent_name}")

        print(f"\n{'='*50}")
        print(f"Starting training for: {agent_name}")
        print(f"Input: {dataset_path}")
        print(f"Output will be saved to: {output_dir}")
        print(f"{'='*50}\n")

        # Initialize tokenizer with correct padding
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load the base model to the GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        # Load and format the training dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.map(format_instruction)

        # Set up the training arguments
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            num_train_epochs=3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none"
        )

        # Initialize and run the trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=2048,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        trainer.train()

        # Save the fine-tuned adapter weights and tokenizer
        print(f"\nSaving weights for {agent_name}...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Critical step: Clear GPU memory to prevent Out-Of-Memory errors during the next iteration
        del trainer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished {agent_name}. GPU memory cleared.")

    print("\nAll available agents have been trained successfully!")

if __name__ == "__main__":
    main()