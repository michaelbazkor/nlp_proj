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
    input_dir = "LoRA inputs"
    output_base_dir = "LoRA Weights"

    os.makedirs(output_base_dir, exist_ok=True)

    dataset_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    if not dataset_files:
        print(f"No .jsonl files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(dataset_files)} dataset(s) to process.")

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    for dataset_path in dataset_files:
        file_name = os.path.basename(dataset_path)
        agent_name = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_base_dir, f"lora_{agent_name}")

        print(f"\n{'=' * 50}")
        print(f"Processing agent: {agent_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load the full dataset
        raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
        raw_dataset = raw_dataset.map(format_instruction)

        # Create a Train / Test split (90% training, 10% validation/testing)
        split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        # Save the test dataset for post-training inference evaluation
        test_file_path = os.path.join(output_base_dir, f"test_data_{agent_name}.jsonl")
        eval_dataset.to_json(test_file_path)
        print(f"Saved {len(eval_dataset)} test samples to: {test_file_path}")
        print(f"Training on {len(train_dataset)} samples.")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        # Configure Training Arguments with Evaluation enabled
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_strategy="epoch",
            eval_strategy="steps",
            eval_steps=50,  # Evaluates and prints loss every 50 steps
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

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Passing the validation set here
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=2048,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        print("\nStarting Training with Evaluation Tracking...")
        trainer.train()

        print(f"\nSaving weights for {agent_name}...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        del trainer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished {agent_name}. GPU memory cleared.")

    print("\nAll available agents have been trained successfully!")


if __name__ == "__main__":
    main()