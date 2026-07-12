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

    # Locate only the training files to drive the loop
    train_files = glob.glob(os.path.join(input_dir, "*_train.jsonl"))

    if not train_files:
        print(f"No '*_train.jsonl' files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(train_files)} agent(s) to train based on train files.")

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

    for train_path in train_files:
        # Extract the base agent name (e.g., 'agent1_motivation' from 'agent1_motivation_train.jsonl')
        file_name = os.path.basename(train_path)
        agent_name = file_name.replace('_train.jsonl', '')

        # Construct paths for validation and test files
        val_path = os.path.join(input_dir, f"{agent_name}_val.jsonl")
        test_path = os.path.join(input_dir, f"{agent_name}_test.jsonl")

        output_dir = os.path.join(output_base_dir, f"lora_{agent_name}")

        print(f"\n{'=' * 50}")
        print(f"Processing agent: {agent_name}")

        if not os.path.exists(val_path) or not os.path.exists(test_path):
            print(f"Skipping {agent_name} - Missing val or test file in '{input_dir}'.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load specific pre-split datasets
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        train_dataset = train_dataset.map(format_instruction)

        eval_dataset = load_dataset("json", data_files=val_path, split="train")
        eval_dataset = eval_dataset.map(format_instruction)

        print(f"Training on {len(train_dataset)} samples.")
        print(f"Validating on {len(eval_dataset)} samples.")
        print(f"Test file is ready at: {test_path} (Will be used for evaluation later)")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_strategy="epoch",
            eval_strategy="steps",
            eval_steps=50,
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
            eval_dataset=eval_dataset,
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