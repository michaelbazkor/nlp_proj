import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score


def extract_json(text):
    """
    Safely extract JSON object from the model's text output
    Handles cases where the model wraps the output in markdown code blocks
    """
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx + 1])
    except Exception:
        pass
    return None


def format_prompt(system_prompt, user_prompt):
    """
    Wraps the system and user prompts into the Llama 3.1 instruct format
    """
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt.strip()}<|eot_id|>\n"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_prompt.strip()}<|eot_id|>\n"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens nested dictionaries so we can compare individual metrics easily
    Example: {"Data": {"BFI_O": 8}} -> {"Data_BFI_O": 8}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    weights_dir = "LoRA Weights"
    base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    test_files = glob.glob(os.path.join(weights_dir, "test_data_*.jsonl"))

    if not test_files:
        print("No test dataset files found. Please ensure training completed successfully.")
        return

    print(f"Loading Base Model in 4-bit: {base_model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    for test_file in test_files:
        filename = os.path.basename(test_file)
        agent_name = filename.replace("test_data_", "").replace(".jsonl", "")
        adapter_path = os.path.join(weights_dir, f"lora_{agent_name}")

        if not os.path.exists(adapter_path):
            print(f"Skipping {agent_name}: Adapter weights not found at {adapter_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating Agent: {agent_name}")
        print(f"{'=' * 60}")

        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        ground_truths = []
        predictions = []

        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Running inference on {len(lines)} test samples...")

        for line in tqdm(lines, desc=f"Predicting {agent_name}"):
            sample = json.loads(line)

            sys_prompt = sample['system']
            user_prompt = sample['user']
            true_assistant = json.loads(sample['assistant'])

            prompt = format_prompt(sys_prompt, user_prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            pred_json = extract_json(response)

            if pred_json and "Data" in pred_json:
                ground_truths.append(flatten_dict(true_assistant["Data"]))
                predictions.append(flatten_dict(pred_json["Data"]))
            else:
                print("\nWarning: Model generated invalid JSON or missed the 'Data' key. Skipping sample.")

        if not predictions:
            print("No valid predictions extracted. Moving to next agent.")
            continue

        print(f"\n--- Evaluation Results for {agent_name} ---")

        keys_to_evaluate = ground_truths[0].keys()

        for key in keys_to_evaluate:
            y_true = []
            y_pred = []

            for t, p in zip(ground_truths, predictions):
                if key in t and key in p:
                    try:
                        y_true.append(float(t[key]))
                        y_pred.append(float(p[key]))
                    except (ValueError, TypeError):
                        pass

            if not y_true:
                continue

            unique_values = set(y_true)

            print(f"\nMetric: {key}")

            if len(unique_values) == 2 and {0.0, 1.0}.issubset(unique_values):
                # Binary Classification Metrics
                y_pred_binary = [1.0 if val >= 0.5 else 0.0 for val in y_pred]
                acc = accuracy_score(y_true, y_pred_binary)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                print(f"  Type: Binary Classification")
                print(f"  Accuracy: {acc:.4f}")
                print(f"  F1 Score: {f1:.4f}")
            else:
                # Regression Metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                print(f"  Type: Continuous / Ordinal Regression")
                print(f"  MAE (Mean Absolute Error): {mae:.4f}")
                print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")

        # Clean up memory before loading the next adapter
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()