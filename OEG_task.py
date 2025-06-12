import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from openai import OpenAI

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

API_KEY = "YOUR_OPENAI_API_KEY"   # ← Make sure to replace with your own key
DATA_PATH = Path("/data/IMPAQTS-PID.txt")
COT_PROMPT_PATH = Path("/data/CoT_prompt.txt")
FS_EXAMPLES_PATH = Path("/data/fs_examples.txt")
OUTPUT_DIR = Path("/results/")  # ← Set your output directory
JUDGE_INSTRUCTION = Path("/home/walter/IMPAQTS-PID/data/oeg_eval_instruct.txt")

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
# 1. Load_dataframe → Loads the IMPAQTS-PIDA Dataset as a pandas DataFrame. 
# 2. load_CoT_prompt → Loads the Chain-of-Thought prompt from a text file.
# 3. sample_fs_examples → Samples examples from the fs_examples DataFrame.
# 4. build_fs_promt → Builds the few-shot prompt by interleaving two examples of both implicatures and presuppositions randomly extracted from fs_examples.
# 5. call_openai_gpt → Calls the OpenAI API to get a completion from gpt-4o-mini.
# 6. load_hf_model_and_tokenizer → Loads a HuggingFace model and tokenizer onto the specified device (GPU or CPU).
# 7. generate_with_hf → Prompts LLMs from HuggingFace using the transformers library.
# 8. generate_for_row → Decides if the generation should be done via OpenAI API or the transformers library checking "model_name".
# 9. GPT4o_as_Judge → Evaluate GPT4o-mini outputs using GPT-4o as judge
#10. judgment_eval → Analyze GPT-4o evaluations and create comparative plot with percentages

def load_dataframe(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path, sep="\t").fillna("")
    return df


def load_CoT_prompt(path: Path) -> str:

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sample_fs_examples(examples_df: pd.DataFrame, category: str, n_samples: int = 2) -> pd.DataFrame:

    subset = examples_df[examples_df["impl"] == category]
    return subset.sample(n=n_samples, replace=False)


def build_fs_prompt(examples_df: pd.DataFrame) -> str:

    impl_samples = sample_fs_examples(examples_df, "impl", n_samples=2)
    ppp_samples = sample_fs_examples(examples_df, "ppp", n_samples=2)

    lines: List[str] = []
    for (row_impl, row_ppp) in zip(impl_samples.itertuples(), ppp_samples.itertuples()):
        lines.append(f"Testo: {row_impl.sentence}\nContenuto Implicito: {row_impl.implicit_content}\n")
        lines.append(f"Testo: {row_ppp.sentence}\nContenuto Implicito: {row_ppp.implicit_content}\n")

    return "\n".join(lines)


def call_openai_gpt(openai_client: OpenAI, prompt_text: str, max_tokens: int = 500) -> str:

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei un assistente virtuale ben capace di comprendere i testi che ti vengono proposti."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def load_hf_model_and_tokenizer(model_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
        
    model.to(device)
    model.eval()
        
    return tokenizer, model
        
def generate_with_hf(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = None,
) -> str:

    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # For zero temperature, we disable sampling completely
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Force deterministic output when temperature=0
        num_beams=1,     # Use greedy decoding
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_config,
        )

    out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return out_text


def generate_for_row(
    base_prompt_text: str,
    content_text: str,
    model_name: str,
    openai_client: OpenAI,
    is_cot: bool,
    hf_tokenizer=None,
    hf_model=None,
    cot_token_limit: int = 1000,
    default_token_limit: int = 500,
) -> str:

    full_prompt = f"{base_prompt_text}Testo: {content_text}\nContenuto Implicito: "
    token_limit = cot_token_limit if is_cot else default_token_limit

    if model_name == "gpt-4o-mini":
        return call_openai_gpt(openai_client, full_prompt, max_tokens=token_limit)
    else:
        # We assume hf_tokenizer and hf_model are not None
        generated_full = generate_with_hf(
            hf_tokenizer,
            hf_model,
            full_prompt,
            max_new_tokens=token_limit,
            temperature=0.0,
            top_p=1.0,
            device=next(hf_model.parameters()).device,
        )
        # The model returns the full text (prompt + generated). We extract only the generated text:
        continuation = generated_full[len(full_prompt) :]
        return continuation.strip()

def GPT4o_as_Judge(df: pd.DataFrame) -> pd.DataFrame:

    with open(JUDGE_INSTRUCTION, 'r', encoding='utf-8') as file:
        instruct = file.read()
    
    zs_eval = []
    fs_eval = []
    CoT_eval = []
    
    openai_client = OpenAI(api_key=API_KEY)
    
    for idx, row in df.iterrows():
        text = row['text']
        zs_output = row['zs_gpt-4o-mini_output']
        fs_output = row['fs_gpt-4o-mini_output']
        CoT_output = row['CoT_gpt-4o-mini_output']
        
        prompts = {
            'zs': f"{instruct}Testo: {text}\n\nOutput: {zs_output}\n\nValutazione: ",
            'fs': f"{instruct}Testo: {text}\n\nOutput: {fs_output}\n\nValutazione: ",
            'CoT': f"{instruct}Testo: {text}\n\nOutput: {CoT_output}\n\nValutazione: "
        }
        
        for prompt_type, prompt in prompts.items():
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Sei un assistente virtuale ben capace di comprendere i testi che ti vengono proposti."},
                    {"role": "user", "content": prompt}
                ]
            )
            eval_result = completion.choices[0].message.content.strip()
            
            if prompt_type == 'zs':
                zs_eval.append(eval_result)
            elif prompt_type == 'fs':
                fs_eval.append(eval_result)
            else:  # CoT
                CoT_eval.append(eval_result)
        
        print(f"Row {idx} evaluated → ZS: {zs_eval[-1]} | FS: {fs_eval[-1]} | CoT: {CoT_eval[-1]}")
    
    df['zs_evaluation'] = zs_eval
    df['fs_evaluation'] = fs_eval
    df['CoT_evaluation'] = CoT_eval
    
    return df

def judgment_eval(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], plt.Figure]:

    categories = [
        "Totalmente corretto",
        "Corretto tra opzioni",
        "Parzialmente corretto",
        "Totalmente sbagliato",
        "Risposta non fornita"
    ]
    
      
    colors = [
        '#98FB98',  
        '#FFF59D',  
        '#FFCC80',  
        '#FFAB91',  
        '#E0E0E0'   
    ]

    eval_columns = ['zs_evaluation', 'fs_evaluation', 'CoT_evaluation']
    prompt_types = ['Zero-shot', 'Few-shot', 'Chain-of-Thought']
    
    results = {prompt: {cat: 0.0 for cat in categories} for prompt in prompt_types}
    
    for col, prompt in zip(eval_columns, prompt_types):
        total_responses = len(df[col].dropna())
        for category in categories:
            count = df[col].str.contains(category, case=False, na=False).sum()
            percentage = (count / total_responses * 100) if total_responses > 0 else 0
            results[prompt][category] = percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.15
    x = np.arange(len(prompt_types))
    
    for i, (category, color) in enumerate(zip(categories, colors)):
        percentages = [results[prompt][category] for prompt in prompt_types]
        position = x + (i - len(categories)/2 + 0.5) * width
        bars = ax.bar(position, percentages, width, label=category, color=color, alpha=0.85)
        

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    

    ax.set_ylabel('Percentage of Responses (%)')
    ax.set_title('GPT4o-mini performance by Prompt Type', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['0-shot', 'Few-shot', 'CoT'])
    

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             title='Annotation Categories', 
             fontsize='small', frameon=False)
    
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

    plt.tight_layout()
    
    return results, fig

# ─── MAIN PROCESSING ──────────────────────────────────────────────────────────

def main() -> None:

    data_df = load_dataframe(DATA_PATH)
    fs_examples_df = load_dataframe(FS_EXAMPLES_PATH)
    CoT_prompt = load_CoT_prompt(COT_PROMPT_PATH)

    zs_prompt = (
        "Esplicita il contenuto implicito del seguente testo. "
        "Considera che esso compare sempre nel periodo più a destra del testo fornito.\n\n"
    )
    base_prompts: Dict[str, str] = {
        "zs": zs_prompt,
        "fs": None,             # Needs to be built inside the row-loop
        "CoT": CoT_prompt,
    }

    models = ["CohereLabs/aya-expanse-8b",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-3B",
            "gpt-4o-mini",
    ]

    openai_client = OpenAI(api_key=API_KEY)

    #  ─── OUTER LOOP: Over each prompt type ────────────────────────────────────
    for prompt_name, prompt_text in base_prompts.items():
        all_model_outputs: Dict[str, List[str]] = {}

        # ─── MIDDLE LOOP: Over each model ─────────────────────────────────────
        for model_name in models:
            outputs: List[str] = []

            # If this is a non-OpenAI model, load it onto GPU (or CPU if no CUDA)
            hf_tokenizer = hf_model = None
            if model_name != "gpt-4o-mini":
                # Load HF model/tokenizer
                print(f"Loading HF model '{model_name.split("/")[0]}' onto GPU...\n")
                hf_tokenizer, hf_model = load_hf_model_and_tokenizer(model_name)
                device_str = next(hf_model.parameters()).device
                print(f" → Loaded on {device_str}.\n")
            else:
                print("Calling API for 'gpt-4o-mini'...\n")

            # ─── INNER LOOP: Over each row in data_df ───────────────────────────
            for idx, row in data_df.iterrows():
                content = row["text"]
                is_cot = (prompt_name == "CoT")

                # Build the base prompt
                if prompt_name == "fs":
                    fs_prompt = build_fs_prompt(fs_examples_df)
                    generated = generate_for_row(
                        base_prompt_text=fs_prompt,
                        content_text=content,
                        model_name=model_name,
                        openai_client=openai_client,
                        is_cot=False,
                        hf_tokenizer=hf_tokenizer,
                        hf_model=hf_model,
                    )
                else:
                    # "zs" or "CoT"
                    generated = generate_for_row(
                        base_prompt_text=prompt_text,
                        content_text=content,
                        model_name=model_name,
                        openai_client=openai_client,
                        is_cot=is_cot,
                        hf_tokenizer=hf_tokenizer,
                        hf_model=hf_model,
                    )

                print(f"[{prompt_name.upper()}|{model_name.split("/")[0]}] Row {idx} → {generated!r}\n")
                outputs.append(generated)

            # After finishing all rows for this model, unload it if it was HF to free GPU memory
            if hf_model is not None:
                del hf_model
            if hf_tokenizer is not None:
                del hf_tokenizer
            torch.cuda.empty_cache()  

            # Store the list of outputs in our dictionary, keyed by column name
            col_key = f"{prompt_name}_{model_name.split("/")[0]}_output"
            all_model_outputs[col_key] = outputs

        # ─── After all models are done for this prompt: add columns & save ───
        for colname, coldata in all_model_outputs.items():
            data_df[colname] = coldata

    # Save to CSV all models outputs for all prompts
    out_path = OUTPUT_DIR / f"OEG_results.csv"
    data_df.to_csv(out_path, index=False)
    print("All models were successfully tested with all prompts.")

    # Run GPT-4o evaluation on GPT4o-mini outputs
    print("\nStarting GPT-4o evaluation of GPT4o-mini...")
    evaluated_df = GPT4o_as_Judge(data_df)
    
    # Save final results with evaluations
    eval_out_path = OUTPUT_DIR / f"GPT4o-mini_OEG_results_evaluated_by_GPT4o.csv"
    evaluated_df.to_csv(eval_out_path, index=False)
    print("Evaluation completed and results saved.")

    # Run evaluation analysis and create plot
    stats, fig = judgment_eval(evaluated_df)
    
    # Print overall statistics summary
    print("\nOverall Statistics Summary:")
    print("=" * 50)
    for prompt_type, categories in stats.items():
        print(f"\n{prompt_type}:")
        print("-" * 30)
        for category, percentage in categories.items():
            print(f"{category}: {percentage:.1f}%")
    
    # Save plot
    plot_path = OUTPUT_DIR / "GPT4o-mini_OEG_results_evaluated_by_GPT4o.png"
    fig.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
