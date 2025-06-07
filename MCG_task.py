import json
import random
from pathlib import Path
from typing import List, Dict, Any
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from openai import OpenAI

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

API_KEY = "YOUR_OPENAI_API_KEY" 
DATA_PATH = Path("/data/IMPAQTS-PID.txt")
OUTPUT_DIR = Path("/results/")  # Set your output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_TOKENS = 25  # keep small to avoid unnecessary costs and speed up inference

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
# 1. Load_dataframe → Loads the IMPAQTS-PIDA Dataset as a pandas DataFrame. 
# 2. build_prompt → Builds the prompt for the model
# 3. clean_answer → Use to format the model's answer in a consistent way for evaluation
# 4. load_hf_model_and_tokenizer → Loads a HuggingFace model and tokenizer onto the specified device (GPU or CPU).
# 5. generate_with_hf → Prompts LLMs from HuggingFace using the transformers library.
# 6. call_openai_gpt → Calls the OpenAI API to get a completion from gpt-4o-mini.
# 7 generate_for_row → Decides if the generation should be done via OpenAI API or the transformers library checking "model_name".

def load_dataframe(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path, sep="\t").fillna("")
    return df

def build_prompt(text: str, choices: str) -> str:
    
    return (
        f"Considera il seguente testo: {text}\n"
        "Quale tra le seguenti opzioni credi che meglio esprima il suo contenuto implicito? "
        "Nota che questo compare nel periodo più a destra del testo. "
        'Rispondi solo con "A", "B", "C" o "D".\n\n'
        f"{choices}"
    )


def clean_answer(generate: str) -> str:
    
    generate = generate.strip()

    match = re.search(r"\b([A-D])(?:\.|\b)(?![a-zA-Z])", generate)
    if match:
        return match.group(1)
    return "X"


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

def generate_for_row(
    prompt_text: str,
    model_name: str,
    openai_client: OpenAI,
    hf_tokenizer=None,
    hf_model=None,
    token_limit: int = MAX_TOKENS,
) -> str:

    if model_name == "gpt-4o-mini":
        return call_openai_gpt(openai_client, prompt_text, max_tokens = MAX_TOKENS)
    else:
        # We assume hf_tokenizer and hf_model are not None
        generated_full = generate_with_hf(
            hf_tokenizer,
            hf_model,
            prompt_text,
            max_new_tokens=token_limit,
            temperature=0.0,
            top_p=1.0,
            device=next(hf_model.parameters()).device,
        )
        # The model returns the full text (prompt + generated). We extract only the generated text:
        continuation = generated_full[len(prompt_text) :]
        return continuation.strip()


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────


def main() -> None:
    data_df = load_dataframe(DATA_PATH)

    models = ["CohereLabs/aya-expanse-8b",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-3B",
            "gpt-4o-mini",
    ]

    openai_client = OpenAI(api_key=API_KEY)

    for model_name in models:
        models_outputs: Dict[str, List[str]] = {}
        models_clean_outputs: Dict[str, List[str]] = {}

        answers: List[str] = []
        clean_answers: List[str] = []

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
        

        for idx, row in data_df.iterrows():
            prompt = build_prompt(row["text"], row["MCs"])

            generated = generate_for_row(
                    prompt_text=prompt,
                    model_name=model_name,
                    openai_client=openai_client,
                    hf_tokenizer=hf_tokenizer,
                    hf_model=hf_model
            )
            print(f"[{model_name.split("/")[0]}] Row {idx} → {generated!r}\n \t\t→ Simplified: {clean_answer(generated)!r}")
            answers.append(generated)
            clean_answers.append(clean_answer(generated))

        # After finishing all rows for this model, unload it if it was HF to free GPU memory
        if hf_model is not None:
            del hf_model
        if hf_tokenizer is not None:
            del hf_tokenizer
        torch.cuda.empty_cache()  


        col_key = f"{model_name.split("/")[0]}_choice"
        col_key_clean = f"{col_key}_clean"
        models_outputs[col_key] = answers
        models_clean_outputs[col_key_clean] = clean_answers

        data_df[col_key] = answers
        data_df[col_key_clean] = clean_answers



    out_file = OUTPUT_DIR / f"MCQ_results.xlsx"
    data_df.to_csv(out_file, index=False)
    print(f"Saved at {out_file}: all models processed!")


if __name__ == "__main__":
    main()
