# IMPAQTS-PID

**IMPAQTS-PID** is a curated dataset and accompanying codebase designed to evaluate Large Language Models' (LLMs) ability to understand and explain implicit content, namely implicatures and presuppositions, in real-world use.

IMPAQTS-PID is a dataset extracted from [IMPAQTS: a multimodal corpus of parliamentary and other political speeches in Italy (1946-2023), annotated with implicit strategies](https://aclanthology.org/2024.parlaclarin-1.15/) (Cominetti et al., ParlaCLARIN 2024). 

This repository accompanies the paper *They want to pretend not to understand*: The Limits of Current LLMs in Interpreting Implicit Content of Political Discourse, presented at ACL 2025.
> https://aclanthology.org/2025.findings-acl.804/

---

## Data

Check the [data](https://github.com/WalterPaci/IMPAQTS-PID/tree/main/data) directory for the dataset description and usage.

It includes:

- **`IMPAQTS-PID.csv`**:  
  A dataset of 31,822 annotated passages from Italian political speeches. Each instance includes an expert-validated explanation of implicit content (implicature or presupposition).

- **`CoT_prompt.txt`**:  
  The Chain-of-Thought prompt used in our experiments, as described in the accompanying paper.

- **`fs_examples.txt`**:  
  A small curated set of examples for use in few-shot prompting within the open-ended generation (OEG) task.

---
## MCG_task script

`MCG_task.py` is **already heavily commented inline**; here what we give is a *bird’s‑eye* companion that walks readers through the key functions that drive them.

This script does the following:

1. **Loads** the *IMPAQTS‑PID* dataset.
2. **Prompts** a set of Large Language Models (LLMs) to do a MCG task where they have to pick one out of four possible options (A,B,C or D).
3. **Cleans & stores** every answer plus a single‑letter version for evaluation.
4. **Evaluates** each model using accuracy, precision, recall, F1. It can optionally print a bar chart of the results.
5. **Exports** both the detailed per‑row results and a metrics table in .tsv format.

## Most Relevant Functions

| Function                      | Purpose                                                                                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_dataframe`              | Reads `IMPAQTS-PID.txt` into a `DataFrame` (tab‑separated, blanks filled).                                                                                |
| `build_prompt`                | Crafts the Italian multiple‑choice prompt (asks for **A/B/C/D** only).                                                                                    |
| `clean_answer`                | Regex‑extracts `A–D`; anything else becomes `"X"`.                                                                                                        |
| `load_hf_model_and_tokenizer` | Downloads a HuggingFace model, patches PAD token, moves it to GPU/CPU, sets `eval()`.                                                                     |
| `generate_with_hf`            | Deterministic *greedy* decoding via `transformers.generate()`.                                                                                            |
| `call_openai_gpt`             | Calls **gpt‑4o‑mini** through the Chat Completions API (temperature 0, system prompt in Italian).                                                         |
| `generate_for_row`            | Routing hub → uses OpenAI for `gpt‑4o‑mini`, else local HF model. Returns only new text.                                                                  |
| `evaluate_results`            | Compares cleaned predictions to gold labels (`right_answer_id`). Computes **accuracy, precision, recall, F1** (macro by default) and can save a bar‑plot. |

*(Each function is documented inline in the code for deeper dives.)*

---

## OEG_task script

`OEG_task.py` is **already heavily commented inline**; here what we give is a *bird’s‑eye* companion that walks readers through the key functions that drive them.

This script is designed to:

* Load IMPAQTS-PID.
* Generate model outputs using **three prompting strategies**:

  * **Zero-shot (ZS)**: A simple instruction with no prior examples built in the prompt.
  * **Few-shot (FS)**: 4-shot based prompt dynamically built from `fs_example.txt`.
  * **Chain-of-Thought (CoT)**: Uses a prompt that encourages intermediate reasoning steps, foaded from `CoT_prompt.txt`.

## Most Relevant Functions

| Function                      | Purpose                                                                                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_dataframe`              | Reads `IMPAQTS-PID.txt` into a `DataFrame`.                                                                                                               |
| `load_CoT_prompt`             | Reads `CoT_prompt.txt` and returns the Chain-of-Thought (CoT) prompt for model use                                                                        |
| `sample_fs_examples`          | Samples a specified number of examples (default 2) from `fs_examples.txt`, filtering implicatures and presuppositions.                                    |
| `load_hf_model_and_tokenizer` | Downloads a HuggingFace model, patches PAD token, moves it to GPU/CPU, sets `eval()`.                                                                     |
| `generate_with_hf`            | Deterministic *greedy* decoding via `transformers.generate()`.                                                                                            |
| `call_openai_gpt`             | Calls **gpt‑4o‑mini** through the Chat Completions API (temperature 0, system prompt in Italian).                                                         |
| `generate_for_row`            | Routing hub → uses OpenAI for `gpt‑4o‑mini`, else local HF model. Returns only new text.                                                                  |


*(Each function is documented inline in the code for deeper dives.)*

---

## 🛠 Installation & Requirements

Make sure to have Python 3.8+ and install the following libraries:

```bash
pip install pandas torch transformers openai scikit-learn matplotlib
```

For optimal execution time when prompting HF models, install a CUDA-enabled `torch` version.

---

## How to Use

To run both scripts clone this repo and run:

```bash
python MCG_task.py
```

or

```bash
python OEG_task.py
```

After the execution of `MCG_task.py` three new files are created in the `results` directory:

* `results/MCQ_results.csv` — full per‑instance predictions.
* `results/MCG_eval.csv`    — aggregated metrics.
* `results/MCG_eval_plot.png` (optional) — visual comparison of model scores (useful for presentations!).

After the execution of `OEG_task.py` one new file is created in the `results` directory:

* `results/OEQ_results.csv` — full per‑instance implicit content explanations.
---

## Authors & Affiliations

* **Walter Paci** – University of Florence
* **Alessandro Panunzi** – University of Florence 
* **Sandro Pezzelle** – University of Amsterdam

---

## 📄 Citation

If you cite this paper in your work or if you use IMPAQTS-PID, please cite it as follows:

```bibTex
@inproceedings{paci-etal-2025-want,
    title = "They want to pretend not to understand: The Limits of Current {LLM}s in Interpreting Implicit Content of Political Discourse",
    author = "Paci, Walter  and
      Panunzi, Alessandro  and
      Pezzelle, Sandro",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.804/",
    pages = "15569--15593",
    ISBN = "979-8-89176-256-5",
    abstract = "Implicit content plays a crucial role in political discourse, where systematically employ pragmatic strategies such as implicatures and presuppositions to influence their audiences. Large Language Models (LLMs) have demonstrated strong performance in tasks requiring complex semantic and pragmatic understanding, highlighting their potential for detecting and explaining the meaning of implicit content. However, their ability to do this within political discourse remains largely underexplored. Leveraging, for the very first time, the large IMPAQTS corpus comprising transcribed Italian political speeches with expert annotations of various types of implicit content, we propose methods to test the effectiveness of LLMs in this challenging problem. Through a multiple-choice task and an open-ended generation task, we demonstrate that all tested models struggle to interpret presuppositions and implicatures. To illustrate, the best-performing model provides a fully correct explanation in only one-fourth of cases in the open-ended generation setup. We conclude that current LLMs lack the key pragmatic capabilities necessary for accurately interpreting highly implicit language, such as that found in political discourse. At the same time, we highlight promising trends and future directions for enhancing model performance. We release our data and code at: $\url{https://github.com/WalterPaci/IMPAQTS-PID}$"
}
```

---

## 🪪 License

* **Code & Prompts**: Released under the [MIT License](LICENSE)
* **Dataset**: Released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.
  This means you may share and adapt the dataset for non-commercial purposes, with attribution.

---

## 📬 Contact

For questions or collaborations, please contact the authors via GitHub.

---
