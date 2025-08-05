
# Liquid-AI Code Aug-Pipeline

A **one-click Colab** turns raw, multi-lingual code into a polished, *Python-only* instruction dataset that is larger, cleaner, and more diverse—ready for fine-tuning code LLMs or repair tasks.

---

## 🛠 How to run (in 3 steps)

1. **Generate a Hugging Face token**  
   *Go to: Profile → Settings → Access Tokens → New token (read+write).*
   → This token is needed to securily download or upload datasets via Colab or scripts.

3. **Open the Colab**  
   👉 [Colab Notebook](https://colab.research.google.com/drive/18s1aTgJQK5DuKrC8wSZRnPtf9_t89X_K?usp=sharing)
   *Make sure to set the runtime type to GPU*

5. **Run all cells**  
   *You’ll enter your token once; the rest runs automatically.*

```

ingest → filter → augment\_dataset → postprocess → upload\_to\_hf

```

📦 Final dataset uploaded to:  
[https://huggingface.co/datasets/Inessssa/liquidai-aug-eb50bee8](https://huggingface.co/datasets/Inessssa/liquidai-aug-eb50bee8) (private)  
⏱️ Total pipeline runtime with Colab's free T4 GPU: **1:26:50**

🔓 The raw dataset is also hosted on Hugging Face to ensure reproducibility and transparency, useful for anyone looking to re-run the pipeline from scratch or compare filtered vs. augmented results.


## 🔍 Filtering: keep only the good stuff

| Step                          | Rule                                                                 | Why                                |
|-------------------------------|----------------------------------------------------------------------|-------------------------------------|
| **Exact + SimHash dedup**     | Drop rows that are byte-for-byte or *h ≤ 3* similar                  | Prevent repetition & model bias     |
| **Low-info outputs**          | Drop literals, lone identifiers, or pure-comment blobs              | Not useful for model learning       |
| **Language-aware invalid keep** | Keep non-compiling rows *only* if task mentions a non-Python language | Useful for code translation tasks   |
| **Compile check**             | Keep only rows whose `output` passes `compile()`                    | Guarantees syntactic validity       |

---

## 🧪 Augmentation: four simple techniques

1. **Translate** every *invalid* snippet to Python 3 (`DeepSeek-Coder`)
2. **Refactor** every *valid* snippet (rename vars, reformat)
3. **Bug-inject** 10% of rows (e.g., off-by-one, logic flip) + fix-it prompt
4. **Paraphrase** 50% of instructions (via LLM self-instruct style)

> All outputs are re-validated via `compile()` before inclusion.

---

## 🤖 Model choice: `DeepSeek-Coder-1.3B-Instruct`

- **Why this one?**  
  Small enough to run in BF16 on a free Colab T4, yet expressive enough for code rewriting and translation.

- **Performance tip:**  
  A single global pipeline instance is reused across augmentation steps to avoid slow re-initialization.

---

## ⏩ Future speed-ups

| Bottleneck                     | Fix                                                               |
|-------------------------------|--------------------------------------------------------------------|
| `pipeline(...)` called per row | Batch prompts via `pipeline(..., batch_size=XX)` + prompt lists  |
| Single-GPU generation         | Use multiprocessing / CPU offload for prompt shards               |
| Compile checks are serial     | Use `multiprocessing.Pool` for parallel compile checks            |
| Repeated LLM outputs          | Cache deterministic prompts with `joblib.Memory`                  |
| Loop + `df.append()`          | Build lists, then concat once                                     |
| Notebook overhead             | Extract to `run_pipeline.py` CLI for pure Python execution        |

> A batching refactor alone (~128 prompts per call) speeds things up **4–5×** on T4 GPUs.

---

## 🗝 Key decisions

- **Compile gate everywhere** → guarantees only runnable code survives.
- **Language-aware heuristics** → salvage useful non-Python instructions for translation.
- **Natural phrasing variants** → swapped in 2/3 of translation outputs to increase diversity.
- **Minimal postprocessing** → drops `input == output`, exact dupes, and failed compiles only.
- **Private HF upload** → temp folder, no local artefacts, token read from `env`.

---

✨ PRs welcome!
