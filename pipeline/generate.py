"""
generate.py
Author : Ines Salem  •  salem.iness00@gmail.com
~~~~~~~~~~~
Synthetic-data augmentation for LiquidAI take-home.

Pipeline
1. Translate INVALID → Python 3
2. Refactor VALID Python
3. Combine + Bug-injection (≈10 %)
4. Paraphrase 50 % of instructions (append rows)
5. Return augmented dataframe

Usage
-----
from generate import augment_dataset
aug_df = augment_dataset(valid_df, invalid_df)
"""

from __future__ import annotations

import random
import re
from typing import Iterator

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from pipeline.ingest import compile_check, find_language

# ─────────────────────────── Config (globals) ──────────────────────────
MODEL_CKPT: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
MAX_NEW: int = 512
BATCH_SIZE: int = 4
BUG_FRACTION: float = 0.10            # 10 % rows get a synthetic bug
PARAPHRASE_FRACTION: float = 0.50     # 50 % instructions paraphrased
SEED_DEFAULT: int = 42

# Human-written instruction variants for translation rows
_TRANSLATE_VARIANTS: list[str] = [
    "Rewrite this code to Python 3.", "Convert this code to Python 3.",
    "Translate this code into Python 3.", "Port this code to Python 3.",
    "Transform this code into Python 3.", "Adapt this code to Python 3.",
    "Refactor this code in Python 3.", "Express this logic in Python 3.",
    "Implement this program in Python 3.", "Update this code to Python 3.",
    "Migrate this code to Python 3.", "Recreate this code using Python 3.",
    "Modify this code to use Python 3 syntax.", "Reimplement this code in Python 3.",
    "Write the Python 3 version of this code.",
]

# Bug-fix instruction pool
_BUG_INSTRUCTIONS: list[str] = [
    "Fix the error in this Python code.", "Correct the bug in this function.",
    "Identify and fix the issue in this code.", "Debug the following Python 3 snippet.",
    "Repair the broken logic in this code.", "Patch the bug in the code below.",
    "Resolve the bug in this Python function.", "Correct the mistake in this implementation.",
    "Make this Python code work as intended.", "Troubleshoot and fix this code.",
]

# ──────────────────────── Lazy HF pipeline (global) ────────────────────
_TOKENIZER: AutoTokenizer | None = None
_GENERATOR = None


def _lazy_pipe():
    global _TOKENIZER, _GENERATOR
    if _GENERATOR is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CKPT, trust_remote_code=True)
        _TOKENIZER.padding_side = "left"
        _TOKENIZER.pad_token = _TOKENIZER.pad_token or _TOKENIZER.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CKPT,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _GENERATOR = pipeline(
            "text-generation",
            model=model,
            tokenizer=_TOKENIZER,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=MAX_NEW,
            do_sample=False,
            return_full_text=False,
        )
    return _GENERATOR


# ───────────────────────────── Utilities ───────────────────────────────
def _extract_code(txt: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S)
    return m.group(1).strip() if m else txt.strip()


def _batched(seq, n) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _run_llm(prompts: list[str]) -> list[str]:
    pipe = _lazy_pipe()
    tokenizer = _TOKENIZER
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]
    outs = pipe(chat_prompts)
    return [o[0]["generated_text"] if o and isinstance(o[0], dict) else "" for o in outs]


# ───────────────────── Bug-injection helpers (globals) ──────────────────
def _invert_condition(c: str) -> str:
    return re.sub(
        r"\b(if\s+.*?)(==|!=|<=|>=|<|>)(.*?)\b",
        lambda m: f"{m[1]}{'!=' if m[2]=='==' else '==' if m[2]=='!=' else '<=' if m[2]=='>' else '>=' if m[2]=='<' else '<' if m[2]=='>=' else '>'}{m[3]}",
        c,
        count=1,
    )


def _off_by_one(c):           return re.sub(r"range\(([^)]+)\)", r"range(\1 - 1)", c, count=1)
def _drop_initialization(c):  return re.sub(r"\b\w+\s*=\s*\d+\s*\n", "", c, count=1)


def _insert_early_return(c):
    lines = c.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("for ", "while ")):
            indent = re.match(r"\s*", lines[i + 1]).group() if i + 1 < len(lines) else ""
            lines.insert(i + 1, f"{indent}return")
            break
    return "\n".join(lines)


def _swap_params(c):          return re.sub(r"range\((\s*\d+),(\s*\d+)\)", r"range(\2,\1)", c, count=1)
def _remove_function_call(c): return re.sub(r"\b\w+\s*\([^)]*\)", "", c, count=1)
def _inject_syntax_bug(c):    return c.replace(":", "", 1)
def _swap_assignment(c):      return re.sub(r"(\w+),\s*(\w+)\s*=\s*(\w+),\s*(\w+)", r"\1 = \3\n\2 = \4", c, count=1)
def _wrong_builtin(c):        return re.sub(r"\bsum\b", "max", c, count=1)
def _remove_return(c):        return re.sub(r"\breturn\b.*", "", c, count=1)

_BUG_FUNCS = [
    _invert_condition, _off_by_one, _drop_initialization, _insert_early_return,
    _swap_params, _remove_function_call, _inject_syntax_bug,
    _swap_assignment, _wrong_builtin, _remove_return,
]

# ======================================================================
#                            PUBLIC API
# ======================================================================
def augment_dataset(
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    bug_fraction: float = BUG_FRACTION,
    paraphrase_fraction: float = PARAPHRASE_FRACTION,
    seed: int = SEED_DEFAULT,
) -> pd.DataFrame:
    """Run full augmentation pipeline and return the combined dataframe."""
    random.seed(seed)

    # ── 1. TRANSLATE INVALID rows ────────────────────────────────────
    trs_rows = []
    for idx_batch in _batched(invalid_df.index.tolist(), BATCH_SIZE):
        prompts, meta = [], []
        for idx in idx_batch:
            row = invalid_df.loc[idx]
            lang = (find_language(row["instruction"]) or "unknown").capitalize()
            prompt = f"Rewrite this {lang} code to Python 3:\n{row['output']}"
            prompts.append(prompt)
            meta.append((idx, prompt.split(":\n")[0]))

        for (idx, stub), raw in zip(meta, _run_llm(prompts)):
            code_py = _extract_code(raw)
            if compile_check(code_py):
                trs_rows.append(
                    {"instruction": stub, "input": invalid_df.loc[idx, "output"], "output": code_py}
                )

    df_translated = pd.DataFrame(trs_rows)
    for idx in df_translated.sample(frac=2 / 3, random_state=seed).index:
        df_translated.at[idx, "instruction"] = random.choice(_TRANSLATE_VARIANTS)

    # ── 2. REFACTOR VALID rows ───────────────────────────────────────
    ref_rows = []
    for idx_batch in _batched(valid_df.index.tolist(), BATCH_SIZE):
        prompts, id_map = [], []
        for idx in idx_batch:
            code = valid_df.loc[idx, "output"]
            prompts.append(
                "Rewrite this Python 3 function to behave the same but with "
                f"different variable names or formatting:\n{code}"
            )
            id_map.append(idx)

        for idx, raw in zip(id_map, _run_llm(prompts)):
            code_new = _extract_code(raw)
            if compile_check(code_new):
                ref_rows.append(
                    {"instruction": valid_df.loc[idx, "instruction"],
                     "input": valid_df.loc[idx, "input"],
                     "output": code_new}
                )

    df_refactored = pd.DataFrame(ref_rows)

    # ── 3. COMBINE & BUG-inject ──────────────────────────────────────
    combined = pd.concat([valid_df, df_translated, df_refactored],
                         ignore_index=True).reset_index(drop=True)

    bug_rows = []
    for i in combined.sample(frac=bug_fraction, random_state=seed).index:
        orig_code = combined.loc[i, "output"]
        buggy_code = random.choice(_BUG_FUNCS)(orig_code)
        if buggy_code != orig_code:
            bug_rows.append(
                {"instruction": random.choice(_BUG_INSTRUCTIONS),
                 "input": buggy_code,
                 "output": orig_code}
            )

    combined = pd.concat([combined, pd.DataFrame(bug_rows)],
                         ignore_index=True).reset_index(drop=True)

    # ── 4. PARAPHRASE subset ─────────────────────────────────────────
    mask = combined.sample(frac=paraphrase_fraction, random_state=seed).index
    para_prompts = [
        "Rewrite this programming instruction with **the same intent** using "
        f"different wording (one variant only):\n\"\"\"\n{combined.loc[i, 'instruction']}\n\"\"\""
        for i in mask
    ]
    paraphrased = _run_llm(para_prompts)

    combined = pd.concat(
        [combined,
         pd.DataFrame([{
             "instruction": t.strip().strip('"""').strip(),
             "input": combined.loc[i, "input"],
             "output": combined.loc[i, "output"],
         } for i, t in zip(mask, paraphrased)])],
        ignore_index=True,
    ).reset_index(drop=True)

    return combined
