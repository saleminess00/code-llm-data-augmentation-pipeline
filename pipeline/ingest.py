"""
ingest.py
Author : Ines Salem  •  salem.iness00@gmail.com
~~~~~~~~~
Light-weight inspection helpers for the LiquidAI take-home project.

"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Optional, Set, Tuple

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import login
from IPython.display import display

PRINTABLE_RATIO_MIN      = 0.95
SHORT_QUANTILE           = 0.02
LONG_QUANTILE            = 0.98
LOW_PRINT_SAMPLE_ROWS    = 10
SHORT_LONG_SAMPLE_ROWS   = 5
RAND_SAMPLE_ROWS_DEFAULT = 5
MAX_NEAR_DUPES_DISPLAY   = 5
SIMHASH_PREFIX_BITS      = 16
HAMMING_MIN              = 0
HAMMING_MAX              = 3

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

COMMON_LANGUAGES: Set[str] = {
    "java", "javascript", "c", "c++", "c#", "go", "ruby", "swift",
    "kotlin", "php", "typescript", "rust", "scala", "perl", "r", "bash",
    "shell", "html", "css", "sql", "dart", "matlab", "objective-c",
}
JOIN_COLS = ["instruction", "input", "output"]
_WORD_RE = re.compile(r"\w+")


def login_to_huggingface() -> None:
    if token := os.getenv("HF_TOKEN"):
        login(token=token); log.info("Logged in to Hugging Face ")
    else:
        log.warning("No token in HF_TOKEN — skipping login")


def load_dataset_from_hub(repo_id: str, *, split: str = "train") -> Dataset:
    return load_dataset(repo_id, split=split)


def find_language(text: str, *, language_vocab: Set[str] = COMMON_LANGUAGES) -> Optional[str]:
    lower = text.lower()
    for lang in language_vocab:
        if re.search(rf"\b{re.escape(lang)}\b", lower):
            return lang
    return None


def compile_check(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except Exception:
        return False


def printable_ratio(s: str) -> float:
    return sum(c.isprintable() for c in s) / max(len(s), 1)


def simhash64(text: str) -> int:
    vec = [0] * 64
    for tok in _WORD_RE.findall(text.lower()):
        h = hash(tok)
        for i in range(64):
            vec[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i, v in enumerate(vec):
        if v > 0:
            fp |= 1 << i
    return fp


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ───────────────────────── MAIN INSPECTOR ───────────────────────── #

def inspect_dataset(
    dataset: Dataset,
    *,
    language_vocab: Set[str] = COMMON_LANGUAGES,
    sample_size: int = RAND_SAMPLE_ROWS_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = dataset.to_pandas()

    # 1 ─ Schema ----------------------------------------------------------- #
    log.info("\n═══════════════════════════════════════")
    log.info("🔍 STEP 1: SCHEMA CHECK")
    expected = {"instruction", "input", "output"}
    actual   = set(df.columns)
    log.info("• Expected : %s", sorted(expected))
    log.info("• Found    : %s", sorted(actual))
    if diff := expected - actual: log.warning("Missing cols → %s", diff)
    if diff := actual - expected: log.warning("Extra cols   → %s", diff)

    # 2 ─ Random sample ---------------------------------------------------- #
    log.info("\n═══════════════════════════════════════")
    log.info("👀 SAMPLE PREVIEW")
    display(df.sample(min(sample_size, len(df)), random_state=42))

    # 3 ─ Basic profiling -------------------------------------------------- #
    log.info("\n═══════════════════════════════════════")
    log.info(" BASIC PROFILING")
    log.info("Nulls:\n%s", df.isnull().sum().to_string())
    log.info("Empty instructions : %d", (df["instruction"].str.strip() == "").sum())
    log.info("Empty outputs      : %d\n", (df["output"].str.strip() == "").sum())

    dup_exact = df.duplicated(subset=JOIN_COLS).sum()
    log.info("Exact row duplicates                : %d", dup_exact)

    # --- Input / Output similarity (exact & approx) ---------------------- #
    _norm = lambda t: " ".join(str(t).lower().split())
    
    io_exact_mask = df["input"].map(_norm) == df["output"].map(_norm)
    io_exact = io_exact_mask.sum()
    log.info("Input == Output (norm-case/space)    : %d", io_exact)
    
    sim_in  = df["input"].map(lambda x: simhash64(_norm(x)))
    sim_out = df["output"].map(lambda x: simhash64(_norm(x)))
    io_approx_mask = [
        HAMMING_MIN < hamming(a, b) <= HAMMING_MAX and not ex
        for a, b, ex in zip(sim_in, sim_out, io_exact_mask)
    ]
    io_approx = sum(io_approx_mask)
    log.info("Input ≈ Output (h ∈ [%d,%d])         : %d",
             HAMMING_MIN + 1, HAMMING_MAX, io_approx)
    
    if io_exact:
        log.info("— showing up to 5 exact I/O dupes —")
        display(df[io_exact_mask].head(5))
    
    if io_approx:
        log.info("— showing up to 5 approx I/O dupes —")
        display(df[pd.Series(io_approx_mask, index=df.index)].head(5))

    # --- Length stats ----------------------------------------------------- #
    log.info("Length stats:\n%s",
             df[["instruction", "input", "output"]]
             .applymap(lambda x: len(str(x))).describe().round(1).to_string())

    # Low printable rows --------------------------------------------------- #
    low_print_mask = (
        (df["instruction"].map(printable_ratio) < PRINTABLE_RATIO_MIN) |
        (df["output"].map(printable_ratio)      < PRINTABLE_RATIO_MIN)
    )
    if low_print_mask.any():
        log.info("Low-printable rows                   : %d (showing %d)",
                 low_print_mask.sum(), LOW_PRINT_SAMPLE_ROWS)
        display(df[low_print_mask].head(LOW_PRINT_SAMPLE_ROWS))
    else:
        log.info("No low-printable rows detected.")

    # Extreme lengths ------------------------------------------------------ #
    for col in ("instruction", "output"):
        lengths = df[col].str.len()
        q_lo, q_hi = lengths.quantile(SHORT_QUANTILE), lengths.quantile(LONG_QUANTILE)
        log.info("%s ≤ %.0f chars (2%%)              : %d rows",
                 col, q_lo, (lengths <= q_lo).sum())
        display(df[lengths <= q_lo].head(SHORT_LONG_SAMPLE_ROWS))
        log.info("%s ≥ %.0f chars (98%%)             : %d rows",
                 col, q_hi, (lengths >= q_hi).sum())
        display(df[lengths >= q_hi].head(SHORT_LONG_SAMPLE_ROWS))

    # Approx-duplicate rows (whole-row SimHash) --------------------------- #
    joined = df[JOIN_COLS].agg(" ".join, axis=1)
    exact_mask = joined.duplicated(keep=False)

    pairs: list[tuple[int, int, int]] = []  # (h, idx1, idx2)
    buckets: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)

    for idx, text in joined[~exact_mask].items():
        fp = simhash64(text)
        key = fp >> (64 - SIMHASH_PREFIX_BITS)
        for prev_idx, prev_fp in buckets[key]:
            hd = hamming(fp, prev_fp)
            if HAMMING_MIN < hd <= HAMMING_MAX:
                pairs.append((hd, prev_idx, idx))
                break
        else:
            buckets[key].append((idx, fp))

    if pairs:
        pairs.sort(reverse=True)  # show most different first
        log.info("Approx row duplicates                : %d (showing %d)",
                 len(pairs), MAX_NEAR_DUPES_DISPLAY)
        for n, (hd, a, b) in enumerate(pairs[:MAX_NEAR_DUPES_DISPLAY], 1):
            display(df.loc[[a, b]]); log.info("— pair %d (h=%d) above —", n, hd)
    else:
        log.info("No approx row duplicates detected.")

    # 4 ─ Python syntax check --------------------------------------------- #
    log.info("\n═══════════════════════════════════════")
    log.info(" PYTHON SYNTAX CHECK")
    df["is_valid_code"] = df["output"].apply(compile_check)
    log.info("Valid code rows                      : %d", df["is_valid_code"].sum())
    log.info("Invalid code rows                    : %d", (~df["is_valid_code"]).sum())

    # 5 ─ Prog-Language-aware inspection --------------------------------------- #
    log.info("\n═══════════════════════════════════════")
    log.info("PROGRAMMING LANGUAGE-AWARE CHECKS")

    def mentions_other_language_only(text: str) -> bool:
        lower = text.lower()
        return any(lang in lower for lang in language_vocab) and "python" not in lower

    other_lang_valid   = df[df["is_valid_code"] & df["instruction"].apply(mentions_other_language_only)]
    python_invalid     = df[~df["is_valid_code"] & df["instruction"].str.contains("python", case=False, na=False)]
    other_lang_invalid = df[~df["is_valid_code"] & df["instruction"].apply(mentions_other_language_only)].copy()
    other_lang_invalid["language_mentioned"] = other_lang_invalid["instruction"].apply(find_language)

    log.info("Valid code + non-Python mention      : %d", len(other_lang_valid));   display(other_lang_valid.head(10))
    log.info("Invalid code but claims Python       : %d", len(python_invalid));     display(python_invalid.head(10))
    log.info("Invalid code + other-language mention: %d", len(other_lang_invalid)); display(other_lang_invalid.head(10))

    return df, other_lang_valid, python_invalid, other_lang_invalid
