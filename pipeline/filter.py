"""
filter.py
Author : Ines Salem  •  salem.iness00@gmail.com
~~~~~~~~~
Row-level cleaning & de-duplication for the LiquidAI take-home project.

Rules
-----
1.  Exact-row + SimHash (h ≤ 3) de-duplication.
2.  Drop outputs that are literals, lone identifiers, or pure-comment blobs.
3.  Heuristics for *non-compiling* outputs:
      • keep if instruction mentions ANY language in ``COMMON_LANGUAGES``    (H-A)
      • keep if instruction has "python" AND one verb-y keyword in PY_KEYWORDS (H-B)
4.  Return two HuggingFace datasets:
      • ``valid_ds``   – output compiles
      • ``invalid_ds`` – output doesn’t compile but kept by rule H-A/B
"""

from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from typing import List, Optional, Set, Tuple

import pandas as pd
from datasets import Dataset

# ────────────────────────── reuse pieces from ingest.py ──────────────────── #
from pipeline.ingest import (
    COMMON_LANGUAGES,
    JOIN_COLS,
    simhash64,
    hamming,
    find_language,
)
# -------------------------------------------------------------------------- #
SIMHASH_PREFIX_BITS = 16
HAMMING_MAX         = 3
PY_KEYWORDS         = {
    "code", "script", "write", "create", "function", "build", "generate"
}

# ——— make sure *something* logs even if basicConfig has already run ——— #
log = logging.getLogger(__name__)
if not log.handlers:                                           # <── key change
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s │ %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# ───────────────────────── low-information detectors ─────────────────────── #
def _is_literal_only(text: str) -> bool:
    try:
        node = ast.parse(str(text).strip())
        return (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        )
    except SyntaxError:
        return False


def _is_single_identifier(text: str) -> bool:
    try:
        node = ast.parse(str(text).strip())
        return (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Name)
        )
    except SyntaxError:
        return False


def _is_comment_only(text: str) -> bool:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("#") for ln in lines)


def _is_useless(text: str) -> bool:
    return (
        _is_literal_only(text)
        or _is_single_identifier(text)
        or _is_comment_only(text)
    )


def _compiles(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────────── main cleaning routine ───────────────────────── #
def filter_dataset(
    dataset: Dataset,
    *,
    external_near_dupes: Optional[set[int]] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Returns
    -------
    valid_ds   – rows whose ``output`` compiles
    invalid_ds – non-compiling rows kept by heuristic rules
    """
    df = dataset.to_pandas()

    # strip whitespace
    for col in JOIN_COLS:
        df[col] = df[col].astype(str).str.strip()

    # drop external near-dupes (if any)
    if external_near_dupes:
        before = len(df)
        df = df.drop(index=list(external_near_dupes), errors="ignore")
        log.info("External near-dupes removed : %d", before - len(df))

    # drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=JOIN_COLS)
    log.info(" Exact duplicates removed    : %d", before - len(df))

    # drop low-information outputs
    mask_keep = ~df["output"].map(_is_useless)
    log.info("Low-info outputs removed    : %d", (~mask_keep).sum())
    df = df[mask_keep]

    # SimHash near-duplicate removal
    joined   = df[JOIN_COLS].agg(" ".join, axis=1)
    keep_row = pd.Series(True, index=df.index)
    buckets: defaultdict[int, List[Tuple[int, int]]] = defaultdict(list)

    for idx, text in joined.items():
        fp  = simhash64(text)
        key = fp >> (64 - SIMHASH_PREFIX_BITS)
        if any(hamming(fp, old_fp) <= HAMMING_MAX for _, old_fp in buckets[key]):
            keep_row[idx] = False
        else:
            buckets[key].append((idx, fp))

    log.info("≈  Near-dupes removed (h≤%d)    : %d", HAMMING_MAX, (~keep_row).sum())
    df = df[keep_row]

    # split compile / non-compile
    df["__compiles__"] = df["output"].map(_compiles)

    def _retain_invalid(row) -> bool:
        """
        Keep non-compiling row if:
        • instruction mentions *any* language keyword (via find_language), OR
        • mentions 'python' AND one of our verb-y keywords.
        """
        instr = row["instruction"].lower()

        # Correct language detection using word boundaries
        if find_language(instr) is not None:
            return True

        # python + keyword check
        if "python" in instr and any(kw in instr for kw in PY_KEYWORDS):
            return True

        return False

    keep_invalid_mask = (~df["__compiles__"]) & df.apply(_retain_invalid, axis=1)

    valid_df   = df[df["__compiles__"]].drop(columns="__compiles__")
    invalid_df = df[keep_invalid_mask].drop(columns="__compiles__")

    log.info("Valid (compiling) rows kept   : %d", len(valid_df))
    log.info("Invalid rows kept (H-A/B)     : %d", len(invalid_df))
    log.info(" Rows dropped in this stage     : %d",
             len(df) - len(valid_df) - len(invalid_df))

    return (
        Dataset.from_pandas(valid_df.reset_index(drop=True)),
        Dataset.from_pandas(invalid_df.reset_index(drop=True)),
    )

