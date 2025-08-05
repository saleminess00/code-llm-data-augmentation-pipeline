"""
postprocess.py
Author : Ines Salem  •  salem.iness00@gmail.com
~~~~~~~~~~~~~~
• Trim whitespace
• Drop rows where input == output
• Drop *exact* duplicate rows
• Keep only rows whose `output` compiles
• (Optional) drop very short examples
• Upload the cleaned dataframe to a **private** HF dataset

Usage
-----
from postprocess import postprocess, upload_to_hf
clean_df = postprocess(augmented_df)
repo_id  = upload_to_hf(clean_df)      # returns the repo name / URL
"""

from __future__ import annotations

import os, shutil, tempfile, uuid
import pandas as pd
from datasets import Dataset
from huggingface_hub import create_repo, upload_folder

from pipeline.ingest import compile_check   # already in your repo

# ───────────────────────────── Cleaning ──────────────────────────────
def postprocess(df: pd.DataFrame,
                *,
                drop_short: bool = False,
                min_input_len: int = 10,
                min_output_len: int = 20) -> pd.DataFrame:
    """Return a cleaned copy of *df* (no mutation in-place)."""
    df = df.copy()

    # trim all text columns
    for col in ("instruction", "input", "output"):
        df[col] = df[col].astype(str).str.strip()

    # drop rows where input and output coincide
    df = df[df["input"] != df["output"]]

    # optional length guard
    if drop_short:
        mask = (df["input"].str.len() >= min_input_len) & (
            df["output"].str.len() >= min_output_len
        )
        df = df[mask]

    # remove *exact-row* duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # keep only rows whose output compiles
    df = df[df["output"].apply(compile_check)].reset_index(drop=True)

    return df


# ──────────────────────────── HF upload ──────────────────────────────
def upload_to_hf(df: pd.DataFrame,
                 *,
                 repo_id: str | None = None,
                 private: bool = True) -> str:
    """
    Save *df* to a temp folder, create (or reuse) a private dataset repo
    and upload the artefacts. HF_TOKEN must be set in the environment.
    Returns the full repo_id.
    """
    if repo_id is None:
        repo_id = f"liquidai-augmented-{uuid.uuid4().hex[:8]}"

    ds = Dataset.from_pandas(df)
    tmp = tempfile.mkdtemp()
    ds.save_to_disk(tmp)

    # create_repo(..., exist_ok=True) => won't fail if already exists / you own it
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=tmp,
        path_in_repo=".",
        commit_message="upload cleaned augmented dataset",
    )

    shutil.rmtree(tmp)
    return repo_id
