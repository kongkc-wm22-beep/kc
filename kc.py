# app.py — Streamlit UI for Laptop Recommender (BMCS2003)
# Works with recommender.py (backend). Supports Google Drive CSV only.

import os
import pandas as pd
import numpy as np
import streamlit as st
import gdown

# === import your backend ===
from recommender import (
    load_data,
    recommend,
    train_test_eval,
    eval_confusion_labstyle,
    USE_PROFILES,
)

st.set_page_config(page_title="Laptop Recommender (BMCS2003)", layout="wide")
st.title("💻 Laptop Recommender – BMCS2003")
st.caption("Content-based scoring with lab-style train/test evaluation")

# ---------- CONFIG: set your Google Drive file here ----------
DRIVE_FILE_ID = "1ys7aa3wdxPcbk7QIzW9tmmnvSlR9HeG3"   # <-- Just the FILE ID
LOCAL_CSV_PATH = "data/_drive_dataset.csv"

# ---------- Data loader (Drive-only) ----------
@st.cache_data(show_spinner=True)
def download_data_from_drive(file_id: str, dest_path: str) -> pd.DataFrame:
    """
    Download the CSV from Google Drive (by FILE ID) to dest_path and return as DataFrame.
    Cache is keyed by file_id, so updating the ID invalidates cache automatically.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=Tr_
