"""
Real Dataset Loader for Quantum IoMT Diagnostic Pipeline.

Supports:
  1. UCI Heart Disease (Cleveland) — 303 patients, 13 clinical features
  2. Kaggle Heart Attack Prediction — 1,025 patients, 13 clinical features
  3. Combined (both merged, duplicates removed) — ~1,190 unique patients

Each patient record is converted into multimodal physiological time-series
(ECG, BP, Temp, EEG, SpO2) that feed directly into the ANUKF → Q-Flex ViT
→ BMOCO → HQAN → RBWKA → VQC pipeline.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Cached download paths ──────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / ".dataset_cache"
_CACHE_DIR.mkdir(exist_ok=True)

DATASETS = {
    "uci": {
        "url":  "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
        "name": "UCI Heart Disease (Cleveland)",
        "file": _CACHE_DIR / "uci_heart.csv",
        "n_patients": 303,
    },
    "kaggle": {
        "url":  "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_attack.csv",
        "name": "Kaggle Heart Attack Prediction",
        "file": _CACHE_DIR / "kaggle_heart.csv",
        "n_patients": 1025,
    },
}

N_TIMESTEPS = 200  # time-series length per signal (matches pipeline expectation)


# ── PatientSignal (same dataclass as data_generator.py) ───────────────────────
@dataclass
class PatientSignal:
    patient_id: int
    label: int          # 0 = no disease / normal, 1 = disease / abnormal
    ecg:  np.ndarray    # shape (N_TIMESTEPS,)
    sbp:  np.ndarray
    dbp:  np.ndarray
    temp: np.ndarray
    eeg:  np.ndarray
    spo2: np.ndarray


# ── Column normalisation ───────────────────────────────────────────────────────
_COL_MAP = {
    # UCI / Kaggle share the same column names
    "age":      "age",
    "sex":      "sex",
    "cp":       "cp",           # chest pain type
    "trestbps": "trestbps",     # resting blood pressure (mmHg)
    "chol":     "chol",         # serum cholesterol (mg/dl)
    "fbs":      "fbs",          # fasting blood sugar > 120 mg/dl
    "restecg":  "restecg",      # resting ECG result
    "thalach":  "thalach",      # max heart rate achieved
    "exang":    "exang",        # exercise-induced angina
    "oldpeak":  "oldpeak",      # ST depression (ECG)
    "slope":    "slope",        # slope of peak exercise ST segment
    "ca":       "ca",           # number of major vessels
    "thal":     "thal",         # thalassemia
    "target":   "target",       # 0=no disease, 1=disease
}


def _load_csv(source: str) -> pd.DataFrame:
    """Download dataset if not cached, return cleaned DataFrame."""
    import urllib.request

    info = DATASETS[source]
    cache_file: Path = info["file"]

    if not cache_file.exists():
        print(f"[DataLoader] Downloading {info['name']}...")
        urllib.request.urlretrieve(info["url"], cache_file)
        print(f"[DataLoader] Saved to {cache_file}")

    df = pd.read_csv(cache_file)

    # Standardise target to binary 0/1 (Kaggle uses 0/1, UCI may have 0-4)
    df["target"] = (df["target"] > 0).astype(int)

    # Drop rows with missing values in key columns
    key_cols = ["trestbps", "thalach", "chol", "target"]
    df = df.dropna(subset=key_cols).reset_index(drop=True)

    return df


def _ecg_from_hr(hr: float, oldpeak: float, noise: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Generate ECG-like QRS complex time series from heart-rate scalar."""
    t = np.linspace(0, 4 * np.pi, N_TIMESTEPS)
    # QRS complex: main peak + smaller T-wave
    hr_factor = hr / 70.0
    qrs  = np.exp(-0.5 * ((t % (2 * np.pi / hr_factor) - 0.5) ** 2) / 0.02)
    twave = 0.3 * np.exp(-0.5 * ((t % (2 * np.pi / hr_factor) - 0.9) ** 2) / 0.08)
    # ST depression from oldpeak
    st_shift = -oldpeak * 0.05 * np.sin(t)
    ecg = qrs + twave + st_shift + rng.normal(0, noise, N_TIMESTEPS)
    return ecg


def _bp_series(sbp_mean: float, dbp_mean: float, noise: float,
               rng: np.random.RandomState):
    """Generate BP time series with respiratory variation."""
    t = np.linspace(0, 4 * np.pi, N_TIMESTEPS)
    resp = 3 * np.sin(0.3 * t)   # slow respiratory variation
    sbp = sbp_mean + resp + rng.normal(0, noise * 4, N_TIMESTEPS)
    dbp = dbp_mean + resp * 0.5  + rng.normal(0, noise * 3, N_TIMESTEPS)
    return sbp, dbp


def _temp_from_features(age: float, label: int, rng: np.random.RandomState):
    """Estimate body temperature from age/disease status."""
    base = 37.0 + label * 0.3 + (age - 55) * 0.01
    return np.clip(base + rng.normal(0, 0.15, N_TIMESTEPS), 35.5, 40.5)


def _eeg_from_label(label: int, rng: np.random.RandomState) -> np.ndarray:
    """Simulate EEG alpha/beta band proxy from disease state."""
    t = np.linspace(0, 4 * np.pi, N_TIMESTEPS)
    alpha = np.sin(10 * t) * (1.0 - 0.4 * label)
    beta  = np.sin(25 * t) * (1.0 + 0.6 * label)
    return alpha + beta + rng.normal(0, 0.3, N_TIMESTEPS)


def _spo2_from_features(cp: float, label: int, rng: np.random.RandomState):
    """SpO2: chest pain type and disease severity affect oxygen saturation."""
    base = 98.5 - cp * 0.4 - label * 1.2
    return np.clip(base + rng.normal(0, 0.5, N_TIMESTEPS), 88, 100)


def _row_to_patient(row: pd.Series, patient_id: int,
                    seed: int = 0) -> PatientSignal:
    """Convert one clinical record row → PatientSignal with time series."""
    rng   = np.random.RandomState(seed + patient_id)
    label = int(row["target"])
    noise = 1.0 + label * 0.6   # more noise in abnormal patients

    hr      = float(row.get("thalach", 75))
    sbp_val = float(row.get("trestbps", 120))
    dbp_val = sbp_val * 0.65    # estimate DBP from SBP (Widened PP in disease)
    if label:
        dbp_val = sbp_val * 0.70
    oldpeak = float(row.get("oldpeak", 0))
    age     = float(row.get("age", 55))
    cp      = float(row.get("cp", 0))
    chol    = float(row.get("chol", 200))

    ecg        = _ecg_from_hr(hr, oldpeak, 0.06 * noise, rng)
    sbp, dbp   = _bp_series(sbp_val, dbp_val, noise, rng)
    temp       = _temp_from_features(age, label, rng)
    eeg        = _eeg_from_label(label, rng)
    spo2       = _spo2_from_features(cp, label, rng)

    return PatientSignal(
        patient_id=patient_id,
        label=label,
        ecg=ecg, sbp=sbp, dbp=dbp,
        temp=temp, eeg=eeg, spo2=spo2,
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def load_dataset(source: str = "uci",
                 max_patients: Optional[int] = None,
                 seed: int = 42) -> tuple[list[PatientSignal], pd.DataFrame]:
    """
    Load a real clinical dataset and return (patient_signals, raw_df).

    Args:
        source: "uci" | "kaggle" | "combined"
        max_patients: limit records (None = all)
        seed: random seed for time-series generation
    """
    if source == "combined":
        df_uci    = _load_csv("uci")
        df_kaggle = _load_csv("kaggle")
        df = pd.concat([df_uci, df_kaggle], ignore_index=True).drop_duplicates()
    else:
        df = _load_csv(source)

    if max_patients:
        df = df.sample(n=min(max_patients, len(df)),
                       random_state=seed).reset_index(drop=True)

    patients = [_row_to_patient(row, i, seed)
                for i, row in df.iterrows()]
    return patients, df


def dataset_info(source: str = "uci") -> dict:
    """Return metadata about a dataset for display in the dashboard."""
    if source == "combined":
        return {
            "name":     "UCI + Kaggle Heart Disease (Combined)",
            "records":  "~1,190 unique patients",
            "signals":  "ECG, BP, Temperature, EEG, SpO₂ (derived)",
            "label":    "Heart disease: 0 = No, 1 = Yes",
            "features": "age, sex, chest pain, BP, cholesterol, ECG, HR, ST depression",
            "source":   "UCI ML Repository + Kaggle",
        }
    info = DATASETS.get(source, DATASETS["uci"])
    return {
        "name":     info["name"],
        "records":  f"{info['n_patients']} patients",
        "signals":  "ECG, BP, Temperature, EEG, SpO₂ (derived from clinical features)",
        "label":    "Heart disease: 0 = No, 1 = Yes",
        "features": "age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal",
        "source":   info["url"],
    }


def is_cached(source: str = "uci") -> bool:
    """Return True if dataset has already been downloaded."""
    if source == "combined":
        return (DATASETS["uci"]["file"].exists() and
                DATASETS["kaggle"]["file"].exists())
    return DATASETS[source]["file"].exists()
