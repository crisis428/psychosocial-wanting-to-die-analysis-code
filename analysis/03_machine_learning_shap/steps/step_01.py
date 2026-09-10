#!/usr/bin/env python3
# Auto-exported from the finalized notebook workflow.
# Cell order is preserved; no notebook outputs are included.

# %% [cell 2]
# 필요할 때만 주석을 제거해 한 번 실행하세요.
# %pip install --upgrade pip
# %pip install numpy==2.2.6 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.6.1
# %pip install xgboost==3.0.5 catboost==1.2.8 shap==0.48.0
# %pip install matplotlib==3.10.3 statsmodels==0.14.4 joblib==1.4.2 threadpoolctl==3.5.0

# %% [cell 4]
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
import warnings

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from catboost import CatBoostClassifier
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

print("Python:", platform.python_version())
print("Platform:", platform.platform())

if platform.python_version() != "3.10.9":
    print(
        "주의: 최종 논문 환경은 Python 3.10.9로 결정했습니다. "
        f"현재 커널은 {platform.python_version()}입니다."
    )

PACKAGES_TO_RECORD = [
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost",
    "catboost", "shap", "matplotlib", "statsmodels",
    "joblib", "threadpoolctl",
]

for package in PACKAGES_TO_RECORD:
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

# %% [cell 6]
SCRIPT_DIR = Path.cwd()
INPUT_CSV = SCRIPT_DIR / "primary_cc.csv"
RESULT_ROOT = SCRIPT_DIR / "full_results_checkpointed"

CHECKPOINT_ROOT = RESULT_ROOT / "checkpoints"
ERROR_ROOT = RESULT_ROOT / "errors"
COMBINED_ROOT = RESULT_ROOT / "combined"
FIGURE_ROOT = RESULT_ROOT / "figures"
SHAP_ROOT = RESULT_ROOT / "shap"

SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 5

SEARCH_N_JOBS = -1
MODEL_THREADS = 1
THRESHOLD = 0.50

BOOTSTRAP_ITERATIONS = 2000
NONINFERIOR_AUROC_MARGIN = 0.01

OUTCOME = "py_thoughts_wanting_to_die"
ID_COLUMN = "participant_id"

PREDICTORS = [
    "age_years",
    "sex_female",
    "married_current",
    "employed_corrected",
    "living_alone",
    "poor_subjective_physical_health",
    "poor_subjective_mental_health",
    "gad7_total",
    "pss14_total",
    "low_self_esteem_score",
    "low_sense_of_belonging",
    "low_perceived_social_equality",
    "low_social_trust",
]

GROUPS = ["total", "male", "female"]

MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM",
    "XGBoost",
    "CatBoost",
]

SEARCH_ITERATIONS = {
    "LogisticRegression": 10,
    "DecisionTree": 20,
    "RandomForest": 20,
    "SVM": 15,
    "XGBoost": 25,
    "CatBoost": 25,
}

FEATURE_LABELS = {
    "age_years": "Age",
    "sex_female": "Female sex",
    "married_current": "Currently married",
    "employed_corrected": "Currently working",
    "living_alone": "Living alone",
    "poor_subjective_physical_health": "Poor subjective physical health",
    "poor_subjective_mental_health": "Poor subjective mental health",
    "gad7_total": "GAD-7",
    "pss14_total": "PSS-14",
    "low_self_esteem_score": "Low self-esteem",
    "low_sense_of_belonging": "Low sense of belonging",
    "low_perceived_social_equality": "Low perceived social equality",
    "low_social_trust": "Low social trust",
}

for folder in [
    RESULT_ROOT, CHECKPOINT_ROOT, ERROR_ROOT,
    COMBINED_ROOT, FIGURE_ROOT, SHAP_ROOT,
]:
    folder.mkdir(parents=True, exist_ok=True)

print("입력 데이터:", INPUT_CSV.resolve())
print("결과 폴더:", RESULT_ROOT.resolve())

# %% [cell 8]
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"{INPUT_CSV} 파일을 찾을 수 없습니다. "
        "스크립트 실행 작업 폴더에 primary_cc.csv를 두세요."
    )

df = pd.read_csv(INPUT_CSV)

required_columns = [ID_COLUMN, OUTCOME] + PREDICTORS
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(f"필수 변수가 없습니다: {missing_columns}")

if df[ID_COLUMN].duplicated().any():
    raise ValueError("participant_id 중복이 있습니다.")

if set(df[OUTCOME].dropna().unique()) - {0, 1}:
    raise ValueError("결과변수는 0과 1이어야 합니다.")

missing_values = int(df[required_columns].isna().sum().sum())
if missing_values:
    raise ValueError(f"주분석 변수에 결측값 {missing_values}개가 있습니다.")

print("데이터 크기:", df.shape)
print("분석대상:", f"{len(df):,}")
print("사건 수:", f"{int(df[OUTCOME].sum()):,}")
print("사건 비율:", f"{df[OUTCOME].mean():.3%}")
print("중복 ID:", int(df[ID_COLUMN].duplicated().sum()))
print("선택 변수 결측:", missing_values)

display(df.head())
