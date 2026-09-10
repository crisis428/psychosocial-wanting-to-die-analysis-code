# %% [cell 10]
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in PACKAGES_TO_RECORD:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=2),
        path,
    )


def atomic_write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    design = sm.add_constant(logits)

    try:
        fit = sm.GLM(
            np.asarray(y_true, dtype=float),
            design,
            family=sm.families.Binomial(),
        ).fit()
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return float("nan"), float("nan")


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    calibration_intercept, calibration_slope = (
        calibration_intercept_slope(y_true, y_prob)
    )

    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "Average_Precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Log_Loss": safe_log_loss(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred)),
        "Specificity": float(specificity),
        "Precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Calibration_Intercept": calibration_intercept,
        "Calibration_Slope": calibration_slope,
        "Threshold": float(threshold),
    }


def group_dataframe(
    source: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, list[str]]:
    if group == "total":
        subset = source.copy()
        predictors = PREDICTORS.copy()
    elif group == "male":
        subset = source.loc[source["sex_female"] == 0].copy()
        predictors = [
            column for column in PREDICTORS
            if column != "sex_female"
        ]
    elif group == "female":
        subset = source.loc[source["sex_female"] == 1].copy()
        predictors = [
            column for column in PREDICTORS
            if column != "sex_female"
        ]
    else:
        raise ValueError(f"알 수 없는 집단: {group}")

    subset["_source_index"] = subset.index.astype(int)
    subset = subset.reset_index(drop=True)
    return subset, predictors


def group_seed_offset(group: str) -> int:
    return {"total": 0, "male": 100_000, "female": 200_000}[group]


def fold_seed(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
) -> int:
    model_offset = MODELS.index(model_name) * 10_000
    return (
        SEED
        + group_seed_offset(group)
        + model_offset
        + repeat * 100
        + fold
    )


def outer_splits(
    group_data: pd.DataFrame,
    group: str,
    repeat: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = group_data[OUTCOME].to_numpy(dtype=int)
    splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=SEED + group_seed_offset(group) + repeat,
    )
    dummy_x = np.zeros((len(group_data), 1))
    return list(splitter.split(dummy_x, y))


def checkpoint_dir(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
) -> Path:
    return (
        CHECKPOINT_ROOT
        / group
        / model_name
        / f"repeat_{repeat + 1:02d}"
        / f"fold_{fold + 1:02d}"
    )


def checkpoint_complete(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
) -> bool:
    folder = checkpoint_dir(group, model_name, repeat, fold)
    required = [
        folder / "predictions.csv",
        folder / "fold_metrics.json",
        folder / "best_parameters.json",
        folder / "split_indices.npz",
        folder / "runtime.json",
        folder / "COMPLETE.flag",
    ]
    return all(path.exists() for path in required)


def model_completion_count(
    group: str,
    model_name: str,
) -> int:
    return sum(
        checkpoint_complete(group, model_name, repeat, fold)
        for repeat in range(OUTER_REPEATS)
        for fold in range(OUTER_SPLITS)
    )


def completion_status() -> pd.DataFrame:
    rows = []
    expected_per_model = OUTER_REPEATS * OUTER_SPLITS

    for group in GROUPS:
        for model_name in MODELS:
            completed = model_completion_count(group, model_name)
            rows.append(
                {
                    "Group": group,
                    "Model": model_name,
                    "Completed": completed,
                    "Expected": expected_per_model,
                    "Pending": expected_per_model - completed,
                    "Completion_Percent": (
                        100 * completed / expected_per_model
                    ),
                }
            )

    status = pd.DataFrame(rows)
    atomic_write_csv(
        status,
        RESULT_ROOT / "completion_summary.csv",
    )
    return status


set_global_seed(SEED)

run_manifest = {
    "run_name": "checkpointed_ml_shap_py3109_v2",
    "created_at_epoch": time.time(),
    "input_path": str(INPUT_CSV.resolve()),
    "input_sha256": sha256_file(INPUT_CSV),
    "rows": len(df),
    "events": int(df[OUTCOME].sum()),
    "seed": SEED,
    "outer_splits": OUTER_SPLITS,
    "outer_repeats": OUTER_REPEATS,
    "inner_splits": INNER_SPLITS,
    "models": MODELS,
    "groups": GROUPS,
    "predictors": PREDICTORS,
    "search_iterations": SEARCH_ITERATIONS,
    "versions": package_versions(),
}
atomic_write_json(run_manifest, RESULT_ROOT / "run_manifest.json")

print("공통 함수 준비 완료")
