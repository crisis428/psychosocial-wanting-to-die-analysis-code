# %% [cell 69]
MANUAL_EXPLANATION_MODEL = None
# 예: MANUAL_EXPLANATION_MODEL = "CatBoost"

TREE_MODELS = [
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "CatBoost",
]


def choose_explanation_model(
    performance_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    manual_model: str | None = None,
) -> dict[str, Any]:
    overall = (
        performance_table.loc[
            performance_table["Group"] == "total"
        ]
        .sort_values("AUROC", ascending=False)
        .reset_index(drop=True)
    )

    best_model = str(overall.loc[0, "Model"])
    best_auroc = float(overall.loc[0, "AUROC"])

    best_tree_row = (
        overall.loc[overall["Model"].isin(TREE_MODELS)]
        .sort_values("AUROC", ascending=False)
        .iloc[0]
    )
    best_tree = str(best_tree_row["Model"])
    best_tree_auroc = float(best_tree_row["AUROC"])

    if manual_model is not None:
        if manual_model not in TREE_MODELS:
            raise ValueError(
                "수동 설명모델은 tree model이어야 합니다: "
                f"{TREE_MODELS}"
            )
        selected = manual_model
        reason = "manual override"
        eligible = True

    elif best_model in TREE_MODELS:
        selected = best_model
        reason = "highest-AUROC model is a tree model"
        eligible = True

    else:
        delta, delta_low, delta_high = comparison_delta(
            comparison_table,
            "total",
            best_tree,
            best_model,
        )
        gap = best_auroc - best_tree_auroc

        eligible = (
            gap <= NONINFERIOR_AUROC_MARGIN
            and delta_high >= 0
        )

        if eligible:
            selected = best_tree
            reason = (
                "best tree model was within the prespecified "
                "0.01 AUROC margin and was not clearly inferior"
            )
        else:
            selected = None
            reason = (
                "no tree model met the prespecified near-best "
                "performance criterion"
            )

    selection = {
        "Best_Overall_Model": best_model,
        "Best_Overall_AUROC": best_auroc,
        "Best_Tree_Model": best_tree,
        "Best_Tree_AUROC": best_tree_auroc,
        "Selected_Explanation_Model": selected,
        "Eligible_For_Primary_SHAP": eligible,
        "Reason": reason,
        "Noninferiority_AUROC_Margin": (
            NONINFERIOR_AUROC_MARGIN
        ),
    }

    atomic_write_json(
        selection,
        RESULT_ROOT / "explanation_model_selection.json",
    )
    return selection


explanation_selection = choose_explanation_model(
    performance,
    comparisons,
    MANUAL_EXPLANATION_MODEL,
)

display(pd.DataFrame([explanation_selection]))

EXPLANATION_MODEL = explanation_selection[
    "Selected_Explanation_Model"
]

if EXPLANATION_MODEL is None:
    print(
        "자동 기준에서는 primary SHAP 설명모델을 선택하지 않았습니다. "
        "성능 결과를 검토한 뒤 MANUAL_EXPLANATION_MODEL을 설정하세요."
    )
else:
    print("SHAP 설명모델:", EXPLANATION_MODEL)

# %% [cell 71]
SHAP_GROUPS = ["total", "male", "female"]


def shap_checkpoint_dir(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
) -> Path:
    return (
        SHAP_ROOT
        / "checkpoints"
        / group
        / model_name
        / f"repeat_{repeat + 1:02d}"
        / f"fold_{fold + 1:02d}"
    )


def shap_checkpoint_complete(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
) -> bool:
    folder = shap_checkpoint_dir(
        group, model_name, repeat, fold
    )
    required = [
        folder / "shap_values.npz",
        folder / "fold_importance.csv",
        folder / "runtime.json",
        folder / "COMPLETE.flag",
    ]
    return all(path.exists() for path in required)


def load_best_parameters(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
) -> dict[str, Any]:
    folder = checkpoint_dir(
        group, model_name, repeat, fold
    )
    record = json.loads(
        (folder / "best_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    return record["Parameters"]


def extract_binary_shap_values(
    shap_output: Any,
) -> np.ndarray:
    if hasattr(shap_output, "values"):
        values = np.asarray(shap_output.values)
    elif isinstance(shap_output, list):
        values = np.asarray(
            shap_output[1] if len(shap_output) > 1
            else shap_output[0]
        )
    else:
        values = np.asarray(shap_output)

    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]

    if values.ndim != 2:
        raise ValueError(
            f"예상하지 못한 SHAP 배열 형태: {values.shape}"
        )

    return values


def run_one_shap_fold(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
    force: bool = False,
) -> dict[str, Any]:
    if model_name not in TREE_MODELS:
        raise ValueError(
            "현재 SHAP 코드는 tree model용입니다."
        )

    if not checkpoint_complete(
        group, model_name, repeat, fold
    ):
        raise RuntimeError(
            "해당 ML fold가 아직 완료되지 않았습니다."
        )

    folder = shap_checkpoint_dir(
        group, model_name, repeat, fold
    )
    folder.mkdir(parents=True, exist_ok=True)

    if (
        shap_checkpoint_complete(
            group, model_name, repeat, fold
        )
        and not force
    ):
        return {
            "status": "skipped",
            "group": group,
            "model": model_name,
            "repeat": repeat + 1,
            "fold": fold + 1,
        }

    complete_flag = folder / "COMPLETE.flag"
    if complete_flag.exists():
        complete_flag.unlink()

    import shap

    started = time.time()
    local_seed = fold_seed(
        group, model_name, repeat, fold
    )

    group_data, predictors = group_dataframe(df, group)
    splits = outer_splits(group_data, group, repeat)
    train_index, test_index = splits[fold]

    x_train = group_data.loc[train_index, predictors]
    y_train = group_data.loc[
        train_index, OUTCOME
    ].to_numpy(dtype=int)
    x_test = group_data.loc[test_index, predictors]
    y_test = group_data.loc[
        test_index, OUTCOME
    ].to_numpy(dtype=int)

    model_pipeline, _ = build_model_and_search_space(
        model_name,
        local_seed,
    )
    best_parameters = load_best_parameters(
        group, model_name, repeat, fold
    )
    model_pipeline.set_params(**best_parameters)
    model_pipeline.fit(x_train, y_train)

    fitted_model = model_pipeline.named_steps["model"]
    explainer = shap.Explainer(fitted_model)
    shap_output = explainer(x_test)
    shap_values = extract_binary_shap_values(shap_output)

    feature_values = x_test.to_numpy(dtype=float)
    participant_ids = group_data.loc[
        test_index, ID_COLUMN
    ].to_numpy()

    importance = np.abs(shap_values).mean(axis=0)
    importance_frame = pd.DataFrame(
        {
            "Group": group,
            "Model": model_name,
            "Repeat": repeat + 1,
            "Fold": fold + 1,
            "Feature": predictors,
            "Feature_Label": [
                FEATURE_LABELS[feature]
                for feature in predictors
            ],
            "Mean_Absolute_SHAP": importance,
        }
    )
    importance_frame["Rank"] = (
        importance_frame["Mean_Absolute_SHAP"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    importance_frame = importance_frame.sort_values(
        "Rank"
    )

    atomic_write_npz(
        folder / "shap_values.npz",
        shap_values=shap_values,
        feature_values=feature_values,
        participant_ids=participant_ids,
        y_true=y_test,
        feature_names=np.asarray(
            predictors,
            dtype="U100",
        ),
    )
    atomic_write_csv(
        importance_frame,
        folder / "fold_importance.csv",
    )
    atomic_write_json(
        {
            "Group": group,
            "Model": model_name,
            "Repeat": repeat + 1,
            "Fold": fold + 1,
            "Seed": local_seed,
            "N_Train": len(train_index),
            "N_Test": len(test_index),
            "Started_At_Epoch": started,
            "Finished_At_Epoch": time.time(),
            "Elapsed_Seconds": time.time() - started,
        },
        folder / "runtime.json",
    )
    atomic_write_text(
        "COMPLETE\n",
        complete_flag,
    )

    return {
        "status": "completed",
        "group": group,
        "model": model_name,
        "repeat": repeat + 1,
        "fold": fold + 1,
        "elapsed_seconds": time.time() - started,
    }


def run_shap_checkpoints(
    group: str,
    model_name: str,
    force: bool = False,
) -> pd.DataFrame:
    rows = []

    for repeat in range(OUTER_REPEATS):
        for fold in range(OUTER_SPLITS):
            label = (
                f"SHAP {group}/{model_name}/"
                f"repeat {repeat + 1}/fold {fold + 1}"
            )

            try:
                result = run_one_shap_fold(
                    group,
                    model_name,
                    repeat,
                    fold,
                    force,
                )
                rows.append(result)

                if result["status"] == "skipped":
                    print("SKIP:", label)
                else:
                    print(
                        "DONE:",
                        label,
                        "| sec:",
                        f"{result['elapsed_seconds']:.1f}",
                    )

            except Exception as error:
                error_path = (
                    ERROR_ROOT
                    / (
                        f"SHAP_{group}_{model_name}_"
                        f"repeat{repeat + 1:02d}_"
                        f"fold{fold + 1:02d}.txt"
                    )
                )
                atomic_write_text(
                    (
                        f"{label}\n"
                        f"{type(error).__name__}: {error}\n\n"
                        f"{traceback.format_exc()}"
                    ),
                    error_path,
                )
                rows.append(
                    {
                        "status": "failed",
                        "group": group,
                        "model": model_name,
                        "repeat": repeat + 1,
                        "fold": fold + 1,
                        "error": str(error),
                    }
                )
                print("FAILED:", label, "|", error)

    return pd.DataFrame(rows)


print("SHAP 체크포인트 함수 준비 완료")
