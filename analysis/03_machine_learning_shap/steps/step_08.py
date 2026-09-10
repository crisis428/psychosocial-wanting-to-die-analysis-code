# %% [cell 73]
if EXPLANATION_MODEL is None:
    raise RuntimeError(
        "설명모델이 선택되지 않았습니다. "
        "Part H에서 수동 또는 자동 선택을 완료하세요."
    )

shap_run_results = {}

for shap_group in SHAP_GROUPS:
    print(
        f"\n===== {shap_group.upper()} SHAP 시작: "
        f"{EXPLANATION_MODEL} ====="
    )
    shap_run_results[shap_group] = run_shap_checkpoints(
        shap_group,
        EXPLANATION_MODEL,
    )

# %% [cell 75]
def combine_shap_results(
    group: str,
    model_name: str,
) -> dict[str, Any]:
    import shap

    all_importance_frames = []
    repeat_one_values = []
    repeat_one_features = []
    repeat_one_ids = []
    repeat_one_outcomes = []
    feature_names = None

    for repeat in range(OUTER_REPEATS):
        for fold in range(OUTER_SPLITS):
            if not shap_checkpoint_complete(
                group, model_name, repeat, fold
            ):
                continue

            folder = shap_checkpoint_dir(
                group, model_name, repeat, fold
            )
            importance = pd.read_csv(
                folder / "fold_importance.csv"
            )
            all_importance_frames.append(importance)

            if repeat == 0:
                with np.load(
                    folder / "shap_values.npz",
                    allow_pickle=False,
                ) as loaded:
                    repeat_one_values.append(
                        loaded["shap_values"]
                    )
                    repeat_one_features.append(
                        loaded["feature_values"]
                    )
                    repeat_one_ids.append(
                        loaded["participant_ids"]
                    )
                    repeat_one_outcomes.append(
                        loaded["y_true"]
                    )
                    feature_names = [
                        str(value)
                        for value in loaded["feature_names"]
                    ]

    if not all_importance_frames:
        raise RuntimeError(
            f"{group}/{model_name}의 완료된 SHAP 결과가 없습니다."
        )

    importance_all = pd.concat(
        all_importance_frames,
        ignore_index=True,
    )

    shap_output_dir = SHAP_ROOT / "combined" / group / model_name
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_csv(
        importance_all,
        shap_output_dir / "all_fold_shap_importance.csv",
    )

    global_importance = (
        importance_all
        .groupby(
            ["Feature", "Feature_Label"],
            as_index=False,
        )
        .agg(
            Mean_Absolute_SHAP=(
                "Mean_Absolute_SHAP",
                "mean",
            ),
            SD_Absolute_SHAP=(
                "Mean_Absolute_SHAP",
                "std",
            ),
            Mean_Rank=("Rank", "mean"),
            SD_Rank=("Rank", "std"),
            Median_Rank=("Rank", "median"),
            Top_5_Frequency=(
                "Rank",
                lambda values: float(
                    np.mean(np.asarray(values) <= 5)
                ),
            ),
        )
        .sort_values("Mean_Absolute_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    global_importance["Overall_Rank"] = (
        np.arange(1, len(global_importance) + 1)
    )

    atomic_write_csv(
        global_importance,
        shap_output_dir / "global_shap_importance.csv",
    )

    rank_wide = importance_all.pivot_table(
        index=["Repeat", "Fold"],
        columns="Feature",
        values="Rank",
        aggfunc="first",
    ).sort_index()

    correlations = []
    for first_index, second_index in combinations(
        range(len(rank_wide)),
        2,
    ):
        correlation = stats.spearmanr(
            rank_wide.iloc[first_index],
            rank_wide.iloc[second_index],
        ).statistic
        correlations.append(float(correlation))

    stability = {
        "Group": group,
        "Model": model_name,
        "N_SHAP_Folds": int(len(rank_wide)),
        "Median_Pairwise_Spearman": float(
            np.nanmedian(correlations)
        ),
        "Minimum_Pairwise_Spearman": float(
            np.nanmin(correlations)
        ),
        "Maximum_Pairwise_Spearman": float(
            np.nanmax(correlations)
        ),
    }

    atomic_write_json(
        stability,
        shap_output_dir / "shap_rank_stability.json",
    )

    if not repeat_one_values:
        raise RuntimeError(
            "repeat 1의 SHAP 결과가 모두 필요합니다."
        )

    shap_values = np.concatenate(
        repeat_one_values,
        axis=0,
    )
    feature_values = np.concatenate(
        repeat_one_features,
        axis=0,
    )
    participant_ids = np.concatenate(
        repeat_one_ids,
        axis=0,
    )
    outcomes = np.concatenate(
        repeat_one_outcomes,
        axis=0,
    )

    if len(np.unique(participant_ids)) != len(
        participant_ids
    ):
        raise ValueError(
            "repeat 1 SHAP에서 participant_id가 중복됩니다."
        )

    feature_labels = [
        FEATURE_LABELS[name]
        for name in feature_names
    ]

    explanation = shap.Explanation(
        values=shap_values,
        data=feature_values,
        feature_names=feature_labels,
    )

    plt.figure()
    shap.plots.beeswarm(
        explanation,
        max_display=len(feature_labels),
        show=False,
    )
    plt.title(
        f"SHAP Summary: {model_name} ({group})"
    )
    save_current_figure(
        f"figure_shap_beeswarm_{group}_{model_name}",
        folder=shap_output_dir,
    )

    plt.figure()
    shap.plots.bar(
        explanation,
        max_display=len(feature_labels),
        show=False,
    )
    plt.title(
        f"Mean Absolute SHAP: {model_name} ({group})"
    )
    save_current_figure(
        f"figure_shap_bar_{group}_{model_name}",
        folder=shap_output_dir,
    )

    atomic_write_npz(
        shap_output_dir / "repeat1_combined_shap_values.npz",
        shap_values=shap_values,
        feature_values=feature_values,
        participant_ids=participant_ids,
        y_true=outcomes,
        feature_names=np.asarray(
            feature_names,
            dtype="U100",
        ),
    )

    return {
        "importance": global_importance,
        "stability": stability,
        "output_dir": shap_output_dir,
    }


shap_combined_results = {}

for shap_group in SHAP_GROUPS:
    shap_combined_results[shap_group] = (
        combine_shap_results(
            shap_group,
            EXPLANATION_MODEL,
        )
    )

    print(
        shap_group,
        shap_combined_results[shap_group]["stability"],
    )
    display(
        shap_combined_results[shap_group]["importance"]
    )

# %% [cell 77]
ORIGINAL_OUTPUT_ROOT = RESULT_ROOT / "original_manuscript_outputs"
ORIGINAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ORIGINAL_GROUP_ORDER = ["total", "male", "female"]
ORIGINAL_MODEL_ORDER = [
    "LogisticRegression",
    "RandomForest",
    "DecisionTree",
    "SVM",
    "XGBoost",
    "CatBoost",
]

ORIGINAL_GROUP_LABELS = {
    "total": "Total sample",
    "male": "Male subgroup",
    "female": "Female subgroup",
}

ORIGINAL_MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "DecisionTree": "Decision Tree",
    "SVM": "SVM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
}

print("기존 원고 형식 출력 폴더:", ORIGINAL_OUTPUT_ROOT)

# %% [cell 79]
supplement_s1_rows = []

for group in ORIGINAL_GROUP_ORDER:
    for model_name in ORIGINAL_MODEL_ORDER:
        subset = performance.loc[
            (performance["Group"] == group)
            & (performance["Model"] == model_name)
        ]

        if subset.empty:
            raise RuntimeError(
                f"성능 결과가 없습니다: {group}/{model_name}"
            )

        row = subset.iloc[0]

        supplement_s1_rows.append(
            {
                "Group": ORIGINAL_GROUP_LABELS[group],
                "Model": ORIGINAL_MODEL_LABELS[model_name],
                "AUROC": float(row["AUROC"]),
                "Recall": float(row["Sensitivity"]),
                "Precision": float(row["Precision"]),
                "Accuracy": float(row["Accuracy"]),
                "F1_score": float(row["F1"]),
            }
        )

supplementary_table_s1 = pd.DataFrame(supplement_s1_rows)

atomic_write_csv(
    supplementary_table_s1,
    ORIGINAL_OUTPUT_ROOT / "Supplementary_Table_S1_recreated.csv",
)

supplementary_table_s1.to_html(
    ORIGINAL_OUTPUT_ROOT / "Supplementary_Table_S1_recreated.html",
    index=False,
    float_format=lambda value: f"{value:.4f}",
)

display(supplementary_table_s1)

# %% [cell 81]
def save_figure_to_original_outputs(filename_stem: str) -> None:
    plt.tight_layout()
    plt.savefig(
        ORIGINAL_OUTPUT_ROOT / f"{filename_stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        ORIGINAL_OUTPUT_ROOT / f"{filename_stem}.pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(8.2, 18),
)

panel_labels = ["A", "B", "C"]

for axis, group, panel_label in zip(
    axes,
    ORIGINAL_GROUP_ORDER,
    panel_labels,
):
    for model_name in ORIGINAL_MODEL_ORDER:
        subset = combined["aggregated"].loc[
            (combined["aggregated"]["Group"] == group)
            & (combined["aggregated"]["Model"] == model_name)
        ]

        if subset.empty:
            continue

        y_true = subset["Y_True"].to_numpy(dtype=int)
        y_prob = subset["Y_Probability"].to_numpy(dtype=float)

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            y_prob,
        )
        auc_value = roc_auc_score(y_true, y_prob)

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{ORIGINAL_MODEL_LABELS[model_name]} "
                f"(AUROC = {auc_value:.3f})"
            ),
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Reference",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title(
        f"{panel_label}. {ORIGINAL_GROUP_LABELS[group]}",
        loc="left",
        fontweight="bold",
    )
    axis.legend(
        loc="lower right",
        fontsize=8,
        frameon=True,
    )

save_figure_to_original_outputs(
    "Figure_1_ROC_total_male_female_recreated"
)
