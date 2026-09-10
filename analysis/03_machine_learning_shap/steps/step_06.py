# %% [cell 65]
def comparison_delta(
    comparison_table: pd.DataFrame,
    group: str,
    model_a: str,
    model_b: str,
) -> tuple[float, float, float]:
    direct = comparison_table.loc[
        (comparison_table["Group"] == group)
        & (comparison_table["Model_A"] == model_a)
        & (comparison_table["Model_B"] == model_b)
    ]

    if not direct.empty:
        row = direct.iloc[0]
        return (
            float(row["Delta_AUROC_A_minus_B"]),
            float(row["Delta_AUROC_CI_Low"]),
            float(row["Delta_AUROC_CI_High"]),
        )

    reverse = comparison_table.loc[
        (comparison_table["Group"] == group)
        & (comparison_table["Model_A"] == model_b)
        & (comparison_table["Model_B"] == model_a)
    ]

    if reverse.empty:
        return float("nan"), float("nan"), float("nan")

    row = reverse.iloc[0]
    return (
        -float(row["Delta_AUROC_A_minus_B"]),
        -float(row["Delta_AUROC_CI_High"]),
        -float(row["Delta_AUROC_CI_Low"]),
    )


main_rows = []

for _, row in performance.loc[
    performance["Group"] == "total"
].iterrows():
    model_name = row["Model"]

    if model_name == "LogisticRegression":
        delta, delta_low, delta_high = 0.0, np.nan, np.nan
    else:
        delta, delta_low, delta_high = comparison_delta(
            comparisons,
            "total",
            model_name,
            "LogisticRegression",
        )

    main_rows.append(
        {
            "Model": model_name,
            "AUROC": row["AUROC"],
            "AUROC_95CI_Low": row["AUROC_CI_Low"],
            "AUROC_95CI_High": row["AUROC_CI_High"],
            "Delta_AUROC_vs_Logistic": delta,
            "Delta_AUROC_95CI_Low": delta_low,
            "Delta_AUROC_95CI_High": delta_high,
            "AUPRC": row["Average_Precision"],
            "AUPRC_95CI_Low": row["AP_CI_Low"],
            "AUPRC_95CI_High": row["AP_CI_High"],
            "Brier": row["Brier"],
            "Calibration_Intercept": row[
                "Calibration_Intercept"
            ],
            "Calibration_Slope": row[
                "Calibration_Slope"
            ],
        }
    )

main_performance_table = pd.DataFrame(main_rows).sort_values(
    "AUROC",
    ascending=False,
)

supplement_performance_table = performance[
    [
        "Group",
        "Model",
        "N",
        "Events",
        "AUROC",
        "AUROC_CI_Low",
        "AUROC_CI_High",
        "Average_Precision",
        "AP_CI_Low",
        "AP_CI_High",
        "Brier",
        "Brier_CI_Low",
        "Brier_CI_High",
        "Log_Loss",
        "Calibration_Intercept",
        "Calibration_Slope",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
        "Threshold",
    ]
].copy()

atomic_write_csv(
    main_performance_table,
    COMBINED_ROOT / "main_table_machine_learning_performance.csv",
)
atomic_write_csv(
    supplement_performance_table,
    COMBINED_ROOT / "supplement_table_detailed_performance.csv",
)

display(main_performance_table)

# %% [cell 67]
def save_current_figure(
    filename_stem: str,
    folder: Path = FIGURE_ROOT,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        folder / f"{filename_stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        folder / f"{filename_stem}.pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def best_model_name(
    performance_table: pd.DataFrame,
    group: str = "total",
) -> str:
    row = (
        performance_table.loc[
            performance_table["Group"] == group
        ]
        .sort_values("AUROC", ascending=False)
        .iloc[0]
    )
    return str(row["Model"])


def best_tree_model_name(
    performance_table: pd.DataFrame,
    group: str = "total",
) -> str:
    tree_models = [
        "DecisionTree", "RandomForest",
        "XGBoost", "CatBoost",
    ]
    row = (
        performance_table.loc[
            (performance_table["Group"] == group)
            & (performance_table["Model"].isin(tree_models))
        ]
        .sort_values("AUROC", ascending=False)
        .iloc[0]
    )
    return str(row["Model"])


# G-1. AUROC forest plot
total_performance = (
    performance.loc[performance["Group"] == "total"]
    .sort_values("AUROC", ascending=True)
)

plt.figure(figsize=(8, 5.5))
x = total_performance["AUROC"].to_numpy()
x_low = (
    x - total_performance["AUROC_CI_Low"].to_numpy()
)
x_high = (
    total_performance["AUROC_CI_High"].to_numpy() - x
)
y_position = np.arange(len(total_performance))

plt.errorbar(
    x,
    y_position,
    xerr=np.vstack([x_low, x_high]),
    fmt="o",
    capsize=4,
)
plt.yticks(
    y_position,
    total_performance["Model"],
)
plt.xlabel("AUROC (95% CI)")
plt.ylabel("")
plt.title("Nested Cross-Validated Discrimination")
plt.axvline(0.5, linestyle="--", linewidth=1)
save_current_figure("figure_ml_auroc_forest_total")


# G-2 to G-4. ROC, PR, calibration
aggregated = combined["aggregated"]
top_model = best_model_name(performance, "total")
top_tree_model = best_tree_model_name(performance, "total")

models_for_curves = []
for candidate in [
    "LogisticRegression",
    top_model,
    top_tree_model,
]:
    if candidate not in models_for_curves:
        models_for_curves.append(candidate)

plt.figure(figsize=(7, 6))
for model_name in MODELS:
    subset = aggregated.loc[
        (aggregated["Group"] == "total")
        & (aggregated["Model"] == model_name)
    ]
    if subset.empty:
        continue

    y_true = subset["Y_True"].to_numpy(dtype=int)
    y_prob = subset["Y_Probability"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} ({auc_value:.3f})",
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False-Positive Rate")
plt.ylabel("True-Positive Rate")
plt.title("Receiver Operating Characteristic Curves")
plt.legend(fontsize=8)
save_current_figure("supplement_roc_all_models_total")


plt.figure(figsize=(7, 6))
for model_name in MODELS:
    subset = aggregated.loc[
        (aggregated["Group"] == "total")
        & (aggregated["Model"] == model_name)
    ]
    if subset.empty:
        continue

    y_true = subset["Y_True"].to_numpy(dtype=int)
    y_prob = subset["Y_Probability"].to_numpy(dtype=float)
    precision_values, recall_values, _ = precision_recall_curve(
        y_true, y_prob
    )
    ap_value = average_precision_score(y_true, y_prob)
    plt.plot(
        recall_values,
        precision_values,
        label=f"{model_name} ({ap_value:.3f})",
    )

prevalence = float(
    aggregated.loc[
        (aggregated["Group"] == "total")
        & (aggregated["Model"] == "LogisticRegression"),
        "Y_True",
    ].mean()
)
plt.axhline(prevalence, linestyle="--")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(fontsize=8)
save_current_figure("supplement_precision_recall_all_models_total")


plt.figure(figsize=(7, 6))
for model_name in models_for_curves:
    subset = aggregated.loc[
        (aggregated["Group"] == "total")
        & (aggregated["Model"] == model_name)
    ]
    if subset.empty:
        continue

    y_true = subset["Y_True"].to_numpy(dtype=int)
    y_prob = subset["Y_Probability"].to_numpy(dtype=float)

    observed, predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )
    plt.plot(
        predicted,
        observed,
        marker="o",
        label=model_name,
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Proportion")
plt.title("Calibration Curves")
plt.legend()
save_current_figure("figure_ml_calibration_selected_models_total")


# G-5. Sex-stratified AUROC comparison
subgroup_performance = performance.loc[
    performance["Group"].isin(["male", "female"])
].copy()

plt.figure(figsize=(9, 6))
model_positions = np.arange(len(MODELS))
offset = 0.12

for group_index, group in enumerate(["male", "female"]):
    group_rows = (
        subgroup_performance.loc[
            subgroup_performance["Group"] == group
        ]
        .set_index("Model")
        .reindex(MODELS)
    )

    x_position = model_positions + (
        -offset if group_index == 0 else offset
    )
    y_values = group_rows["AUROC"].to_numpy()
    y_low = (
        y_values - group_rows["AUROC_CI_Low"].to_numpy()
    )
    y_high = (
        group_rows["AUROC_CI_High"].to_numpy()
        - y_values
    )

    plt.errorbar(
        x_position,
        y_values,
        yerr=np.vstack([y_low, y_high]),
        fmt="o",
        capsize=3,
        label=group.capitalize(),
    )

plt.xticks(
    model_positions,
    MODELS,
    rotation=25,
    ha="right",
)
plt.ylabel("AUROC (95% CI)")
plt.title("Sex-Stratified Model Performance")
plt.legend()
save_current_figure("supplement_auroc_sex_stratified")

print("성능 그림 생성 완료")
