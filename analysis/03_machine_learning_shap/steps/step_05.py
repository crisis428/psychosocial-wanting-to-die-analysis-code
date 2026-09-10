# %% [cell 61]
def load_completed_checkpoints() -> dict[str, pd.DataFrame]:
    prediction_frames = []
    metric_rows = []
    parameter_rows = []

    for group in GROUPS:
        for model_name in MODELS:
            for repeat in range(OUTER_REPEATS):
                for fold in range(OUTER_SPLITS):
                    if not checkpoint_complete(
                        group, model_name, repeat, fold
                    ):
                        continue

                    folder = checkpoint_dir(
                        group, model_name, repeat, fold
                    )

                    prediction_frames.append(
                        pd.read_csv(folder / "predictions.csv")
                    )

                    metric_rows.append(
                        json.loads(
                            (folder / "fold_metrics.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    )

                    parameter_record = json.loads(
                        (folder / "best_parameters.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    flat_record = {
                        "Group": parameter_record["Group"],
                        "Model": parameter_record["Model"],
                        "Repeat": parameter_record["Repeat"],
                        "Fold": parameter_record["Fold"],
                        "Predictors": json.dumps(
                            parameter_record["Predictors"],
                            ensure_ascii=False,
                        ),
                        "Parameters": json.dumps(
                            parameter_record["Parameters"],
                            ensure_ascii=False,
                        ),
                    }
                    parameter_rows.append(flat_record)

    if not prediction_frames:
        raise RuntimeError("완료된 체크포인트가 없습니다.")

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    metrics = pd.DataFrame(metric_rows)
    parameters = pd.DataFrame(parameter_rows)

    atomic_write_csv(
        predictions,
        COMBINED_ROOT / "all_outer_fold_predictions.csv",
    )
    atomic_write_csv(
        metrics,
        COMBINED_ROOT / "all_outer_fold_metrics.csv",
    )
    atomic_write_csv(
        parameters,
        COMBINED_ROOT / "best_parameters_by_fold.csv",
    )

    aggregated = (
        predictions
        .groupby(
            ["Group", "Model", ID_COLUMN],
            as_index=False,
        )
        .agg(
            Y_True=("Y_True", "first"),
            Y_Probability=("Y_Probability", "mean"),
            N_Outer_Predictions=("Y_Probability", "size"),
        )
    )
    aggregated["Y_Predicted_0_5"] = (
        aggregated["Y_Probability"] >= THRESHOLD
    ).astype(int)

    atomic_write_csv(
        aggregated,
        COMBINED_ROOT / "aggregated_participant_predictions.csv",
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "parameters": parameters,
        "aggregated": aggregated,
    }


combined = load_completed_checkpoints()

print("Fold predictions:", combined["predictions"].shape)
print("Fold metrics:", combined["metrics"].shape)
print("Parameter rows:", combined["parameters"].shape)
print("Participant-level aggregated:", combined["aggregated"].shape)

display(combined["metrics"].head())

# %% [cell 63]
def bootstrap_metric_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    iterations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []

    for _ in range(iterations):
        sample = rng.integers(0, n, size=n)
        sampled_y = y_true[sample]
        sampled_p = y_prob[sample]

        if len(np.unique(sampled_y)) < 2:
            continue

        if metric_name == "AUROC":
            value = roc_auc_score(sampled_y, sampled_p)
        elif metric_name == "Average_Precision":
            value = average_precision_score(
                sampled_y, sampled_p
            )
        elif metric_name == "Brier":
            value = brier_score_loss(sampled_y, sampled_p)
        else:
            raise ValueError(metric_name)

        values.append(float(value))

    return np.asarray(values, dtype=float)


def percentile_ci(
    values: np.ndarray,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")

    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def build_model_performance(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for group in GROUPS:
        for model_index, model_name in enumerate(MODELS):
            subset = aggregated.loc[
                (aggregated["Group"] == group)
                & (aggregated["Model"] == model_name)
            ].copy()

            if subset.empty:
                continue

            y_true = subset["Y_True"].to_numpy(dtype=int)
            y_prob = subset["Y_Probability"].to_numpy(dtype=float)

            metrics = calculate_metrics(
                y_true, y_prob, THRESHOLD
            )

            auroc_boot = bootstrap_metric_distribution(
                y_true,
                y_prob,
                "AUROC",
                BOOTSTRAP_ITERATIONS,
                SEED + group_seed_offset(group) + model_index,
            )
            ap_boot = bootstrap_metric_distribution(
                y_true,
                y_prob,
                "Average_Precision",
                BOOTSTRAP_ITERATIONS,
                SEED + 10_000 + group_seed_offset(group) + model_index,
            )
            brier_boot = bootstrap_metric_distribution(
                y_true,
                y_prob,
                "Brier",
                BOOTSTRAP_ITERATIONS,
                SEED + 20_000 + group_seed_offset(group) + model_index,
            )

            auroc_low, auroc_high = percentile_ci(auroc_boot)
            ap_low, ap_high = percentile_ci(ap_boot)
            brier_low, brier_high = percentile_ci(brier_boot)

            rows.append(
                {
                    "Group": group,
                    "Model": model_name,
                    "N": len(subset),
                    "Events": int(y_true.sum()),
                    **metrics,
                    "AUROC_CI_Low": auroc_low,
                    "AUROC_CI_High": auroc_high,
                    "AP_CI_Low": ap_low,
                    "AP_CI_High": ap_high,
                    "Brier_CI_Low": brier_low,
                    "Brier_CI_High": brier_high,
                }
            )

    performance = pd.DataFrame(rows)
    atomic_write_csv(
        performance,
        COMBINED_ROOT / "model_performance_summary.csv",
    )
    return performance


def paired_bootstrap_comparisons(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for group in GROUPS:
        group_data = aggregated.loc[
            aggregated["Group"] == group
        ].copy()

        probability_wide = group_data.pivot(
            index=ID_COLUMN,
            columns="Model",
            values="Y_Probability",
        )
        outcome = (
            group_data
            .drop_duplicates(ID_COLUMN)
            .set_index(ID_COLUMN)["Y_True"]
            .reindex(probability_wide.index)
            .to_numpy(dtype=int)
        )

        for pair_index, (model_a, model_b) in enumerate(
            combinations(MODELS, 2)
        ):
            if (
                model_a not in probability_wide.columns
                or model_b not in probability_wide.columns
            ):
                continue

            prob_a = probability_wide[model_a].to_numpy(dtype=float)
            prob_b = probability_wide[model_b].to_numpy(dtype=float)

            point_auroc_delta = (
                roc_auc_score(outcome, prob_a)
                - roc_auc_score(outcome, prob_b)
            )
            point_ap_delta = (
                average_precision_score(outcome, prob_a)
                - average_precision_score(outcome, prob_b)
            )

            rng = np.random.default_rng(
                SEED
                + 50_000
                + group_seed_offset(group)
                + pair_index
            )
            auroc_deltas = []
            ap_deltas = []
            n = len(outcome)

            for _ in range(BOOTSTRAP_ITERATIONS):
                sample = rng.integers(0, n, size=n)
                sampled_y = outcome[sample]

                if len(np.unique(sampled_y)) < 2:
                    continue

                sampled_a = prob_a[sample]
                sampled_b = prob_b[sample]

                auroc_deltas.append(
                    roc_auc_score(sampled_y, sampled_a)
                    - roc_auc_score(sampled_y, sampled_b)
                )
                ap_deltas.append(
                    average_precision_score(sampled_y, sampled_a)
                    - average_precision_score(sampled_y, sampled_b)
                )

            auroc_low, auroc_high = percentile_ci(
                np.asarray(auroc_deltas)
            )
            ap_low, ap_high = percentile_ci(
                np.asarray(ap_deltas)
            )

            rows.append(
                {
                    "Group": group,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Delta_AUROC_A_minus_B": point_auroc_delta,
                    "Delta_AUROC_CI_Low": auroc_low,
                    "Delta_AUROC_CI_High": auroc_high,
                    "Delta_AP_A_minus_B": point_ap_delta,
                    "Delta_AP_CI_Low": ap_low,
                    "Delta_AP_CI_High": ap_high,
                }
            )

    comparisons = pd.DataFrame(rows)
    atomic_write_csv(
        comparisons,
        COMBINED_ROOT / "paired_model_comparisons.csv",
    )
    return comparisons


performance = build_model_performance(
    combined["aggregated"]
)
comparisons = paired_bootstrap_comparisons(
    combined["aggregated"]
)

print("성능표 생성 완료:", performance.shape)
print("모델 비교표 생성 완료:", comparisons.shape)

display(
    performance.sort_values(
        ["Group", "AUROC"],
        ascending=[True, False],
    )
)
