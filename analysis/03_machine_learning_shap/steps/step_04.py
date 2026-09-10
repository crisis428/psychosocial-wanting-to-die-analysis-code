# %% [cell 14]
def run_one_outer_fold(
    group: str,
    model_name: str,
    repeat: int,
    fold: int,
    force: bool = False,
) -> dict[str, Any]:
    folder = checkpoint_dir(group, model_name, repeat, fold)
    folder.mkdir(parents=True, exist_ok=True)

    if checkpoint_complete(group, model_name, repeat, fold) and not force:
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

    started = time.time()
    local_seed = fold_seed(group, model_name, repeat, fold)
    set_global_seed(local_seed)

    group_data, predictors = group_dataframe(df, group)
    splits = outer_splits(group_data, group, repeat)
    train_index, test_index = splits[fold]

    x_train = group_data.loc[train_index, predictors]
    y_train = group_data.loc[train_index, OUTCOME].to_numpy(dtype=int)
    x_test = group_data.loc[test_index, predictors]
    y_test = group_data.loc[test_index, OUTCOME].to_numpy(dtype=int)

    model, search_space = build_model_and_search_space(
        model_name,
        local_seed,
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=local_seed + 1_000_000,
    )

    requested_iterations = SEARCH_ITERATIONS[model_name]
    n_iter = min(
        requested_iterations,
        discrete_space_size(search_space),
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=SEARCH_N_JOBS,
        cv=inner_cv,
        refit=True,
        random_state=local_seed,
        return_train_score=False,
        error_score="raise",
        verbose=0,
    )

    search.fit(x_train, y_train)

    y_prob = search.best_estimator_.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)
    metrics = calculate_metrics(y_test, y_prob, THRESHOLD)

    prediction_frame = pd.DataFrame(
        {
            "Group": group,
            "Model": model_name,
            "Repeat": repeat + 1,
            "Fold": fold + 1,
            ID_COLUMN: group_data.loc[
                test_index, ID_COLUMN
            ].to_numpy(),
            "Source_Index": group_data.loc[
                test_index, "_source_index"
            ].to_numpy(),
            "Y_True": y_test,
            "Y_Probability": y_prob,
            "Y_Predicted_0_5": y_pred,
        }
    )

    fold_metrics = {
        "Group": group,
        "Model": model_name,
        "Repeat": repeat + 1,
        "Fold": fold + 1,
        "N_Train": len(train_index),
        "N_Test": len(test_index),
        "Train_Events": int(y_train.sum()),
        "Test_Events": int(y_test.sum()),
        "Best_Inner_AUROC": float(search.best_score_),
        **metrics,
    }

    best_parameters = {
        "Group": group,
        "Model": model_name,
        "Repeat": repeat + 1,
        "Fold": fold + 1,
        "Predictors": predictors,
        "Parameters": search.best_params_,
    }

    runtime = {
        "Group": group,
        "Model": model_name,
        "Repeat": repeat + 1,
        "Fold": fold + 1,
        "Seed": local_seed,
        "Started_At_Epoch": started,
        "Finished_At_Epoch": time.time(),
        "Elapsed_Seconds": time.time() - started,
        "Python": platform.python_version(),
    }

    atomic_write_csv(
        prediction_frame,
        folder / "predictions.csv",
    )
    atomic_write_json(
        fold_metrics,
        folder / "fold_metrics.json",
    )
    atomic_write_json(
        best_parameters,
        folder / "best_parameters.json",
    )
    atomic_write_npz(
        folder / "split_indices.npz",
        train_index=np.asarray(train_index, dtype=int),
        test_index=np.asarray(test_index, dtype=int),
    )
    atomic_write_json(
        runtime,
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
        "AUROC": metrics["AUROC"],
        "elapsed_seconds": runtime["Elapsed_Seconds"],
    }


def run_model_checkpoints(
    group: str,
    model_name: str,
    force: bool = False,
) -> pd.DataFrame:
    results = []
    total = OUTER_REPEATS * OUTER_SPLITS

    print(
        f"\n[{group.upper()} | {model_name}] "
        f"총 {total}개 fold 확인"
    )

    for repeat in range(OUTER_REPEATS):
        for fold in range(OUTER_SPLITS):
            label = (
                f"{group}/{model_name}/"
                f"repeat {repeat + 1}/{OUTER_REPEATS}/"
                f"fold {fold + 1}/{OUTER_SPLITS}"
            )

            try:
                result = run_one_outer_fold(
                    group=group,
                    model_name=model_name,
                    repeat=repeat,
                    fold=fold,
                    force=force,
                )
                results.append(result)

                if result["status"] == "skipped":
                    print("SKIP:", label)
                else:
                    print(
                        "DONE:",
                        label,
                        "| AUROC:",
                        f"{result['AUROC']:.4f}",
                        "| sec:",
                        f"{result['elapsed_seconds']:.1f}",
                    )

            except Exception as error:
                error_path = (
                    ERROR_ROOT
                    / (
                        f"{group}_{model_name}_"
                        f"repeat{repeat + 1:02d}_"
                        f"fold{fold + 1:02d}.txt"
                    )
                )
                error_text = (
                    f"{label}\n"
                    f"{type(error).__name__}: {error}\n\n"
                    f"{traceback.format_exc()}"
                )
                atomic_write_text(error_text, error_path)

                results.append(
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

    status = completion_status()
    completed = model_completion_count(group, model_name)

    print(
        f"\n[{group.upper()} | {model_name}] "
        f"완료 {completed}/{total}"
    )

    return pd.DataFrame(results)


def rerun_failed_or_pending(
    group: str,
    model_name: str,
) -> pd.DataFrame:
    return run_model_checkpoints(
        group=group,
        model_name=model_name,
        force=False,
    )


print("Fold 체크포인트 실행 함수 준비 완료")

# %% [cell 18]
result_total_1 = run_model_checkpoints("total", "LogisticRegression")

# %% [cell 20]
result_total_2 = run_model_checkpoints("total", "DecisionTree")

# %% [cell 22]
result_total_3 = run_model_checkpoints("total", "RandomForest")

# %% [cell 24]
result_total_4 = run_model_checkpoints("total", "SVM")

# %% [cell 26]
result_total_5 = run_model_checkpoints("total", "XGBoost")

# %% [cell 28]
result_total_6 = run_model_checkpoints("total", "CatBoost")

# %% [cell 29]
display(
    completion_status().query("Group == 'total'")
)

# %% [cell 32]
result_male_1 = run_model_checkpoints("male", "LogisticRegression")

# %% [cell 34]
result_male_2 = run_model_checkpoints("male", "DecisionTree")

# %% [cell 36]
result_male_3 = run_model_checkpoints("male", "RandomForest")

# %% [cell 38]
result_male_4 = run_model_checkpoints("male", "SVM")

# %% [cell 40]
result_male_5 = run_model_checkpoints("male", "XGBoost")

# %% [cell 42]
result_male_6 = run_model_checkpoints("male", "CatBoost")

# %% [cell 43]
display(
    completion_status().query("Group == 'male'")
)

# %% [cell 46]
result_female_1 = run_model_checkpoints("female", "LogisticRegression")

# %% [cell 48]
result_female_2 = run_model_checkpoints("female", "DecisionTree")

# %% [cell 50]
result_female_3 = run_model_checkpoints("female", "RandomForest")

# %% [cell 52]
result_female_4 = run_model_checkpoints("female", "SVM")

# %% [cell 54]
result_female_5 = run_model_checkpoints("female", "XGBoost")

# %% [cell 56]
result_female_6 = run_model_checkpoints("female", "CatBoost")

# %% [cell 57]
display(
    completion_status().query("Group == 'female'")
)

# %% [cell 59]
status = completion_status()
display(status)

print(
    "\n전체 완료:",
    int(status["Completed"].sum()),
    "/",
    int(status["Expected"].sum()),
)
print("전체 미완료:", int(status["Pending"].sum()))

error_files = sorted(ERROR_ROOT.glob("*.txt"))
print("오류 로그 수:", len(error_files))
for path in error_files[:20]:
    print("-", path.name)
