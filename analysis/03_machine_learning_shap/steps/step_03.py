# %% [cell 12]
def build_model_and_search_space(
    model_name: str,
    seed: int,
) -> tuple[Pipeline, dict[str, list[Any]]]:
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=seed,
        )
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        search_space = {
            "model__C": [
                0.001, 0.003, 0.01, 0.03, 0.1,
                0.3, 1.0, 3.0, 10.0, 30.0,
            ],
            "model__penalty": ["l1", "l2"],
        }

    elif model_name == "DecisionTree":
        estimator = DecisionTreeClassifier(random_state=seed)
        pipeline = Pipeline([("model", estimator)])
        search_space = {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None, 3, 4, 5, 6, 8, 10],
            "model__min_samples_split": [2, 5, 10, 20, 40],
            "model__min_samples_leaf": [1, 2, 5, 10, 20, 40],
            "model__max_features": [None, "sqrt", "log2", 0.5, 0.8],
            "model__ccp_alpha": [0.0, 0.0001, 0.0005, 0.001, 0.005],
        }

    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(
            random_state=seed,
            n_jobs=MODEL_THREADS,
        )
        pipeline = Pipeline([("model", estimator)])
        search_space = {
            "model__n_estimators": [300, 500, 800, 1000],
            "model__max_depth": [None, 4, 6, 8, 10, 15],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", 0.5, 0.8],
            "model__bootstrap": [True],
        }

    elif model_name == "SVM":
        estimator = SVC(
            probability=True,
            random_state=seed,
        )
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        search_space = {
            "model__C": [
                0.03, 0.1, 0.3, 1.0,
                3.0, 10.0, 30.0,
            ],
            "model__gamma": [
                "scale", "auto",
                0.001, 0.003, 0.01, 0.03, 0.1,
            ],
            "model__kernel": ["rbf"],
        }

    elif model_name == "XGBoost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=MODEL_THREADS,
        )
        pipeline = Pipeline([("model", estimator)])
        search_space = {
            "model__n_estimators": [300, 500, 800, 1000],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
            "model__reg_lambda": [0.1, 1.0, 3.0, 10.0],
        }

    elif model_name == "CatBoost":
        estimator = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=MODEL_THREADS,
        )
        pipeline = Pipeline([("model", estimator)])
        search_space = {
            "model__iterations": [300, 500, 800, 1000],
            "model__depth": [3, 4, 5, 6, 7],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__l2_leaf_reg": [1, 3, 5, 10, 20],
            "model__random_strength": [0.0, 0.5, 1.0, 2.0],
            "model__bagging_temperature": [0.0, 0.5, 1.0, 2.0],
            "model__border_count": [32, 64, 128],
        }

    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")

    return pipeline, search_space


def discrete_space_size(
    search_space: dict[str, list[Any]],
) -> int:
    size = 1
    for values in search_space.values():
        size *= len(values)
    return size


for model_name in MODELS:
    _, space = build_model_and_search_space(model_name, SEED)
    print(
        model_name,
        "| search iterations:",
        SEARCH_ITERATIONS[model_name],
        "| possible combinations:",
        f"{discrete_space_size(space):,}",
    )
