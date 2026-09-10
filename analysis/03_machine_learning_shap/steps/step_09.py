# %% [cell 83]
def read_global_shap_importance(
    group: str,
    model_name: str,
) -> pd.DataFrame:
    path = (
        SHAP_ROOT
        / "combined"
        / group
        / model_name
        / "global_shap_importance.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"SHAP 중요도 파일이 없습니다: {path}\n"
            "Part I와 Part J를 전체·남성·여성에 대해 먼저 실행하세요."
        )

    return pd.read_csv(path)


fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(18, 7),
)

for axis, group, panel_label in zip(
    axes,
    ORIGINAL_GROUP_ORDER,
    panel_labels,
):
    importance = read_global_shap_importance(
        group,
        EXPLANATION_MODEL,
    ).copy()

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=True,
    )

    axis.barh(
        importance["Feature_Label"],
        importance["Mean_Absolute_SHAP"],
    )
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title(
        f"{panel_label}. {ORIGINAL_GROUP_LABELS[group]}",
        loc="left",
        fontweight="bold",
    )

save_figure_to_original_outputs(
    "Figure_2_SHAP_importance_total_male_female_recreated"
)

# %% [cell 85]
beeswarm_paths = []

for group in ORIGINAL_GROUP_ORDER:
    path = (
        SHAP_ROOT
        / "combined"
        / group
        / EXPLANATION_MODEL
        / f"figure_shap_beeswarm_{group}_{EXPLANATION_MODEL}.png"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"SHAP beeswarm이 없습니다: {path}\n"
            "Part I와 Part J를 전체·남성·여성에 대해 먼저 실행하세요."
        )

    beeswarm_paths.append(path)


fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10, 24),
)

for axis, path, group, panel_label in zip(
    axes,
    beeswarm_paths,
    ORIGINAL_GROUP_ORDER,
    panel_labels,
):
    image = plt.imread(path)
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"{panel_label}. {ORIGINAL_GROUP_LABELS[group]}",
        loc="left",
        fontweight="bold",
        fontsize=13,
    )

save_figure_to_original_outputs(
    "Supplementary_Figure_S1_SHAP_summary_total_male_female_recreated"
)

# %% [cell 87]
original_output_inventory = pd.DataFrame(
    [
        {
            "Original_item": "Table 1",
            "Content": "Participant characteristics by outcome status",
            "Environment": "Python statistical analysis",
            "Current_status": "Completed separately",
        },
        {
            "Original_item": "Figure 1",
            "Content": "ROC curves: total, male, female",
            "Environment": "This notebook",
            "Current_status": "Recreated",
        },
        {
            "Original_item": "Figure 2",
            "Content": "SHAP importance: total, male, female",
            "Environment": "This notebook",
            "Current_status": "Recreated after all-group SHAP",
        },
        {
            "Original_item": "Figure 3",
            "Content": "Networks: total, male, female",
            "Environment": "R network analysis",
            "Current_status": "Pending network completion",
        },
        {
            "Original_item": "Supplementary Table S1",
            "Content": "Six-model performance: total, male, female",
            "Environment": "This notebook",
            "Current_status": "Recreated",
        },
        {
            "Original_item": "Supplementary Figure S1",
            "Content": "SHAP beeswarm: total, male, female",
            "Environment": "This notebook",
            "Current_status": "Recreated after all-group SHAP",
        },
        {
            "Original_item": "Supplementary Figure S2",
            "Content": "Strength and expected influence: total, male, female",
            "Environment": "R network analysis",
            "Current_status": "Pending network completion",
        },
        {
            "Original_item": "Supplementary Figure S3",
            "Content": "Bridge expected influence: total, male, female",
            "Environment": "R network analysis",
            "Current_status": "Pending network completion",
        },
        {
            "Original_item": "Supplementary Table S2",
            "Content": "Centrality indices: total, male, female",
            "Environment": "R network analysis",
            "Current_status": "Pending network completion",
        },
        {
            "Original_item": "Supplementary Table S3",
            "Content": "SHAP vs network integrated interpretation",
            "Environment": "Python + R integration",
            "Current_status": "Pending network completion",
        },
        {
            "Original_item": "Supplementary Figure S4",
            "Content": "Network edge accuracy and centrality stability",
            "Environment": "R network analysis",
            "Current_status": "Pending network completion",
        },
    ]
)

atomic_write_csv(
    original_output_inventory,
    ORIGINAL_OUTPUT_ROOT / "original_output_inventory.csv",
)

display(original_output_inventory)

# %% [cell 89]
final_status = completion_status()
display(final_status)

print("\n주요 결과 파일")
for path in sorted(COMBINED_ROOT.glob("*")):
    print("-", path.relative_to(RESULT_ROOT))

print("\n머신러닝 그림")
for path in sorted(FIGURE_ROOT.glob("*.png")):
    print("-", path.relative_to(RESULT_ROOT))

print("\nSHAP 결과")
for path in sorted(SHAP_ROOT.rglob("*.png")):
    print("-", path.relative_to(RESULT_ROOT))

zip_path = shutil.make_archive(
    str(SCRIPT_DIR / "full_results_checkpointed"),
    "zip",
    root_dir=RESULT_ROOT,
)

print("\n기존 원고 재생성 출력")
for path in sorted(ORIGINAL_OUTPUT_ROOT.glob("*")):
    print("-", path.relative_to(RESULT_ROOT))

print("\n업로드할 ZIP:", zip_path)
