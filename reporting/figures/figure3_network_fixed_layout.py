from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

OUTCOME = "py_thoughts_wanting_to_die"

NODE_CODE = {
    "py_thoughts_wanting_to_die": "SI",
    "gad7_total": "GA",
    "pss14_total": "PS",
    "low_self_esteem_score": "SE",
    "poor_subjective_physical_health": "PH",
    "poor_subjective_mental_health": "MH",
    "low_sense_of_belonging": "BE",
    "low_perceived_social_equality": "EQ",
    "low_social_trust": "TR",
    "age_years": "AG",
    "sex_female": "FE",
    "married_current": "MA",
    "employed_corrected": "EM",
    "living_alone": "LA",
}

NODE_DOMAIN = {
    "py_thoughts_wanting_to_die": "Outcome",
    "gad7_total": "Psychological",
    "pss14_total": "Psychological",
    "low_self_esteem_score": "Psychological",
    "poor_subjective_physical_health": "Subjective health",
    "poor_subjective_mental_health": "Subjective health",
    "low_sense_of_belonging": "Social-contextual",
    "low_perceived_social_equality": "Social-contextual",
    "low_social_trust": "Social-contextual",
    "age_years": "Demographic",
    "sex_female": "Demographic",
    "married_current": "Demographic",
    "employed_corrected": "Demographic",
    "living_alone": "Demographic",
}

DOMAIN_COLOR = {
    "Outcome": "#000000",
    "Psychological": "#A65628",
    "Subjective health": "#1F9E89",
    "Social-contextual": "#4DAF4A",
    "Demographic": "#8E44AD",
}

POSITIVE_EDGE_COLOR = "#D73027"
NEGATIVE_EDGE_COLOR = "#4575B4"

GROUP_SPECS = {
    "total": ("A", "Total sample", "Figure_3A_Total_Sample"),
    "male": ("B", "Male subgroup", "Figure_3B_Male_Subgroup"),
    "female": ("C", "Female subgroup", "Figure_3C_Female_Subgroup"),
}

ALL_NODE_ORDER = list(NODE_CODE)
NODE_INDEX = {node: index for index, node in enumerate(ALL_NODE_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-layout network panels A, B, and C without a legend "
            "from network_results_FINAL_v1.zip."
        )
    )
    parser.add_argument(
        "--results-zip",
        type=Path,
        default=Path("network_results_FINAL_v1.zip"),
        help="Final network result ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figure3_network_panels"),
        help="Directory for PNG and PDF outputs.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_weight_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix


def edge_curvature(node_a: str, node_b: str) -> float:
    if OUTCOME in (node_a, node_b):
        return 0.0
    first, second = sorted((NODE_INDEX[node_a], NODE_INDEX[node_b]))
    distance = second - first
    magnitude = min(0.055 + 0.014 * distance, 0.22)
    sign = -1 if ((first + second) % 2) else 1
    return sign * magnitude


def draw_panel(
    weight_matrix: pd.DataFrame,
    coords: dict[str, dict[str, float]],
    common_edge_scale: float,
    panel: str,
    title: str,
    output_stem: str,
    output_dir: Path,
    dpi: int,
) -> None:
    nodes = list(weight_matrix.columns)
    figure, axis = plt.subplots(figsize=(8.6, 6.6))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for i, node_a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node_b = nodes[j]
            weight = float(weight_matrix.loc[node_a, node_b])
            if np.isclose(weight, 0.0):
                continue

            x1, y1 = coords[node_a]["x"], coords[node_a]["y"]
            x2, y2 = coords[node_b]["x"], coords[node_b]["y"]
            magnitude_ratio = min(abs(weight) / common_edge_scale, 1.0)
            linewidth = 0.18 + 5.4 * (magnitude_ratio ** 0.82)
            alpha = 0.08 + 0.82 * (magnitude_ratio ** 0.72)
            color = POSITIVE_EDGE_COLOR if weight > 0 else NEGATIVE_EDGE_COLOR

            axis.add_patch(
                FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={edge_curvature(node_a, node_b):.3f}",
                    linewidth=linewidth,
                    color=to_rgba(color, alpha),
                    capstyle="round",
                    joinstyle="round",
                    zorder=1,
                )
            )

    for node in nodes:
        x, y = coords[node]["x"], coords[node]["y"]
        node_color = DOMAIN_COLOR[NODE_DOMAIN[node]]
        outer_size = 1120 if node == OUTCOME else 840
        inner_size = 990 if node == OUTCOME else 730

        axis.scatter([x], [y], s=outer_size, c="#B8B8B8", edgecolors="none", zorder=3)
        axis.scatter(
            [x], [y], s=inner_size, c=node_color,
            edgecolors="white", linewidths=2.0, zorder=4,
        )
        axis.text(
            x, y, NODE_CODE[node], ha="center", va="center", color="white",
            fontsize=12.5 if node == OUTCOME else 11.5,
            fontweight="bold", zorder=5,
        )

    axis.text(
        0.01, 0.985, f"{panel}. {title}", transform=axis.transAxes,
        ha="left", va="top", fontsize=16, fontweight="bold",
    )
    axis.set_xlim(-1.58, 1.58)
    axis.set_ylim(-0.80, 1.30)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    plt.tight_layout(pad=0.6)

    figure.savefig(output_dir / f"{output_stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_zip = args.results_zip.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_zip.exists():
        raise FileNotFoundError(results_zip)

    extract_dir = output_dir / "_network_results_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(results_zip) as archive:
        archive.extractall(extract_dir)

    coords_frame = pd.read_csv(
        extract_dir / "total" / "12b_final_layered_layout_coordinates.csv"
    )
    coords = coords_frame.set_index("node")[["x", "y"]].to_dict("index")

    edge_scale = pd.read_csv(extract_dir / "05_common_edge_scale.csv")
    common_edge_scale = float(edge_scale["plotting_scale_maximum"].iloc[0])

    for group_name, (panel, title, output_stem) in GROUP_SPECS.items():
        weights = read_weight_matrix(
            extract_dir / group_name / "11_EBICglasso_weight_matrix.csv"
        )
        draw_panel(
            weights, coords, common_edge_scale, panel, title, output_stem,
            output_dir, args.dpi,
        )

    shutil.rmtree(extract_dir)
    print(f"Completed: {output_dir}")


if __name__ == "__main__":
    main()
