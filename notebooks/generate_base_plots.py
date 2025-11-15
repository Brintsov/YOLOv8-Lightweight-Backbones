import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH_QUANT = Path("../docs/profiling_quantized_results_cpu.csv")
CSV_PATH_NANO = Path("../docs/profiling_results_cpu_and_gpu.csv")

OUTPUT_DIR = Path("../docs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAP_OVERALL_COL = "MaP"
MAP_SMALL_COL = "MaP@[area=small]"
MAP_MEDIUM_COL = "MaP@[area=medium]"
MAP_LARGE_COL = "MaP@[area=large]"

RECALL_OVERALL_COL = "Recall@[max_detections=100]"
FLOPS_G_COL = "flops_g"
FPS_COL = "GPU:0_fps"
CPU_FPS_COL = "CPU:0_fps"
LATENCY_COL = "avg_latency_sec"
RAM_COL = "ram_delta_mb"

sns.set_theme(style="whitegrid", context="notebook")


def load_data(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if LATENCY_COL in df.columns:
        df[LATENCY_COL] = df[LATENCY_COL] * 1000
    return df.sort_values("model")


def add_value_labels_horizontal(ax, decimals=2):
    for p in ax.patches:
        value = round(p.get_width(), decimals)
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(x + 0.01 * abs(x), y, f"{value:.2f}", va="center", fontsize=6)


def barplot_h(summary, x, y, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(summary[y], summary[x])
    add_value_labels_horizontal(ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(y, fontsize=6)
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def grouped_map_plot(summary, output_dir):
    df_melt = summary.melt(
        id_vars=["model"],
        value_vars=[MAP_SMALL_COL, MAP_MEDIUM_COL, MAP_LARGE_COL],
        var_name="Object Size",
        value_name="mAP"
    )
    df_melt["Object Size"] = df_melt["Object Size"].replace({
        MAP_SMALL_COL: "Small",
        MAP_MEDIUM_COL: "Medium",
        MAP_LARGE_COL: "Large",
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax = sns.barplot(
        data=df_melt,
        y="model", x="mAP",
        hue="Object Size", orient="h",
        edgecolor="black",
        ax=ax,
    )
    for p in ax.patches:
        value = p.get_width()
        if value:
            x = p.get_x() + p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.annotate(
                f"{value:.1%}",
                (x, y),
                ha="left",
                va="center",
                fontsize=5,
                xytext=(4, 0),
                textcoords="offset points"
            )
    ax.set_title("Models mAP by Object Sizes", fontsize=9, pad=10)
    ax.set_xlabel("mAP", fontsize=7)
    ax.set_ylabel("Model", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        title="Object Size",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.7")
    legend.get_frame().set_facecolor("white")
    legend.get_title().set_fontsize(8)
    for text in legend.get_texts():
        text.set_fontsize(7)
    fig.tight_layout()
    fig.savefig(output_dir / "map_by_area_grouped.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap_map(summary, output_dir):
    heat_df = summary.set_index("model")[
        [MAP_SMALL_COL, MAP_MEDIUM_COL, MAP_LARGE_COL]
    ].rename(columns={"MaP@[area=small]": "Small", "MaP@[area=medium]": "Medium", "MaP@[area=large]": "Large"})
    plt.figure(figsize=(6, 4))
    sns.heatmap(heat_df, annot=True, fmt=f".1%", cmap="viridis", cbar_kws={"shrink": .7}, annot_kws={"fontsize": 7})
    plt.title("Models mAP by Object Sizes: small / medium / large", fontsize=11)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "map_heatmap.png", dpi=220)
    plt.close()


def scatter(summary, x, y, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(summary[x], summary[y])
    for _, row in summary.iterrows():
        ax.text(row[x], row[y], row["model"], fontsize=6, ha="left", va="bottom")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def generate_plots_base_nano(summary):
    barplot_h(summary, MAP_OVERALL_COL, "model",
              "Overall mAP", "mAP",
              OUTPUT_DIR / "nano_base" / "map_overall_hbar.png")

    grouped_map_plot(summary, OUTPUT_DIR / "nano_base")
    heatmap_map(summary, OUTPUT_DIR / "nano_base")

    scatter(summary, FLOPS_G_COL, MAP_OVERALL_COL,
            "mAP vs FLOPs", "GFLOPs", "mAP",
            OUTPUT_DIR / "nano_base" / "map_vs_flops.png")

    scatter(summary, FPS_COL, MAP_OVERALL_COL,
            "mAP vs FPS", "FPS", "mAP",
            OUTPUT_DIR / "nano_base" / "map_vs_fps.png")

    barplot_h(summary, FPS_COL, "model",
              "FPS per model with CPU", "FPS",
              OUTPUT_DIR / "nano_base" / "fps_hbar.png")

    barplot_h(summary, CPU_FPS_COL, "model",
              "FPS per Model with CPU", "FPS",
              OUTPUT_DIR / "nano_base" / "cpu_fps_hbar.png")

    barplot_h(summary, LATENCY_COL, "model",
              "Latency per model", "Latency (ms)",
              OUTPUT_DIR / "nano_base" / "latency_hbar.png")

    barplot_h(summary, RAM_COL, "model",
              "RAM delta", "RAM (KB)",
              OUTPUT_DIR / "nano_base" / "ram_hbar.png")

    barplot_h(summary, FLOPS_G_COL, "model",
              "Model complexity (GFLOPs)", "GFLOPs",
              OUTPUT_DIR / "nano_base" / "flops_hbar.png")


def generate_plots_quantized(summary):

    barplot_h(summary, MAP_OVERALL_COL, "model",
              "Overall mAP", "mAP",
              OUTPUT_DIR/ "quantized" / "map_overall_hbar.png")

    grouped_map_plot(summary, OUTPUT_DIR / "quantized")
    heatmap_map(summary, OUTPUT_DIR / "quantized")

    barplot_h(summary, CPU_FPS_COL, "model",
              "FPS per Model with CPU", "FPS",
              OUTPUT_DIR / "quantized" / "cpu_fps_hbar.png")

    barplot_h(summary, LATENCY_COL, "model",
              "Latency per model", "Latency (ms)",
              OUTPUT_DIR / "quantized" / "latency_hbar.png")

    barplot_h(summary, RAM_COL, "model",
              "RAM delta", "RAM (KB)",
              OUTPUT_DIR / "quantized" / "ram_hbar.png")


def main():
    (OUTPUT_DIR / "quantized").mkdir(exist_ok=True)
    (OUTPUT_DIR / "nano_base").mkdir(exist_ok=True)
    df = load_data(CSV_PATH_QUANT)
    print(df.to_string(index=False, float_format="%.3f"))
    generate_plots_quantized(df)
    df = load_data(CSV_PATH_NANO)
    print(df.to_string(index=False, float_format="%.3f"))
    generate_plots_base_nano(df)
    print("\nSaved all plots to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
