# src/utils/plotting.py

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def set_style():
    mpl.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 17,
        "figure.titlesize": 22,
    })


STYLES = {
    "PALM": dict(color="black", linestyle="--", marker="o", linewidth=2),
    "iPALM": dict(color="green", linestyle=":", marker="s", linewidth=2),
    "FISTA-PALM": dict(color="red", linestyle="-", marker="^", linewidth=2),
}


BAR_COLORS = {
    "PALM": "black",
    "iPALM": "green",
    "FISTA-PALM": "red",
}


def _handle_output(save_path=None, dpi=300, show=True, tight=True):
    if tight:
        plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_histories(histories, title="Convergence", save_path=None, dpi=300, show=True):
    set_style()

    plt.figure(figsize=(10, 6), facecolor="white")

    for name, hist in histories.items():
        plt.plot(hist, label=name, **STYLES.get(name, {}))

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend()

    _handle_output(save_path, dpi, show)


def plot_random_vs_svd(plot_data, sparsities, save_path=None, dpi=300, show=True):
    set_style()

    fig, axes = plt.subplots(1, len(sparsities), figsize=(18, 6), sharey=True)

    if len(sparsities) == 1:
        axes = [axes]

    init_styles = {
        "Random": dict(linestyle="-", marker="o"),
        "SVD": dict(linestyle="--", marker=None),
    }

    method_colors = {
        "PALM": "black",
        "iPALM": "green",
        "FISTA-PALM": "red",
    }

    for i, s in enumerate(sparsities):
        ax = axes[i]
        key = f"{int(s * 100)}%"

        for init in ["Random", "SVD"]:
            for method, hist in plot_data[key][init].items():
                ax.plot(
                    hist,
                    label=f"{method} ({init})",
                    color=method_colors.get(method),
                    linewidth=2,
                    markevery=5,
                    **init_styles[init],
                )

        ax.set_title(f"Sparsity {key}")
        ax.set_xlabel("Iteration")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":")

        if i == 0:
            ax.set_ylabel("Objective")

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.25)
    _handle_output(save_path, dpi, show, tight=False)


def plot_bar(results, configs, sparsities, save_path=None, dpi=300, show=True):
    set_style()

    methods = ["PALM", "iPALM", "FISTA-PALM"]
    fig, axs = plt.subplots(1, len(configs), figsize=(20, 6), facecolor="white")

    if len(configs) == 1:
        axs = [axs]

    width = 0.25
    x = np.arange(len(sparsities))
    handles = []

    for i, (m, n, r) in enumerate(configs):
        ax = axs[i]
        config_label = f"{m}x{n}, r={r}"

        for j, method in enumerate(methods):
            vals = [
                results[config_label][f"{int(s * 100)}%"][method]
                for s in sparsities
            ]

            bars = ax.bar(
                x + j * width,
                vals,
                width,
                label=method,
                color=BAR_COLORS.get(method),
            )

            if i == 0:
                handles.append(bars[0])

        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{int(s * 100)}%" for s in sparsities])
        ax.set_title(config_label)
        ax.set_xlabel("Sparsity")
        ax.grid(True, linestyle=":", axis="y")

        if i == 0:
            ax.set_ylabel("Objective")

    fig.legend(
        handles,
        methods,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(methods),
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.25, wspace=0.25)
    _handle_output(save_path, dpi, show, tight=False)


def plot_time_grid(results, save_path=None, dpi=300, show=True):
    set_style()

    fig, axs = plt.subplots(3, 3, figsize=(18, 16), facecolor="white")
    axs = axs.flatten()

    handles = []
    labels = []

    for i, (label, data) in enumerate(results):
        ax = axs[i]

        for method, (times, hist) in data.items():
            style = STYLES.get(method, {}).copy()
            style.pop("marker", None)

            line, = ax.plot(times, hist, label=method, **style)

            if i == 0:
                handles.append(line)
                labels.append(method)

        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective")
        ax.set_title(label)
        ax.grid(True, linestyle=":")

    for j in range(len(results), len(axs)):
        axs[j].axis("off")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.1, hspace=0.35, wspace=0.3)
    _handle_output(save_path, dpi, show, tight=False)


def plot_milestone_grid(
    df,
    matrix_configs,
    sparsity_levels,
    milestones,
    title="Objective Values at Iteration Milestones",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    nrows = len(sparsity_levels)
    ncols = len(matrix_configs)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(20, 14),
        sharex=True,
        sharey=False,
        facecolor="white",
    )

    if nrows == 1:
        axs = np.array([axs])
    if ncols == 1:
        axs = axs.reshape(nrows, 1)

    method_styles = {
        "PALM": {"color": "black", "linestyle": "--", "marker": "o"},
        "FISTA-PALM": {"color": "red", "linestyle": "-", "marker": "^"},
    }

    handles = []
    labels = []

    for row, sparsity in enumerate(sparsity_levels):
        sparsity_label = f"{int(sparsity * 100)}%"

        for col, (m, n, r) in enumerate(matrix_configs):
            ax = axs[row, col]

            sub = df[
                (df["m"] == m)
                & (df["n"] == n)
                & (df["rank"] == r)
                & (df["Sparsity"] == sparsity_label)
            ]

            for method in ["PALM", "FISTA-PALM"]:
                method_row = sub[sub["Method"] == method]

                values = [
                    method_row[f"Iter {k}"].values[0]
                    for k in milestones
                ]

                line, = ax.plot(
                    milestones,
                    values,
                    label=method,
                    linewidth=2,
                    **method_styles[method],
                )

                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(method)

            ax.set_yscale("log")
            ax.grid(True, linestyle=":")

            if row == 0:
                ax.set_title(f"{m}x{n}, r={r}")

            if col == 0:
                ax.set_ylabel(f"{sparsity_label}\nObjective")

            if row == nrows - 1:
                ax.set_xlabel("Iteration")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.09, hspace=0.35, wspace=0.25)

    _handle_output(save_path, dpi, show, tight=False)


def plot_singular_value_scaling_panels(
    df,
    small_scales,
    large_scales,
    title="Objective Value vs Singular Value Scale",
    subtitle=None,
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    method_styles = {
        "PALM": {"color": "black", "marker": "s", "linestyle": "-"},
        "FISTA-PALM": {"color": "red", "marker": "o", "linestyle": "-"},
    }

    panels = [
        ("Large Singular Values", large_scales, axs[0]),
        ("Small Singular Values", small_scales, axs[1]),
    ]

    for panel_title, scales, ax in panels:
        sub = df[df["Scale"].isin(scales)]

        for method in ["FISTA-PALM", "PALM"]:
            method_df = sub[sub["Method"] == method].sort_values("Scale")

            ax.plot(
                method_df["Scale"],
                method_df["Final Objective"],
                label=method,
                linewidth=2,
                **method_styles[method],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_title(panel_title)
        ax.set_xlabel("Singular value scale")
        ax.grid(True, which="both", linestyle=":")

    axs[0].set_ylabel("Objective Value at Iter 100")

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(title, y=0.99)

    if subtitle is not None:
        fig.text(0.5, 0.91, subtitle, ha="center", fontsize=14)

    fig.subplots_adjust(top=0.82, bottom=0.25, wspace=0.25)

    _handle_output(save_path, dpi, show, tight=False)


def plot_conditioning_comparison(
    df,
    scales,
    title="Well-Conditioned vs Ill-Conditioned Matrices",
    subtitle=None,
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    method_styles = {
        "PALM": {"color": "black", "marker": "s", "linestyle": "-"},
        "FISTA-PALM": {"color": "red", "marker": "o", "linestyle": "-"},
    }

    panels = [
        ("Well-conditioned", axs[0]),
        ("Ill-conditioned", axs[1]),
    ]

    for condition, ax in panels:
        sub = df[df["Condition"] == condition]

        for method in ["PALM", "FISTA-PALM"]:
            method_df = sub[sub["Method"] == method].sort_values("Scale")

            ax.plot(
                method_df["Scale"],
                method_df["Final Objective"],
                label=method,
                linewidth=2,
                **method_styles[method],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_title(condition)
        ax.set_xlabel("Singular value scale")
        ax.grid(True, which="both", linestyle=":")

    axs[0].set_ylabel("Objective Value at Iter 100")

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(title, y=0.99)

    if subtitle is not None:
        fig.text(0.5, 0.91, subtitle, ha="center", fontsize=14)

    fig.subplots_adjust(top=0.82, bottom=0.25, wspace=0.25)

    _handle_output(save_path, dpi, show, tight=False)
    

def plot_reconstruction_grid(
    image_stack,
    labels,
    title="Reconstructions",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    ax.imshow(image_stack, cmap="gray")
    ax.axis("off")
    ax.set_title(title)

    row_height = image_stack.shape[0] / len(labels)

    for i, label in enumerate(labels):
        ax.text(
            -5,
            i * row_height + row_height / 2,
            label,
            va="center",
            ha="right",
        )

    _handle_output(save_path, dpi, show)


def plot_recon_and_loss(
    img1,
    img2,
    loss_p,
    loss_f,
    loss_i,
    title="Reconstructions and Convergence",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    axs[0].imshow(img1, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Recon (100)")

    axs[1].imshow(img2, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Recon (200)")

    axs[2].plot(loss_p, label="PALM", **STYLES["PALM"])
    axs[2].plot(loss_f, label="FISTA-PALM", **STYLES["FISTA-PALM"])
    axs[2].plot(loss_i, label="iPALM", **STYLES["iPALM"])
    axs[2].set_yscale("log")
    axs[2].set_title("Convergence")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Objective")
    axs[2].grid(True, linestyle=":")
    axs[2].legend()

    fig.suptitle(title)
    fig.subplots_adjust(top=0.85)
    _handle_output(save_path, dpi, show)


def plot_orl_recon_and_loss(
    img_100,
    img_200,
    histories,
    row_labels=None,
    title="ORL Face Reconstructions and Convergence",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    if row_labels is None:
        row_labels = ["Original", "PALM", "FISTA", "iPALM"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    axs[0].imshow(img_100, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Reconstructions (100 Iter)")

    axs[1].imshow(img_200, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Reconstructions (200 Iter)")

    row_height = img_100.shape[0] / len(row_labels)

    for ax in axs[:2]:
        for j, label in enumerate(row_labels):
            ax.text(
                -5,
                j * row_height + row_height / 2,
                label,
                va="center",
                ha="right",
                color="black",
            )

    for method, hist in histories.items():
        axs[2].plot(hist, label=method, **STYLES.get(method, {}))

    axs[2].set_yscale("log")
    axs[2].set_title("Convergence (200 Iter)")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Objective")
    axs[2].grid(True, linestyle=":")
    axs[2].legend()

    fig.suptitle(title)
    _handle_output(save_path, dpi, show)


def plot_berkeley_recon_and_loss(
    img_100,
    img_200,
    histories,
    image_size=(128, 128),
    row_labels=None,
    title="BSDS500 Image Denoising and Convergence",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    if row_labels is None:
        row_labels = ["Original", "Noisy", "PALM", "FISTA", "iPALM"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 7), facecolor="white")

    axs[0].imshow(img_100, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Reconstructions (100 Iter)")

    axs[1].imshow(img_200, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Reconstructions (200 Iter)")

    row_height = image_size[0]

    for ax in axs[:2]:
        for j, label in enumerate(row_labels):
            ax.text(
                -5,
                j * row_height + row_height / 2,
                label,
                va="center",
                ha="right",
                color="black",
            )

    for method, hist in histories.items():
        axs[2].plot(hist, label=method, **STYLES.get(method, {}))

    axs[2].set_yscale("log")
    axs[2].set_title("Convergence (200 Iter)")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Objective")
    axs[2].grid(True, linestyle=":")
    axs[2].legend(loc="lower center", ncol=3)

    fig.suptitle(title)
    _handle_output(save_path, dpi, show)


def plot_three_block_loss_comparison(
    histories,
    title="COIL-20 3-Block Sparse Factorisation",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")

    methods = ["PALM", "FISTA-PALM"]

    for method in methods:
        if method in histories:
            ax.plot(
                histories[method],
                label=method,
                **STYLES.get(method, {}),
            )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend(frameon=False)

    _handle_output(save_path, dpi, show)


def plot_three_block_reconstruction_grid(
    original,
    reconstruction,
    image_shape=(32, 32),
    n_images=10,
    title="3-Block Reconstruction",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    n_images = min(n_images, original.shape[1], reconstruction.shape[1])

    fig, axs = plt.subplots(
        2,
        n_images,
        figsize=(2 * n_images, 4.5),
        facecolor="white",
    )

    for i in range(n_images):
        axs[0, i].imshow(original[:, i].reshape(image_shape), cmap="gray")
        axs[0, i].axis("off")

        axs[1, i].imshow(reconstruction[:, i].reshape(image_shape), cmap="gray")
        axs[1, i].axis("off")

    axs[0, 0].set_ylabel("Original", rotation=0, labelpad=45, va="center")
    axs[1, 0].set_ylabel("Recon", rotation=0, labelpad=45, va="center")

    fig.suptitle(title)
    fig.subplots_adjust(top=0.82, hspace=0.05, wspace=0.05)

    _handle_output(save_path, dpi, show, tight=False)


def plot_three_block_combined_reconstruction_grid(
    A,
    reconstructions,
    image_shape=(32, 32),
    n_images=10,
    row_labels=None,
    title="COIL-20 Three-Block Reconstructions",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    if row_labels is None:
        row_labels = ["Original"] + list(reconstructions.keys())

    rows = [A] + list(reconstructions.values())
    n_rows = len(rows)
    n_images = min(n_images, A.shape[1])

    fig, axs = plt.subplots(
        n_rows,
        n_images,
        figsize=(2 * n_images, 2.2 * n_rows),
        facecolor="white",
    )

    if n_rows == 1:
        axs = np.array([axs])

    for row_idx, row_data in enumerate(rows):
        for col_idx in range(n_images):
            ax = axs[row_idx, col_idx]
            ax.imshow(row_data[:, col_idx].reshape(image_shape), cmap="gray")
            ax.axis("off")

        axs[row_idx, 0].text(
            -0.28,
            0.5,
            row_labels[row_idx],
            transform=axs[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=18,
            color="black",
        )

    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.88,
        bottom=0.03,
        hspace=0.08,
        wspace=0.05,
    )

    _handle_output(save_path, dpi, show, tight=False)
    

def plot_three_block_dataset_comparison(
    coil_histories,
    synthetic_histories,
    title="Loss Curves: 3-Block Sparse Factorisation",
    save_path=None,
    dpi=300,
    show=True,
):
    set_style()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

    panels = [
        ("COIL-20 Dataset", coil_histories, axs[0]),
        ("Synthetic Matrix", synthetic_histories, axs[1]),
    ]

    handles = []
    labels = []

    methods = ["PALM", "FISTA-PALM"]

    for panel_title, histories, ax in panels:
        for method in methods:
            if method in histories:
                line, = ax.plot(
                    histories[method],
                    label=method,
                    **STYLES.get(method, {}),
                )

                if panel_title == "COIL-20 Dataset":
                    handles.append(line)
                    labels.append(method)

        ax.set_title(panel_title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.25, wspace=0.3)

    _handle_output(save_path, dpi, show, tight=False)