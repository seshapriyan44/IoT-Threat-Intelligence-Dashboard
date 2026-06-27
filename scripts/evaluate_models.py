"""
evaluate_models.py
==================
Full evaluation pipeline for the IoT Botnet Detection project.

Evaluates both the Centralized and Federated Autoencoder models against a
BALANCED dataset (equal benign and attack samples), computes classification
metrics including Specificity, and generates publication-ready charts saved
under static/charts/.

Usage:
    python evaluate_models.py

Outputs:
    static/charts/centralized_confusion.png
    static/charts/federated_confusion.png
    static/charts/model_comparison.png
    static/charts/model_metrics.json
    static/charts/centralized_metrics.json
    static/charts/federated_metrics.json
"""
import suppress_logs
import json
import os
import sys
import time
import joblib
import json

from pathlib import Path

import numpy as np
import pandas as pd

# Use non-interactive Agg backend — safe for headless/server environments
# Must be called BEFORE any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ── Constants ──────────────────────────────────────────────────────────────────

# Filesystem paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Decision thresholds (from predict.py — do not alter)
with open(BASE_DIR / "models" / "thresholds.json", "r") as f:
    THRESHOLDS = json.load(f)

DATASET_DIR: Path = BASE_DIR / "dataset"
BENIGN_CSV: Path = DATASET_DIR / "1.benign.csv"
CENTRALIZED_MODEL_PATH: Path = (BASE_DIR / "models" / "autoencoder_model.h5")
FEDERATED_MODEL_PATH: Path = (BASE_DIR / "models" / "federated_autoencoder.keras")
CHARTS_DIR: Path = (BASE_DIR / "static" / "charts")

# All attack CSV files — each row is ground-truth label 1
ATTACK_CSVS: list[Path] = [
    DATASET_DIR / "1.mirai.ack.csv",
    DATASET_DIR / "1.mirai.scan.csv",
    DATASET_DIR / "1.mirai.syn.csv",
    DATASET_DIR / "1.mirai.udp.csv",
    DATASET_DIR / "1.mirai.udpplain.csv",
    DATASET_DIR / "1.gafgyt.combo.csv",
    DATASET_DIR / "1.gafgyt.junk.csv",
    DATASET_DIR / "1.gafgyt.scan.csv",
    DATASET_DIR / "1.gafgyt.tcp.csv",
    DATASET_DIR / "1.gafgyt.udp.csv",
]

# Chart rendering
FIGURE_DPI: int = 300

# Dark-theme colour palette — mirrors the Flask dashboard design
PALETTE: dict[str, str] = {
    "bg":          "#0d1117",
    "surface":     "#161b22",
    "surface_alt": "#1c2128",
    "border":      "#30363d",
    "text":        "#e6edf3",
    "muted":       "#8b949e",
    "accent":      "#2f81f7",   # blue  — Centralized
    "green":       "#3fb950",   # green — Federated
    "red":         "#f85149",
    "yellow":      "#e3b341",
}


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_dataset() -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Load all CSVs, assign ground-truth labels, fit a MinMaxScaler
    exclusively on the FULL benign dataset, then build a BALANCED
    evaluation set by sampling the same number of benign rows as
    total attack rows.

    Preprocessing replicates predict.py exactly:
      - MinMaxScaler fitted only on the COMPLETE benign.csv
      - Sampled benign rows transformed with that same scaler
      - All attack rows transformed with that same scaler
      - Cast to float32

    Balancing strategy:
      - Load all attack CSVs → total_attack_rows
      - benign_df.sample(n=total_attack_rows, random_state=42)
      - Prevents accuracy inflation caused by class imbalance

    Returns:
        X_scaled : float32 ndarray of shape (n_samples, n_features)
        y        : int ndarray of shape (n_samples,)
                   0 = Benign, 1 = Attack
        scaler   : the fitted MinMaxScaler instance
    """
    # ── Validate paths before loading ─────────────────────────────────────────
    missing = [str(p) for p in [BENIGN_CSV, *ATTACK_CSVS] if not p.exists()]
    if missing:
        print("[ERROR] The following dataset files were not found:")
        for m in missing:
            print(f"        {m}")
        sys.exit(1)

    # ── Load FULL benign dataset — used exclusively to fit the scaler ──────────
    print(f"  Loading benign data  : {BENIGN_CSV}")
    benign_df = pd.read_csv(BENIGN_CSV, header=0)
    print(f"  Full benign rows     : {len(benign_df):,}")

    # ── Fit scaler ONLY on the COMPLETE benign dataset (mirrors predict.py) ────
    
    scaler = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")

    print("  Loaded scaler        : models/scaler.pkl")

    # ── Attack data → label 1 ─────────────────────────────────────────────────
    attack_frames: list[pd.DataFrame] = []
    for csv_path in ATTACK_CSVS:
        print(f"  Loading attack data  : {csv_path.name}")
        df = pd.read_csv(csv_path, header=0)
        attack_frames.append(df)

    attack_df = pd.concat(attack_frames, ignore_index=True)
    total_attack_rows = len(attack_df)
    print(f"  Total attack rows    : {total_attack_rows:,}")

    # ── Balance: sample the same number of benign rows as total attack rows ────
    # The scaler was already fitted on the full benign set above — this sample
    # only affects which rows enter the evaluation, not the scaler fit.
    sample_size = min(len(benign_df),total_attack_rows)
    benign_sampled = benign_df.sample(n=sample_size,random_state=42)
    attack_df = attack_df.sample(n=sample_size,random_state=42)
    attack_labels = np.ones(len(attack_df), dtype=int)
    benign_labels  = np.zeros(sample_size, dtype=int)
    print(f"  Benign sampled       : {sample_size:,} rows (balanced to attack count)")

    # ── Concatenate sampled benign + all attack rows ───────────────────────────
    all_features = pd.concat([benign_sampled, attack_df], ignore_index=True)
    all_labels   = np.concatenate([benign_labels, attack_labels])

    X_scaled = scaler.transform(all_features.values).astype(np.float32)

    # Guard against NaN values that can arise from zero-variance features
    if np.isnan(X_scaled).any():
        nan_count = int(np.isnan(X_scaled).sum())
        print(f"  [WARN] {nan_count:,} NaN value(s) found after scaling — replaced with 0.0")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    n_benign = int((all_labels == 0).sum())
    n_attack = int((all_labels == 1).sum())
    print(
        f"\n  Dataset ready — "
        f"Total: {len(X_scaled):,} | "
        f"Benign: {n_benign:,} | "
        f"Attack: {n_attack:,}"
    )

    return X_scaled, all_labels, scaler
    scaler.fit(benign_df.values)

    joblib.dump(scaler, "models/scaler.pkl")


# ── Model Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(
    model_path: Path,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict:
    """
    Load a Keras autoencoder, compute per-row reconstruction error (MSE),
    classify each row independently against the threshold, and return metrics.

    Classification rule (per-row, NOT average):
        MSE(row) >  threshold  →  Attack  (1)
        MSE(row) <= threshold  →  Benign  (0)

    Args:
        model_path  : Path to the .h5 or .keras model file.
        X           : Scaled feature matrix, float32, shape (n, features).
        y_true      : Ground-truth integer labels, shape (n,).
        threshold   : Decision boundary for anomaly classification.
        model_name  : Human-readable label for console output.

    Returns:
        dict containing:
            accuracy, precision, recall,
            f1, specificity               — float scalar metrics
            tp, fp, tn, fn               — int confusion matrix counts
            y_pred                        — int ndarray of predictions
            mse                           — float64 ndarray of per-row MSE
    """
    # ── Validate model path ────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    print(f"  Loading model        : {model_path}")
    # compile=False avoids errors if the saved model used custom losses/metrics
    model = load_model(str(model_path), compile=False)

    print(f"  Running inference on : {len(X):,} samples ...")
    # batch_size=1024 balances GPU/CPU memory and throughput
    X_reconstructed = model.predict(X, batch_size=1024, verbose=0)

    # Per-row mean squared reconstruction error
    # Each row is classified independently — do NOT average across rows
    mse: np.ndarray = np.mean((X - X_reconstructed) ** 2, axis=1)
    
    benign_errors = mse[y_true == 0]
    attack_errors = mse[y_true == 1]

    print("\n===== BENIGN =====")
    print(np.percentile(benign_errors,[50,90,95,99]))

    print("\n===== ATTACK =====")
    print(np.percentile(attack_errors,[1,5,10,25,50]))

    # Apply threshold: error > threshold signals anomalous (Attack) traffic
    y_pred: np.ndarray = (mse > threshold).astype(int)

    fn_mask = (y_true == 1) & (y_pred == 0)

    print("\n===== FALSE NEGATIVES =====")
    print(f"Count : {fn_mask.sum()}")

    fn_errors = mse[fn_mask]

    if len(fn_errors) > 0:
        print("Min FN Error :", fn_errors.min())
        print("Max FN Error :", fn_errors.max())
        print("Mean FN Error:", fn_errors.mean())
    else:
        print("No false negatives.")
    
    # ── Extract TP, FP, TN, FN from confusion matrix ──────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── Compute classification metrics ─────────────────────────────────────────
    # zero_division=0 prevents divide-by-zero warnings on degenerate splits
    accuracy    = float(accuracy_score(y_true, y_pred))
    precision   = float(precision_score(y_true, y_pred, zero_division=0))
    recall      = float(recall_score(y_true, y_pred, zero_division=0))
    f1          = float(f1_score(y_true, y_pred, zero_division=0))
    # Specificity = TN / (TN + FP) — true negative rate; guards against /0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Sample composition derived from ground-truth labels
    n_samples = len(X)
    n_benign  = int((y_true == 0).sum())
    n_attack  = int((y_true == 1).sum())

    metrics = {
        "accuracy":    round(accuracy,    6),
        "precision":   round(precision,   6),
        "recall":      round(recall,      6),
        "f1":          round(f1,          6),
        "specificity": round(specificity, 6),
        "tp":          int(tp),
        "fp":          int(fp),
        "tn":          int(tn),
        "fn":          int(fn),
        "y_pred":      y_pred,
        "mse":         mse,
    }

    # ── Console summary ────────────────────────────────────────────────────────
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {model_name.upper()}")
    print(sep)
    print(f"  {'Samples':<14}: {n_samples:,}")
    print(f"  {'Benign':<14}: {n_benign:,}")
    print(f"  {'Attack':<14}: {n_attack:,}")
    print()
    print(f"  {'TP':<14}: {tp:,}")
    print(f"  {'FP':<14}: {fp:,}")
    print(f"  {'TN':<14}: {tn:,}")
    print(f"  {'FN':<14}: {fn:,}")
    print()
    print(f"  {'Accuracy':<14}: {accuracy:.6f}")
    print(f"  {'Precision':<14}: {precision:.6f}")
    print(f"  {'Recall':<14}: {recall:.6f}")
    print(f"  {'Specificity':<14}: {specificity:.6f}")
    print(f"  {'F1 Score':<14}: {f1:.6f}")
    print(sep)

    return metrics


# ── Confusion Matrix Plot ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
    metrics: dict,
) -> None:
    """
    Generate and save a dark-themed confusion matrix as a PNG.

    Each cell shows the abbreviated label (TN/FP/FN/TP), the raw count,
    and the row-normalised percentage, giving both absolute and relative
    information at a glance.

    Args:
        y_true     : Ground-truth labels.
        y_pred     : Predicted labels.
        model_name : Display string used in the chart title.
        save_path  : Full file path for the saved PNG.
        metrics    : Metrics dict from evaluate_model(); drives the
                     summary footer rendered below the matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Row-normalise for colour intensity (each row sums to 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    # ── Figure — extra height provides space for the metrics footer ────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    # Dark-to-light blue heatmap; colour driven by row-normalised values
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised ratio", color=PALETTE["muted"], fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["border"], length=0)
    cbar.outline.set_edgecolor(PALETTE["border"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["muted"], fontsize=8)

    # ── Cell annotations ───────────────────────────────────────────────────────
    cell_labels = [
        [f"TN\n{tn:,}\n{cm_norm[0, 0]:.1%}",  f"FP\n{fp:,}\n{cm_norm[0, 1]:.1%}"],
        [f"FN\n{fn:,}\n{cm_norm[1, 0]:.1%}",  f"TP\n{tp:,}\n{cm_norm[1, 1]:.1%}"],
    ]
    for row in range(2):
        for col in range(2):
            # Use dark text on bright cells, light text on dark cells
            text_colour = PALETTE["text"] if cm_norm[row, col] > 0.55 else PALETTE["bg"]
            ax.text(
                col, row, cell_labels[row][col],
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=text_colour,
                linespacing=1.45,
            )

    # ── Axes ───────────────────────────────────────────────────────────────────
    class_names = ["Benign (0)", "Attack (1)"]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_names, color=PALETTE["text"], fontsize=10)
    ax.set_xlabel("Predicted Label", color=PALETTE["muted"], fontsize=10, labelpad=10)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names, color=PALETTE["text"], fontsize=10)
    ax.set_ylabel("True Label", color=PALETTE["muted"], fontsize=10, labelpad=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
        spine.set_linewidth(0.8)

    ax.tick_params(which="both", length=0, colors=PALETTE["border"])

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.set_title(
        f"{model_name}\nConfusion Matrix",
        color=PALETTE["text"],
        fontsize=12,
        fontweight="bold",
        pad=14,
        linespacing=1.5,
    )

    # ── Metrics footer — rendered below the matrix via figure coordinates ───────
    # subplots_adjust creates bottom margin so fig.text() is never clipped
    fig.subplots_adjust(top=0.87, bottom=0.26, left=0.14, right=0.84)

    line1 = (
        f"Accuracy:{metrics['accuracy']*100:.2f}% "
        f"Precision:{metrics['precision']*100:.2f}% "
        f"Recall:{metrics['recall']*100:.2f}% "
    )
    line2 = (
        f"Specificity:{metrics['specificity']*100:.2f}% "
        f"F1 Score:{metrics['f1']*100:.2f}% "
    )
    fig.text(
        0.50, 0.14, line1,
        ha="center", va="center",
        fontsize=8, color=PALETTE["muted"],
        family="monospace",
    )
    fig.text(
        0.50, 0.08, line2,
        ha="center", va="center",
        fontsize=8, color=PALETTE["muted"],
        family="monospace",
    )

    fig.savefig(
        save_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Model Comparison Chart ─────────────────────────────────────────────────────

def plot_model_comparison(
    centralized_metrics: dict,
    federated_metrics: dict,
    save_path: Path,
) -> None:
    """
    Generate and save a grouped bar chart comparing Accuracy, Precision,
    Recall, Specificity, and F1 Score between the Centralized and Federated
    models.

    The y-axis auto-zooms to the data range so small differences between
    models remain visible. Value labels are displayed as percentages.

    Args:
        centralized_metrics : Metrics dict returned by evaluate_model().
        federated_metrics   : Metrics dict returned by evaluate_model().
        save_path           : Full file path for the saved PNG.
    """
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity\n(TNR)", "F1 Score"]
    metric_keys   = ["accuracy", "precision", "recall", "specificity", "f1"]

    c_vals = [centralized_metrics[k] for k in metric_keys]
    f_vals = [federated_metrics[k]   for k in metric_keys]

    x         = np.arange(len(metric_labels))
    bar_width  = 0.32

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    # ── Grouped bars ───────────────────────────────────────────────────────────
    bars_c = ax.bar(
        x - bar_width / 2, c_vals, bar_width,
        label="Centralized",
        color=PALETTE["accent"],
        linewidth=0,
        zorder=3,
    )
    bars_f = ax.bar(
        x + bar_width / 2, f_vals, bar_width,
        label="Federated",
        color=PALETTE["green"],
        linewidth=0,
        zorder=3,
    )

    # ── Value labels above each bar — shown as percentages (e.g. 99.54%) ───────
    for bars in (bars_c, bars_f):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.002,
                f"{height:.2%}",
                ha="center", va="bottom",
                fontsize=7,
                color=PALETTE["muted"],
            )

    # ── Grid ───────────────────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.yaxis.grid(
        True,
        color=PALETTE["border"],
        linewidth=0.6,
        linestyle="--",
        zorder=0,
    )
    ax.xaxis.grid(False)

    # ── Axes ───────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=PALETTE["text"], fontsize=10)
    # Auto-zoom: floor at 90% or 2 pp below the worst metric, cap above 100%
    # to leave headroom for percentage labels above the tallest bars
    ax.set_ylim(0.75, 1.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.tick_params(axis="y", colors=PALETTE["muted"],  labelsize=9, length=0)
    ax.tick_params(axis="x", colors=PALETTE["border"], length=0)
    ax.set_ylabel("Score", color=PALETTE["muted"], fontsize=10, labelpad=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
        spine.set_linewidth(0.8)

    # ── Legend ─────────────────────────────────────────────────────────────────
    ax.legend(
        fontsize=10,
        framealpha=0.0,
        labelcolor=PALETTE["text"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.set_title(
        "Centralized vs. Federated Autoencoder — Performance Comparison",
        color=PALETTE["text"],
        fontsize=12,
        fontweight="bold",
        pad=16,
    )

    fig.tight_layout(pad=0.8)
    fig.savefig(
        save_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Metrics JSON ───────────────────────────────────────────────────────────────

def save_metrics_json(
    centralized_metrics: dict,
    federated_metrics: dict,
    save_path: Path,
) -> None:
    """
    Persist scalar evaluation metrics for both models to a JSON file.

    Non-serialisable fields (y_pred, mse arrays) are intentionally
    excluded so the file can be loaded directly by the Flask dashboard.

    Args:
        centralized_metrics : Full metrics dict from evaluate_model().
        federated_metrics   : Full metrics dict from evaluate_model().
        save_path           : Output path for the JSON file.
    """
    # Include specificity in all persisted files
    scalar_keys = ["accuracy", "precision", "recall", "specificity", "f1"]

    # ── Combined file (model_metrics.json) ────────────────────────────────────
    output = {
        "Centralized": {k: round(centralized_metrics[k] * 100, 4) for k in scalar_keys},
        "Federated":   {k: round(federated_metrics[k] * 100, 4) for k in scalar_keys},
    }

    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=4)

    print(f"  Saved → {save_path}")

    # ── Individual files (centralized_metrics.json, federated_metrics.json) ───
    # Derived from save_path.parent so no extra parameters are needed
    charts_dir = save_path.parent

    c_path = charts_dir / "centralized_metrics.json"
    with open(c_path, "w", encoding="utf-8") as fh:
        json.dump({k: round(centralized_metrics[k] * 100, 4) for k in scalar_keys}, fh, indent=4)
    print(f"  Saved → {c_path}")

    f_path = charts_dir / "federated_metrics.json"
    with open(f_path, "w", encoding="utf-8") as fh:
        json.dump({k: round(federated_metrics[k] * 100, 4) for k in scalar_keys}, fh, indent=4)
    print(f"  Saved → {f_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full evaluation pipeline:

        1.  Create output directory.
        2.  Load and preprocess the balanced dataset.
        3.  Evaluate the Centralized Autoencoder.
        4.  Evaluate the Federated Autoencoder.
        5.  Generate and save both confusion matrices (with metrics footer).
        6.  Generate and save the model comparison chart.
        7.  Save combined and individual metrics JSON files.
        8.  Print a final human-readable summary.
        9.  Report total execution time.
    """
    sep = "=" * 50
    start_time = time.perf_counter()

    # ── 1. Output directory ────────────────────────────────────────────────────
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{sep}")
    print(f"  OUTPUT DIRECTORY")
    print(sep)
    print(f"  {CHARTS_DIR.resolve()}")

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  LOADING DATASET")
    print(sep)
    X, y_true, _ = load_dataset()

    # ── 3. Evaluate Centralized model ─────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  EVALUATING — CENTRALIZED AUTOENCODER")
    print(sep)
    c_metrics = evaluate_model(
        model_path=CENTRALIZED_MODEL_PATH,
        X=X,
        y_true=y_true,
        threshold=THRESHOLDS["centralized"],
        model_name="Centralized",
    )

    # ── 4. Evaluate Federated model ────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  EVALUATING — FEDERATED AUTOENCODER")
    print(sep)
    f_metrics = evaluate_model(
        model_path=FEDERATED_MODEL_PATH,
        X=X,
        y_true=y_true,
        threshold=THRESHOLDS["federated"],
        model_name="Federated",
    )

    # ── 5. Confusion matrices ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  GENERATING CONFUSION MATRICES")
    print(sep)
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=c_metrics["y_pred"],
        model_name="Centralized Autoencoder",
        save_path=CHARTS_DIR / "centralized_confusion.png",
        metrics=c_metrics,
    )
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=f_metrics["y_pred"],
        model_name="Federated Autoencoder",
        save_path=CHARTS_DIR / "federated_confusion.png",
        metrics=f_metrics,
    )

    # ── 6. Model comparison chart ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  GENERATING MODEL COMPARISON CHART")
    print(sep)
    plot_model_comparison(
        centralized_metrics=c_metrics,
        federated_metrics=f_metrics,
        save_path=CHARTS_DIR / "model_comparison.png",
    )

    # ── 7. Metrics JSON ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  SAVING METRICS JSON")
    print(sep)
    save_metrics_json(
        centralized_metrics=c_metrics,
        federated_metrics=f_metrics,
        save_path=CHARTS_DIR / "model_metrics.json",
    )

    # ── 8. Final summary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  EVALUATION SUMMARY")
    print(sep)

    for label, m in [("CENTRALIZED", c_metrics), ("FEDERATED", f_metrics)]:
        print(f"\n  {label}")
        print(f"  {'Accuracy':<14}: {m['accuracy']:.6f} ")
        print(f"  {'Precision':<14}: {m['precision']:.6f} ")
        print(f"  {'Recall':<14}: {m['recall']:.6f} ")
        print(f"  {'Specificity':<14}: {m['specificity']:.6f} ")
        print(f"  {'F1-Score':<14}: {m['f1']:.6f} ")

    # ── 9. Execution time ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    print(f"\n{sep}")
    print(f"  All charts saved to      : {CHARTS_DIR.resolve()}")
    print(f"  Evaluation completed in  : {elapsed:.2f} seconds")
    print(f"{sep}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()