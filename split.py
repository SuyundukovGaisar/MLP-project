import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

COLUMN_NAMES = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
FEATURE_COLS = COLUMN_NAMES[2:]
LABEL_COL = "diagnosis"


def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    first_val = str(df.iloc[0, 0])
    if not first_val.lstrip("-").isdigit():
        df = pd.read_csv(path)
        if df.shape[1] == 32:
            df.columns = COLUMN_NAMES
    return df


def eda_report(df):
    print("\n" + "=" * 55)
    print("  EDA — Разведочный анализ данных")
    print("=" * 55)
    print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
    print(f"Признаков: {len(FEATURE_COLS)}")
    print("\nРаспределение классов:")
    vc = df[LABEL_COL].value_counts()
    for cls, cnt in vc.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")
    missing = df[FEATURE_COLS].isnull().sum().sum()
    print(f"\nПропущенных значений: {missing}")
    print("\nСтатистика (первые 5 признаков):")
    print(df[FEATURE_COLS[:5]].describe().round(4).to_string())
    print("=" * 55 + "\n")


def fill_missing_by_class(df, class_means=None):
    df = df.copy()
    classes = df[LABEL_COL].unique()
    if class_means is None:
        class_means = {}
        for cls in classes:
            mask = df[LABEL_COL] == cls
            class_means[cls] = df.loc[mask, FEATURE_COLS].mean()
    for cls in classes:
        mask = df[LABEL_COL] == cls
        df.loc[mask, FEATURE_COLS] = df.loc[mask, FEATURE_COLS].fillna(class_means[cls])
    if df[FEATURE_COLS].isnull().sum().sum() > 0:
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())
    return df, class_means


def stratified_split(df, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in df[LABEL_COL].unique():
        idx = np.array(df.index[df[LABEL_COL] == cls].tolist())
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)} строк | Val: {len(val_df)} строк")
    for cls in df[LABEL_COL].unique():
        print(f"  {cls}: train={( train_df[LABEL_COL]==cls).sum()}, val={(val_df[LABEL_COL]==cls).sum()}")
    return train_df, val_df


def compute_norm_params(train_df):
    X = train_df[FEATURE_COLS].values.astype(np.float64)
    mu  = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return mu, std


def normalize(df, mu, std):
    df = df.copy()
    df[FEATURE_COLS] = (df[FEATURE_COLS].values.astype(np.float64) - mu) / std
    return df


def encode_labels(df):
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].map({"M": 1, "B": 0})
    return df


# --- Графики ---

def plot_class_distribution(df, out_dir):
    vc = df[LABEL_COL].value_counts()
    colors = ["#e74c3c", "#2ecc71"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Распределение классов", fontsize=13, fontweight="bold")
    axes[0].pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=colors,
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Доля классов")
    bars = axes[1].bar(vc.index, vc.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vc.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(val), ha="center", va="bottom", fontweight="bold")
    axes[1].set_xlabel("Диагноз"); axes[1].set_ylabel("Количество")
    axes[1].set_title("Абсолютное количество"); axes[1].grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_class_distribution.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] {path}")


def plot_feature_distributions(df, out_dir, n=12):
    feats = FEATURE_COLS[:n]
    ncols, nrows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.2))
    fig.suptitle(f"Распределение первых {n} признаков по классам", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()
    df_M = df[df[LABEL_COL] == "M"]
    df_B = df[df[LABEL_COL] == "B"]
    for i, feat in enumerate(feats):
        ax = axes_flat[i]
        ax.hist(df_M[feat].dropna(), bins=25, alpha=0.6, color="#e74c3c", label="M", density=True)
        ax.hist(df_B[feat].dropna(), bins=25, alpha=0.6, color="#2ecc71", label="B", density=True)
        ax.set_title(feat, fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if i == 0: ax.legend(fontsize=7)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_feature_distributions.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] {path}")


def plot_boxplots(df, out_dir, n=12):
    feats = FEATURE_COLS[:n]
    ncols, nrows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.2))
    fig.suptitle(f"Boxplot первых {n} признаков по классам", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()
    df_M = df[df[LABEL_COL] == "M"]
    df_B = df[df[LABEL_COL] == "B"]
    for i, feat in enumerate(feats):
        ax = axes_flat[i]
        bp = ax.boxplot([df_B[feat].dropna().values, df_M[feat].dropna().values],
                        tick_labels=["B", "M"], patch_artist=True,
                        boxprops=dict(linewidth=1.2),
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.set_title(feat, fontsize=8); ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.3)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_boxplots.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] {path}")


def plot_correlation_heatmap(df, out_dir):
    X = df[FEATURE_COLS].values.astype(np.float64)
    X_c = X - X.mean(axis=0)
    std = X_c.std(axis=0, ddof=0); std[std == 0] = 1.0
    X_c /= std
    corr = (X_c.T @ X_c) / X_c.shape[0]
    short = [c.replace("_mean","_m").replace("_worst","_w") for c in FEATURE_COLS]
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title("Матрица корреляции признаков", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_correlation_heatmap.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] {path}")


def plot_normalized_comparison(train_df, val_df, out_dir, n=6):
    feats = FEATURE_COLS[:n]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Нормализованные признаки: Train vs Val", fontsize=13, fontweight="bold")
    for i, feat in enumerate(feats):
        ax = axes.flatten()[i]
        ax.hist(train_df[feat].values, bins=30, alpha=0.6, color="#3498db", label="Train", density=True)
        ax.hist(val_df[feat].values,   bins=30, alpha=0.6, color="#f39c12", label="Val",   density=True)
        ax.set_title(feat, fontsize=9); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_normalized_comparison.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] {path}")


def parse_args():
    p = argparse.ArgumentParser(description="split.py — предобработка и разделение датасета")
    p.add_argument("--input",      default="data.csv")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--val_ratio",  type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_raw = load_data(args.input)
    eda_report(df_raw)

    print("[plot] Строим EDA-графики...")
    plot_class_distribution(df_raw, args.output_dir)
    plot_feature_distributions(df_raw, args.output_dir)
    plot_boxplots(df_raw, args.output_dir)
    plot_correlation_heatmap(df_raw, args.output_dir)

    # Разделяем ДО заполнения пропусков, чтобы не допустить утечки данных
    train_raw, val_raw = stratified_split(df_raw, val_ratio=args.val_ratio, seed=args.seed)

    # Параметры заполнения считаем только по train
    train_filled, class_means = fill_missing_by_class(train_raw)
    val_filled,   _           = fill_missing_by_class(val_raw, class_means=class_means)
    print("Пропуски заполнены средним по классу (без утечки из val).")

    # Параметры нормализации считаем только по train
    mu, std = compute_norm_params(train_filled)
    train_norm = normalize(train_filled, mu, std)
    val_norm   = normalize(val_filled,   mu, std)
    print(f"Z-score нормализация выполнена. mu[:3]={mu[:3].round(3)}, std[:3]={std[:3].round(3)}")

    train_final = encode_labels(train_norm)
    val_final   = encode_labels(val_norm)

    plot_normalized_comparison(train_final, val_final, args.output_dir)

    train_final.to_csv(os.path.join(args.output_dir, "data_train.csv"), index=False)
    val_final.to_csv(  os.path.join(args.output_dir, "data_val.csv"),   index=False)
    np.savez(os.path.join(args.output_dir, "normalization_params.npz"),
             mu=mu, std=std, feature_names=np.array(FEATURE_COLS))

    print(f"\nСохранено: data_train.csv ({len(train_final)} строк), data_val.csv ({len(val_final)} строк)")
    print(f"Выходная папка: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
