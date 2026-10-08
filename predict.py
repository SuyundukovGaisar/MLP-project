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

LABEL_COL = "diagnosis"
FEATURE_COLS = [
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
N_CLASSES = 2
CLASS_NAMES = {0: "B (доброкачественная)", 1: "M (злокачественная)"}


# --- Функции активации ---

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0.0, z)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATION_FN = {"sigmoid": sigmoid, "relu": relu}


# --- Модель MLP---

class MLP:
    def __init__(self, weights, biases, activation="sigmoid"):
        self.weights    = weights
        self.biases     = biases
        self.activation = activation
        self.n_layers   = len(weights)

    def predict_proba(self, X):
        act_fn = ACTIVATION_FN[self.activation]
        A = X.astype(np.float64)
        for i in range(self.n_layers):
            Z = A @ self.weights[i] + self.biases[i]
            A = act_fn(Z) if i < self.n_layers - 1 else softmax(Z)
        return A

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        n = int(data["n_layers"][0])
        return cls(
            weights    = [data[f"W_{i}"] for i in range(n)],
            biases     = [data[f"b_{i}"] for i in range(n)],
            activation = str(data["activation"][0]),
        )

    @property
    def layer_sizes(self):
        sizes = [self.weights[0].shape[0]]
        for W in self.weights:
            sizes.append(W.shape[1])
        return sizes

    @property
    def total_params(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


# --- Метрики ---

def to_onehot(y):
    oh = np.zeros((len(y), N_CLASSES))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh

def cross_entropy(proba, y_oh):
    clipped = np.clip(proba, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.sum(y_oh * np.log(clipped), axis=1)))

def binary_cross_entropy(proba, y):
    p = np.clip(proba[:, 1], 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm

def precision_recall_f1(cm):
    res = {}
    for cls in range(2):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        res[cls] = {"precision": pr, "recall": rc, "f1": f1}
    return res


# --- Загрузка данных ---

def load_dataset(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(int) if LABEL_COL in df.columns else None
    return X, y


# --- Графики ---

def plot_confusion_matrix(cm, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["B (0)", "M (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказано"); ax.set_ylabel("Истинное")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{val}\n({val/total*100:.1f}%)",
                    ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_confusion_matrix.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"Confusion matrix -> {path}")


def plot_proba_distribution(proba, y_true, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    p = proba[:, 1]
    if y_true is not None:
        ax.hist(p[y_true == 1], bins=30, alpha=0.65, color="#e74c3c", label="M (злокач.)", density=True)
        ax.hist(p[y_true == 0], bins=30, alpha=0.65, color="#2ecc71", label="B (доброкач.)", density=True)
        ax.legend(fontsize=10)
    else:
        ax.hist(p, bins=30, alpha=0.75, color="#3498db", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("P(M) — вероятность злокачественности"); ax.set_ylabel("Плотность")
    ax.set_title("Распределение предсказанных вероятностей", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.35)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_predict_proba.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"Proba distribution -> {path}")


def plot_metrics_bar(metrics, out_dir):
    cls_names = ["B (доброкач.)", "M (злокач.)"]
    m_names   = ["precision", "recall", "f1"]
    colors    = ["#3498db", "#e74c3c", "#2ecc71"]
    x = np.arange(2); width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (mn, col) in enumerate(zip(m_names, colors)):
        vals = [metrics[cls][mn] for cls in range(2)]
        bars = ax.bar(x + i*width, vals, width, label=mn.capitalize(),
                      color=col, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x + width); ax.set_xticklabels(cls_names)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Значение")
    ax.set_title("Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_predict_metrics.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"Metrics bar -> {path}")


def save_predictions(data_path, preds, proba, out_dir):
    df_src = pd.read_csv(data_path)
    df_out = pd.DataFrame({
        "predicted_label": preds,
        "predicted_class": [CLASS_NAMES[p] for p in preds],
        "proba_B": proba[:, 0].round(6),
        "proba_M": proba[:, 1].round(6),
    })
    if LABEL_COL in df_src.columns:
        df_out.insert(0, "true_label", df_src[LABEL_COL].values)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    path = os.path.join(out_dir, f"predictions_{stem}.csv")
    df_out.to_csv(path, index=False)
    print(f"Предсказания -> {path}")


def print_report(proba, preds, y_true, data_path, model_path):
    print("\n" + "=" * 55)
    print("  ОТЧЁТ О ПРЕДСКАЗАНИЯХ")
    print("=" * 55)
    print(f"  Модель   : {model_path}")
    print(f"  Данные   : {data_path}")
    print(f"  Образцов : {len(preds)}")
    n_M = (preds == 1).sum(); n_B = (preds == 0).sum()
    print(f"\n  Предсказано: M={n_M} ({n_M/len(preds)*100:.1f}%), B={n_B} ({n_B/len(preds)*100:.1f}%)")

    if y_true is not None:
        bce  = binary_cross_entropy(proba, y_true)
        ce   = cross_entropy(proba, to_onehot(y_true))
        acc  = float(np.mean(preds == y_true))
        cm   = confusion_matrix(y_true, preds)
        prf  = precision_recall_f1(cm)
        TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        print(f"\n  Binary Cross-Entropy : {bce:.6f}")
        print(f"  Cross-Entropy        : {ce:.6f}")
        print(f"  Accuracy             : {acc:.4f} ({acc*100:.2f}%)")
        print(f"\n  Confusion Matrix:")
        print(f"    {'':12s}  Pred B   Pred M")
        print(f"    True B  :  {TN:>5}    {FP:>5}")
        print(f"    True M  :  {FN:>5}    {TP:>5}")
        print(f"\n  {'Класс':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*52}")
        for cls in range(2):
            name = "B (доброкач.)" if cls == 0 else "M (злокач.)"
            print(f"  {name:<20} {prf[cls]['precision']:>10.4f}"
                  f" {prf[cls]['recall']:>10.4f} {prf[cls]['f1']:>10.4f}")
        mp = np.mean([prf[c]["precision"] for c in range(2)])
        mr = np.mean([prf[c]["recall"]    for c in range(2)])
        mf = np.mean([prf[c]["f1"]        for c in range(2)])
        print(f"  {'macro avg':<20} {mp:>10.4f} {mr:>10.4f} {mf:>10.4f}")
    print("=" * 55 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="predict.py — предсказание с помощью обученного MLP")
    p.add_argument("--model",      default="model_weights.npz")
    p.add_argument("--data",       default="data_val.csv")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--no_plots",   action="store_true")
    p.add_argument("--no_save",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model):
        print(f"[ERROR] Файл модели не найден: '{args.model}'. Сначала запустите train.py")
        sys.exit(1)

    print(f"Загрузка модели '{args.model}' ...", end=" ")
    model = MLP.load(args.model)
    print(f"OK  |  Архитектура: {' -> '.join(str(s) for s in model.layer_sizes)}")
    print(f"Параметров: {model.total_params:,} | Активация: {model.activation}")

    if not os.path.exists(args.data):
        print(f"[ERROR] Файл данных не найден: '{args.data}'")
        sys.exit(1)

    print(f"\nЗагрузка '{args.data}' ...", end=" ")
    X, y_true = load_dataset(args.data)
    print(f"shape: {X.shape}")

    proba = model.predict_proba(X)
    preds = (proba[:, 1] >= args.threshold).astype(int) if args.threshold != 0.5 \
            else np.argmax(proba, axis=1)

    print_report(proba, preds, y_true, args.data, args.model)

    if not args.no_save:
        save_predictions(args.data, preds, proba, args.output_dir)

    if not args.no_plots:
        plot_proba_distribution(proba, y_true, args.output_dir)
        if y_true is not None:
            cm = confusion_matrix(y_true, preds)
            plot_confusion_matrix(cm, args.output_dir)
            plot_metrics_bar(precision_recall_f1(cm), args.output_dir)

    print("predict.py завершён.\n")


if __name__ == "__main__":
    main()
