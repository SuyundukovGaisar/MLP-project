import os
import sys
import argparse
import time
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


# --- Функции активации ---

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0.0, z)

def relu_deriv(a):
    return (a > 0).astype(np.float64)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu":    (relu,    relu_deriv),
}


# --- Инициализация весов (He Uniform) ---

def he_uniform(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, (fan_in, fan_out))


# --- Модель MLP ---

class MLP:
    def __init__(self, layer_sizes, activation="sigmoid", seed=42):
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.n_layers    = len(layer_sizes) - 1
        rng = np.random.default_rng(seed)
        self.weights = [he_uniform(layer_sizes[i], layer_sizes[i+1], rng)
                        for i in range(self.n_layers)]
        self.biases  = [np.zeros((1, layer_sizes[i+1]))
                        for i in range(self.n_layers)]

    def forward(self, X):
        act_fn, _ = ACTIVATIONS[self.activation]
        A = X
        pre_acts, acts = [], [X]
        for i in range(self.n_layers):
            Z = A @ self.weights[i] + self.biases[i]
            pre_acts.append(Z)
            A = act_fn(Z) if i < self.n_layers - 1 else softmax(Z)
            acts.append(A)
        return pre_acts, acts

    def backward(self, X, y_oh, pre_acts, acts):
        _, act_d = ACTIVATIONS[self.activation]
        m = X.shape[0]
        dW_list = [None] * self.n_layers
        db_list = [None] * self.n_layers
        dZ = acts[-1] - y_oh
        for i in reversed(range(self.n_layers)):
            dW_list[i] = (acts[i].T @ dZ) / m
            db_list[i] = dZ.mean(axis=0, keepdims=True)
            if i > 0:
                dZ = (dZ @ self.weights[i].T) * act_d(acts[i])
        return dW_list, db_list

    def predict_proba(self, X):
        _, acts = self.forward(X)
        return acts[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path):
        arrays = {f"W_{i}": W for i, W in enumerate(self.weights)}
        arrays.update({f"b_{i}": b for i, b in enumerate(self.biases)})
        arrays["layer_sizes"] = np.array(self.layer_sizes)
        arrays["activation"]  = np.array([self.activation])
        arrays["n_layers"]    = np.array([self.n_layers])
        np.savez(path, **arrays)
        print(f"Модель сохранена -> {path}")

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        layer_sizes = data["layer_sizes"].tolist()
        activation  = str(data["activation"][0])
        n_layers    = int(data["n_layers"][0])
        obj = cls.__new__(cls)
        obj.layer_sizes = layer_sizes
        obj.activation  = activation
        obj.n_layers    = n_layers
        obj.weights = [data[f"W_{i}"] for i in range(n_layers)]
        obj.biases  = [data[f"b_{i}"] for i in range(n_layers)]
        return obj


# --- Функция потерь и метрики ---

def cross_entropy(proba, y_oh):
    clipped = np.clip(proba, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.sum(y_oh * np.log(clipped), axis=1)))

def accuracy(proba, y):
    return float(np.mean(np.argmax(proba, axis=1) == y))

def to_onehot(y):
    oh = np.zeros((len(y), N_CLASSES))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


# --- Оптимизаторы ---

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def init_state(self, weights, biases):
        pass

    def update(self, weights, biases, dW_list, db_list, t):
        for i in range(len(weights)):
            weights[i] -= self.lr * dW_list[i]
            biases[i]  -= self.lr * db_list[i]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps

    def init_state(self, weights, biases):
        self.mW = [np.zeros_like(W) for W in weights]
        self.vW = [np.zeros_like(W) for W in weights]
        self.mb = [np.zeros_like(b) for b in biases]
        self.vb = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, dW_list, db_list, t):
        b1t = self.beta1 ** t
        b2t = self.beta2 ** t
        for i in range(len(weights)):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * dW_list[i]
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * db_list[i]
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * dW_list[i]**2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * db_list[i]**2
            mW_hat = self.mW[i] / (1 - b1t)
            mb_hat = self.mb[i] / (1 - b1t)
            vW_hat = self.vW[i] / (1 - b2t)
            vb_hat = self.vb[i] / (1 - b2t)
            weights[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            biases[i]  -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


def build_optimizer(name, lr):
    if name == "sgd":
        return SGD(lr=lr)
    elif name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {name}. Доступны: sgd, adam")


# --- Загрузка данных ---

def load_dataset(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(int)
    return X, y


# --- Генератор мини-батчей ---

def batch_iter(X, y, batch_size, rng):
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    for start in range(0, X.shape[0], batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


# --- Цикл обучения ---

def train(model, optimizer, X_tr, y_tr, X_val, y_val, epochs, batch_size, seed):
    optimizer.init_state(model.weights, model.biases)
    rng = np.random.default_rng(seed)
    y_tr_oh  = to_onehot(y_tr)
    y_val_oh = to_onehot(y_val)
    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        for X_b, y_b in batch_iter(X_tr, y_tr, batch_size, rng):
            t += 1
            pre, acts = model.forward(X_b)
            dW, db    = model.backward(X_b, to_onehot(y_b), pre, acts)
            optimizer.update(model.weights, model.biases, dW, db, t)

        p_tr  = model.predict_proba(X_tr)
        p_val = model.predict_proba(X_val)
        l_tr  = cross_entropy(p_tr,  y_tr_oh)
        l_val = cross_entropy(p_val, y_val_oh)
        a_tr  = accuracy(p_tr,  y_tr)
        a_val = accuracy(p_val, y_val)
        history["train_loss"].append(l_tr)
        history["val_loss"].append(l_val)
        history["train_acc"].append(a_tr)
        history["val_acc"].append(a_val)

        log_every = max(1, epochs // 50) if epochs > 50 else 1
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:>4}/{epochs} - loss: {l_tr:.4f} - val_loss: {l_val:.4f}"
                  f" - acc: {a_tr:.4f} - val_acc: {a_val:.4f} [{time.time()-t0:.2f}s]")

    return history


# --- Графики обучения ---

def plot_learning_curves(history, out_dir, optimizer_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Кривые обучения (оптимизатор: {optimizer_name.upper()})",
                 fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], color="#3498db", label="training loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   color="#e74c3c", label="validation loss",
                 linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curves — Loss"); axes[0].legend(); axes[0].grid(alpha=0.35)

    axes[1].plot(epochs, history["train_acc"], color="#2ecc71", label="training acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   color="#f39c12", label="validation acc",
                 linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Learning Curves — Accuracy"); axes[1].legend()
    axes[1].grid(alpha=0.35); axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot_learning_curves.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"Кривые обучения -> {path}")


# --- Интерактивный ввод архитектуры ---

def interactive_setup():
    print("\n" + "=" * 55)
    print("  Настройка архитектуры нейросети")
    print("=" * 55)

    while True:
        raw = input("\nКоличество скрытых слоёв [по умолчанию: 2]: ").strip()
        if raw == "": n_hidden = 2; break
        try:
            n_hidden = int(raw)
            if n_hidden >= 1: break
            print("  Минимум 1.")
        except ValueError:
            print("  Введите целое число.")

    defaults = [64, 32] + [16] * max(0, n_hidden - 2)
    hidden = []
    for i in range(n_hidden):
        d = defaults[i] if i < len(defaults) else 16
        while True:
            raw = input(f"  Нейронов в слое {i+1} [по умолчанию: {d}]: ").strip()
            if raw == "": hidden.append(d); break
            try:
                n = int(raw)
                if n >= 1: hidden.append(n); break
                print("  Минимум 1.")
            except ValueError:
                print("  Введите целое число.")

    while True:
        raw = input("\nЧисло эпох [по умолчанию: 100]: ").strip()
        if raw == "": epochs = 100; break
        try:
            epochs = int(raw)
            if epochs >= 1: break
            print("  Минимум 1.")
        except ValueError:
            print("  Введите целое число.")

    while True:
        raw = input("Размер батча [по умолчанию: 32]: ").strip()
        if raw == "": batch_size = 32; break
        try:
            batch_size = int(raw)
            if batch_size >= 1: break
        except ValueError:
            print("  Введите целое число.")

    while True:
        raw = input("Learning rate [по умолчанию: 0.01]: ").strip()
        if raw == "": lr = 0.01; break
        try:
            lr = float(raw)
            if lr > 0: break
            print("  Должно быть > 0.")
        except ValueError:
            print("  Введите число.")

    while True:
        raw = input("Оптимизатор [sgd / adam] [по умолчанию: sgd]: ").strip().lower()
        if raw == "": optimizer = "sgd"; break
        if raw in ("sgd", "adam"): optimizer = raw; break
        print("  Введите sgd или adam.")

    print(f"\n  Архитектура: 30 -> {' -> '.join(str(n) for n in hidden)} -> 2")
    print(f"  Эпохи: {epochs}, Батч: {batch_size}, LR: {lr}, Оптимизатор: {optimizer.upper()}\n")
    return hidden, epochs, batch_size, lr, optimizer


# --- Аргументы командной строки ---

def parse_args():
    p = argparse.ArgumentParser(description="train.py — обучение MLP")
    p.add_argument("--train",      default="data_train.csv")
    p.add_argument("--val",        default="data_val.csv")
    p.add_argument("--layers",     type=int, nargs="+", default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--optimizer",  default=None, choices=["sgd", "adam"])
    p.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"])
    p.add_argument("--output",     default="model_weights.npz")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    interactive = all(v is None for v in [args.layers, args.epochs,
                                           args.batch_size, args.lr, args.optimizer])
    if interactive:
        hidden, epochs, batch_size, lr, optimizer_name = interactive_setup()
    else:
        hidden         = args.layers     or [64, 32]
        epochs         = args.epochs     or 100
        batch_size     = args.batch_size or 32
        optimizer_name = args.optimizer  or "sgd"
        lr = args.lr if args.lr is not None else (0.001 if optimizer_name == "adam" else 0.01)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Загрузка {args.train} ...", end=" ")
    X_tr, y_tr = load_dataset(args.train)
    print(f"x_train shape : {X_tr.shape}")

    print(f"Загрузка {args.val} ...", end=" ")
    X_val, y_val = load_dataset(args.val)
    print(f"x_valid shape : {X_val.shape}")

    layer_sizes = [X_tr.shape[1]] + hidden + [N_CLASSES]
    model = MLP(layer_sizes=layer_sizes, activation=args.activation, seed=args.seed)
    total = sum(W.size + b.size for W, b in zip(model.weights, model.biases))
    print(f"\nАрхитектура: {' -> '.join(str(s) for s in layer_sizes)}")
    print(f"Параметров: {total:,} | Активация: {args.activation}")
    print(f"Оптимизатор: {optimizer_name.upper()} | lr={lr} | batch={batch_size} | epochs={epochs}\n")

    optimizer = build_optimizer(optimizer_name, lr)
    t_start = time.time()
    history = train(model, optimizer, X_tr, y_tr, X_val, y_val, epochs, batch_size, args.seed)
    elapsed = time.time() - t_start

    print(f"\n{'='*55}")
    print(f"  Обучение завершено за {elapsed:.1f} сек.")
    print(f"  train_loss: {history['train_loss'][-1]:.4f} | val_loss: {history['val_loss'][-1]:.4f}")
    print(f"  train_acc:  {history['train_acc'][-1]:.4f} | val_acc:  {history['val_acc'][-1]:.4f}")
    print(f"  Лучший val_acc: {max(history['val_acc']):.4f}")
    print(f"{'='*55}\n")

    model_path = os.path.join(args.output_dir, args.output)
    model.save(model_path)

    meta_path = os.path.join(args.output_dir, "model_meta.npz")
    np.savez(meta_path,
             layer_sizes=np.array(layer_sizes),
             activation=np.array([args.activation]),
             optimizer=np.array([optimizer_name]),
             lr=np.array([lr]),
             epochs=np.array([epochs]),
             batch_size=np.array([batch_size]),
             final_train_acc=np.array([history["train_acc"][-1]]),
             final_val_acc=np.array([history["val_acc"][-1]]))
    print(f"Метаданные -> {meta_path}")

    plot_learning_curves(history, args.output_dir, optimizer_name)
    print(f"\n> saving model '{model_path}' to disk...")


if __name__ == "__main__":
    main()
