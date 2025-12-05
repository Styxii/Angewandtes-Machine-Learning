"""
Erstellt Plots für die trainierten Modelle.
Speichert die Bilder in reports/figures/.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from glob import glob
import math

# keras falls verfügbar
try:
    from tensorflow import keras

    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_SEED

parser = argparse.ArgumentParser(description="Plot training results")
parser.add_argument("--no-show", action="store_true", help="Plots nicht anzeigen")
args = parser.parse_args()

# normalerweise plots anzeigen außer --no-show
SHOW_PLOTS = not args.no_show

OUTPUT_DIR = os.path.join("reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# daten laden
processed = DATA_PROCESSED / "techuse_processed.csv"
if not processed.exists():
    raise FileNotFoundError(f"Processed data not found: {processed}")

df = pd.read_csv(processed)
if "sleep_quality" not in df.columns:
    raise ValueError("Processed data must include 'sleep_quality'")

X = df.drop(columns=["sleep_quality"])
y = df["sleep_quality"]

# gleicher split wie beim training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Find model files
model_pkl_files = sorted(
    [f for f in glob(str(MODELS_DIR / "*.pkl")) if "scaler" not in f],
    key=os.path.getmtime,
    reverse=True,
)
keras_file = MODELS_DIR / "keras_ffn.keras"

models = []

# xgboost modelle laden
for fp in model_pkl_files[:2]:
    try:
        m = joblib.load(fp)
        if not hasattr(m, "predict"):
            continue
        model_name = Path(fp).stem
        models.append((Path(fp), m, "xgboost"))
    except Exception:
        pass

# keras falls vorhanden
if keras_file.exists() and HAS_KERAS:
    try:
        m = keras.models.load_model(keras_file)
        models.append((keras_file, m, "keras"))
    except Exception:
        pass

if not models:
    raise FileNotFoundError(f"Keine Modelle in {MODELS_DIR} gefunden.")

# scaler für keras
scaler = None
scaler_path = MODELS_DIR / "scaler.pkl"
if HAS_KERAS and scaler_path.exists():
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None

# alle modelle durchgehen
preds = {}
metrics = {}
for path_obj, mdl, model_type in models:
    if model_type == "xgboost":
        y_pred_i = mdl.predict(X_test)
    elif model_type == "keras":
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        y_pred_i = mdl.predict(X_test_scaled, verbose=0).flatten()
    else:
        continue

    model_name = path_obj.name
    preds[model_name] = y_pred_i
    rmse = math.sqrt(mean_squared_error(y_test, y_pred_i))
    mae = mean_absolute_error(y_test, y_pred_i)
    r2 = r2_score(y_test, y_pred_i)
    metrics[model_name] = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "path": str(path_obj),
        "type": model_type,
    }

# zusammenfassung speichern
summary_lines = []
summary_lines.append(f"Models: {', '.join([m[0].name for m in models])}\n")
for name, m in metrics.items():
    summary_lines.append(f"Model: {name}")
    summary_lines.append(f"  R²:   {m['r2']:.4f}")
    summary_lines.append(f"  RMSE: {m['rmse']:.4f}")
    summary_lines.append(f"  MAE:  {m['mae']:.4f}\n")

summary_file = os.path.join(OUTPUT_DIR, "model_metrics.txt")
with open(summary_file, "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary_lines))

# plot mit allen modellen
names = list(preds.keys())

plt.figure(figsize=(10, 7))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for idx, name in enumerate(names):
    y_pred = preds[name]
    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5,
        s=30,
        label=f"{name} (R²={metrics[name]['r2']:.3f})",
        color=colors[idx % len(colors)],
    )

# ideallinie
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
    label="Perfect prediction",
)
plt.xlabel("Actual sleep_quality", fontsize=11)
plt.ylabel("Predicted sleep_quality", fontsize=11)
plt.title("Model Comparison: All Models", fontsize=12, fontweight="bold")
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()

fn = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(fn, dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

print(f"Saved: {fn}")
print(f"Saved: {summary_file}")

# keras trainings-verlauf
history_path = DATA_PROCESSED / "keras_training_history.csv"
if history_path.exists():
    try:
        df_history = pd.read_csv(history_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(
            df_history.index + 1, df_history["loss"], label="Training Loss", linewidth=2
        )
        ax1.plot(
            df_history.index + 1,
            df_history["val_loss"],
            label="Validation Loss",
            linewidth=2,
        )
        ax1.set_xlabel("Epoche", fontsize=11)
        ax1.set_ylabel("Loss (Huber)", fontsize=11)
        ax1.set_title("Keras FFN: Loss Verlauf", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax2.plot(
            df_history.index + 1, df_history["mae"], label="Training MAE", linewidth=2
        )
        ax2.plot(
            df_history.index + 1,
            df_history["val_mae"],
            label="Validation MAE",
            linewidth=2,
        )
        ax2.set_xlabel("Epoche", fontsize=11)
        ax2.set_ylabel("MAE", fontsize=11)
        ax2.set_title("Keras FFN: MAE Verlauf", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        keras_history_fn = os.path.join(OUTPUT_DIR, "keras_training_history.png")
        plt.savefig(keras_history_fn, dpi=150)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        print(f"Saved: {keras_history_fn}")
    except Exception as e:
        print(f"Keras history plot failed: {e}")
