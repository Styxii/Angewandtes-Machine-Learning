"""
Explorative Datenanalyse (EDA)

Autor: Henry Marx
Datum: .12.2025
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import joblib

from tensorflow import keras
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Projekt-Root zum Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROCESSED, REPORTS_DIR

# Matplotlib & Seaborn Styling
plt.style.use("default")
sns.set_palette("husl")

# Sicherstellen, dass figures/EDA/ Ordner existiert
figures_dir = REPORTS_DIR
figures_dir.mkdir(parents=True, exist_ok=True)

print("EDA-Visualisierungen werden erstellt...")

# Rohdaten laden
df_raw = pd.read_csv(DATA_RAW / "Tech_Use_Stress_Wellness.csv")

# Numerische Features extrahieren (ohne Zielvariable)
numeric_features = df_raw.select_dtypes(include=[np.number]).columns.tolist()
if "sleep_quality" in numeric_features:
    numeric_features.remove("sleep_quality")  # Zielvariable raus
if "user_id" in numeric_features:
    numeric_features.remove("user_id")  # user_id raus

# PLOT 1 Multi-Panel Histogramme

n_features = len(numeric_features)
n_cols = 4  # 4 Spalten
n_rows = (n_features + n_cols - 1) // n_cols  # Aufrunden

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    # Histogramm mit KDE (Kernel Density Estimation)
    axes[idx].hist(
        df_raw[feature].dropna(),
        bins=30,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        density=False,
    )

    # Titel und Labels
    axes[idx].set_title(feature, fontsize=10, fontweight="bold")
    axes[idx].set_ylabel("Häufigkeit", fontsize=9)
    axes[idx].grid(True, alpha=0.3)

    # Statistiken als Text einfügen
    mean_val = df_raw[feature].mean()
    median_val = df_raw[feature].median()
    axes[idx].axvline(mean_val, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[idx].axvline(
        median_val, color="green", linestyle="--", linewidth=1.5, alpha=0.7
    )

    # Legende
    axes[idx].legend(["Mean", "Median", "Verteilung"], fontsize=8, loc="upper right")

# Verstecke leere Subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis("off")

plt.suptitle(
    "Verteilung aller numerischen Features", fontsize=16, fontweight="bold", y=0.995
)
plt.tight_layout()

# Speichern
output_file_1 = figures_dir / "eda_histograms.png"
plt.savefig(output_file_1, dpi=300, bbox_inches="tight")
plt.close()

# PLOT 2 Grouped Boxplots

key_features = numeric_features  # Alle numerischen Features

n_key = len(key_features)
n_cols_box = 4  # 4 Spalten
n_rows_box = (n_key + n_cols_box - 1) // n_cols_box

fig, axes = plt.subplots(n_rows_box, n_cols_box, figsize=(18, n_rows_box * 4))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    # Grouped Boxplot aufgeteilt nach sleep_quality Klasse
    sns.boxplot(
        x="sleep_quality",
        y=feature,
        data=df_raw,
        ax=axes[idx],
        hue="sleep_quality",
        palette="Set2",
        legend=False,
    )

    # Titel und Labels
    axes[idx].set_title(
        f"{feature} nach Schlafqualität", fontsize=11, fontweight="bold"
    )
    axes[idx].set_xlabel("Sleep Quality (1=sehr schlecht, 5=sehr gut)", fontsize=9)
    axes[idx].set_ylabel(feature, fontsize=9)
    axes[idx].grid(True, alpha=0.3, axis="y")

    # Mittelwerte als Linie einzeichnen
    means = df_raw.groupby("sleep_quality")[feature].mean()
    axes[idx].plot(
        range(len(means)),
        means,
        color="red",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Mean",
        alpha=0.7,
    )
    axes[idx].legend(fontsize=8)

# Verstecke leere Subplots
for idx in range(n_key, len(axes)):
    axes[idx].axis("off")

plt.suptitle(
    "Feature-Verteilung nach Schlafqualitäts-Klassen",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()

# Speichern
output_file_2 = figures_dir / "eda_boxplots_by_class.png"
plt.savefig(output_file_2, dpi=300, bbox_inches="tight")
plt.close()

# PLOT 3 Korrelations-Barchart

numeric_cols = [
    col for col in df_raw.select_dtypes(include=[np.number]).columns if col != "user_id"
]

# Korrelationen mit Zielvariable berechnen
correlations = df_raw[numeric_cols].corr()["sleep_quality"].sort_values(ascending=False)
correlations_features = correlations.drop("sleep_quality")

fig, ax = plt.subplots(figsize=(10, 10))

# Farbe nach positiv/negativ
colors = ["green" if x > 0 else "red" for x in correlations_features.values]

ax.barh(
    range(len(correlations_features)),
    correlations_features.values,
    color=colors,
    edgecolor="black",
    alpha=0.7,
)

ax.set_yticks(range(len(correlations_features)))
ax.set_yticklabels(correlations_features.index, fontsize=10)
ax.set_xlabel("Korrelation mit Sleep Quality", fontsize=12, fontweight="bold")
ax.set_ylabel("Features", fontsize=12, fontweight="bold")
ax.set_title(
    "Feature-Korrelationen mit Schlafqualität (sortiert)",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, alpha=0.3, axis="x")
ax.axvline(0, color="black", linewidth=1.5)  # Null-Linie

plt.tight_layout()

output_file_3 = figures_dir / "eda_correlations_barchart.png"
plt.savefig(output_file_3, dpi=300, bbox_inches="tight")
plt.close()

# PLOT 4 Heatmap für ALLE Features

corr_all = df_raw[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr_all,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    square=True,
    ax=ax,
    cbar_kws={"label": "Korrelationskoeffizient"},
    linewidths=0.5,
    linecolor="white",
)

ax.set_title(
    "Korrelationsmatrix: Alle Features",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

plt.tight_layout()

output_file_4 = figures_dir / "eda_correlations_heatmap.png"
plt.savefig(output_file_4, dpi=300, bbox_inches="tight")
plt.close()

# PLOT 5 Class Imbalance

# Klassenverteilung berechnen
class_counts = df_raw["sleep_quality"].round().value_counts().sort_index()
total_samples = len(df_raw)

# Imbalance Ratio
max_class = class_counts.max()
min_class = class_counts.min()
imbalance_ratio = max_class / min_class

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LINKER PLOT: Bar Chart mit Prozent-Anteilen
colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]  # Rot→Grün für 1-5
bars = axes[0].bar(
    class_counts.index, class_counts.values, color=colors, edgecolor="black", alpha=0.8
)

# Prozent-Werte über Balken
for idx, (cls, count) in enumerate(class_counts.items()):
    pct = count / total_samples * 100
    # Kleinere Schrift und mehr Abstand für bessere Lesbarkeit
    axes[0].text(
        cls,
        count + 150,
        f"{count}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

axes[0].set_xlabel("Sleep Quality Klasse", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Anzahl Samples", fontsize=11, fontweight="bold")
axes[0].set_title(
    "Klassenverteilung (absolut + relativ)", fontsize=12, fontweight="bold"
)
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xticklabels(
    [
        "1\n(sehr\nschlecht)",
        "2\n(schlecht)",
        "3\n(mittel)",
        "4\n(gut)",
        "5\n(sehr\ngut)",
    ]
)
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].set_ylim(0, max_class * 1.15)  # Mehr Platz für Labels

# RECHTER PLOT Pie Chart
axes[1].pie(
    class_counts.values,
    labels=[f"Klasse {int(i)}" for i in class_counts.index],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    textprops={"fontsize": 10},
)
axes[1].set_title(
    f"Klassenverteilung (prozentual)\nImbalance Ratio: {imbalance_ratio:.0f}:1",
    fontsize=12,
    fontweight="bold",
)

plt.tight_layout()

output_file_5 = figures_dir / "eda_class_imbalance.png"
plt.savefig(output_file_5, dpi=300, bbox_inches="tight")
plt.close()

# Zusammenfassung

warnings.filterwarnings("ignore")

# Testdaten laden (wie im Training)
df_processed = pd.read_csv(DATA_PROCESSED / "techuse_processed.csv")
feature_cols = [
    "age",
    "daily_screen_time_hours",
    "phone_usage_hours",
    "laptop_usage_hours",
    "tablet_usage_hours",
    "tv_usage_hours",
    "social_media_hours",
    "work_related_hours",
    "entertainment_hours",
    "gaming_hours",
    "physical_activity_hours_per_week",
    "caffeine_mg",
    "mindfulness_minutes_per_day",
    "mental_health_score",
    "mood_rating",
    "stress_level",
    "uses_wellness_apps",
    "eats_healthy",
]
# One-hot columns
one_hot_cols = [
    c
    for c in df_processed.columns
    if c.startswith("gender_") or c.startswith("location_type_")
]
feature_cols += one_hot_cols
feature_cols = [c for c in feature_cols if c in df_processed.columns]

X = df_processed[feature_cols].fillna(0)
y = df_processed["sleep_quality"].astype(int)
y_class = y - 1  # 0-4 für Modelle

_, X_test, _, y_test = train_test_split(X, y_class, test_size=0.2, random_state=135)

MODELS_DIR = Path(__file__).parent.parent / "models"
model_paths = {
    "XGBoost Baseline": MODELS_DIR / "xgb_classifier_baseline.pkl",
    "XGBoost Tuned": MODELS_DIR / "xgb_classifier_tuned.pkl",
    "Keras": MODELS_DIR / "keras_classifier.keras",
}

# Skaler für Keras laden
scaler_path = MODELS_DIR / "scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
else:
    scaler = None

# Vorhersagen berechnen
preds = {}
labels = [1, 2, 3, 4, 5]

# XGBoost Baseline
if model_paths["XGBoost Baseline"].exists():
    xgb_base = joblib.load(model_paths["XGBoost Baseline"])
    preds["XGBoost Baseline"] = xgb_base.predict(X_test)

# XGBoost Tuned
if model_paths["XGBoost Tuned"].exists():
    xgb_tuned = joblib.load(model_paths["XGBoost Tuned"])
    preds["XGBoost Tuned"] = xgb_tuned.predict(X_test)

# Keras
try:
    if model_paths["Keras"].exists() and scaler is not None:
        keras_model = keras.models.load_model(model_paths["Keras"])
        X_test_scaled = scaler.transform(X_test)
        keras_pred = keras_model.predict(X_test_scaled, verbose=0)
        preds["Keras"] = keras_pred.argmax(axis=1)
except ImportError:
    pass

# Multi-Plot erstellen
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, y_pred) in enumerate(preds.items()):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Vorhergesagte Klasse")
    axes[idx].set_ylabel("Wahre Klasse")
    axes[idx].set_xticklabels(labels)
    axes[idx].set_yticklabels(labels)
plt.suptitle(
    "Confusion Matrices aller Modelle (Testdaten)", fontsize=16, fontweight="bold"
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Speichern
output_file_cm = REPORTS_DIR / "confusion_matrices.png"
plt.savefig(output_file_cm, dpi=300, bbox_inches="tight")
plt.close()

print(f"Confusion Matrices gespeichert: {output_file_cm}")
print(f"Fertig! 6 Plots gespeichert in: {figures_dir}")
