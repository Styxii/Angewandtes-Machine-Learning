#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sleep Prediction - Trainingsskript
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# damit wir src importieren können
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, RANDOM_SEED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import time

# falls tensorflow installiert ist, nutzen wir das auch
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential

    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("Keras nicht gefunden, machen wir halt nur XGBoost")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED) if HAS_KERAS else None


print("Training: Tech_Use_Stress_Wellness.csv")

# CSV laden
raw_file = DATA_RAW / "Tech_Use_Stress_Wellness.csv"
if not raw_file.exists():
    raise FileNotFoundError(
        f"{raw_file} nicht gefunden. Bitte lege die CSV in data/raw/ ab."
    )

df = pd.read_csv(raw_file)
print(f"Rohdaten: {raw_file} ({len(df)} Zeilen)")

# daten vorbereiten
print("Feature-Engineering...")

df = df.copy()

# Zielvariable
if "sleep_quality" not in df.columns:
    raise ValueError("sleep_quality Spalte nicht gefunden in CSV")

y = df["sleep_quality"].astype(float)

# alle device nutzungen einzeln (nicht zusammenrechnen)
device_cols = [
    "phone_usage_hours",
    "laptop_usage_hours",
    "tablet_usage_hours",
    "tv_usage_hours",
    "social_media_hours",
    "work_related_hours",
    "entertainment_hours",
    "gaming_hours",
]
for c in device_cols:
    if c not in df.columns:
        df[c] = 0.0

# koffein spalte umbenennen falls nötig
if "caffeine_intake_mg_per_day" in df.columns:
    df.rename(columns={"caffeine_intake_mg_per_day": "caffeine_mg"}, inplace=True)
elif "caffeine_mg" not in df.columns:
    df["caffeine_mg"] = 0.0

# ja/nein features
if "uses_wellness_apps" in df.columns:
    df["uses_wellness_apps"] = df["uses_wellness_apps"].astype(int)
else:
    df["uses_wellness_apps"] = 0

if "eats_healthy" in df.columns:
    df["eats_healthy"] = df["eats_healthy"].astype(int)
else:
    df["eats_healthy"] = 0

# manchmal sind true/false als text gespeichert
for col in ["uses_wellness_apps", "eats_healthy"]:
    if df[col].dtype == object:
        df[col] = df[col].map({"True": 1, "False": 0}).fillna(0).astype(int)

# fehlende werte auffüllen
df["age"] = df["age"].fillna(df["age"].median())
if "physical_activity_hours_per_week" in df.columns:
    df["physical_activity_hours_per_week"] = df[
        "physical_activity_hours_per_week"
    ].fillna(0)
else:
    df["physical_activity_hours_per_week"] = 0

df["mindfulness_minutes_per_day"] = df.get("mindfulness_minutes_per_day", 0).fillna(0)
df["mental_health_score"] = df.get("mental_health_score", 0).fillna(
    df.get("mental_health_score", df["age"].median())
)

df["mood_rating"] = df.get("mood_rating", 0).fillna(0)
df["stress_level"] = df.get("stress_level", 0).fillna(0)

# geschlecht und standort in zahlen umwandeln
cat_cols = []
if "gender" in df.columns:
    cat_cols.append("gender")
if "location_type" in df.columns:
    cat_cols.append("location_type")

if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# alle features die wir nutzen wollen
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

# die umgewandelten kategorien auch noch dazu
one_hot_cols = [
    c for c in df.columns if c.startswith("gender_") or c.startswith("location_type_")
]
feature_cols += one_hot_cols

# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(0)

print(f"Features: {len(X)} Samples, {len(feature_cols)} Features")

# sichern damit man später nachschauen kann
processed_file = DATA_PROCESSED / "techuse_processed.csv"
X.assign(sleep_quality=y).to_csv(processed_file, index=False)
print(f"Prozessierte Daten: {processed_file}")

print("Trainiere XGBoost (Baseline)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

model = XGBRegressor(
    n_estimators=200, max_depth=5, random_state=RANDOM_SEED, verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

within_05 = np.mean(np.abs(y_test - y_pred) <= 0.5) * 100
within_10 = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100

print(f"Baseline Metriken:")
print(f"  R²:   {r2:.4f}")
print(f"  RMSE: {rmse:.4f} (Skala 1-5)")
print(f"  MAE:  {mae:.4f} (durchschnittlicher Fehler)")
print(
    f"  Genauigkeit: {within_05:.1f}% innerhalb ±0.5, {within_10:.1f}% innerhalb ±1.0"
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / "xgb_baseline.pkl"
joblib.dump(model, model_path)
print(f"Baseline gespeichert: {model_path}")

try:
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    pass
except Exception:
    pass

print("Hyperparameter-Tuning (RandomizedSearchCV)...")
try:
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    base = XGBRegressor(random_state=RANDOM_SEED, verbosity=0)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=25,
        scoring="r2",
        cv=3,
        verbose=2,  # Zeigt Fortschritt: 0=keine, 1=kurz, 2=ausführlich
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    pass

    best_model = search.best_estimator_
    tuned_path = MODELS_DIR / "xgb_tuned.pkl"
    joblib.dump(best_model, tuned_path)
    print(f"Tuned gespeichert: {tuned_path}")

    # Evaluate auf Test-Set mit interpretierbaren Metriken
    y_pred_tuned = best_model.predict(X_test)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    within_05_tuned = np.mean(np.abs(y_test - y_pred_tuned) <= 0.5) * 100
    within_10_tuned = np.mean(np.abs(y_test - y_pred_tuned) <= 1.0) * 100

    print(f"Tuned Metriken:")
    print(f"  R²:   {r2_tuned:.4f}")
    print(f"  RMSE: {rmse_tuned:.4f} (Skala 1-5)")
    print(f"  MAE:  {mae_tuned:.4f} (durchschnittlicher Fehler)")
    print(
        f"  Genauigkeit: {within_05_tuned:.1f}% innerhalb ±0.5, {within_10_tuned:.1f}% innerhalb ±1.0"
    )

    try:
        fi2 = pd.DataFrame(
            {"feature": X.columns, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        pass
    except Exception:
        pass
except Exception as e:
    print("Tuning fehlgeschlagen:", e)

if HAS_KERAS:
    print("Trainiere Keras FFN (max Qualität)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # netzwerk mit mehreren schichten
    nn_model = Sequential(
        [
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001),
                input_shape=(X_train_scaled.shape[1],),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            layers.Dense(
                96, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            layers.Dense(
                64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(
                32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)
            ),
            layers.Dropout(0.05),
            layers.Dense(1),  # Output-Schicht
        ]
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=0.005,
        clipvalue=1.0,
    )

    nn_model.compile(
        optimizer=optimizer,
        loss="huber",
        metrics=["mae", "mse"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,  # Großzügiger bei Geduld
        restore_best_weights=True,
        verbose=0,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=0,
    )

    print("Keras Training startet (max 500 Epochen, Early Stopping aktiv)...")
    history = nn_model.fit(
        X_train_scaled,
        y_train,
        epochs=500,
        batch_size=8,
        validation_split=0.2,
        verbose=1,
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
    )

    nn_test_loss, nn_test_mae, nn_test_mse = nn_model.evaluate(
        X_test_scaled, y_test, verbose=0
    )
    y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_r2 = r2_score(y_test, y_pred_nn)
    nn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))

    print(
        f"Keras: Loss {nn_test_loss:.4f} | MAE {nn_test_mae:.4f} | RMSE {nn_rmse:.4f} | R² {nn_r2:.4f}"
    )

    nn_path = MODELS_DIR / "keras_ffn.keras"
    nn_model.save(nn_path)
    print(f"Keras gespeichert: {nn_path}")

    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler gespeichert: {scaler_path}")

    history_df = pd.DataFrame(history.history)
    history_path = DATA_PROCESSED / "keras_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"History gespeichert: {history_path}")

else:
    print("Keras nicht verfügbar. Überspringe.")

print("Training abgeschlossen.")
