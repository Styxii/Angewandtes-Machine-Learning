"""
Sleep Quality Classification - Trainingsskript

Autor: Henry Marx
Datum: 15.12.2025
"""

import pandas as pd
import numpy as np
import sys
import joblib

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, RANDOM_SEED

# falls tensorflow installiert ist
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

# alle device nutzungen einzeln
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

# wie viele Spalten existieren tatsächlich im DataFrame
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(0)

print(f"Features: {len(X)} Samples, {len(feature_cols)} Features")

# sichern damit man später nachschauen kann
processed_file = DATA_PROCESSED / "techuse_processed.csv"
X.assign(sleep_quality=y).to_csv(processed_file, index=False)
print(f"Prozessierte Daten: {processed_file}")

# sleep_quality als diskrete Klassen (1-5)
print("KLASSIFIKATION (sleep_quality als diskrete Klassen 1-5)")

# für klassifikation brauchen wir die labels als integers (0-4)
y_class = (y.astype(int) - 1).values  # Konvertiere 1-5 zu 0-4

# Kein stratify, da manche Klassen nur 1 Sample haben
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=RANDOM_SEED
)

print(f"Training-Set: {len(X_train)} Samples")
print(f"Test-Set: {len(X_test)} Samples")
print(f"Klassen: 0-4 (entspricht Schlafqualität 1-5)")

# ...........................................................
# XGBoost Classifier - Baseline
print("\n1. Trainiere XGBoost Classifier (Baseline)...")

sample_weights = compute_sample_weight("balanced", y_train_class)
print(f"   Class Weights aktiviert (Imbalance Ausgleich)")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

clf_model = XGBClassifier(
    n_estimators=300,  # Erhöht von 200 für mehr Lernkapazität
    max_depth=6,  # Erhöht von 5 für komplexere Patterns
    learning_rate=0.05,  # Konservative LR für stabile Konvergenz
    min_child_weight=3,  # Höher für Class Imbalance (verhindert Overfitting auf seltene Klassen)
    random_state=RANDOM_SEED,
    verbosity=0,
    eval_metric="mlogloss",
)
clf_model.fit(X_train, y_train_class, sample_weight=sample_weights)

# Training Metriken
y_pred_train = clf_model.predict(X_train)
y_pred_train_proba = clf_model.predict_proba(X_train)
acc_train = accuracy_score(y_train_class, y_pred_train)
f1_train = f1_score(y_train_class, y_pred_train, average="weighted")

loss_train = log_loss(y_train_class, y_pred_train_proba, labels=[0, 1, 2, 3, 4])

# Test Metriken
y_pred_class = clf_model.predict(X_test)
y_pred_proba = clf_model.predict_proba(X_test)
acc = accuracy_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class, average="weighted")
loss_test = log_loss(y_test_class, y_pred_proba, labels=[0, 1, 2, 3, 4])

# Für Ausgabe: zurück zu 1-5
y_test_display = y_test_class + 1
y_pred_display = y_pred_class + 1

print(f"   Training: Acc={acc_train:.4f}, Loss={loss_train:.4f}, F1={f1_train:.4f}")

clf_path = MODELS_DIR / "xgb_classifier_baseline.pkl"
joblib.dump(clf_model, clf_path)
print(f"   Gespeichert: {clf_path}")

# ...........................................................
# XGBoost Classifier - Hyperparameter Tuning

print("\n2. Hyperparameter-Tuning...")
try:
    param_dist_clf = {
        # Weniger Overfitting: kleinere max_depth, mehr Regularisierung, kleinere learning_rate
        "n_estimators": [100, 200, 300],  # Weniger Bäume, um Overfitting zu vermeiden
        "max_depth": [2, 3, 4, 5],  # Flachere Bäume
        "learning_rate": [0.01, 0.03, 0.05],  # Noch konservativer
        "subsample": [0.6, 0.7, 0.8],  # Weniger Daten pro Baum
        "colsample_bytree": [0.5, 0.6, 0.7],  # Weniger Features pro Baum
        "min_child_weight": [5, 7, 10],  # Höher für robustere Blätter
        "gamma": [0.1, 0.2, 0.3],  # Mehr Pruning
        "reg_alpha": [0.1, 0.5, 1.0],  # Mehr L1 Regularisierung
        "reg_lambda": [1.0, 2.0, 3.0],  # Mehr L2 Regularisierung
    }

    base_clf = XGBClassifier(
        random_state=RANDOM_SEED, verbosity=0, eval_metric="mlogloss"
    )

    # PredefinedSplit mit einem einzigen Train-Validation-Split
    # Damit existieren alle Klassen in beiden Splits
    print("Erstelle Train-Validation Split für Hyperparameter-Tuning...")

    # 80% Training, 20% Validation Split
    split_index = int(len(X_train) * 0.8)
    test_fold = [-1] * split_index + [0] * (
        len(X_train) - split_index
    )  # -1 = train, 0 = validation
    cv_strategy = PredefinedSplit(test_fold)

    search_clf = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist_clf,
        n_iter=40,  # Erhöht von 30 für bessere Exploration
        scoring="f1_weighted",  # Besser für Class Imbalance
        cv=cv_strategy,  # Single Train-Val Split
        verbose=2,  # Zeige Fortschritt
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Nutze alle CPU-Kerne
        refit=True,
    )

    print("   Starte Tuning (Single Train-Val Split, F1-Weighted)...")

    # Fit mit sample_weight für konsistentes Training
    # sample_weights für den gesamten X_train berechnen
    search_clf.fit(X_train, y_train_class, sample_weight=sample_weights)

    best_clf = search_clf.best_estimator_

    # Jetzt mit sample_weight auf vollem Training Set nachtrainieren
    best_clf.fit(X_train, y_train_class, sample_weight=sample_weights)

    # Training Metriken
    y_pred_train_tuned = best_clf.predict(X_train)
    y_pred_train_tuned_proba = best_clf.predict_proba(X_train)
    acc_train_tuned = accuracy_score(y_train_class, y_pred_train_tuned)
    f1_train_tuned = f1_score(y_train_class, y_pred_train_tuned, average="weighted")
    loss_train_tuned = log_loss(
        y_train_class, y_pred_train_tuned_proba, labels=[0, 1, 2, 3, 4]
    )

    # Test Metriken
    y_pred_clf_tuned = best_clf.predict(X_test)
    y_pred_clf_tuned_proba = best_clf.predict_proba(X_test)
    y_pred_clf_tuned_display = y_pred_clf_tuned + 1

    acc_tuned = accuracy_score(y_test_class, y_pred_clf_tuned)
    f1_tuned = f1_score(y_test_class, y_pred_clf_tuned, average="weighted")
    loss_tuned = log_loss(y_test_class, y_pred_clf_tuned_proba, labels=[0, 1, 2, 3, 4])

    print(
        f"   Training: Acc={acc_train_tuned:.4f}, Loss={loss_train_tuned:.4f}, F1={f1_train_tuned:.4f}"
    )

    clf_tuned_path = MODELS_DIR / "xgb_classifier_tuned.pkl"
    joblib.dump(best_clf, clf_tuned_path)
    print(f"   Gespeichert: {clf_tuned_path}")

except Exception as e:
    print(f" Classifier Tuning fehlgeschlagen: {e}")

# ...........................................................
# Keras Neural Network Classifier
if HAS_KERAS:
    print("\n3. Trainiere Keras Neural Network Classifier...")

    # IQR-basiertes Outlier-Capping für Keras
    print("   Outlier-Capping für Neural Network...")
    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()

    # Identifiziere binäre Features
    binary_features = [col for col in X_train.columns if X_train[col].nunique() <= 2]
    continuous_features = [col for col in X_train.columns if col not in binary_features]

    # Nur kontinuierliche Features cappen
    for col in continuous_features:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers in Training data
        X_train_capped[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
        # Cap outliers in Test data (using Training bounds!)
        X_test_capped[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)

    # StandardScaler für Keras
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_capped)
    X_test_scaled = scaler.transform(X_test_capped)

    # Neural Network mit Softmax Output für 5 Klassen
    clf_nn_model = Sequential(
        [
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001),
                input_shape=(X_train_scaled.shape[1],),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(
                96, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(
                64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(
                32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)
            ),
            layers.Dropout(0.2),
            # Output: 5 Klassen mit Softmax
            layers.Dense(5, activation="softmax"),
        ]
    )

    optimizer_clf = keras.optimizers.Adam(
        learning_rate=0.003, clipvalue=1.0
    )  # Reduziert von 0.005

    # sparse_categorical_crossentropy für diskrete Klassen
    clf_nn_model.compile(
        optimizer=optimizer_clf,
        loss="sparse_categorical_crossentropy",  # CrossEntropyLoss
        metrics=["accuracy"],
    )

    early_stop_clf = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=0,  # Reduziert von 30
    )

    reduce_lr_clf = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=0.0001,
        verbose=0,  # Patience reduziert von 10
    )

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_class), y=y_train_class
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("   Keras Training (max 500 Epochen, Early Stopping aktiv)...")
    history_clf = clf_nn_model.fit(
        X_train_scaled,
        y_train_class,
        epochs=500,
        batch_size=64,  # Erhöht von 32 für stabileres Training
        validation_split=0.2,
        verbose=1,
        shuffle=True,
        callbacks=[early_stop_clf, reduce_lr_clf],
        class_weight=class_weight_dict,
    )

    # Beste Epoche aus History extrahieren
    best_epoch = np.argmin(history_clf.history["val_loss"])
    train_acc_final = history_clf.history["accuracy"][best_epoch]
    train_loss_final = history_clf.history["loss"][best_epoch]
    val_acc_final = history_clf.history["val_accuracy"][best_epoch]
    val_loss_final = history_clf.history["val_loss"][best_epoch]

    # Test Set Evaluation
    y_pred_nn_clf_probs = clf_nn_model.predict(X_test_scaled, verbose=0)
    y_pred_nn_clf = np.argmax(
        y_pred_nn_clf_probs, axis=1
    )  # Klasse mit höchster Wahrscheinlichkeit
    y_pred_nn_clf_display = y_pred_nn_clf + 1

    acc_nn = accuracy_score(y_test_class, y_pred_nn_clf)
    f1_nn = f1_score(y_test_class, y_pred_nn_clf, average="weighted")

    loss_nn = log_loss(y_test_class, y_pred_nn_clf_probs, labels=[0, 1, 2, 3, 4])

    print(f"   Training:   Acc={train_acc_final:.4f}, Loss={train_loss_final:.4f}")
    print(f"   Validation: Acc={val_acc_final:.4f}, Loss={val_loss_final:.4f}")
    print(f"   (Best Epoch: {best_epoch + 1}/{len(history_clf.history['loss'])})")
    print(f"   F1 Score: {f1_nn:.4f}")

    # Speichern
    clf_nn_path = MODELS_DIR / "keras_classifier.keras"
    clf_nn_model.save(clf_nn_path)
    print(f"   Gespeichert: {clf_nn_path}")

    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Training History speichern für Plots
    history_df = pd.DataFrame(history_clf.history)
    history_file = DATA_PROCESSED / "keras_classifier_history.csv"
    history_df.to_csv(history_file, index=False)

else:
    print("\n Keras nicht verfügbar. Nur XGBoost trainiert.")


joblib.dump(
    {
        "X_train": X_train,  # Werte für XGBoost und spätere Analysen speichern
        "X_test": X_test,
        "y_train": y_train_class,
        "y_test": y_test_class,
    },
    MODELS_DIR / "train_test_splits.pkl",
)

joblib.dump(
    {"y_pred_test_keras": y_pred_nn_clf}, MODELS_DIR / "keras_results.pkl"
)  # Gespeicherte Vorhersagen für Keras für spätere Analysen

# Zusammenfassung
print("TRAINING ABGESCHLOSSEN")
print("." * 50)
print("\nTrainierte Modelle:")
print("  1. XGBoost Classifier Baseline")
print("  2. XGBoost Classifier Tuned")
if HAS_KERAS:
    print("  3. Keras Neural Network Classifier")
print(f"\nGespeichert in: {MODELS_DIR}")
print("\nNächste Schritte:")
print("  - Visualisierungen: python scripts/plot_classification_results.py")
print("  - Vorhersagen: python scripts/predict_interactive.py")
print("." * 50)
