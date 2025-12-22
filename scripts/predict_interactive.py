"""
Autor: Henry Marx
Datum: 15.12.2025

Interaktive Schlafqualitäts-Klassifikation
Gib eigene Werte ein und erhalte eine Klassenvorhersage (1-5) mit Wahrscheinlichkeiten.
"""

import pandas as pd
import joblib
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELS_DIR, DATA_RAW, DATA_PROCESSED

print("\n" + "=" * 70)
print("Schlafqualitäts-KLASSIFIKATION (Interaktiv)")
print("=" * 70)

# lade KLASSIFIKATIONS-modelle
xgb_classifier = None
for name in ["xgb_classifier_tuned.pkl", "xgb_classifier_baseline.pkl"]:
    p = MODELS_DIR / name
    if p.exists():
        try:
            xgb_classifier = joblib.load(p)
            print(f" XGBoost Classifier geladen: {p.name}")
            break
        except Exception:
            pass

keras_classifier = None
scaler = None
keras_path = MODELS_DIR / "keras_classifier.keras"
scaler_path = MODELS_DIR / "scaler.pkl"

try:
    from tensorflow import keras as tfkeras

    if keras_path.exists():
        keras_classifier = tfkeras.models.load_model(keras_path)
        print(" Keras Classifier geladen")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("Scaler geladen")
except Exception as e:
    print(f"WARNUNG: Keras konnte nicht geladen werden: {e}")

if not xgb_classifier and not keras_classifier:
    print("\nFehler: Keine Klassifikations-Modelle gefunden.")
    print("Bitte zuerst train_model.py ausführen.")
    exit(1)

print("Wählen Sie einen Eingabemodus:")
print("\n1 - Eigene Werte manuell eingeben")
print("2 - Beispiel aus dem Datensatz verwenden")
print()

try:
    mode = input("Wähle einen Modus (1 oder 2): ").strip()

    actual_sleep_quality = None  # Für Beispiel-Modus

    if mode == "2":
        # Beispiel aus Datensatz
        print("\n" + "=" * 70)
        print("BEISPIEL AUS DATENSATZ")
        print("=" * 70)

        # Lade beide Datensätze: RAW für user_id, PROCESSED für Features
        raw_file = DATA_RAW / "Tech_Use_Stress_Wellness.csv"
        processed_file = DATA_PROCESSED / "techuse_processed.csv"

        if not raw_file.exists() or not processed_file.exists():
            print(f"Fehler: Datensätze nicht gefunden.")
            exit(1)

        df_raw = pd.read_csv(raw_file)
        df_processed = pd.read_csv(processed_file)

        # Prüfe ob user_id vorhanden
        if "user_id" not in df_raw.columns:
            print("Fehler: user_id nicht im Datensatz gefunden.")
            exit(1)

        print(
            f"\nVerfügbare User IDs: {df_raw['user_id'].min()} bis {df_raw['user_id'].max()}"
        )
        print(f"Total: {len(df_raw)} User")

        user_id_input = input(
            f"\nWähle eine User ID (oder Enter für zufällig): "
        ).strip()

        if user_id_input:
            user_id = int(user_id_input)
            # Finde Index der user_id im RAW Datensatz
            matching_rows = df_raw[df_raw["user_id"] == user_id]
            if len(matching_rows) == 0:
                print(f"User ID {user_id} nicht gefunden. Nutze zufällige User ID.")
                user_idx = np.random.randint(0, len(df_raw))
                user_id = df_raw.iloc[user_idx]["user_id"]
            else:
                user_idx = matching_rows.index[0]
        else:
            user_idx = np.random.randint(0, len(df_raw))
            user_id = df_raw.iloc[user_idx]["user_id"]

        # Hole die entsprechende Zeile aus PROCESSED (gleicher Index)
        user_row = df_processed.iloc[user_idx]
        actual_sleep_quality = int(user_row["sleep_quality"])
        actual_class = actual_sleep_quality - 1  # 0-4

        df_input = pd.DataFrame([user_row.drop("sleep_quality")])

        print(f"\n{'=' * 70}")
        print(f"Beispiel-Daten:")
        print(f"  User ID: {int(user_id)}")
        print(f"  DataFrame-Index: {user_idx}")
        print(
            f"  Tatsächliche Sleep Quality: {actual_sleep_quality} (Klasse {actual_class})"
        )
        print(f"{'=' * 70}")

    elif mode == "1":
        # Manuelle Eingabe
        print("\n" + "=" * 70)
        print("Gib deine Werte ein:")
        print("=" * 70)

        age = float(input("Alter: "))
        daily_screen_time = float(input("Bildschirmzeit (Stunden/Tag): "))
        phone_usage = float(input("Handy-Nutzung (Stunden/Tag): "))
        laptop_usage = float(input("Laptop-Nutzung (Stunden/Tag): "))
        tablet_usage = float(input("Tablet-Nutzung (Stunden/Tag): "))
        tv_usage = float(input("TV-Nutzung (Stunden/Tag): "))
        social_media = float(input("Social Media (Stunden/Tag): "))
        work_related = float(input("Arbeit am Gerät (Stunden/Tag): "))
        entertainment = float(input("Entertainment (Stunden/Tag): "))
        gaming = float(input("Gaming (Stunden/Tag): "))
        physical_activity = float(input("Körperliche Aktivität (Stunden/Woche): "))
        caffeine = float(input("Koffein (mg/Tag): "))
        mindfulness = float(input("Achtsamkeit/Meditation (Minuten/Tag): "))
        mental_health = float(input("Mental Health Score (0-100): "))
        mood = float(input("Stimmung (1-10): "))
        stress = float(input("Stress-Level (1-10): "))
        wellness_apps = int(input("Nutzt du Wellness Apps? (0=Nein, 1=Ja): "))
        healthy_eating = int(input("Isst du gesund? (0=Nein, 1=Ja): "))

        print("\nGeschlecht: 0=Weiblich, 1=Männlich, 2=Andere")
        gender = int(input("Geschlecht: "))

        print("\nStandort: 0=Stadt, 1=Vorort, 2=Land")
        location = int(input("Standort: "))

        features = {
            "age": age,
            "daily_screen_time_hours": daily_screen_time,
            "phone_usage_hours": phone_usage,
            "laptop_usage_hours": laptop_usage,
            "tablet_usage_hours": tablet_usage,
            "tv_usage_hours": tv_usage,
            "social_media_hours": social_media,
            "work_related_hours": work_related,
            "entertainment_hours": entertainment,
            "gaming_hours": gaming,
            "physical_activity_hours_per_week": physical_activity,
            "caffeine_mg": caffeine,
            "mindfulness_minutes_per_day": mindfulness,
            "mental_health_score": mental_health,
            "mood_rating": mood,
            "stress_level": stress,
            "uses_wellness_apps": wellness_apps,
            "eats_healthy": healthy_eating,
            "gender_Male": 1 if gender == 1 else 0,
            "gender_Other": 1 if gender == 2 else 0,
            "location_type_Suburban": 1 if location == 1 else 0,
            "location_type_Urban": 1 if location == 0 else 0,
        }

        df_input = pd.DataFrame([features])

    else:
        print("Ungültige Wahl. Beende Programm.")
        exit(1)

except ValueError:
    print("\nFehler: Bitte gültige Zahlen eingeben.")
    exit(1)
except Exception as e:
    print(f"\nFehler: {e}")
    exit(1)

# VORHERSAGEN
print("\n" + "=" * 70)
print("KLASSIFIKATIONS-VORHERSAGEN")
print("=" * 70)

if actual_sleep_quality is not None:
    quality_labels = {
        1: "1 (sehr schlecht)",
        2: "2 (schlecht)",
        3: "3 (mittel)",
        4: "4 (gut)",
        5: "5 (sehr gut)",
    }
    print(f"Tatsächlich: {quality_labels[actual_sleep_quality]}\n")

# XGBoost Classifier
if xgb_classifier:
    try:
        pred_class = int(xgb_classifier.predict(df_input)[0])
        pred_sleep_quality = pred_class + 1  # Konvertiere 0-4 zu 1-5

        print(f"XGBoost Classifier:")
        print(f"  Vorhersage: Klasse {pred_class} → Sleep Quality {pred_sleep_quality}")

        if actual_sleep_quality is not None:
            error = abs(pred_sleep_quality - actual_sleep_quality)
            if error == 0:
                assessment = "KORREKT!"
            elif error == 1:
                assessment = "Off-by-1 (akzeptabel)"
            else:
                assessment = f"Off-by-{error}"
            print(f"  Bewertung: {assessment}")

        print()
    except Exception as e:
        print(f"XGBoost Fehler: {e}\n")

# Keras Classifier
if keras_classifier:
    try:
        X = df_input.values
        if scaler:
            X = scaler.transform(X)

        pred_probs = keras_classifier.predict(X, verbose=0)[0]
        pred_class = int(np.argmax(pred_probs))
        pred_sleep_quality = pred_class + 1

        print(f"Keras Classifier:")
        print(f"  Vorhersage: Klasse {pred_class} → Sleep Quality {pred_sleep_quality}")
        print(f"  Wahrscheinlichkeitsverteilung:")

        for i, prob in enumerate(pred_probs):
            class_label = i + 1
            marker = (
                "*"
                if (
                    actual_sleep_quality is not None
                    and class_label == actual_sleep_quality
                )
                else (">" if i == pred_class else " ")
            )
            bar = "" * int(prob * 50)  # Balken für Visualisierung
            print(f"    {marker} Klasse {i} (Quality {class_label}): {prob:.1%} {bar}")

        if actual_sleep_quality is not None:
            error = abs(pred_sleep_quality - actual_sleep_quality)
            if error == 0:
                assessment = "KORREKT!"
            elif error == 1:
                assessment = "Off-by-1 (akzeptabel)"
            else:
                assessment = f"Off-by-{error}"
            print(f"  Bewertung: {assessment}")

        print()
    except Exception as e:
        print(f"Keras Fehler: {e}\n")

# FEATURE IMPORTANCE (SHAP oder einfach)
if xgb_classifier:
    print("." * 70)
    print("EINFLUSS-ANALYSE (Top 5 wichtigste Faktoren)")
    print("." * 70)

    try:
        import shap

        explainer = shap.TreeExplainer(xgb_classifier)
        shap_values = explainer.shap_values(df_input)

        # Bei Multi-Class: Liste von Arrays (eines pro Klasse)
        if isinstance(shap_values, list):
            # Nutze SHAP-Werte der vorhergesagten Klasse
            pred_class_idx = int(xgb_classifier.predict(df_input)[0])
            shap_values_pred = shap_values[pred_class_idx]
            # Flatten zu 1D wenn nötig
            shap_values_pred = shap_values_pred.flatten()
        else:
            # Flatten zu 1D wenn nötig
            shap_values_pred = shap_values.flatten()

        feature_names = df_input.columns.tolist()

        # Konvertiere SHAP-Werte zu Python-Skalaren (keine Arrays mehr)
        contributions = []
        for name, val in zip(feature_names, shap_values_pred):
            # Falls val noch ein Array ist, extrahiere Skalar
            if hasattr(val, "item"):
                contributions.append((name, val.item()))
            else:
                contributions.append((name, float(val)))

        contributions_sorted = sorted(
            contributions, key=lambda x: abs(x[1]), reverse=True
        )

        feature_de = {
            "stress_level": "Stress-Level",
            "mental_health_score": "Mental Health Score",
            "physical_activity_hours_per_week": "Körperliche Aktivität",
            "daily_screen_time_hours": "Bildschirmzeit",
            "caffeine_mg": "Koffein",
            "mood_rating": "Stimmung",
            "mindfulness_minutes_per_day": "Achtsamkeit",
            "social_media_hours": "Social Media",
            "gaming_hours": "Gaming",
            "work_related_hours": "Arbeit am Gerät",
            "phone_usage_hours": "Handy-Nutzung",
            "age": "Alter",
        }

        print("\nTop 5 Faktoren für diese Vorhersage:")
        for i, (feature, impact) in enumerate(contributions_sorted[:5], 1):
            direction = "erhöht" if impact > 0 else "senkt"
            feature_name = feature_de.get(feature, feature)
            print(
                f"{i}. {feature_name}: {direction} Wahrscheinlichkeit um {abs(impact):.3f}"
            )

    except ImportError:
        print("\nHinweis: SHAP nicht installiert (pip install shap)")
        print("Zeige einfache Feature-Wichtigkeit:\n")

        importance = xgb_classifier.feature_importances_
        feature_names = df_input.columns.tolist()
        feat_imp = sorted(
            zip(feature_names, importance), key=lambda x: x[1], reverse=True
        )

        feature_de = {
            "stress_level": "Stress-Level",
            "mental_health_score": "Mental Health Score",
            "physical_activity_hours_per_week": "Körperliche Aktivität",
            "daily_screen_time_hours": "Bildschirmzeit",
            "caffeine_mg": "Koffein",
            "mood_rating": "Stimmung",
            "mindfulness_minutes_per_day": "Achtsamkeit",
        }

        for i, (feature, imp) in enumerate(feat_imp[:5], 1):
            feature_name = feature_de.get(feature, feature)
            print(f"{i}. {feature_name} (Wichtigkeit: {imp:.3f})")
