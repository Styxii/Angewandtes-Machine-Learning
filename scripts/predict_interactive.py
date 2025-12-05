"""
Interaktive Schlafqualitäts-Prognose
Gibt eigene Werte ein und erhalte eine Vorhersage von trainierten Modellen.
"""

import pandas as pd
import joblib
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELS_DIR

print("\n" + "=" * 70)
print("Schlafqualitäts-Prognose (Interaktiv)")
print("=" * 70)
xgb_model = None
for name in ["xgb_tuned.pkl", "xgb_baseline.pkl"]:
    p = MODELS_DIR / name
    if p.exists():
        try:
            xgb_model = joblib.load(p)
            print(f"✓ XGBoost geladen: {p.name}")
            break
        except Exception:
            pass
keras_model = None
scaler = None
keras_path = MODELS_DIR / "keras_ffn.keras"
scaler_path = MODELS_DIR / "scaler.pkl"
try:
    from tensorflow import keras as tfkeras

    if keras_path.exists():
        keras_model = tfkeras.models.load_model(keras_path)
        print("✓ Keras FFN geladen")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
except Exception:
    pass

if not xgb_model and not keras_model:
    print("\nFehler: Keine Modelle gefunden. Bitte zuerst train_model.py ausführen.")
    exit(1)

print("\n" + "=" * 70)
print("Gib deine Werte ein (basierend auf deinen täglichen Daten):")
print("=" * 70)

try:
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
    mental_health = float(input("Mental Health Score (1-10): "))
    mood = float(input("Stimmung (1-10): "))
    stress = float(input("Stress-Level (1-10): "))
    wellness_apps = int(input("Nutzt du Wellness Apps? (0=Nein, 1=Ja): "))
    healthy_eating = int(input("Isst du gesund? (0=Nein, 1=Ja): "))

    print("\nGeschlecht: 0=Weiblich, 1=Männlich, 2=Andere")
    gender = int(input("Geschlecht: "))

    print("\nStandort: 0=Stadt, 1=Vorort, 2=Land")
    location = int(input("Standort: "))

except ValueError:
    print("\nFehler: Bitte gültige Zahlen eingeben.")
    exit(1)

# features zusammenstellen
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

print("\n" + "=" * 70)
print("PROGNOSE")
print("=" * 70)
pred_xgb = None
if xgb_model:
    try:
        pred_xgb = float(xgb_model.predict(df_input)[0])
        print(f"XGBoost Vorhersage: {pred_xgb:.2f} / 5.0")
    except Exception as e:
        print(f"XGBoost Fehler: {e}")

if keras_model:
    try:
        X = df_input.values
        if scaler:
            X = scaler.transform(X)
        pred_keras = float(keras_model.predict(X, verbose=0).flatten()[0])
        print(f"Keras FFN Vorhersage: {pred_keras:.2f} / 5.0")
    except Exception as e:
        print(f"Keras Fehler: {e}")
if xgb_model and pred_xgb is not None:
    print("\n" + "=" * 70)
    print("EINFLUSS-ANALYSE (Top 5 wichtigste Faktoren)")
    print("=" * 70)

    try:
        import shap

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(df_input)
        feature_names = df_input.columns.tolist()
        contributions = list(zip(feature_names, shap_values[0]))
        contributions_sorted = sorted(
            contributions, key=lambda x: abs(x[1]), reverse=True
        )
        for i, (feature, impact) in enumerate(contributions_sorted[:5], 1):
            direction = "erhöht" if impact > 0 else "senkt"
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
            }.get(feature, feature)

            print(f"{i}. {feature_de}: {direction} die Vorhersage um {abs(impact):.3f}")

    except ImportError:
        print("\nHinweis: SHAP nicht installiert. Installiere mit: pip install shap")
        print("Zeige einfache Feature-Wichtigkeit:")
        importance = xgb_model.feature_importances_
        feature_names = df_input.columns.tolist()

        feat_imp = sorted(
            zip(feature_names, importance), key=lambda x: x[1], reverse=True
        )

        for i, (feature, imp) in enumerate(feat_imp[:5], 1):
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
            }.get(feature, feature)
            print(f"{i}. {feature_de} (Wichtigkeit: {imp:.3f})")

print("\n" + "=" * 70)
print("Interpretation:")
print("  4.5-5.0: Sehr gute Schlafqualität erwartet")
print("  4.0-4.5: Gute Schlafqualität erwartet")
print("  3.0-4.0: Mittelmäßige Schlafqualität")
print("  1.0-3.0: Schlechte Schlafqualität - Verbesserungen empfohlen")
print("=" * 70)
