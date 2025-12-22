"""
Projekt Setup - ertellt notwendige Verzeichnisse und Konfiguration
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

print("Sleep Prediction Project - Setup")

# Verzeichnisse erstellen
print("\nErstelle Verzeichnisse...")
dirs = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "interim",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "models",
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"   {d.relative_to(PROJECT_ROOT)}")

# config.py erstellen
print("\nErstelle src/config.py...")
config_file = PROJECT_ROOT / "src" / "config.py"
config_content = '''"""Zentrale Konfiguration"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
'''
with open(config_file, "w", encoding="utf-8") as f:
    f.write(config_content)
print("   src/config.py")

# Python-Pakete installieren
print("\nInstalliere benötigte Pakete...")
requirements_file = PROJECT_ROOT / "requirements.txt"
if requirements_file.exists():
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_file),
                "--quiet",
            ]
        )
        print("  Alle Pakete installiert/aktualisiert")
    except subprocess.CalledProcessError:
        print("  WARNUNG: Fehler bei der Installation - bitte manuell prüfen")
else:
    print("  WARNUNG: requirements.txt nicht gefunden")

print("Setup abgeschlossen!")
