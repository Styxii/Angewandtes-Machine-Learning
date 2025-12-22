
### Bedienung des Projekts und weitere Erläuterungen

1. Es gibt eine setup.py-Datei die alle nötigen Packages installieren sollte, führen sie diese aus
2. Danach ist die train_model.py die Hauptdatei, mit der trainiert werden kann.
    - dabei zieth sich das Script die Daten aus der "tech_use_stress_wellness.csv" im raw-Ordner und nutzt diese um 3 verschiedene Modelle automatisch zu trainieren.
    - XGBoost Baseline, XGBoost tuned sowie ein Keras-Modell
3. Die eda.py-Datei erstellt einige Plots die die Daten in Diagrammen darstellen und einige Rückschlüsse erkennen lassen, die in der Dokumentation festgehalten wurden
    - diese Plots finden sie unter reports/figures/eda
    - zusätzlich wird eine confusion_matrix im Ordner figures erstellt
4. Predict_interactive lässt auf Grundlage des trainierten Modells eine Prognose erstellen. Hier gibt es 2 Optionen. Ein aus dem Datensatz zufällig über die User-Id ausgewähltes Beispiel wählen um zu schauen wie das Modell auf Grundlage dieser Daten vorhersagt. Die zweite Option ist eine Abfrage der verwendeten Features, bei der man persönliche Werte eingeben kann und schließlich eine Prognose über die eigene Schlafqualität erhält.
5. Die Dokumentation ist unter dem Projekt-Ordner "notebooks" gespeichert. 

Viel Spaß mit dem Projekt!
