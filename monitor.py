import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Données de référence (entraînement)
ref_df = pd.read_csv("reference_data.csv")

# Données de production (logs de l'API)
prod_df = pd.read_csv("logs/current_data.csv")

# Rapport de drift
report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=ref_df, current_data=prod_df)
result.save_html("monitoring_report.html")
print("Rapport généré : monitoring_report.html")