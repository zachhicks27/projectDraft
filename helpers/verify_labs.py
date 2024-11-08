# verify_vitals.py
import pandas as pd
from pathlib import Path

# Load data
data_path = Path('data/mimic-iii-clinical-database-demo-1.4')
chartevents = pd.read_csv(data_path / 'CHARTEVENTS.csv')
d_items = pd.read_csv(data_path / 'D_ITEMS.csv')

# Vital sign ItemIDs we're looking for
vital_items = {
    '220050': 'Systolic BP',
    '220051': 'Diastolic BP',
    '223835': 'FiO2',
    '220739': 'GCS',
    '220045': 'Heart Rate',
    '220210': 'Respiratory Rate',
    '220277': 'SpO2',
    '223761': 'Temperature',
    '226559': 'Urine Output'
}

# Print info about these vital signs
print("\nVital Signs Information:")
for itemid, name in vital_items.items():
    count = chartevents[chartevents['itemid'] == int(itemid)].shape[0]
    print(f"{name} (ItemID {itemid}): {count} measurements")
    
    if count > 0:
        # Show example values
        examples = chartevents[chartevents['itemid'] == int(itemid)]['valuenum'].head(3)
        print(f"Example values: {examples.tolist()}\n")