from src.preprocessing.atc_mapper import ATCMapper, map_to_atc_codes
from collections import defaultdict
from pathlib import Path
from sklearn.impute import KNNImputer
from tqdm import tqdm
import json
import logging
import numpy as np
import os
import pandas as pd

import time

def load_and_preprocess_mimic(base_path):
    """Enhanced preprocessing with ATC mapping."""
    # Setup logging
    setup_logging()
    
    # Load core tables
    logging.info("Loading core tables...")
    patients = pd.read_csv(f'{base_path}/PATIENTS.csv', parse_dates=['dob'])
    admissions = pd.read_csv(f'{base_path}/ADMISSIONS.csv', parse_dates=['admittime'])
    
    # Load and process diagnoses
    logging.info("Loading diagnoses...")
    diagnoses = pd.read_csv(f'{base_path}/DIAGNOSES_ICD.csv')
    diagnoses_processed, top_diagnoses = get_top_k_diagnoses(diagnoses)
    
    # Load and process medications with ATC mapping
    logging.info("Loading and processing prescriptions...")
    prescriptions = pd.read_csv(f'{base_path}/PRESCRIPTIONS.csv')
    prescriptions_processed, top_medications = get_top_k_medications(
        prescriptions,
        atc_csv_path='atc_classes.csv'
    )
    
    # Load time series data
    print("Loading time series data...")
    chartevents = pd.read_csv(
        f'{base_path}/CHARTEVENTS.csv',
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        parse_dates=['charttime']
    )
    
    labevents = pd.read_csv(
        f'{base_path}/LABEVENTS.csv',
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        parse_dates=['charttime']
    )
    
    outputevents = pd.read_csv(
        f'{base_path}/OUTPUTEVENTS.csv',
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'value'],
        parse_dates=['charttime']
    )
    
    # Process time series
    print("Processing time series data...")
    timeseries = process_timeseries_enhanced(chartevents, labevents, outputevents)
    
    # Get weight and height
    print("Extracting weight and height...")
    weight_height = extract_weight_height(chartevents)
    
    # Get demographics
    print("Processing demographics...")
    demographics = process_demographics(patients, admissions, weight_height)
    
    return {
        'demographics': demographics,
        'diagnoses': diagnoses_processed,
        'prescriptions': prescriptions_processed,
        'timeseries': timeseries,
        'top_diagnoses': top_diagnoses,
        'top_medications': top_medications
    }

def process_diagnoses(diagnoses):
    """Extract top 2000 diseases by frequency."""
    # Count diagnosis frequencies
    diagnosis_counts = diagnoses['icd9_code'].value_counts()
    
    # Get top 2000 diagnoses
    top_diagnoses = diagnosis_counts.head(2000).index.tolist()
    
    # Filter diagnoses to only include top 2000
    diagnoses_processed = diagnoses[diagnoses['icd9_code'].isin(top_diagnoses)]
    
    return diagnoses_processed

def extract_weight_height(chartevents):
    """Extract weight and height measurements from CHARTEVENTS."""
    # Define relevant ITEMID values
    weight_ids = [226512, 226531]  # Admission Weight and Daily Weight
    height_ids = [226707, 226730]  # Height (inches) and Height (cm)
    
    # Extract weight measurements
    weights = chartevents[chartevents['itemid'].isin(weight_ids)].copy()
    weights = weights.sort_values('charttime').groupby(['subject_id', 'hadm_id'])['valuenum'].first().reset_index()
    weights = weights.rename(columns={'valuenum': 'weight'})
    
    # Extract height measurements
    heights = chartevents[chartevents['itemid'].isin(height_ids)].copy()
    heights = heights.sort_values('charttime').groupby(['subject_id', 'hadm_id'])['valuenum'].first().reset_index()
    heights = heights.rename(columns={'valuenum': 'height'})
    
    # Merge weight and height
    weight_height = pd.merge(weights, heights, on=['subject_id', 'hadm_id'], how='outer')
    
    return weight_height

def process_demographics(patients, admissions, weight_height):
    """Process demographic data and remove minors."""
    # Merge demographics
    demographics = pd.merge(patients, admissions, on='subject_id')
    
    # Add weight and height information
    demographics = pd.merge(demographics, weight_height, on=['subject_id', 'hadm_id'], how='left')
    
    # Remove rows with missing dates
    demographics = demographics.dropna(subset=['dob', 'admittime'])
    
    # Calculate age safely
    demographics['age'] = demographics.apply(
        lambda x: calculate_age(x['dob'], x['admittime']),
        axis=1
    )
    
    # Remove minors
    demographics = demographics[demographics['age'] >= 18]
    
    # Select relevant demographic features
    demographic_features = [
        'subject_id', 'hadm_id', 'gender', 'age', 'weight', 'height',
        'religion', 'language', 'marital_status', 'ethnicity'
    ]
    
    # Only keep columns that exist
    existing_features = [col for col in demographic_features if col in demographics.columns]
    demographics = demographics[existing_features]
    
    # Impute missing values
    numeric_columns = demographics.select_dtypes(include=['number']).columns
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = pd.DataFrame(
        imputer.fit_transform(demographics[numeric_columns]),
        columns=numeric_columns
    )
    demographics.update(imputed_data)
    
    return demographics

def add_data_quality_checks(demographics, timeseries, diagnoses, medications):
    """
    Add additional quality checks for the full dataset processing.
    """
    print("\nDetailed Data Quality Report:")
    
    # Check coverage of physiological parameters
    param_coverage = (
        timeseries.groupby('itemid')
        .agg({
            'hadm_id': 'nunique',
            'valuenum': ['count', 'mean', 'std']
        })
        .sort_values(('hadm_id', 'nunique'), ascending=False)
    )
    
    # Check temporal distribution
    temporal_dist = (
        timeseries.groupby('hadm_id')
        .agg({
            'charttime': ['min', 'max', 'count'],
            'itemid': 'nunique'
        })
    )
    
    # Check diagnosis and medication distributions
    diag_per_admission = diagnoses.groupby('hadm_id')['icd9_code'].nunique()
    med_per_admission = medications.groupby('hadm_id')['drug'].nunique()
    
    # Generate report
    report = {
        'Parameter Coverage': param_coverage,
        'Temporal Distribution': temporal_dist,
        'Diagnoses per Admission': diag_per_admission.describe(),
        'Medications per Admission': med_per_admission.describe()
    }
    
    return report

def add_split_validation(train, val, test):
    """
    Validate that the splits maintain proper proportions and characteristics.
    """
    splits = {'Train': train, 'Validation': val, 'Test': test}
    
    print("\nSplit Validation:")
    for name, split in splits.items():
        patients = len(split['demographics']['subject_id'].unique())
        admissions = len(split['demographics'])
        diagnoses = len(split['diagnoses'])
        medications = len(split['medications'])
        measurements = len(split['timeseries'])
        
        print(f"\n{name} Split:")
        print(f"Patients: {patients}")
        print(f"Admissions: {admissions}")
        print(f"Diagnoses: {diagnoses}")
        print(f"Medications: {medications}")
        print(f"Measurements: {measurements}")
        print(f"Measurements per admission: {measurements/admissions:.2f}")

def process_timeseries_enhanced(chartevents, labevents, outputevents):
    """Enhanced time series processing with both lab values and vital signs."""
    vital_items = {
        220050: 'sbp',         # Systolic BP
        220051: 'dbp',         # Diastolic BP
        223835: 'fio2',        # FiO2
        220739: 'gcs',         # GCS
        220045: 'hr',          # Heart Rate
        220210: 'rr',          # Respiratory Rate
        220277: 'spo2',        # SpO2
        223761: 'temp',        # Temperature
        50931: 'glucose',      # Blood Glucose (from labs)
    }
    
    print("\nProcessing time series data:")
    print(f"Total items to extract: {len(vital_items)}")
    
    # Combine relevant data from both sources
    vital_data = chartevents[chartevents['itemid'].isin(vital_items.keys())].copy()
    lab_data = labevents[labevents['itemid'].isin(vital_items.keys())].copy()
    output_data = outputevents[outputevents['itemid'].isin(vital_items.keys())].copy()
    
    print(f"Found chart events: {vital_data['itemid'].nunique()} unique items")
    print(f"Found lab events: {lab_data['itemid'].nunique()} unique items")
    print(f"Found output events: {output_data['itemid'].nunique()} unique items")
    
    # Combine all data
    all_data = pd.concat([vital_data, lab_data, output_data])
    print(f"Total events after combining: {len(all_data)}")
    
    # Create pivot table with mean values for each measurement
    timeseries = (all_data.pivot_table(
        values='valuenum',
        index=['subject_id', 'hadm_id', 'charttime'],
        columns='itemid',
        aggfunc='mean'
    ).reset_index())
    
    # Rename columns to meaningful names
    timeseries = timeseries.rename(columns=vital_items)
    
    print(f"Final timeseries shape: {timeseries.shape}")
    print("Available columns:", timeseries.columns.tolist())
    
    return timeseries

def process_24h_units(timeseries):
    """Process time series data into 24-hour units."""
    # Sort by time
    timeseries = timeseries.sort_values(['subject_id', 'hadm_id', 'charttime'])
    
    # Set time index
    timeseries.set_index('charttime', inplace=True)
    
    # Resample to 24H periods and take mean
    resampled = []
    for (subject, hadm, item), group in timeseries.groupby(['subject_id', 'hadm_id', 'itemid']):
        # Resample this group
        group_resampled = group['valuenum'].resample('24H').mean().reset_index()
        group_resampled['subject_id'] = subject
        group_resampled['hadm_id'] = hadm
        group_resampled['itemid'] = item
        resampled.append(group_resampled)
    
    # Combine resampled data
    timeseries_resampled = pd.concat(resampled, ignore_index=True)
    
    # Remove admissions with too many missing values
    missing_counts = (
        timeseries_resampled
        .groupby(['subject_id', 'hadm_id'])
        .apply(lambda x: x['valuenum'].isna().sum())
    )
    valid_admissions = missing_counts[missing_counts <= 10].index
    
    timeseries_resampled = timeseries_resampled[
        timeseries_resampled.set_index(['subject_id', 'hadm_id']).index.isin(valid_admissions)
    ]
    
    return timeseries_resampled.reset_index(drop=True)

def validate_timeseries_data(chartevents, labevents, outputevents):
    """Validate time series data before processing."""
    print("\nData Validation:")
    
    # Check for required columns
    required_columns = ['subject_id', 'hadm_id', 'itemid', 'charttime']
    for df, name in [(chartevents, 'chartevents'), 
                     (labevents, 'labevents'), 
                     (outputevents, 'outputevents')]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name}: {missing_cols}")
    
    # Check for null values in key columns
    for df, name in [(chartevents, 'chartevents'),
                     (labevents, 'labevents'),
                     (outputevents, 'outputevents')]:
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            print(f"\nNull values in {name}:")
            print(null_counts[null_counts > 0])
            print(f"Total rows in {name}: {len(df)}")
            print(f"Rows with complete data: {len(df.dropna(subset=required_columns))}")

def get_valid_admissions(demographics, timeseries, diagnoses, medications):
    """
    Get valid admission IDs that have sufficient data across all aspects.
    Returns intersection of valid admissions from all data sources.
    """
    # Get admissions with complete demographics
    demo_admissions = set(demographics['hadm_id'])
    
    # Get admissions with sufficient time series data
    timeseries_admissions = set(timeseries['hadm_id'])
    
    # Get admissions with diagnosis data
    diagnosis_admissions = set(diagnoses['hadm_id'])
    
    # Get admissions with medication data
    medication_admissions = set(medications['hadm_id'])
    
    # Get intersection of all valid admissions
    valid_admissions = demo_admissions.intersection(
        timeseries_admissions,
        diagnosis_admissions,
        medication_admissions
    )
    
    return valid_admissions

def calculate_age(dob, admittime):
    """Safely calculate age between two dates."""
    try:
        return (admittime - dob).days // 365
    except:
        # Handle overflow by converting to datetime if needed
        if isinstance(dob, str):
            dob = pd.to_datetime(dob)
        if isinstance(admittime, str):
            admittime = pd.to_datetime(admittime)
        
        # Calculate difference in years using datetime
        return admittime.year - dob.year - (
            (admittime.month, admittime.day) < (dob.month, dob.day)
        )

def train_val_test_split(data, train_size=0.8, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets based on hadm_id.
    Ensures all data for one admission stays together.
    """
    # Get unique admission IDs
    unique_admissions = data['hadm_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_admissions)
    
    # Calculate split indices
    train_idx = int(len(unique_admissions) * train_size)
    val_idx = int(len(unique_admissions) * (train_size + val_size))
    
    # Split admission IDs
    train_admissions = unique_admissions[:train_idx]
    val_admissions = unique_admissions[train_idx:val_idx]
    test_admissions = unique_admissions[val_idx:]
    
    return train_admissions, val_admissions, test_admissions
        
def get_top_k_diagnoses(diagnoses_df, k=2000):
    """
    Extract top k most frequent diagnoses from ICD-9 codes.
    
    Args:
        diagnoses_df (pd.DataFrame): The DIAGNOSES_ICD table
        k (int): Number of top diagnoses to keep (default 2000)
    
    Returns:
        tuple: (processed diagnoses df, set of top k diagnoses)
    """
    print(f"\nProcessing top {k} diagnoses...")
    
    # Count diagnosis frequencies
    diagnosis_counts = diagnoses_df['icd9_code'].value_counts()
    print(f"Total unique diagnoses: {len(diagnosis_counts)}")
    
    # Get top k diagnoses
    top_diagnoses = set(diagnosis_counts.head(k).index)
    print(f"Coverage of top {k} diagnoses: {(diagnosis_counts.head(k).sum() / diagnosis_counts.sum()) * 100:.1f}%")
    
    # Filter diagnoses to only include top k
    diagnoses_processed = diagnoses_df[diagnoses_df['icd9_code'].isin(top_diagnoses)].copy()
    
    return diagnoses_processed, top_diagnoses

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_top_k_medications(prescriptions_df, k=1000, atc_csv_path='atc_codes.csv'):
    """
    Extract top k most frequent medications and map to ATC codes.
    
    Args:
        prescriptions_df (pd.DataFrame): The PRESCRIPTIONS table
        k (int): Number of top medications to keep (default 1000)
        atc_csv_path (str): Path to ATC codes CSV file
    
    Returns:
        tuple: (processed prescriptions df with ATC codes, set of top medications)
    """
    print(f"\nProcessing top {k} medications...")
    
    # Count medication frequencies
    med_counts = prescriptions_df['drug'].value_counts()
    print(f"Total unique medications: {len(med_counts)}")
    
    # Get top k medications
    top_medications = set(med_counts.head(k).index)
    print(f"Coverage of top {k} medications: {(med_counts.head(k).sum() / med_counts.sum()) * 100:.1f}%")
    
    # Filter prescriptions to only include top k
    prescriptions_processed = prescriptions_df[prescriptions_df['drug'].isin(top_medications)].copy()
    
    # Map to ATC codes using CSV file
    prescriptions_processed = map_to_atc_codes(prescriptions_processed, r'data/atc_classes.csv')
    
    return prescriptions_processed, set(prescriptions_processed['drug'].unique())

def save_processed_data(data_dict, prefix, output_dir='data/processed'):
    """
    Save all components of the processed data with consistent naming.
    
    Args:
        data_dict (dict): Dictionary of dataframes to save
        prefix (str): Prefix for filenames (train/val/test)
        output_dir (str): Directory to save files to
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for key, df in data_dict.items():
        filename = output_path / f"{prefix}_{key}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")

if __name__ == "__main__":
    # Set paths
    base_path = Path('data/mimic-iii-clinical-database-demo-1.4')
    processed_path = Path('data/processed')
    
    # Create directories if they don't exist
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Process the data
    data = load_and_preprocess_mimic(base_path)
    
    # Create cohort dictionary for easier handling
    cohort = {
        'demographics': data['demographics'],
        'timeseries': data['timeseries'],
        'diagnoses': data['diagnoses'],
        'prescriptions': data['prescriptions']
    }
    
    # Split data into train/val/test
    train_admissions, val_admissions, test_admissions = train_val_test_split(data['demographics'])
    
    # Create dictionary for splits
    splits = {
        'train': train_admissions,
        'val': val_admissions,
        'test': test_admissions
    }
    
    # Split and save each portion to processed directory
    for split_name, split_admissions in splits.items():
        split_data = {
            key: df[df['hadm_id'].isin(split_admissions)]
            for key, df in cohort.items()
        }
        save_processed_data(split_data, split_name, processed_path)
        
    # Save additional metadata
    metadata = {
        'top_diagnoses': list(data['top_diagnoses']),
        'top_medications': list(data['top_medications'])
    }
    with open(processed_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f)

    # Print warnings about dataset size
    if data['demographics']['hadm_id'].nunique() < 100:
        print("\nWARNING: Using demo dataset - results will not match paper")
        logging.warning("Using demo dataset - results will not match paper")

    if len(val_admissions) < 1000:
        print("\nWARNING: Small number of valid admissions")
        logging.warning("Small number of valid admissions")
    
    # Print summary statistics
    print("\nDataset Statistics:")
    stats = {
        "Total number of unique patients": data['demographics']['subject_id'].nunique(),
        "Total number of hospital admissions": data['demographics']['hadm_id'].nunique(),
        "Number of unique diagnoses": data['diagnoses']['icd9_code'].nunique(),
        "Number of unique medications": data['prescriptions']['drug'].nunique(),
        "Training admissions": len(train_admissions),
        "Validation admissions": len(val_admissions),
        "Testing admissions": len(test_admissions),
        # Change this line:
        "Unique physiological parameters": len(data['timeseries'].columns) - 3,  # Subtract subject_id, hadm_id, charttime
        "Average measurements per admission": len(data['timeseries']) / data['demographics']['hadm_id'].nunique()
    }
    
    # Save statistics
    with open(processed_path / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Print statistics
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\nDemographic Statistics:")
    age_stats = data['demographics']['age'].describe().to_dict()
    gender_stats = data['demographics']['gender'].value_counts(normalize=True).to_dict()
    
    # Save demographic statistics
    demo_stats = {
        'age_statistics': age_stats,
        'gender_distribution': gender_stats
    }
    with open(processed_path / 'demographic_statistics.json', 'w') as f:
        json.dump(demo_stats, f, indent=4)
    
    # Print demographic statistics
    print(f"Age statistics:\n{pd.Series(age_stats)}")
    print(f"\nGender distribution:\n{pd.Series(gender_stats)}")
