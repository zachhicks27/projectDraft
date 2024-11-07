import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from collections import Counter

def load_and_preprocess_mimic(base_path):
    """
    Enhanced preprocessing following the paper's specifications.
    """
    # Load core tables
    patients = pd.read_csv(f'{base_path}/PATIENTS.csv', parse_dates=['dob'])
    admissions = pd.read_csv(f'{base_path}/ADMISSIONS.csv', parse_dates=['admittime'])
    
    # Load diagnosis and medication data
    diagnoses = pd.read_csv(f'{base_path}/DIAGNOSES_ICD.csv')
    prescriptions = pd.read_csv(f'{base_path}/PRESCRIPTIONS.csv')
    
    # Load time series data
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

    # Process diagnoses and medications
    diagnoses_processed = process_diagnoses(diagnoses)
    medications_processed = process_medications(prescriptions)
    
    # Get weight and height
    weight_height = extract_weight_height(chartevents)
    
    # Process demographics
    demographics = process_demographics(patients, admissions, weight_height)
    
    # Process time series data with enhanced parameters
    timeseries = process_timeseries_enhanced(chartevents, labevents, outputevents)
    
    # Filter admissions based on complete data
    valid_admissions = get_valid_admissions(demographics, timeseries, diagnoses_processed, medications_processed)
    
    # Filter all dataframes to valid admissions
    demographics = demographics[demographics['hadm_id'].isin(valid_admissions)]
    timeseries = timeseries[timeseries['hadm_id'].isin(valid_admissions)]
    diagnoses_processed = diagnoses_processed[diagnoses_processed['hadm_id'].isin(valid_admissions)]
    medications_processed = medications_processed[medications_processed['hadm_id'].isin(valid_admissions)]
    
    return demographics, timeseries, diagnoses_processed, medications_processed

def process_medications(prescriptions):
    """Extract top 1000 medications and map to ATC codes."""
    # Count medication frequencies
    med_counts = prescriptions['drug'].value_counts()
    
    # Get top 1000 medications
    top_meds = med_counts.head(1000).index.tolist()
    
    # Filter prescriptions to only include top 1000
    medications_processed = prescriptions[prescriptions['drug'].isin(top_meds)]
    
    # TODO: Add ATC mapping when you have access to the mapping tool
    # The paper mentions using a public tool for ATC mapping
    # medications_processed['atc_code'] = medications_processed['drug'].map(atc_mapping)
    
    return medications_processed

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
    """Enhanced time series processing with additional parameters."""
    print("\nInitial data shapes:")
    print(f"Chartevents: {chartevents.shape}")
    print(f"Labevents: {labevents.shape}")
    print(f"Outputevents: {outputevents.shape}")
    
    # Filter out null hadm_id values first
    labevents = labevents.dropna(subset=['hadm_id'])
    
    print("\nAfter removing null hadm_id values:")
    print(f"Labevents: {labevents.shape}")
    
    # Define relevant physiological parameters (enhanced from paper)
    vital_signs_ids = {
        'blood_pressure_diastolic': [220051, 220180],
        'blood_pressure_systolic': [220050, 220179],
        'heart_rate': [220045],
        'respiratory_rate': [220210],
        'temperature': [223761, 223762],
        'o2_saturation': [220277],
        'glucose': [220621, 225664],
        'gcs': [220739, 223900, 223901],  # Glasgow Coma Scale
        'fio2': [223835],  # Fraction of inspiration O2
        'ph': [220274, 220734],  # pH
        'urine_output': [226559]  # Urine output
    }
    
    # Filter chartevents to only include relevant vital signs
    chartevents_filtered = chartevents[chartevents['itemid'].isin(
        [id for ids in vital_signs_ids.values() for id in ids]
    )]
    
    print(f"\nChartevents after filtering for relevant vital signs: {chartevents_filtered.shape}")
    
    # Combine all time series data
    timeseries = pd.concat([
        chartevents_filtered,
        labevents,
        outputevents.rename(columns={'value': 'valuenum'})
    ])
    
    print(f"\nCombined time series shape: {timeseries.shape}")
    
    # Sort by time
    timeseries.sort_values(['subject_id', 'hadm_id', 'charttime'], inplace=True)
    
    # Group by subject, admission, and itemid before resampling
    timeseries.set_index('charttime', inplace=True)
    
    # Perform resampling with proper handling of the multi-index
    timeseries_resampled = []
    
    for (subject, hadm, item), group in timeseries.groupby(['subject_id', 'hadm_id', 'itemid']):
        # Resample each group
        resampled = group['valuenum'].resample('24H').mean().reset_index()
        resampled['subject_id'] = subject
        resampled['hadm_id'] = hadm
        resampled['itemid'] = item
        timeseries_resampled.append(resampled)
    
    # Combine all resampled data
    timeseries_resampled = pd.concat(timeseries_resampled, ignore_index=True)
    
    # Count missing values per admission
    missing_counts = (
        timeseries_resampled
        .groupby(['subject_id', 'hadm_id'])
        .apply(lambda x: x['valuenum'].isna().sum())
    )
    
    # Get valid admissions (those with 10 or fewer missing values)
    valid_admissions = missing_counts[missing_counts <= 10].index
    
    # Filter to valid admissions
    timeseries_resampled = timeseries_resampled[
        timeseries_resampled.set_index(['subject_id', 'hadm_id']).index.isin(valid_admissions)
    ]
    
    # Ensure columns are in a consistent order
    timeseries_resampled = timeseries_resampled[[
        'subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum'
    ]]
    
    # Print summary statistics
    print("\nFinal Processing Statistics:")
    print(f"Number of admissions: {timeseries_resampled['hadm_id'].nunique()}")
    print(f"Number of unique patients: {timeseries_resampled['subject_id'].nunique()}")
    print(f"Number of measurements: {len(timeseries_resampled)}")
    print(f"Number of unique parameters: {timeseries_resampled['itemid'].nunique()}")
    
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

def save_processed_data(data_dict, prefix):
    """
    Save all components of the processed data with consistent naming.
    """
    for key, df in data_dict.items():
        filename = f"{prefix}_{key}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")

if __name__ == "__main__":
    
    # Set path to your MIMIC-III database
    base_path = 'mimic-iii-clinical-database-demo-1.4'
    
    # Process the data
    demographics, timeseries, diagnoses, medications = load_and_preprocess_mimic(base_path)
    
    # Create cohort dictionary for easier handling
    cohort = {
        'demographics': demographics,
        'timeseries': timeseries,
        'diagnoses': diagnoses,
        'medications': medications
    }
    
    # Split data into train/val/test
    train_admissions, val_admissions, test_admissions = train_val_test_split(demographics)
    
    # Create train/val/test datasets
    splits = {
        'train': train_admissions,
        'val': val_admissions,
        'test': test_admissions
    }
    
    # Split and save each portion
    for split_name, split_admissions in splits.items():
        split_data = {
            key: df[df['hadm_id'].isin(split_admissions)]
            for key, df in cohort.items()
        }
        save_processed_data(split_data, split_name)

    # In main:
    if demographics['hadm_id'].nunique() < 100:
        print("\nWARNING: Using demo dataset - results will not match paper")

    # Add minimum admission threshold
    if len(val_admissions) < 1000:
        print("\nWARNING: Small number of valid admissions")
    
    # Print summary statistics
    print("\nDataset Statistics:")
    print(f"Total number of unique patients: {demographics['subject_id'].nunique()}")
    print(f"Total number of hospital admissions: {demographics['hadm_id'].nunique()}")
    print(f"Number of unique diagnoses: {diagnoses['icd9_code'].nunique()}")
    print(f"Number of unique medications: {medications['drug'].nunique()}")
    
    print("\nSplit Statistics:")
    print(f"Training admissions: {len(train_admissions)}")
    print(f"Validation admissions: {len(val_admissions)}")
    print(f"Testing admissions: {len(test_admissions)}")
    
    print("\nDemographic Statistics:")
    print(f"Age statistics:\n{demographics['age'].describe()}")
    print(f"\nGender distribution:\n{demographics['gender'].value_counts(normalize=True)}")
    
    print("\nTime Series Statistics:")
    print(f"Unique physiological parameters: {timeseries['itemid'].nunique()}")
    print(f"Average measurements per admission: {len(timeseries) / demographics['hadm_id'].nunique():.2f}")