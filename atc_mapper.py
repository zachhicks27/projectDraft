import pandas as pd
import re
from pathlib import Path
import logging
from tqdm import tqdm

class ATCMapper:
    """
    ATC mapper using direct CSV mapping file.
    """
    def __init__(self, atc_csv_path):
        """
        Initialize ATC mapper with CSV file.
        
        Args:
            atc_csv_path (str): Path to ATC code CSV file
        """
        # Load ATC codes
        self.atc_df = pd.read_csv(atc_csv_path)
        
        # Create dictionary of standardized names to ATC codes
        self.atc_dict = {}
        for _, row in self.atc_df.iterrows():
            if pd.notna(row['atc_name']):
                std_name = self.standardize_name(row['atc_name'])
                # Keep only level 3 ATC code (first 4 characters)
                atc_code = row['atc_code'][:4] if len(row['atc_code']) >= 4 else row['atc_code']
                self.atc_dict[std_name] = atc_code
        
        # Common drug name mappings for exact matches
        self.common_drugs = {
            'ACETAMINOPHEN': 'N02B',
            'MORPHINE': 'N02A',
            'HEPARIN': 'B01A',
            'INSULIN': 'A10A',
            'WARFARIN': 'B01A',
            'METOPROLOL': 'C07A',
            'ASPIRIN': 'B01A',
            'FUROSEMIDE': 'C03C',
            'POTASSIUM CHLORIDE': 'A12B',
            'MAGNESIUM SULFATE': 'A12C',
            # Add more as needed
        }
        
        print(f"Loaded {len(self.atc_dict)} ATC codes")

    def standardize_name(self, name):
        """Standardize drug name for matching."""
        if not isinstance(name, str):
            return ''
        
        # Convert to uppercase and remove extra spaces
        name = re.sub(r'\s+', ' ', name.upper().strip())
        
        # Remove common suffixes and prefixes
        name = re.sub(r'\([^)]*\)', '', name)  # Remove parenthetical content
        name = re.sub(r'\d+(\.\d+)?(%|MG|MCG|G|ML|MEQ).*', '', name)  # Remove dosage info
        
        # Remove specific words that might interfere with matching
        remove_words = ['SOLN', 'TAB', 'CAP', 'INJ', 'ORAL', 'LIQUID', 'SUSPENSION']
        for word in remove_words:
            name = re.sub(fr'\b{word}\b', '', name)
        
        return name.strip()

    def get_atc_code(self, drug_name):
        """
        Get ATC code for drug name using multiple matching attempts.
        
        Args:
            drug_name (str): Drug name to look up
            
        Returns:
            str: ATC code or None if no match found
        """
        if not drug_name:
            return None
            
        # Standardize the input name
        std_name = self.standardize_name(drug_name)
        
        # Try exact match in common drugs first
        if std_name in self.common_drugs:
            return self.common_drugs[std_name]
        
        # Try exact match in ATC dictionary
        if std_name in self.atc_dict:
            return self.atc_dict[std_name]
            
        # Try partial matching
        for atc_name, code in self.atc_dict.items():
            if std_name in atc_name or atc_name in std_name:
                return code
        
        return None

def map_to_atc_codes(prescriptions_df, atc_csv_path):
    """
    Map medications to ATC codes using CSV file.
    
    Args:
        prescriptions_df (pd.DataFrame): Prescriptions DataFrame
        atc_csv_path (str): Path to ATC code CSV file
    
    Returns:
        pd.DataFrame: Prescriptions with ATC codes
    """
    # Initialize mapper
    mapper = ATCMapper(atc_csv_path)
    
    # Get unique drugs
    unique_drugs = prescriptions_df['drug'].unique()
    print(f"\nMapping {len(unique_drugs)} unique drugs to ATC codes...")
    
    # Create mapping dictionary
    atc_mapping = {}
    failed_drugs = []
    
    # Process all drugs with progress bar
    for drug in tqdm(unique_drugs):
        atc_code = mapper.get_atc_code(drug)
        if atc_code:
            atc_mapping[drug] = atc_code
        else:
            failed_drugs.append(drug)
    
    # Add ATC codes to DataFrame
    prescriptions_df['atc_code'] = prescriptions_df['drug'].map(atc_mapping)
    
    # Remove rows with unmapped drugs
    original_len = len(prescriptions_df)
    prescriptions_df = prescriptions_df.dropna(subset=['atc_code'])
    removed_rows = original_len - len(prescriptions_df)
    
    # Print statistics
    print("\nATC Mapping Statistics:")
    print(f"Total unique drugs: {len(unique_drugs)}")
    print(f"Successfully mapped: {len(atc_mapping)}")
    print(f"Failed to map: {len(failed_drugs)}")
    print(f"Removed {removed_rows} rows with unmapped drugs")
    print(f"Unique ATC codes: {prescriptions_df['atc_code'].nunique()}")
    print(f"Final number of prescriptions: {len(prescriptions_df)}")
    
    # Save failed drugs for review
    if failed_drugs:
        pd.DataFrame({
            'drug': failed_drugs,
            'standardized_name': [mapper.standardize_name(d) for d in failed_drugs]
        }).to_csv('unmapped_drugs.csv', index=False)
        print(f"\nList of unmapped drugs saved to 'unmapped_drugs.csv'")
    
    return prescriptions_df