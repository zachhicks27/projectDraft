from pyhealth.medcode import CrossMap, RxNorm, ATC
import pandas as pd
import json
import os
from tqdm import tqdm
import time
import logging
from pathlib import Path

class ATCMapper:
    """
    Class to handle ATC code mapping with caching functionality.
    """
    def __init__(self, cache_file='atc_mapping_cache.json'):
        """
        Initialize ATC mapper with cache file.
        
        Args:
            cache_file (str): Path to cache file
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.rxnorm = RxNorm()
        self.atc = ATC()
        
    def _load_cache(self):
        """Load existing cache if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save current cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
            
    def get_atc_code(self, drug):
        """
        Get ATC code for drug, using cache if available.
        
        Args:
            drug (str): Drug name
            
        Returns:
            str: ATC code or None if mapping fails
        """
        # Check cache first
        if drug in self.cache:
            return self.cache[drug]
        
        try:
            # Use a mapping dictionary for common drugs
            common_drugs = {
                'Acetaminophen': 'N02B',
                'Heparin': 'B01A',
                'Insulin': 'A10A',
                'Morphine': 'N02A',
                'Aspirin': 'B01A',
                'Furosemide': 'C03C',
                'Warfarin': 'B01A',
                'Metoprolol': 'C07A',
                'Potassium Chloride': 'A12B',
                'Magnesium Sulfate': 'A12C',
                # Add more common mappings as needed
            }
            
            # Try direct mapping first
            if drug in common_drugs:
                atc_code = common_drugs[drug]
            else:
                # Try to get RxNorm concept first
                rxnorm_concepts = self.rxnorm.lookup(drug)
                if rxnorm_concepts:
                    mapping = CrossMap.load("RxNorm", "ATC")
                    atc_codes = mapping.map(drug, target_kwargs={"level": 3})
                    if atc_codes and len(atc_codes) > 0:
                        atc_code = atc_codes[0]
                    else:
                        return None
                else:
                    return None
                
            self.cache[drug] = atc_code
            self._save_cache()
            return atc_code
            
        except Exception as e:
            logging.warning(f"Failed to get ATC code for {drug}: {str(e)}")
            return None

def map_to_atc_codes(prescriptions_df, cache_file='atc_mapping_cache.json'):
    """
    Map medications to ATC level 3 codes using PyHealth with caching.
    
    Args:
        prescriptions_df (pd.DataFrame): Processed prescriptions
        cache_file (str): Path to cache file
        
    Returns:
        pd.DataFrame: Prescriptions with added ATC codes
    """
    # Initialize ATC mapper
    mapper = ATCMapper(cache_file)
    
    # Get unique drugs
    unique_drugs = prescriptions_df['drug'].unique()
    logging.info(f"Mapping {len(unique_drugs)} unique drugs to ATC codes...")
    
    # Create mapping dictionary
    atc_mapping = {}
    failed_drugs = []
    
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
    
    if len(prescriptions_df) == 0:
        logging.warning("No prescriptions remained after ATC mapping!")
        # Keep the original data but mark unmapped
        prescriptions_df = prescriptions_df.copy()
        prescriptions_df['atc_code'] = 'UNKNOWN'
    
    # Print statistics
    print("\nATC Mapping Statistics:")
    print(f"Total unique drugs: {len(unique_drugs)}")
    print(f"Successfully mapped: {len(atc_mapping)}")
    print(f"Failed to map: {len(failed_drugs)}")
    print(f"Removed {removed_rows} rows with unmapped drugs")
    print(f"Unique ATC codes: {prescriptions_df['atc_code'].nunique()}")
    print(f"Final number of prescriptions: {len(prescriptions_df)}")
    
    # Save failed drugs for reference
    if failed_drugs:
        pd.Series(failed_drugs, name='drug').to_csv('unmapped_drugs.csv', index=False)
        print(f"\nList of unmapped drugs saved to 'unmapped_drugs.csv'")
    
    return prescriptions_df
