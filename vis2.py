"""
COPY THIS ENTIRE CELL TO YOUR NOTEBOOK
=======================================
This will clean your data and prepare df_clean for analysis
"""

import pandas as pd
import numpy as np

# ============================================================================
# DATA CLEANING - Remove mortgages without exactly 200 records
# ============================================================================

def clean_mortgage_data(df, mtg_col='mtgnum', expected_records=200):
    """
    Remove mortgages that don't have exactly expected_records rows.
    Returns cleaned dataframe ready for analysis.
    """
    
    print("="*80)
    print("DATA CLEANING")
    print("="*80)
    
    # Count records per mortgage
    records_per_mtg = df.groupby(mtg_col).size()
    
    print(f"\nOriginal data:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique mortgages: {df[mtg_col].nunique():,}")
    
    print(f"\nRecord counts per mortgage:")
    print(records_per_mtg.value_counts().sort_index())
    
    # Keep only mortgages with exactly expected_records
    valid_mtgs = records_per_mtg[records_per_mtg == expected_records].index
    df_clean = df[df[mtg_col].isin(valid_mtgs)].copy()
    
    # Calculate exclusions
    n_excluded = df[mtg_col].nunique() - len(valid_mtgs)
    pct_kept = len(valid_mtgs) / df[mtg_col].nunique() * 100
    
    print(f"\n" + "-"*80)
    print(f"RESULTS:")
    print(f"  ✓ Kept: {len(valid_mtgs):,} mortgages ({pct_kept:.1f}%)")
    print(f"  ✗ Excluded: {n_excluded:,} mortgages ({100-pct_kept:.1f}%)")
    print(f"\n  Cleaned data: {len(df_clean):,} rows")
    print("="*80)
    
    # Verify all mortgages have exactly expected_records
    final_counts = df_clean.groupby(mtg_col).size()
    assert (final_counts == expected_records).all(), "ERROR: Some mortgages don't have exactly 200 records!"
    print(f"✅ Validation passed: All mortgages have exactly {expected_records} records\n")
    
    return df_clean

# Clean the data
df_clean = clean_mortgage_data(df_x, mtg_col='mtgnum', expected_records=200)

# Now use df_clean for analysis (not df_x)
print("Ready! Use df_clean for your analysis")