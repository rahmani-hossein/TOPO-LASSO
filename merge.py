"""
MERGE RANKED_DF WITH FEATURES AND VOLUME
=========================================
Combines your ranked_df with features from df_clean and volume from df_volume_input
"""

import pandas as pd
import numpy as np

# ============================================================================
# COMPLETE WORKFLOW - COPY THIS CELL
# ============================================================================

# You have:
# - ranked_df: mtgnum, sens_mean, sens_abs, rank, segment
# - df_clean: all the feature columns (200 rows per mortgage)
# - df_volume_input: volume/amount data (separate dataframe)

# ----------------------------------------------------------------------------
# STEP 1: Get one row per mortgage from df_clean (take first row)
# ----------------------------------------------------------------------------

mtg_features = df_clean.groupby('mtgnum').first().reset_index()

print(f"✓ Extracted features: {len(mtg_features):,} mortgages")


# ----------------------------------------------------------------------------
# STEP 2: Merge with your ranked_df to add segments
# ----------------------------------------------------------------------------

mtg_features_with_segments = mtg_features.merge(
    ranked_df[['mtgnum', 'sens_mean', 'sens_abs', 'rank', 'segment']], 
    on='mtgnum', 
    how='inner'
)

print(f"✓ Merged with segments: {len(mtg_features_with_segments):,} mortgages")


# ----------------------------------------------------------------------------
# STEP 3: Add volume from df_volume_input (filter to valid mortgages first)
# ----------------------------------------------------------------------------

# Filter volume df to only mortgages in ranked_df
valid_mtgs = ranked_df['mtgnum'].unique()
df_volume_filtered = df_volume_input[df_volume_input['mtgnum'].isin(valid_mtgs)]

# Get one value per mortgage (take first row if multiple)
volumes = df_volume_filtered.groupby('mtgnum').first()[['amount']].reset_index()
volumes = volumes.rename(columns={'amount': 'volume'})

# Merge with features
mtg_features_complete = mtg_features_with_segments.merge(
    volumes, 
    on='mtgnum', 
    how='left'
)

print(f"✓ Added volume: {mtg_features_complete['volume'].notna().sum():,} mortgages have volume")


# ----------------------------------------------------------------------------
# STEP 4: Create segment table
# ----------------------------------------------------------------------------

from segment_table_simple import create_segment_table_simple

segment_table = create_segment_table_simple(
    mtg_features=mtg_features_complete,   # Now has: segments + features + volume
    df_volume=None,                        # Already merged, so set to None
    mtg_col='mtgnum'
)

# Display
print("\n" + "="*80)
print("SEGMENT TABLE:")
print("="*80)
print(segment_table)

# Copy to clipboard
segment_table.to_clipboard(index=False)
print("\n✓ Copied to clipboard! Paste into Excel/PowerPoint")

# Save
segment_table.to_csv('/mnt/user-data/outputs/segment_table.csv', index=False)
print("✓ Saved to segment_table.csv")


# ============================================================================
# OPTIONAL: Check the merged dataframe
# ============================================================================

print("\n" + "="*80)
print("MERGED DATAFRAME (mtg_features_complete):")
print("="*80)
print(f"Shape: {mtg_features_complete.shape}")
print(f"Columns: {mtg_features_complete.columns.tolist()}")
print(f"\nSample rows:")
print(mtg_features_complete.head(3))

# Save the complete merged data if you want
mtg_features_complete.to_csv('/mnt/user-data/outputs/mortgages_complete.csv', index=False)
print("\n✓ Saved complete data to mortgages_complete.csv")