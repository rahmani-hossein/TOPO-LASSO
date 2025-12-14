"""
CREATE SEGMENT DISTRIBUTION TABLE - VOLUME ALREADY MERGED
==========================================================
Simple function that returns dataframe only (no plots)
"""

import pandas as pd
import numpy as np

def create_segment_table(mtg_features_complete):
    """
    Create segment distribution table from mortgage data with volume already merged
    
    Parameters:
    -----------
    mtg_features_complete : pd.DataFrame
        Must have: 'mtgnum', 'segment', 'volume' (already merged)
        Optional: 'beacon', 'insured', 'PDBA_underlying_channel', 'TDS_revised', 
                  'LTV_current', 'closed_balance_x', 'remaining_amort_months_x', etc.
    
    Returns:
    --------
    segment_table : pd.DataFrame
        Table with statistics by segment
    """
    
    results = []
    
    for seg in sorted(mtg_features_complete['segment'].unique()):
        seg_data = mtg_features_complete[mtg_features_complete['segment'] == seg]
        
        row = {
            'Segment': int(seg),
            'Count': len(seg_data)
        }
        
        # Total Volume & Avg Amount (from merged 'volume' column)
        if 'volume' in seg_data.columns:
            row['Total Volume'] = seg_data['volume'].fillna(0).sum()
            row['Avg Amount'] = seg_data['volume'].mean()
        
        # Avg Beacon
        if 'beacon' in seg_data.columns:
            row['Avg Beacon'] = seg_data['beacon'].mean()
        
        # Insured %
        if 'insured' in seg_data.columns:
            if seg_data['insured'].dtype == 'object':
                pct = (seg_data['insured'].str.upper() == 'YES').sum() / len(seg_data) * 100
            else:
                pct = (seg_data['insured'] == 1).sum() / len(seg_data) * 100
            row['Insured %'] = pct
        elif 'insurance_flag' in seg_data.columns:
            if seg_data['insurance_flag'].dtype == 'object':
                pct = (seg_data['insurance_flag'].str.upper() == 'YES').sum() / len(seg_data) * 100
            else:
                pct = (seg_data['insurance_flag'] == 1).sum() / len(seg_data) * 100
            row['Insured %'] = pct
        
        # Broker %
        if 'PDBA_underlying_channel' in seg_data.columns:
            pct = (seg_data['PDBA_underlying_channel'].str.upper() == 'BROKER').sum() / len(seg_data) * 100
            row['Broker %'] = pct
        
        # Avg TDS
        if 'TDS_revised' in seg_data.columns:
            row['Avg TDS'] = seg_data['TDS_revised'].mean()
        
        # Avg LTV
        if 'LTV_current' in seg_data.columns:
            row['Avg LTV'] = seg_data['LTV_current'].mean()
        
        # Avg Balance
        if 'closed_balance_x' in seg_data.columns:
            row['Avg Balance'] = seg_data['closed_balance_x'].mean()
        elif 'closed_balance_y' in seg_data.columns:
            row['Avg Balance'] = seg_data['closed_balance_y'].mean()
        
        # Avg Amortization (in months)
        if 'remaining_amort_months_x' in seg_data.columns:
            row['Avg AM (months)'] = seg_data['remaining_amort_months_x'].mean()
        
        results.append(row)
    
    # Create DataFrame
    segment_table = pd.DataFrame(results)
    
    # Add segment labels
    segment_table['Segment'] = segment_table['Segment'].apply(
        lambda x: f"{x} (Lowest)" if x == 1 else (
            f"{x} (Highest)" if x == segment_table['Segment'].max() else str(x)
        )
    )
    
    # Round numbers for cleaner display
    for col in segment_table.columns:
        if segment_table[col].dtype in ['float64', 'float32']:
            if '%' in col:
                segment_table[col] = segment_table[col].round(1)
            elif 'Volume' in col or 'Amount' in col or 'Balance' in col:
                segment_table[col] = segment_table[col].round(0)
            else:
                segment_table[col] = segment_table[col].round(2)
    
    return segment_table


# ============================================================================
# USAGE
# ============================================================================

"""
# After merging everything into mtg_features_complete:

segment_table = create_segment_table(mtg_features_complete)

# View it
print(segment_table)

# Copy to clipboard
segment_table.to_clipboard(index=False)

# Save to CSV
segment_table.to_csv('segment_table.csv', index=False)
"""