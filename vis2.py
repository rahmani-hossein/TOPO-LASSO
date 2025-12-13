"""
SEGMENT DISTRIBUTION TABLE - MATCHING YOUR PRESENTATION FORMAT
===============================================================
Creates the exact table format from your slide
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CREATE SEGMENT DISTRIBUTION TABLE
# ============================================================================

def create_segment_distribution_table(mtg_features, segment_col='segment'):
    """
    Create segment distribution table matching presentation format
    
    Parameters:
    -----------
    mtg_features : pd.DataFrame
        Mortgage-level data with segment assignments
    segment_col : str
        Column name for segment (default: 'segment')
    
    Returns:
    --------
    segment_table : pd.DataFrame
        Table with all statistics by segment
    """
    
    print("\n" + "="*80)
    print("CREATING SEGMENT DISTRIBUTION TABLE")
    print("="*80)
    
    # Initialize results
    results = []
    
    for seg in sorted(mtg_features[segment_col].unique()):
        seg_data = mtg_features[mtg_features[segment_col] == seg].copy()
        
        stats = {
            'Segment': int(seg),
            'Count': len(seg_data),
        }
        
        # ====================================================================
        # TOTAL VOLUME (use closed_balance_x or closed_balance_y)
        # ====================================================================
        if 'closed_balance_x' in seg_data.columns:
            stats['Total_Volume'] = seg_data['closed_balance_x'].sum()
        elif 'closed_balance_y' in seg_data.columns:
            stats['Total_Volume'] = seg_data['closed_balance_y'].sum()
        else:
            stats['Total_Volume'] = 0
        
        # ====================================================================
        # AVERAGE BEACON SCORE
        # ====================================================================
        if 'beacon' in seg_data.columns:
            stats['Avg_Beacon'] = seg_data['beacon'].mean()
        else:
            stats['Avg_Beacon'] = np.nan
        
        # ====================================================================
        # INSURED PERCENTAGE (from 'insured' or 'insurance_flag' column)
        # ====================================================================
        if 'insured' in seg_data.columns:
            # Assume 'insured' is Yes/No or 1/0
            if seg_data['insured'].dtype == 'object':
                insured_count = (seg_data['insured'].str.upper() == 'YES').sum()
            else:
                insured_count = (seg_data['insured'] == 1).sum()
            stats['Insured_Pct'] = (insured_count / len(seg_data) * 100) if len(seg_data) > 0 else 0
        elif 'insurance_flag' in seg_data.columns:
            if seg_data['insurance_flag'].dtype == 'object':
                insured_count = (seg_data['insurance_flag'].str.upper() == 'YES').sum()
            else:
                insured_count = (seg_data['insurance_flag'] == 1).sum()
            stats['Insured_Pct'] = (insured_count / len(seg_data) * 100) if len(seg_data) > 0 else 0
        else:
            stats['Insured_Pct'] = np.nan
        
        # ====================================================================
        # BROKER PERCENTAGE (from 'PDBA_underlying_channel' column)
        # ====================================================================
        if 'PDBA_underlying_channel' in seg_data.columns:
            # Count where channel is 'Broker'
            broker_count = (seg_data['PDBA_underlying_channel'].str.upper() == 'BROKER').sum()
            stats['Broker_Pct'] = (broker_count / len(seg_data) * 100) if len(seg_data) > 0 else 0
        else:
            stats['Broker_Pct'] = np.nan
        
        # ====================================================================
        # AVERAGE TDS (from 'TDS_revised' column)
        # ====================================================================
        if 'TDS_revised' in seg_data.columns:
            stats['Avg_TDS'] = seg_data['TDS_revised'].mean()
        else:
            stats['Avg_TDS'] = np.nan
        
        # ====================================================================
        # AVERAGE LTV (from 'LTV_current' column)
        # ====================================================================
        if 'LTV_current' in seg_data.columns:
            stats['Avg_LTV'] = seg_data['LTV_current'].mean()
        else:
            stats['Avg_LTV'] = np.nan
        
        # ====================================================================
        # AVERAGE BALANCE (from closed_balance_x or closed_balance_y)
        # ====================================================================
        if 'closed_balance_x' in seg_data.columns:
            stats['Avg_Balance'] = seg_data['closed_balance_x'].mean()
        elif 'closed_balance_y' in seg_data.columns:
            stats['Avg_Balance'] = seg_data['closed_balance_y'].mean()
        else:
            stats['Avg_Balance'] = np.nan
        
        # ====================================================================
        # AVERAGE AMORTIZATION MONTHS (if available)
        # ====================================================================
        if 'remaining_amort_months_x' in seg_data.columns:
            stats['Avg_Amort_Months'] = seg_data['remaining_amort_months_x'].mean()
        else:
            stats['Avg_Amort_Months'] = np.nan
        
        # ====================================================================
        # PLACEHOLDER FOR FTR%, AVG MONEY-IN, etc. (add when columns known)
        # ====================================================================
        stats['FTR_Pct'] = np.nan  # Add when column is identified
        stats['Avg_Money_In'] = np.nan  # Add when column is identified
        
        results.append(stats)
    
    # Create DataFrame
    segment_table = pd.DataFrame(results)
    
    # Add labels for lowest/highest segments
    segment_table['Segment_Label'] = segment_table['Segment'].astype(str)
    segment_table.loc[segment_table['Segment'] == 1, 'Segment_Label'] = '1 (Lowest)'
    segment_table.loc[segment_table['Segment'] == segment_table['Segment'].max(), 'Segment_Label'] = \
        f"{int(segment_table['Segment'].max())} (Highest)"
    
    print(f"✓ Created segment table with {len(segment_table)} segments")
    
    return segment_table


# ============================================================================
# DISPLAY TABLE (FORMATTED)
# ============================================================================

def display_segment_table(segment_table):
    """
    Display formatted segment table matching presentation style
    """
    
    print("\n" + "="*80)
    print("SENSITIVITY SEGMENT DISTRIBUTION")
    print("="*80)
    
    # Select columns in presentation order
    display_cols = [
        'Segment_Label',
        'Count',
        'Total_Volume',
        'Avg_Beacon',
        'Insured_Pct',
        'Broker_Pct',
        'Avg_TDS',
        'Avg_LTV',
        'Avg_Balance',
        'Avg_Amort_Months'
    ]
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in segment_table.columns]
    
    # Format for display
    display_df = segment_table[display_cols].copy()
    
    # Format numbers
    if 'Total_Volume' in display_df.columns:
        display_df['Total_Volume'] = display_df['Total_Volume'].apply(
            lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else 'N/A'
        )
    
    if 'Avg_Beacon' in display_df.columns:
        display_df['Avg_Beacon'] = display_df['Avg_Beacon'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else 'N/A'
        )
    
    for pct_col in ['Insured_Pct', 'Broker_Pct']:
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else 'N/A'
            )
    
    for avg_col in ['Avg_TDS', 'Avg_LTV']:
        if avg_col in display_df.columns:
            display_df[avg_col] = display_df[avg_col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
            )
    
    if 'Avg_Balance' in display_df.columns:
        display_df['Avg_Balance'] = display_df['Avg_Balance'].apply(
            lambda x: f"${x/1000:.0f}k" if pd.notna(x) else 'N/A'
        )
    
    if 'Avg_Amort_Months' in display_df.columns:
        display_df['Avg_Amort_Months'] = display_df['Avg_Amort_Months'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else 'N/A'
        )
    
    # Rename columns for display
    rename_map = {
        'Segment_Label': 'Segment',
        'Total_Volume': 'Total Volume',
        'Avg_Beacon': 'Avg. Beacon',
        'Insured_Pct': 'Insured %',
        'Broker_Pct': 'Broker %',
        'Avg_TDS': 'Avg. TDS',
        'Avg_LTV': 'Avg. LTV',
        'Avg_Balance': 'Avg. Balance',
        'Avg_Amort_Months': 'Avg. AM (years)'
    }
    
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    
    print("\n" + display_df.to_string(index=False))
    print("\n" + "="*80)


# ============================================================================
# VISUALIZE TABLE (AS PNG LIKE YOUR SLIDE)
# ============================================================================

def visualize_segment_table(segment_table, output_path=None):
    """
    Create visual table matching your presentation format
    """
    
    print("\nCreating visual table...")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    
    for _, row in segment_table.iterrows():
        table_row = [
            row['Segment_Label'],
            f"{int(row['Count']):,}",
        ]
        
        # Total Volume
        if pd.notna(row['Total_Volume']):
            table_row.append(f"${row['Total_Volume']/1e6:.1f}M")
        else:
            table_row.append('N/A')
        
        # Avg Beacon
        if pd.notna(row['Avg_Beacon']):
            table_row.append(f"{row['Avg_Beacon']:.0f}")
        else:
            table_row.append('N/A')
        
        # Insured %
        if pd.notna(row['Insured_Pct']):
            table_row.append(f"{row['Insured_Pct']:.1f}%")
        else:
            table_row.append('N/A')
        
        # Broker %
        if pd.notna(row['Broker_Pct']):
            table_row.append(f"{row['Broker_Pct']:.1f}%")
        else:
            table_row.append('N/A')
        
        # Avg TDS
        if pd.notna(row['Avg_TDS']):
            table_row.append(f"{row['Avg_TDS']:.2f}")
        else:
            table_row.append('N/A')
        
        # Avg LTV
        if pd.notna(row['Avg_LTV']):
            table_row.append(f"{row['Avg_LTV']:.2f}")
        else:
            table_row.append('N/A')
        
        # Avg Balance
        if pd.notna(row['Avg_Balance']):
            table_row.append(f"${row['Avg_Balance']/1000:.0f}k")
        else:
            table_row.append('N/A')
        
        table_data.append(table_row)
    
    # Column headers
    columns = [
        'Segment',
        'Count',
        'Total\nVolume',
        'Avg.\nBeacon',
        'Insured\n%',
        'Broker\n%',
        'Avg.\nTDS',
        'Avg.\nLTV',
        'Avg.\nBalance'
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.10, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    # Style header (matching your green header)
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2d5f3f')  # Dark green like your slide
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows (alternating colors)
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f5f5f5')
            else:
                cell.set_facecolor('white')
            
            # Bold first column (segment labels)
            if j == 0:
                cell.set_text_props(weight='bold')
            
            # Highlight lowest and highest segments
            if '(Lowest)' in str(table_data[i-1][0]) or '(Highest)' in str(table_data[i-1][0]):
                cell.set_text_props(weight='bold')
    
    plt.title('Sensitivity Segment Distributions', 
              fontsize=18, fontweight='bold', pad=20)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def create_segment_distribution_report(mtg_features, output_dir='/mnt/user-data/outputs'):
    """
    Complete pipeline to create segment distribution table
    
    Parameters:
    -----------
    mtg_features : pd.DataFrame
        Mortgage-level data with segment assignments
        Must have 'segment' column
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    segment_table : pd.DataFrame
        Segment distribution statistics
    """
    
    # Check required column
    if 'segment' not in mtg_features.columns:
        raise ValueError("mtg_features must have 'segment' column. Run segmentation first!")
    
    # Create table
    segment_table = create_segment_distribution_table(mtg_features)
    
    # Display formatted version
    display_segment_table(segment_table)
    
    # Create visualization
    fig = visualize_segment_table(segment_table, 
                                  output_path=f'{output_dir}/segment_distribution_table.png')
    
    # Save CSV
    segment_table.to_csv(f'{output_dir}/segment_distribution.csv', index=False)
    print(f"\n✓ Saved: segment_distribution.csv")
    
    print("\n" + "="*80)
    print("✅ SEGMENT DISTRIBUTION REPORT COMPLETE")
    print("="*80)
    
    return segment_table


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# After running sensitivity analysis:

from sensitivity_concise import run_analysis

results = run_analysis(
    df_clean,
    metric='sens_mean',
    ranking_method='absolute',
    n_segments=10
)

# Create segment distribution table
from segment_distribution import create_segment_distribution_report

segment_table = create_segment_distribution_report(
    mtg_features=results['features'],
    output_dir='/mnt/user-data/outputs'
)
"""