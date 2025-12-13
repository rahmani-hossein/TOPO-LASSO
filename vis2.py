"""
CONCISE SENSITIVITY ANALYSIS WITH RANKING
==========================================
Handles both positive and negative sensitivities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: CALCULATE 1D SENSITIVITY (MEAN & AUC)
# ============================================================================

def calculate_sensitivity_1d(df, mtg_col='mtgnum', treatment_col='margin_x', 
                             outcome_col='sensitivity'):
    """Calculate mean and AUC for each mortgage"""
    
    results = []
    for mtg in df[mtg_col].unique():
        data = df[df[mtg_col] == mtg].sort_values(treatment_col)
        x = data[treatment_col].values
        y = data[outcome_col].values
        
        # Normalize x to [0,1] for AUC
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        
        results.append({
            mtg_col: mtg,
            'sens_mean': np.mean(y),
            'sens_auc': np.trapz(y, x_norm),
            'sens_min': np.min(y),
            'sens_max': np.max(y)
        })
    
    return pd.DataFrame(results)


# ============================================================================
# STEP 2: ANALYZE POSITIVE VS NEGATIVE SENSITIVITIES
# ============================================================================

def analyze_sensitivity_signs(sens_df, metric='sens_mean'):
    """Analyze how many sensitivities are positive vs negative"""
    
    print("="*80)
    print("SENSITIVITY SIGN ANALYSIS")
    print("="*80)
    
    n_positive = (sens_df[metric] > 0).sum()
    n_negative = (sens_df[metric] < 0).sum()
    n_zero = (sens_df[metric] == 0).sum()
    total = len(sens_df)
    
    print(f"\nUsing metric: {metric}")
    print(f"  Positive (wrong direction): {n_positive:,} ({n_positive/total*100:.1f}%)")
    print(f"  Negative (correct direction): {n_negative:,} ({n_negative/total*100:.1f}%)")
    print(f"  Zero: {n_zero:,} ({n_zero/total*100:.1f}%)")
    
    print(f"\nSensitivity statistics:")
    print(f"  Min: {sens_df[metric].min():.4f}")
    print(f"  Max: {sens_df[metric].max():.4f}")
    print(f"  Mean: {sens_df[metric].mean():.4f}")
    print(f"  Median: {sens_df[metric].median():.4f}")
    
    return {
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'pct_positive': n_positive/total*100,
        'pct_negative': n_negative/total*100
    }


# ============================================================================
# STEP 3: RANKING STRATEGIES
# ============================================================================

def rank_mortgages(sens_df, metric='sens_mean', ranking_method='absolute'):
    """
    Rank mortgages by sensitivity
    
    ranking_method options:
    - 'absolute': Rank by absolute value (|sensitivity|) - RECOMMENDED
    - 'raw': Rank by raw value (negative = high rank)
    - 'negative_only': Rank only negative sensitivities, exclude positive
    """
    
    print("\n" + "="*80)
    print(f"RANKING METHOD: {ranking_method.upper()}")
    print("="*80)
    
    if ranking_method == 'absolute':
        # Rank by absolute value - most sensitive regardless of sign
        sens_df['sens_abs'] = sens_df[metric].abs()
        sens_df = sens_df.sort_values('sens_abs', ascending=False)
        sens_df['rank'] = range(1, len(sens_df) + 1)
        print("\n✓ Ranked by ABSOLUTE VALUE")
        print("  High rank = large |sensitivity| (strong effect, any direction)")
        
    elif ranking_method == 'raw':
        # Rank by raw value - most negative = highest rank
        sens_df = sens_df.sort_values(metric, ascending=True)
        sens_df['rank'] = range(1, len(sens_df) + 1)
        print("\n✓ Ranked by RAW VALUE (most negative first)")
        print("  High rank = most negative (correct direction)")
        
    elif ranking_method == 'negative_only':
        # Only rank negative sensitivities
        neg_only = sens_df[sens_df[metric] < 0].copy()
        neg_only = neg_only.sort_values(metric, ascending=True)
        neg_only['rank'] = range(1, len(neg_only) + 1)
        sens_df = neg_only
        print(f"\n✓ Ranked NEGATIVE SENSITIVITIES ONLY")
        print(f"  Excluded {(sens_df[metric] >= 0).sum():,} positive sensitivities")
        print(f"  Remaining: {len(neg_only):,} mortgages")
    
    return sens_df


# ============================================================================
# STEP 4: CREATE SEGMENTS
# ============================================================================

def create_segments(sens_df, n_segments=10):
    """Create segments from ranked data"""
    
    # Quantile-based segmentation
    sens_df['segment'] = pd.qcut(sens_df['rank'], q=n_segments, 
                                  labels=range(1, n_segments+1), duplicates='drop')
    
    print(f"\n✓ Created {n_segments} segments")
    print(f"  Segment 1 = Lowest rank (least sensitive)")
    print(f"  Segment {n_segments} = Highest rank (most sensitive)")
    
    return sens_df


# ============================================================================
# STEP 5: MERGE WITH FEATURES & ANALYZE
# ============================================================================

def analyze_segments_by_features(df_original, sens_ranked, mtg_col='mtgnum',
                                feature_cols=None):
    """Merge with original data and analyze features by segment"""
    
    # Get one row per mortgage
    mtg_features = df_original.groupby(mtg_col).first().reset_index()
    
    # Merge with sensitivity
    mtg_features = mtg_features.merge(sens_ranked, on=mtg_col, how='inner')
    
    # Auto-detect features if not provided
    if feature_cols is None:
        numeric_cols = mtg_features.select_dtypes(include=[np.number]).columns
        exclude = [mtg_col, 'rank', 'segment', 'sens_mean', 'sens_auc', 'sens_abs', 
                  'sens_min', 'sens_max']
        feature_cols = [c for c in numeric_cols if c not in exclude]
    
    # Analyze by segment
    seg_analysis = mtg_features.groupby('segment').agg({
        mtg_col: 'count',
        **{col: ['mean', 'median', 'std'] for col in feature_cols if col in mtg_features.columns}
    }).round(4)
    
    seg_analysis.columns = ['_'.join(map(str, c)) if c[1] else c[0] 
                           for c in seg_analysis.columns]
    seg_analysis = seg_analysis.rename(columns={f'{mtg_col}_count': 'count'})
    
    return mtg_features, seg_analysis


# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================

def plot_sensitivity_distribution(sens_df, metric='sens_mean'):
    """Plot distribution showing positive/negative split"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram
    ax1 = axes[0]
    ax1.hist(sens_df[metric], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.set_xlabel('Sensitivity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Sensitivity Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    n_pos = (sens_df[metric] > 0).sum()
    n_neg = (sens_df[metric] < 0).sum()
    stats_text = f'Positive: {n_pos:,}\nNegative: {n_neg:,}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([sens_df[sens_df[metric] < 0][metric].values,
                      sens_df[sens_df[metric] > 0][metric].values],
                     labels=['Negative\n(Correct)', 'Positive\n(Wrong)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Sensitivity', fontsize=12, fontweight='bold')
    ax2.set_title('Positive vs Negative', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def run_analysis(df_clean, mtg_col='mtgnum', treatment_col='margin_x',
                outcome_col='sensitivity', metric='sens_mean',
                ranking_method='absolute', n_segments=10,
                feature_cols=None, output_dir='/mnt/user-data/outputs'):
    """
    Complete sensitivity analysis pipeline
    
    Parameters:
    -----------
    ranking_method : str
        'absolute' (recommended), 'raw', or 'negative_only'
    metric : str
        'sens_mean' (recommended) or 'sens_auc'
    """
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Calculate 1D sensitivity
    print("\nStep 1: Calculating 1D sensitivity...")
    sens_df = calculate_sensitivity_1d(df_clean, mtg_col, treatment_col, outcome_col)
    
    # Step 2: Analyze signs
    sign_stats = analyze_sensitivity_signs(sens_df, metric)
    
    # Step 3: Rank
    sens_ranked = rank_mortgages(sens_df, metric, ranking_method)
    
    # Step 4: Create segments
    sens_ranked = create_segments(sens_ranked, n_segments)
    
    # Step 5: Analyze features
    print("\nStep 5: Analyzing features by segment...")
    mtg_features, seg_analysis = analyze_segments_by_features(
        df_clean, sens_ranked, mtg_col, feature_cols
    )
    
    # Step 6: Visualize
    print("\nStep 6: Creating visualizations...")
    fig = plot_sensitivity_distribution(sens_df, metric)
    fig.savefig(f'{output_dir}/sensitivity_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: sensitivity_distribution.png")
    
    # Save outputs
    print("\nSaving outputs...")
    sens_ranked.to_csv(f'{output_dir}/sensitivity_ranked.csv', index=False)
    seg_analysis.to_csv(f'{output_dir}/segment_analysis.csv')
    mtg_features.to_csv(f'{output_dir}/mortgages_with_segments.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTop 10 most sensitive mortgages:")
    print(sens_ranked.head(10)[[mtg_col, metric, 'sens_abs', 'rank', 'segment']])
    
    return {
        'sensitivity': sens_ranked,
        'features': mtg_features,
        'segments': seg_analysis,
        'sign_stats': sign_stats
    }