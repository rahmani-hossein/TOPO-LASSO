"""
MODULAR SENSITIVITY ANALYSIS
=============================
Production-ready code for mortgage sensitivity segmentation and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


# ============================================================================
# MODULE 1: DATA PREPARATION
# ============================================================================

def prepare_data(df, mtg_col='mtgnum', treatment_col='margin_x', 
                 outcome_col='sensitivity', verbose=True):
    """
    Prepare and validate input data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with mortgage records
    mtg_col : str
        Column name for mortgage identifier
    treatment_col : str  
        Column name for treatment variable (margin_x)
    outcome_col : str
        Column name for outcome/sensitivity
    verbose : bool
        Print diagnostic info
    
    Returns:
    --------
    pd.DataFrame : Validated data
    """
    if verbose:
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        print(f"Total rows: {len(df):,}")
        print(f"Unique mortgages: {df[mtg_col].nunique():,}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check records per mortgage
        records_per_mtg = df.groupby(mtg_col).size()
        print(f"\nRecords per mortgage:")
        print(f"  Mean: {records_per_mtg.mean():.1f}")
        print(f"  Min: {records_per_mtg.min()}")
        print(f"  Max: {records_per_mtg.max()}")
        print(f"  Mode: {records_per_mtg.mode().values[0] if len(records_per_mtg.mode()) > 0 else 'N/A'}")
    
    return df.copy()


# ============================================================================
# MODULE 2: 1D SENSITIVITY CALCULATION
# ============================================================================

def calculate_1d_sensitivity(df, mtg_col='mtgnum', treatment_col='margin_x',
                             outcome_col='sensitivity', method='mean', 
                             verbose=True):
    """
    Calculate 1D sensitivity metric from response curves
    
    Parameters:
    -----------
    method : str
        Method for 1D calculation:
        - 'mean': Average sensitivity (RECOMMENDED for CATE)
        - 'max': Maximum sensitivity  
        - 'std': Standard deviation (variability)
        - 'auc': Area under curve
        - 'range': Max - Min
        - 'all': Calculate all metrics
    
    Returns:
    --------
    pd.DataFrame : One row per mortgage with 1D sensitivity
    """
    if verbose:
        print("\n" + "="*80)
        print(f"1D SENSITIVITY CALCULATION - Method: {method.upper()}")
        print("="*80)
    
    results = []
    
    for mtg_num in df[mtg_col].unique():
        # Get all records for this mortgage
        mtg_data = df[df[mtg_col] == mtg_num].copy()
        mtg_data = mtg_data.sort_values(treatment_col)
        
        x_values = mtg_data[treatment_col].values
        y_values = mtg_data[outcome_col].values
        
        record = {mtg_col: mtg_num}
        
        if method == 'mean' or method == 'all':
            record['sens_mean'] = np.mean(y_values)
        
        if method == 'max' or method == 'all':
            record['sens_max'] = np.max(y_values)
        
        if method == 'std' or method == 'all':
            record['sens_std'] = np.std(y_values)
        
        if method == 'auc' or method == 'all':
            # Normalize x-values to [0,1] to make AUC comparable
            x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-10)
            record['sens_auc'] = np.trapz(y_values, x_norm)
        
        if method == 'range' or method == 'all':
            record['sens_range'] = np.max(y_values) - np.min(y_values)
        
        if method == 'min' or method == 'all':
            record['sens_min'] = np.min(y_values)
        
        results.append(record)
    
    sensitivity_1d = pd.DataFrame(results)
    
    if verbose:
        metric_col = [c for c in sensitivity_1d.columns if c.startswith('sens_')][0]
        print(f"✓ Calculated 1D sensitivity for {len(sensitivity_1d):,} mortgages")
        print(f"\nStatistics ({metric_col}):")
        print(f"  Min: {sensitivity_1d[metric_col].min():.4f}")
        print(f"  Max: {sensitivity_1d[metric_col].max():.4f}")
        print(f"  Mean: {sensitivity_1d[metric_col].mean():.4f}")
        print(f"  Median: {sensitivity_1d[metric_col].median():.4f}")
        print(f"  Std: {sensitivity_1d[metric_col].std():.4f}")
    
    return sensitivity_1d


# ============================================================================
# MODULE 3: SEGMENTATION / QUANTILE RANKING
# ============================================================================

def create_sensitivity_segments(df, sensitivity_1d, mtg_col='mtgnum',
                                n_segments=10, sensitivity_metric='sens_mean',
                                verbose=True):
    """
    Rank mortgages by sensitivity and create segments (quantiles)
    
    Parameters:
    -----------
    n_segments : int
        Number of segments (default 10)
    sensitivity_metric : str
        Column name to use for ranking
    
    Returns:
    --------
    pd.DataFrame : Mortgage-level data with segment assignments
    """
    if verbose:
        print("\n" + "="*80)
        print(f"CREATING {n_segments} SENSITIVITY SEGMENTS")
        print("="*80)
    
    # Get one row per mortgage with all features
    mtg_features = df.groupby(mtg_col).first().reset_index()
    
    # Merge sensitivity
    mtg_features = mtg_features.merge(sensitivity_1d, on=mtg_col, how='left')
    
    # Create segments using NumPy
    sorted_indices = np.argsort(mtg_features[sensitivity_metric].values)
    n_per_segment = len(sorted_indices) // n_segments
    
    segment_assignments = np.zeros(len(sorted_indices), dtype=int)
    
    for seg in range(n_segments):
        start_idx = seg * n_per_segment
        if seg == n_segments - 1:  # Last segment gets remaining
            end_idx = len(sorted_indices)
        else:
            end_idx = (seg + 1) * n_per_segment
        
        # Segment 1 = lowest, Segment 10 = highest
        segment_assignments[sorted_indices[start_idx:end_idx]] = seg + 1
    
    mtg_features['sensitivity_segment'] = segment_assignments
    
    if verbose:
        print(f"✓ Assigned {len(mtg_features):,} mortgages to {n_segments} segments")
        print("\nSegment Distribution:")
        for seg in range(1, n_segments + 1):
            count = np.sum(segment_assignments == seg)
            avg_sens = mtg_features[mtg_features['sensitivity_segment'] == seg][sensitivity_metric].mean()
            print(f"  Segment {seg:2d}: {count:6,} mortgages (Avg sens: {avg_sens:.4f})")
    
    return mtg_features


# ============================================================================
# MODULE 4: SEGMENT ANALYSIS
# ============================================================================

def analyze_segments(mtg_features, feature_cols=None, verbose=True):
    """
    Analyze features by sensitivity segment
    
    Parameters:
    -----------
    mtg_features : pd.DataFrame
        Mortgage-level data with segment assignments
    feature_cols : list
        Feature columns to analyze (auto-detect if None)
    
    Returns:
    --------
    pd.DataFrame : Segment-level statistics
    """
    if verbose:
        print("\n" + "="*80)
        print("SEGMENT FEATURE ANALYSIS")
        print("="*80)
    
    # Auto-detect numeric features if not specified
    if feature_cols is None:
        numeric_cols = mtg_features.select_dtypes(include=[np.number]).columns
        exclude_cols = ['mtgnum', 'sensitivity_segment']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Filter to available features
    feature_cols = [c for c in feature_cols if c in mtg_features.columns]
    
    if verbose:
        print(f"Analyzing {len(feature_cols)} features: {feature_cols}")
    
    segment_stats = []
    
    for seg in sorted(mtg_features['sensitivity_segment'].unique()):
        seg_data = mtg_features[mtg_features['sensitivity_segment'] == seg]
        
        stats = {
            'segment': seg,
            'count': len(seg_data),
            'total_volume': seg_data.get('volume', seg_data.get('balance', pd.Series([0]))).sum()
        }
        
        for feature in feature_cols:
            values = seg_data[feature].dropna().values
            if len(values) > 0:
                stats[f'{feature}_mean'] = np.mean(values)
                stats[f'{feature}_median'] = np.median(values)
                stats[f'{feature}_std'] = np.std(values)
        
        segment_stats.append(stats)
    
    segment_analysis = pd.DataFrame(segment_stats)
    
    if verbose:
        print("\n✓ Segment analysis complete")
        display_cols = ['segment', 'count'] + [c for c in segment_analysis.columns 
                                               if c.endswith('_mean')][:5]
        print("\nSegment Summary:")
        print(segment_analysis[display_cols].to_string(index=False))
    
    return segment_analysis


# ============================================================================
# MODULE 5: RESPONSE CURVE VISUALIZATION
# ============================================================================

def visualize_response_curves(df, mtg_features, mtg_col='mtgnum',
                              treatment_col='margin_x', outcome_col='sensitivity',
                              n_samples=6, by_segment=False, output_path=None):
    """
    Visualize response curves Y vs T
    
    Parameters:
    -----------
    by_segment : bool
        If True, sample mortgages from different segments
    output_path : str or Path
        Where to save the figure
    """
    print("\n" + "="*80)
    print("VISUALIZING RESPONSE CURVES")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Sample mortgages
    if by_segment and 'sensitivity_segment' in mtg_features.columns:
        # Sample from different segments
        segments = sorted(mtg_features['sensitivity_segment'].unique())
        step = max(1, len(segments) // n_samples)
        sample_segments = segments[::step][:n_samples]
        
        sampled_mtgs = []
        for seg in sample_segments:
            seg_mtgs = mtg_features[mtg_features['sensitivity_segment'] == seg][mtg_col].values
            if len(seg_mtgs) > 0:
                sampled_mtgs.append(np.random.choice(seg_mtgs))
    else:
        unique_mtgs = df[mtg_col].unique()
        np.random.seed(42)
        sampled_mtgs = np.random.choice(unique_mtgs, min(n_samples, len(unique_mtgs)), replace=False)
    
    for idx, mtg_num in enumerate(sampled_mtgs[:6]):
        ax = axes[idx]
        
        # Get response curve data
        mtg_data = df[df[mtg_col] == mtg_num].sort_values(treatment_col)
        
        x_vals = mtg_data[treatment_col].values
        y_vals = mtg_data[outcome_col].values
        
        # Plot curve
        ax.plot(x_vals, y_vals, linewidth=2.5, marker='o', markersize=4,
                color='steelblue', alpha=0.8)
        
        # Get segment info
        seg_info = ""
        if 'sensitivity_segment' in mtg_features.columns:
            seg = mtg_features[mtg_features[mtg_col] == mtg_num]['sensitivity_segment'].values
            if len(seg) > 0:
                seg_info = f" (Segment {seg[0]})"
        
        ax.set_xlabel('Treatment (T): Margin X', fontsize=11, fontweight='bold')
        ax.set_ylabel('Outcome (Y): Sensitivity', fontsize=11, fontweight='bold')
        ax.set_title(f'Mortgage {mtg_num:,}{seg_info}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add stats box
        stats_text = (f'Mean Y: {np.mean(y_vals):.2f}\n'
                     f'Range: {np.max(y_vals) - np.min(y_vals):.2f}\n'
                     f'N points: {len(y_vals)}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(sampled_mtgs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Response Curves: Y vs. T (Sensitivity vs. Margin)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# MODULE 6: SEGMENT DISTRIBUTION TABLE
# ============================================================================

def create_segment_distribution_table(mtg_features, segment_analysis, 
                                      feature_cols=None, output_path=None):
    """
    Create segment distribution table matching your presentation format
    """
    print("\n" + "="*80)
    print("CREATING SEGMENT DISTRIBUTION TABLE")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    columns = ['Segment', 'Count', 'Total\nVolume', 'Avg.\nBeacon']
    
    # Add feature columns if available
    if feature_cols is None:
        feature_cols = []
        for col in mtg_features.columns:
            if any(x in col.lower() for x in ['insured', 'broker', 'tds', 'ltv', 
                                               'balance', 'beacon', 'amort']):
                feature_cols.append(col)
    
    # Build table data
    table_data = []
    
    for seg in sorted(segment_analysis['segment'].unique()):
        seg_stats = segment_analysis[segment_analysis['segment'] == seg].iloc[0]
        
        row = [
            f"{int(seg)} {'(Lowest)' if seg == 1 else '(Highest)' if seg == 10 else ''}",
            f"{int(seg_stats['count']):,}",
            f"${seg_stats.get('total_volume', 0)/1e6:.1f}M" if 'total_volume' in seg_stats else 'N/A',
        ]
        
        # Add feature columns
        for feature in feature_cols[:5]:  # Limit to 5 features
            mean_col = f'{feature}_mean'
            if mean_col in seg_stats:
                val = seg_stats[mean_col]
                if 'pct' in feature.lower() or '%' in feature.lower():
                    row.append(f"{val:.1f}%")
                else:
                    row.append(f"{val:.2f}")
            else:
                row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=columns[:len(table_data[0])],
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12] + [0.12] * (len(table_data[0]) - 2))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns[:len(table_data[0])])):
        cell = table[(0, i)]
        cell.set_facecolor('#2d5f3f')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
    
    plt.title('Sensitivity Segment Distributions', fontsize=16, 
             fontweight='bold', pad=20)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# MODULE 7: SEGMENT COMPARISON VISUALIZATIONS  
# ============================================================================

def create_segment_comparison_plots(segment_analysis, mtg_features,
                                   output_path=None):
    """
    Create comprehensive segment comparison plots
    """
    print("\n" + "="*80)
    print("CREATING SEGMENT COMPARISON PLOTS")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    segments = sorted(segment_analysis['segment'].unique())
    
    # Plot 1: Count distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(segments, segment_analysis['count'], color='steelblue', 
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Sensitivity Segment', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Mortgages', fontsize=12, fontweight='bold')
    ax1.set_title('Mortgage Count by Segment', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average sensitivity
    sens_cols = [c for c in segment_analysis.columns if c.startswith('sens_')]
    if sens_cols:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(segments, segment_analysis[sens_cols[0]], marker='o',
                linewidth=2.5, markersize=8, color='coral')
        ax2.fill_between(segments, segment_analysis[sens_cols[0]], alpha=0.3, color='coral')
        ax2.set_xlabel('Sensitivity Segment', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Sensitivity', fontsize=12, fontweight='bold')
        ax2.set_title('Sensitivity by Segment', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3-6: Feature distributions
    feature_cols = [c for c in segment_analysis.columns if c.endswith('_mean') 
                   and not c.startswith('sens_')]
    
    plot_positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    colors = ['seagreen', 'purple', 'orange', 'teal']
    
    for idx, feature in enumerate(feature_cols[:4]):
        if idx < len(plot_positions):
            ax = fig.add_subplot(gs[plot_positions[idx]])
            ax.bar(segments, segment_analysis[feature], color=colors[idx],
                  alpha=0.7, edgecolor='black', linewidth=1.5)
            
            feature_name = feature.replace('_mean', '').replace('_', ' ').title()
            ax.set_xlabel('Sensitivity Segment', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Avg {feature_name}', fontsize=12, fontweight='bold')
            ax.set_title(f'{feature_name} by Segment', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Box plot of sensitivity distribution
    ax7 = fig.add_subplot(gs[2, :])
    if 'sensitivity_segment' in mtg_features.columns:
        sens_metric = [c for c in mtg_features.columns if c.startswith('sens_')][0]
        box_data = [mtg_features[mtg_features['sensitivity_segment'] == seg][sens_metric].values
                   for seg in segments]
        bp = ax7.boxplot(box_data, labels=segments, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        ax7.set_xlabel('Sensitivity Segment', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Sensitivity Distribution', fontsize=12, fontweight='bold')
        ax7.set_title('Sensitivity Distribution by Segment', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Segment Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# MODULE 8: MAIN PIPELINE
# ============================================================================

def run_sensitivity_analysis(df, output_dir, 
                             mtg_col='mtgnum',
                             treatment_col='margin_x',
                             outcome_col='sensitivity',
                             sensitivity_method='mean',
                             n_segments=10,
                             feature_cols=None):
    """
    Complete sensitivity analysis pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    output_dir : str or Path
        Directory to save outputs
    sensitivity_method : str
        Method for 1D calculation ('mean', 'max', 'auc', 'all')
    n_segments : int
        Number of sensitivity segments (default 10)
    feature_cols : list
        Features to analyze by segment
    
    Returns:
    --------
    dict : Analysis results and outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("MORTGAGE SENSITIVITY ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Prepare data
    df_clean = prepare_data(df, mtg_col, treatment_col, outcome_col)
    
    # Step 2: Calculate 1D sensitivity
    sensitivity_1d = calculate_1d_sensitivity(df_clean, mtg_col, treatment_col,
                                              outcome_col, sensitivity_method)
    
    # Step 3: Create segments
    sens_metric = [c for c in sensitivity_1d.columns if c.startswith('sens_')][0]
    mtg_features = create_sensitivity_segments(df_clean, sensitivity_1d, mtg_col,
                                               n_segments, sens_metric)
    
    # Step 4: Analyze segments
    segment_analysis = analyze_segments(mtg_features, feature_cols)
    
    # Step 5: Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig1 = visualize_response_curves(df_clean, mtg_features, mtg_col, treatment_col,
                                     outcome_col, n_samples=6, by_segment=True,
                                     output_path=output_dir / 'response_curves.png')
    
    fig2 = create_segment_distribution_table(mtg_features, segment_analysis,
                                             feature_cols,
                                             output_path=output_dir / 'segment_table.png')
    
    fig3 = create_segment_comparison_plots(segment_analysis, mtg_features,
                                           output_path=output_dir / 'segment_comparison.png')
    
    # Step 6: Save data outputs
    print("\n" + "="*80)
    print("SAVING DATA FILES")
    print("="*80)
    
    sensitivity_1d.to_csv(output_dir / 'sensitivity_1d.csv', index=False)
    print(f"✓ sensitivity_1d.csv")
    
    segment_analysis.to_csv(output_dir / 'segment_analysis.csv', index=False)
    print(f"✓ segment_analysis.csv")
    
    mtg_features.to_csv(output_dir / 'mortgages_with_segments.csv', index=False)
    print(f"✓ mortgages_with_segments.csv")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to: {output_dir}")
    
    return {
        'sensitivity_1d': sensitivity_1d,
        'mtg_features': mtg_features,
        'segment_analysis': segment_analysis,
        'figures': [fig1, fig2, fig3]
    }