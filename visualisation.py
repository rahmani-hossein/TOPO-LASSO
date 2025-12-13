import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")

def visualize_200_curves(df, n_mortgages=6):
    """
    For each mortgage, visualize all 200 sensitivity curves (one per margin_x value)
    """
    print("\n" + "="*80)
    print("STEP 1: Visualizing 200 Sensitivity Curves per Mortgage")
    print("="*80)
    
    # Get unique mortgages
    unique_mtg = df['mtgnum'].unique()
    
    # Sample some mortgages
    np.random.seed(42)
    if len(unique_mtg) > n_mortgages:
        sampled_mtg = np.random.choice(unique_mtg, n_mortgages, replace=False)
    else:
        sampled_mtg = unique_mtg
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, mtg_num in enumerate(sampled_mtg[:6]):
        ax = axes[idx]
        
        # Get all 200 records for this mortgage
        mtg_data = df[df['mtgnum'] == mtg_num].copy()
        
        print(f"Mortgage {mtg_num}: {len(mtg_data)} records")
        
        # Sort by margin_x for proper curve visualization
        mtg_data = mtg_data.sort_values('margin_x')
        
        # Extract the 200 margin_x values and 200 sensitivity values
        x_values = mtg_data['margin_x'].values  # 200 points
        y_values = mtg_data['sensitivity'].values  # 200 points
        
        # Plot the curve
        ax.plot(x_values, y_values, 
                linewidth=2, 
                marker='o', 
                markersize=4,
                alpha=0.7,
                color='steelblue')
        
        ax.set_xlabel('Margin X', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=11, fontweight='bold')
        ax.set_title(f'Mortgage {mtg_num:,}\n({len(mtg_data)} points)', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add some statistics
        y_mean = np.mean(y_values)
        y_std = np.std(y_values)
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        
        stats_text = f'Mean: {y_mean:.2f}\nStd: {y_std:.2f}\nRange: [{y_min:.2f}, {y_max:.2f}]'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(sampled_mtg), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Sensitivity Curves: 200 Points per Mortgage', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    print(f"✓ Visualized {len(sampled_mtg)} mortgages")
    
    return fig

def calculate_1d_sensitivity_numpy(df):
    """
    Calculate 1D sensitivity from 200 values using NumPy
    Multiple methods available - using Area Under Curve (trapezoidal rule)
    """
    print("\n" + "="*80)
    print("STEP 2: Calculating 1D Sensitivity from 200 Points")
    print("="*80)
    
    results = []
    
    for mtg_num in df['mtgnum'].unique():
        # Get 200 records for this mortgage
        mtg_data = df[df['mtgnum'] == mtg_num].copy()
        mtg_data = mtg_data.sort_values('margin_x')
        
        x_values = mtg_data['margin_x'].values
        y_values = mtg_data['sensitivity'].values
        
        # Method 1: Area Under Curve using NumPy trapezoidal rule
        auc = np.trapz(y_values, x_values)
        
        # Method 2: Mean sensitivity (alternative)
        mean_sens = np.mean(y_values)
        
        # Method 3: Max sensitivity (alternative)
        max_sens = np.max(y_values)
        
        # Method 4: Standard deviation (variability)
        std_sens = np.std(y_values)
        
        results.append({
            'mtgnum': mtg_num,
            'sensitivity_1d_auc': auc,
            'sensitivity_mean': mean_sens,
            'sensitivity_max': max_sens,
            'sensitivity_std': std_sens
        })
    
    sensitivity_1d = pd.DataFrame(results)
    
    print(f"✓ Calculated 1D sensitivity for {len(sensitivity_1d):,} mortgages")
    print(f"\nSensitivity Statistics (AUC):")
    print(f"  Min: {sensitivity_1d['sensitivity_1d_auc'].min():.2f}")
    print(f"  Max: {sensitivity_1d['sensitivity_1d_auc'].max():.2f}")
    print(f"  Mean: {sensitivity_1d['sensitivity_1d_auc'].mean():.2f}")
    print(f"  Median: {sensitivity_1d['sensitivity_1d_auc'].median():.2f}")
    
    return sensitivity_1d

def rank_and_create_quantiles(df, sensitivity_1d, n_quantiles=10):
    """
    Rank mortgages by 1D sensitivity and assign to quantiles
    """
    print("\n" + "="*80)
    print(f"STEP 3: Ranking and Creating {n_quantiles} Quantiles")
    print("="*80)
    
    # Merge sensitivity back to original data
    df_merged = df.merge(sensitivity_1d[['mtgnum', 'sensitivity_1d_auc']], 
                         on='mtgnum', how='left')
    
    # Get one row per mortgage for ranking
    mtg_level = df_merged.groupby('mtgnum').first().reset_index()
    
    # Create quantiles using NumPy
    # Sort by sensitivity
    sorted_indices = np.argsort(mtg_level['sensitivity_1d_auc'].values)
    n_per_quantile = len(sorted_indices) // n_quantiles
    
    quantile_assignments = np.zeros(len(sorted_indices), dtype=int)
    
    for q in range(n_quantiles):
        start_idx = q * n_per_quantile
        if q == n_quantiles - 1:  # Last quantile gets remaining
            end_idx = len(sorted_indices)
        else:
            end_idx = (q + 1) * n_per_quantile
        
        # Assign quantile (0 = lowest, 9 = highest)
        quantile_assignments[sorted_indices[start_idx:end_idx]] = q
    
    mtg_level['sensitivity_quantile'] = quantile_assignments
    
    print(f"✓ Assigned {len(mtg_level):,} mortgages to {n_quantiles} quantiles")
    print("\nMortgages per quantile:")
    unique, counts = np.unique(quantile_assignments, return_counts=True)
    for q, count in zip(unique, counts):
        print(f"  Quantile {q}: {count} mortgages")
    
    return mtg_level

def analyze_quantiles_numpy(mtg_level):
    """
    Analyze features by quantile using NumPy
    """
    print("\n" + "="*80)
    print("STEP 4: Analyzing Features by Quantile")
    print("="*80)
    
    # Features to analyze
    feature_cols = [
        'sensitivity_1d_auc',
        'remaining_amort_months_x',
        'cost_of_funds_x',
        'prediction'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in mtg_level.columns]
    
    quantile_stats = []
    
    for q in range(10):
        # Get mortgages in this quantile
        q_data = mtg_level[mtg_level['sensitivity_quantile'] == q]
        
        stats = {'quantile': q, 'n_mortgages': len(q_data)}
        
        for feature in available_features:
            values = q_data[feature].values
            stats[f'{feature}_mean'] = np.mean(values)
            stats[f'{feature}_median'] = np.median(values)
            stats[f'{feature}_std'] = np.std(values)
            stats[f'{feature}_min'] = np.min(values)
            stats[f'{feature}_max'] = np.max(values)
        
        quantile_stats.append(stats)
    
    quantile_analysis = pd.DataFrame(quantile_stats)
    quantile_analysis = quantile_analysis.set_index('quantile')
    
    print("✓ Quantile analysis complete")
    print("\nSummary (Quantile 0=Lowest, 9=Highest):")
    display_cols = ['n_mortgages', 'sensitivity_1d_auc_mean', 
                   'remaining_amort_months_x_mean', 'prediction_mean']
    display_cols = [c for c in display_cols if c in quantile_analysis.columns]
    print(quantile_analysis[display_cols].to_string())
    
    return quantile_analysis

def visualize_quantile_analysis(quantile_analysis, mtg_level):
    """
    Create comprehensive quantile visualizations
    """
    print("\n" + "="*80)
    print("STEP 5: Creating Quantile Visualizations")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    quantiles = quantile_analysis.index.values
    
    # Plot 1: Number of mortgages per quantile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(quantiles, quantile_analysis['n_mortgages'], 
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Mortgages', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Across Quantiles', fontsize=13, fontweight='bold')
    ax1.set_xticks(quantiles)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mean sensitivity (AUC) by quantile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(quantiles, quantile_analysis['sensitivity_1d_auc_mean'], 
             marker='o', linestyle='-', linewidth=2.5, markersize=8, color='coral')
    ax2.fill_between(quantiles, quantile_analysis['sensitivity_1d_auc_mean'], 
                     alpha=0.3, color='coral')
    ax2.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Sensitivity (AUC)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Sensitivity by Quantile', fontsize=13, fontweight='bold')
    ax2.set_xticks(quantiles)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Remaining amortization
    ax3 = fig.add_subplot(gs[0, 2])
    if 'remaining_amort_months_x_mean' in quantile_analysis.columns:
        ax3.bar(quantiles, quantile_analysis['remaining_amort_months_x_mean'], 
                color='seagreen', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Avg Remaining Months', fontsize=12, fontweight='bold')
        ax3.set_title('Remaining Amortization by Quantile', fontsize=13, fontweight='bold')
        ax3.set_xticks(quantiles)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cost of funds
    ax4 = fig.add_subplot(gs[1, 0])
    if 'cost_of_funds_x_mean' in quantile_analysis.columns:
        ax4.bar(quantiles, quantile_analysis['cost_of_funds_x_mean'], 
                color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Avg Cost of Funds', fontsize=12, fontweight='bold')
        ax4.set_title('Cost of Funds by Quantile', fontsize=13, fontweight='bold')
        ax4.set_xticks(quantiles)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Prediction mean
    ax5 = fig.add_subplot(gs[1, 1])
    if 'prediction_mean' in quantile_analysis.columns:
        ax5.bar(quantiles, quantile_analysis['prediction_mean'], 
                color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax5.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Avg Prediction', fontsize=12, fontweight='bold')
        ax5.set_title('Model Prediction by Quantile', fontsize=13, fontweight='bold')
        ax5.set_xticks(quantiles)
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Box plot of sensitivity distribution
    ax6 = fig.add_subplot(gs[1, 2])
    box_data = [mtg_level[mtg_level['sensitivity_quantile'] == q]['sensitivity_1d_auc'].values 
                for q in quantiles]
    bp = ax6.boxplot(box_data, labels=quantiles, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    ax6.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Sensitivity (AUC)', fontsize=12, fontweight='bold')
    ax6.set_title('Sensitivity Distribution by Quantile', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Sensitivity range by quantile (min, mean, max)
    ax7 = fig.add_subplot(gs[2, :2])
    x_pos = np.arange(len(quantiles))
    width = 0.25
    
    ax7.bar(x_pos - width, quantile_analysis['sensitivity_1d_auc_min'], 
            width, label='Min', color='lightcoral', alpha=0.7, edgecolor='black')
    ax7.bar(x_pos, quantile_analysis['sensitivity_1d_auc_mean'], 
            width, label='Mean', color='steelblue', alpha=0.7, edgecolor='black')
    ax7.bar(x_pos + width, quantile_analysis['sensitivity_1d_auc_max'], 
            width, label='Max', color='lightgreen', alpha=0.7, edgecolor='black')
    
    ax7.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Sensitivity (AUC)', fontsize=12, fontweight='bold')
    ax7.set_title('Sensitivity Range by Quantile (Min, Mean, Max)', fontsize=13, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(quantiles)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Cumulative distribution
    ax8 = fig.add_subplot(gs[2, 2])
    sorted_sens = np.sort(mtg_level['sensitivity_1d_auc'].values)
    cumulative = np.arange(1, len(sorted_sens) + 1) / len(sorted_sens) * 100
    ax8.plot(sorted_sens, cumulative, linewidth=2.5, color='darkblue')
    ax8.set_xlabel('Sensitivity (AUC)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Cumulative %', fontsize=12, fontweight='bold')
    ax8.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Median')
    ax8.legend()
    
    # Plot 9: Sensitivity variance by quantile
    ax9 = fig.add_subplot(gs[3, 0])
    ax9.bar(quantiles, quantile_analysis['sensitivity_1d_auc_std'], 
            color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax9.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Std Dev of Sensitivity', fontsize=12, fontweight='bold')
    ax9.set_title('Sensitivity Variability by Quantile', fontsize=13, fontweight='bold')
    ax9.set_xticks(quantiles)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Plot 10: Quantile comparison heatmap
    ax10 = fig.add_subplot(gs[3, 1:])
    
    # Select key metrics for heatmap
    heatmap_cols = ['sensitivity_1d_auc_mean', 'remaining_amort_months_x_mean', 
                    'cost_of_funds_x_mean', 'prediction_mean']
    heatmap_cols = [c for c in heatmap_cols if c in quantile_analysis.columns]
    
    if heatmap_cols:
        heatmap_data = quantile_analysis[heatmap_cols].T
        
        # Normalize for better visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        heatmap_normalized = scaler.fit_transform(heatmap_data.values)
        
        im = ax10.imshow(heatmap_normalized, cmap='RdYlGn', aspect='auto')
        ax10.set_xticks(np.arange(len(quantiles)))
        ax10.set_yticks(np.arange(len(heatmap_cols)))
        ax10.set_xticklabels(quantiles)
        ax10.set_yticklabels([c.replace('_mean', '').replace('_', ' ').title() 
                              for c in heatmap_cols])
        ax10.set_xlabel('Sensitivity Quantile', fontsize=12, fontweight='bold')
        ax10.set_title('Feature Comparison Across Quantiles (Normalized)', 
                      fontsize=13, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax10)
        cbar.set_label('Normalized Value', fontsize=10)
        
        # Add values on heatmap
        for i in range(len(heatmap_cols)):
            for j in range(len(quantiles)):
                text = ax10.text(j, i, f'{heatmap_normalized[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.suptitle('Comprehensive Quantile Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    print("✓ Created visualization dashboard")
    
    return fig

def main_analysis(df, output_dir='/mnt/user-data/outputs'):
    """
    Main analysis pipeline - simple NumPy-based approach
    """
    print("\n" + "="*80)
    print("MORTGAGE SENSITIVITY ANALYSIS - NUMPY IMPLEMENTATION")
    print("="*80)
    print(f"Dataset: {len(df):,} rows, {df['mtgnum'].nunique():,} mortgages")
    
    # Step 1: Visualize 200 curves
    fig1 = visualize_200_curves(df, n_mortgages=6)
    fig1.savefig(f'{output_dir}/sensitivity_curves_200pts.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Saved: sensitivity_curves_200pts.png")
    
    # Step 2: Calculate 1D sensitivity from 200 points
    sensitivity_1d = calculate_1d_sensitivity_numpy(df)
    
    # Step 3: Rank and create quantiles
    mtg_level = rank_and_create_quantiles(df, sensitivity_1d, n_quantiles=10)
    
    # Step 4: Analyze quantiles
    quantile_analysis = analyze_quantiles_numpy(mtg_level)
    
    # Step 5: Visualize
    fig2 = visualize_quantile_analysis(quantile_analysis, mtg_level)
    fig2.savefig(f'{output_dir}/quantile_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Saved: quantile_analysis_dashboard.png")
    
    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUT FILES")
    print("="*80)
    
    sensitivity_1d.to_csv(f'{output_dir}/sensitivity_1d.csv', index=False)
    print(f"✓ sensitivity_1d.csv")
    
    quantile_analysis.to_csv(f'{output_dir}/quantile_analysis.csv')
    print(f"✓ quantile_analysis.csv")
    
    mtg_level.to_csv(f'{output_dir}/mortgages_with_quantiles.csv', index=False)
    print(f"✓ mortgages_with_quantiles.csv")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    
    return {
        'sensitivity_1d': sensitivity_1d,
        'mtg_level': mtg_level,
        'quantile_analysis': quantile_analysis
    }