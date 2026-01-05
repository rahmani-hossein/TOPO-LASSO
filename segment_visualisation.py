"""
Visualize Renewal Probability Curves by Segment
================================================

This plots all 10 segments to see:
1. Overall curve shape
2. Where monotonicity violations occur (curve goes UP instead of DOWN)
3. Which segments need isotonic regression the most

Run this BEFORE smoothing to diagnose your data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_all_segments(df, save_path='segment_analysis.png'):
    """
    Plot renewal probability vs margin for all 10 segments.
    
    Highlights violations where higher margin → higher renewal probability
    (which is unrealistic and will be fixed by isotonic regression)
    
    Args:
        df: DataFrame with columns [segment, margin, renewal_probability]
        save_path: Where to save the plot
    """
    
    # Check required columns
    required = ['segment', 'margin', 'renewal_probability']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must have columns: {required}")
    
    # Get unique segments
    segments = sorted(df['segment'].unique())
    n_segments = len(segments)
    
    print(f"Plotting {n_segments} segments...")
    
    # Create subplot grid (2 rows x 5 columns for 10 segments)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Track statistics
    violation_stats = []
    
    for i, segment in enumerate(segments):
        ax = axes[i]
        
        # Get data for this segment
        seg_data = df[df['segment'] == segment].sort_values('margin')
        
        margins = seg_data['margin'].values
        probs = seg_data['renewal_probability'].values
        
        # Identify monotonicity violations
        # Violation = when prob[i+1] > prob[i] (goes UP instead of DOWN)
        violations = []
        for j in range(len(margins) - 1):
            if probs[j+1] > probs[j]:
                violations.append(j)
        
        n_violations = len(violations)
        pct_violations = 100 * n_violations / (len(margins) - 1)
        
        # Store stats
        violation_stats.append({
            'segment': segment,
            'n_violations': n_violations,
            'pct_violations': pct_violations,
            'total_points': len(margins)
        })
        
        # Plot the curve
        ax.plot(margins, probs, 'o-', linewidth=1, markersize=2, 
               alpha=0.6, color='steelblue', label='Actual')
        
        # Highlight violation regions in RED
        if violations:
            for v_idx in violations:
                ax.plot([margins[v_idx], margins[v_idx+1]], 
                       [probs[v_idx], probs[v_idx+1]], 
                       'r-', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Margin', fontsize=9)
        ax.set_ylabel('Renewal Probability', fontsize=9)
        ax.set_title(f'Segment {segment}\n{n_violations} violations ({pct_violations:.1f}%)', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add reference line for ideal monotonic decrease
        # (just for visual reference)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Overall title
    fig.suptitle('Renewal Probability Curves by Segment\n(Red = Monotonicity Violations)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend
    blue_patch = mpatches.Patch(color='steelblue', label='Predicted Curve')
    red_patch = mpatches.Patch(color='red', label='Violation (goes UP)')
    fig.legend(handles=[blue_patch, red_patch], loc='lower center', 
              ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.show()
    
    # Print violation statistics
    print("\n" + "="*70)
    print("MONOTONICITY VIOLATION ANALYSIS")
    print("="*70)
    
    stats_df = pd.DataFrame(violation_stats).sort_values('n_violations', ascending=False)
    print(stats_df.to_string(index=False))
    
    total_violations = stats_df['n_violations'].sum()
    total_points = stats_df['total_points'].sum()
    overall_pct = 100 * total_violations / total_points
    
    print(f"\nOverall: {total_violations} violations out of {total_points} transitions ({overall_pct:.2f}%)")
    
    # Identify worst segments
    worst_segments = stats_df[stats_df['pct_violations'] > 10]['segment'].tolist()
    if worst_segments:
        print(f"\n⚠ Segments with >10% violations: {worst_segments}")
        print("  → These need isotonic regression the most!")
    
    return stats_df


def plot_single_segment_detailed(df, segment_id, save_path=None):
    """
    Detailed plot for a single segment showing:
    1. Original curve
    2. Where violations occur
    3. What isotonic regression will do
    
    Args:
        df: DataFrame with columns [segment, margin, renewal_probability]
        segment_id: Which segment to plot (0-9)
        save_path: Optional path to save figure
    """
    
    from sklearn.isotonic import IsotonicRegression
    
    # Get data for this segment
    seg_data = df[df['segment'] == segment_id].sort_values('margin')
    
    if len(seg_data) == 0:
        print(f"No data found for segment {segment_id}")
        return
    
    margins = seg_data['margin'].values
    probs = seg_data['renewal_probability'].values
    
    # Apply isotonic regression
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    probs_iso = iso.fit_transform(margins, probs)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT PLOT: Before (with violations highlighted) =====
    ax1.plot(margins, probs, 'o-', linewidth=1.5, markersize=4, 
            alpha=0.6, color='steelblue', label='Original')
    
    # Highlight violations
    for i in range(len(margins) - 1):
        if probs[i+1] > probs[i]:
            ax1.plot([margins[i], margins[i+1]], 
                    [probs[i], probs[i+1]], 
                    'r-', linewidth=3, alpha=0.7)
            ax1.scatter([margins[i], margins[i+1]], 
                       [probs[i], probs[i+1]], 
                       color='red', s=50, zorder=5)
    
    ax1.set_xlabel('Margin', fontsize=11)
    ax1.set_ylabel('Renewal Probability', fontsize=11)
    ax1.set_title('BEFORE: Original Predictions\n(Red = Violations)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.legend()
    
    # ===== RIGHT PLOT: After isotonic regression =====
    ax2.plot(margins, probs, 'o', markersize=4, alpha=0.3, 
            color='lightgray', label='Original (ghosted)')
    ax2.plot(margins, probs_iso, 'o-', linewidth=2, markersize=4, 
            color='darkgreen', label='After Isotonic', alpha=0.8)
    
    ax2.set_xlabel('Margin', fontsize=11)
    ax2.set_ylabel('Renewal Probability', fontsize=11)
    ax2.set_title('AFTER: Isotonic Regression\n(Monotonic Decreasing)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    ax2.legend()
    
    # Overall title
    fig.suptitle(f'Segment {segment_id}: Isotonic Regression Effect', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()
    
    # Print statistics
    n_violations = sum(1 for i in range(len(probs)-1) if probs[i+1] > probs[i])
    max_violation = max([probs[i+1] - probs[i] for i in range(len(probs)-1) if probs[i+1] > probs[i]], 
                       default=0)
    
    print(f"\nSegment {segment_id} Statistics:")
    print(f"  Violations: {n_violations} out of {len(margins)-1} transitions")
    print(f"  Largest violation: {max_violation:.4f}")
    print(f"  Range before: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Range after:  [{probs_iso.min():.3f}, {probs_iso.max():.3f}]")


def compare_before_after_all_segments(df, smoothing_strength=0.03, save_path='before_after_comparison.png'):
    """
    Show all 10 segments in a 2x10 grid:
    - Top row: Original (noisy)
    - Bottom row: After isotonic + spline smoothing
    
    Args:
        df: DataFrame
        smoothing_strength: Spline parameter
        save_path: Where to save
    """
    
    from scipy.interpolate import UnivariateSpline
    from sklearn.isotonic import IsotonicRegression
    
    segments = sorted(df['segment'].unique())
    
    fig, axes = plt.subplots(2, 10, figsize=(25, 6))
    
    for i, segment in enumerate(segments):
        seg_data = df[df['segment'] == segment].sort_values('margin')
        
        margins = seg_data['margin'].values
        probs = seg_data['renewal_probability'].values
        
        # Apply smoothing
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        probs_iso = iso.fit_transform(margins, probs)
        
        try:
            spline = UnivariateSpline(margins, probs_iso, s=smoothing_strength, k=3)
            probs_smooth = np.clip(spline(margins), 0, 1)
        except:
            probs_smooth = probs_iso
        
        # Top row: Original
        axes[0, i].plot(margins, probs, 'o-', linewidth=1, markersize=1.5, 
                       alpha=0.5, color='steelblue')
        axes[0, i].set_ylim([0, 1])
        axes[0, i].set_title(f'Seg {segment}', fontsize=9)
        axes[0, i].grid(True, alpha=0.2)
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10, fontweight='bold')
        axes[0, i].set_xticks([])
        
        # Bottom row: Smoothed
        axes[1, i].plot(margins, probs_smooth, '-', linewidth=2, 
                       color='darkgreen', alpha=0.8)
        axes[1, i].set_ylim([0, 1])
        axes[1, i].grid(True, alpha=0.2)
        if i == 0:
            axes[1, i].set_ylabel('Smoothed', fontsize=10, fontweight='bold')
        axes[1, i].set_xlabel('Margin', fontsize=8)
    
    fig.suptitle('Before vs After: All Segments', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to: {save_path}")
    plt.show()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_analysis():
    """
    Example showing how to analyze your data.
    """
    
    print("="*70)
    print("EXAMPLE: Analyzing Renewal Probability Curves")
    print("="*70)
    
    # Create synthetic data (replace with your actual data)
    np.random.seed(42)
    
    n_mortgages = 100
    n_segments = 10
    n_margins = 200
    
    data = []
    for mort_id in range(n_mortgages):
        segment = mort_id % n_segments
        
        # Different segments have different behaviors
        sensitivity = 1.0 + (segment / n_segments) * 2.0
        
        for margin in np.linspace(0, 2, n_margins):
            # True curve (decreasing)
            true_prob = 0.95 * np.exp(-sensitivity * margin)
            
            # Add noise (this creates violations!)
            noise = np.random.normal(0, 0.05)
            renewal_prob = np.clip(true_prob + noise, 0, 1)
            
            data.append({
                'mortgage_id': f'M{mort_id:04d}',
                'segment': segment,
                'margin': margin,
                'renewal_probability': renewal_prob
            })
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} rows for analysis\n")
    
    # 1. Plot all segments to see violations
    print("1. Plotting all segments...")
    stats = plot_all_segments(df, save_path='all_segments_violations.png')
    
    # 2. Detailed view of worst segment
    print("\n2. Detailed analysis of worst segment...")
    worst_segment = stats.iloc[0]['segment']
    plot_single_segment_detailed(df, segment_id=worst_segment, 
                                save_path=f'segment_{worst_segment}_detailed.png')
    
    # 3. Before/after comparison
    print("\n3. Before/after comparison for all segments...")
    compare_before_after_all_segments(df, smoothing_strength=0.03, 
                                     save_path='before_after_all.png')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. all_segments_violations.png - Overview of all segments")
    print(f"  2. segment_{worst_segment}_detailed.png - Detailed view of worst segment")
    print("  3. before_after_all.png - Before/after comparison")


# =============================================================================
# YOUR ACTUAL USAGE
# =============================================================================

"""
HOW TO USE WITH YOUR DATA:

import pandas as pd
from segment_visualization import plot_all_segments, plot_single_segment_detailed

# Load your data
df = pd.read_csv('your_data.csv')

# Must have columns: segment, margin, renewal_probability
# segment should be 0-9 (10 values)
# margin should have 200 values per segment

# 1. Plot all segments to see violations
stats = plot_all_segments(df, save_path='my_segment_analysis.png')

# 2. Look at a specific segment in detail (e.g., segment 3)
plot_single_segment_detailed(df, segment_id=3, save_path='segment_3_detail.png')

# 3. See before/after for all segments
compare_before_after_all_segments(df, save_path='my_before_after.png')
"""

if __name__ == '__main__':
    # Run example
    example_analysis()