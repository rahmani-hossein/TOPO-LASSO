"""
Simple Segment Visualization - All Segments on One Plot
========================================================

Shows all 10 segments as colored curves on a single plot.
Each curve shows the MEAN prediction at each margin for that segment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_segments_together(df, save_path='segments_plot.png', show_plot=True):
    """
    Plot all 10 segments on one graph, each with its own color.
    
    Each curve shows the mean prediction at each margin for that segment.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        save_path: Where to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        DataFrame with aggregated data (segment, margin, mean_prediction)
    """
    
    print("Creating plot with all segments...")
    
    # Calculate mean prediction for each (segment, margin) combination
    agg_data = df.groupby(['segment', 'margin']).agg({
        'prediction': 'mean'
    }).reset_index()
    
    agg_data = agg_data.rename(columns={'prediction': 'mean_prediction'})
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Define colors for each segment
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot each segment
    segments = sorted(agg_data['segment'].unique())
    
    for i, seg in enumerate(segments):
        seg_data = agg_data[agg_data['segment'] == seg].sort_values('margin')
        
        plt.plot(seg_data['margin'], 
                seg_data['mean_prediction'], 
                linewidth=2, 
                label=f'Segment {seg}',
                color=colors[i],
                alpha=0.8)
    
    # Formatting
    plt.xlabel('Margin', fontsize=12)
    plt.ylabel('Mean Prediction (Renewal Probability)', fontsize=12)
    plt.title('Prediction Curves by Segment\n(Mean prediction at each margin)', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return agg_data


def plot_segments_with_smoothing(df, smoothing_strength=0.03, 
                                 save_path='segments_smooth.png', show_plot=True):
    """
    Plot all segments with both original and smoothed curves.
    
    Shows:
    - Dotted lines: Original mean predictions
    - Solid lines: Smoothed predictions
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        smoothing_strength: Spline smoothing parameter
        save_path: Where to save
        show_plot: Whether to display
    """
    
    from scipy.interpolate import UnivariateSpline
    from sklearn.isotonic import IsotonicRegression
    
    print("Creating plot with smoothing...")
    
    # Calculate mean prediction for each (segment, margin)
    agg_data = df.groupby(['segment', 'margin']).agg({
        'prediction': 'mean'
    }).reset_index()
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    segments = sorted(agg_data['segment'].unique())
    
    for i, seg in enumerate(segments):
        seg_data = agg_data[agg_data['segment'] == seg].sort_values('margin')
        
        margins = seg_data['margin'].values
        probs = seg_data['prediction'].values
        
        # Apply smoothing
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        probs_iso = iso.fit_transform(margins, probs)
        
        try:
            spline = UnivariateSpline(margins, probs_iso, s=smoothing_strength, k=3)
            probs_smooth = np.clip(spline(margins), 0, 1)
        except:
            probs_smooth = probs_iso
        
        # Plot original (dotted)
        plt.plot(margins, probs, '--', 
                linewidth=1, color=colors[i], alpha=0.4)
        
        # Plot smoothed (solid)
        plt.plot(margins, probs_smooth, '-', 
                linewidth=2.5, color=colors[i], alpha=0.9,
                label=f'Segment {seg}')
    
    # Formatting
    plt.xlabel('Margin', fontsize=12)
    plt.ylabel('Mean Prediction (Renewal Probability)', fontsize=12)
    plt.title('Smoothed Prediction Curves by Segment\n(Solid = Smoothed, Dotted = Original)', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_segments_interactive(df, save_path='segments_interactive.png', show_plot=True):
    """
    Plot with larger figure and better readability.
    Good for presentations or detailed analysis.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        save_path: Where to save
        show_plot: Whether to display
    """
    
    print("Creating interactive-style plot...")
    
    # Aggregate
    agg_data = df.groupby(['segment', 'margin']).agg({
        'prediction': 'mean'
    }).reset_index()
    
    # Create larger figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Nice colors
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    segments = sorted(agg_data['segment'].unique())
    
    for i, seg in enumerate(segments):
        seg_data = agg_data[agg_data['segment'] == seg].sort_values('margin')
        
        ax.plot(seg_data['margin'], 
               seg_data['prediction'], 
               linewidth=3, 
               label=f'Segment {seg}',
               color=colors[i],
               marker='o',
               markersize=2,
               alpha=0.85)
    
    # Formatting
    ax.set_xlabel('Margin', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Prediction (Renewal Probability)', fontsize=14, fontweight='bold')
    ax.set_title('Prediction Curves by Segment', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    # Add minor gridlines
    ax.grid(True, which='minor', alpha=0.1)
    ax.minorticks_on()
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_segments_stats(df):
    """
    Show statistics about each segment's prediction curve.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
    
    Returns:
        DataFrame with segment statistics
    """
    
    print("\nSegment Statistics:")
    print("="*70)
    
    stats = []
    
    for seg in sorted(df['segment'].unique()):
        seg_data = df[df['segment'] == seg].sort_values('margin')
        
        # Calculate mean prediction curve
        mean_curve = seg_data.groupby('margin')['prediction'].mean()
        
        # Count violations
        violations = sum(1 for i in range(len(mean_curve)-1) 
                        if mean_curve.iloc[i+1] > mean_curve.iloc[i])
        
        stats.append({
            'segment': seg,
            'mean_prediction': seg_data['prediction'].mean(),
            'std_prediction': seg_data['prediction'].std(),
            'min_prediction': seg_data['prediction'].min(),
            'max_prediction': seg_data['prediction'].max(),
            'violations_in_mean_curve': violations,
            'n_points': len(seg_data)
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    return stats_df


# =============================================================================
# COMPLETE ANALYSIS
# =============================================================================

def analyze_segments(df, smoothing_strength=0.03, output_dir='.'):
    """
    Complete analysis with multiple visualizations.
    
    Args:
        df: DataFrame with [segment, margin, prediction]
        smoothing_strength: For smoothed plot
        output_dir: Where to save files
    
    Returns:
        Dictionary with stats and aggregated data
    """
    
    print("="*70)
    print("SEGMENT ANALYSIS")
    print("="*70)
    
    # 1. Statistics
    print("\n1. Computing statistics...")
    stats = compare_segments_stats(df)
    
    # 2. Basic plot
    print("\n2. Creating basic plot...")
    agg_data = plot_all_segments_together(df, 
                                          save_path=f'{output_dir}/segments_all.png',
                                          show_plot=False)
    
    # 3. Smoothed plot
    print("\n3. Creating smoothed plot...")
    plot_segments_with_smoothing(df, 
                                smoothing_strength=smoothing_strength,
                                save_path=f'{output_dir}/segments_smoothed.png',
                                show_plot=False)
    
    # 4. Interactive style
    print("\n4. Creating presentation-quality plot...")
    plot_segments_interactive(df, 
                             save_path=f'{output_dir}/segments_presentation.png',
                             show_plot=False)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nFiles created in {output_dir}:")
    print("  1. segments_all.png - All segments on one plot")
    print("  2. segments_smoothed.png - With smoothing applied")
    print("  3. segments_presentation.png - High quality for presentations")
    
    return {
        'stats': stats,
        'aggregated_data': agg_data
    }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
EXAMPLE 1: Quick plot
---------------------
import pandas as pd
from plot_segments import plot_all_segments_together

df = pd.read_csv('your_data.csv')
# Must have: segment, margin, prediction

plot_all_segments_together(df, save_path='my_plot.png')


EXAMPLE 2: With smoothing
--------------------------
from plot_segments import plot_segments_with_smoothing

plot_segments_with_smoothing(df, smoothing_strength=0.03, 
                            save_path='smooth_plot.png')


EXAMPLE 3: Complete analysis
-----------------------------
from plot_segments import analyze_segments

results = analyze_segments(df, smoothing_strength=0.03)
# Creates 3 plots + statistics


EXAMPLE 4: Just statistics
---------------------------
from plot_segments import compare_segments_stats

stats = compare_segments_stats(df)
print(stats)
"""


if __name__ == '__main__':
    """
    Test with sample data
    """
    
    import pandas as pd
    import sys
    
    # Create or load test data
    if len(sys.argv) > 1:
        # Load from file
        filename = sys.argv[1]
        print(f"Loading {filename}...")
        df = pd.read_csv(filename)
    else:
        # Create sample data
        print("Creating sample data...")
        np.random.seed(42)
        
        data = []
        for seg in range(10):
            # Each segment has different elasticity
            elasticity = 1.0 + (seg / 10) * 2.0
            
            # Multiple mortgages per segment
            for mort in range(20):
                for margin in np.linspace(0, 2, 200):
                    pred = 0.9 * np.exp(-elasticity * margin) + np.random.normal(0, 0.05)
                    pred = np.clip(pred, 0, 1)
                    
                    data.append({
                        'segment': seg,
                        'margin': margin,
                        'prediction': pred
                    })
        
        df = pd.DataFrame(data)
    
    print(f"Loaded {len(df)} rows, {df['segment'].nunique()} segments")
    
    # Run analysis
    results = analyze_segments(df, smoothing_strength=0.03)
    
    print("\n✓ Done! Check the PNG files.")