"""
SIMPLE & FAST Segment Visualization
====================================

Minimal code, maximum speed.
Just shows your curves - no fancy calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# FUNCTION 1: Quick plot of all 10 segments (SIMPLE)
# =============================================================================

def plot_segments_simple(df, save_path='segments.png'):
    """
    Dead simple: plot all 10 segments.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        save_path: Where to save
    """
    
    segments = sorted(df['segment'].unique())
    
    # Create 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, seg in enumerate(segments):
        # Get data for this segment
        data = df[df['segment'] == seg].sort_values('margin')
        
        # Simple plot
        axes[i].plot(data['margin'], data['prediction'], 'o-', 
                    markersize=1, linewidth=0.5, alpha=0.7)
        axes[i].set_title(f'Segment {seg}', fontsize=9)
        axes[i].set_ylim([0, 1])
        axes[i].grid(alpha=0.3)
        
        # Only label outer plots
        if i >= 5:
            axes[i].set_xlabel('Margin', fontsize=8)
        if i % 5 == 0:
            axes[i].set_ylabel('Prediction', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"✓ Saved to {save_path}")


# =============================================================================
# FUNCTION 2: Plot just ONE segment (FASTEST)
# =============================================================================

def plot_one_segment(df, segment_id, save_path=None):
    """
    Plot a single segment. Super fast.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        segment_id: Which segment (0-9)
        save_path: Optional save path
    """
    
    # Get data
    data = df[df['segment'] == segment_id].sort_values('margin')
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(data['margin'], data['prediction'], 'o-', 
            markersize=3, linewidth=1, alpha=0.7)
    plt.xlabel('Margin', fontsize=11)
    plt.ylabel('Prediction (Renewal Probability)', fontsize=11)
    plt.title(f'Segment {segment_id}', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# FUNCTION 3: Show violations (RED highlights)
# =============================================================================

def plot_with_violations(df, segment_id, save_path=None):
    """
    Plot one segment with violations highlighted in RED.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        segment_id: Which segment (0-9)
        save_path: Optional save path
    """
    
    # Get data
    data = df[df['segment'] == segment_id].sort_values('margin')
    margins = data['margin'].values
    probs = data['prediction'].values
    
    # Plot base curve
    plt.figure(figsize=(8, 5))
    plt.plot(margins, probs, 'o-', markersize=3, linewidth=1, 
            alpha=0.5, color='blue', label='Curve')
    
    # Highlight violations (where it goes UP)
    for i in range(len(margins) - 1):
        if probs[i+1] > probs[i]:
            plt.plot([margins[i], margins[i+1]], 
                    [probs[i], probs[i+1]], 
                    'r-', linewidth=2, alpha=0.8)
    
    plt.xlabel('Margin', fontsize=11)
    plt.ylabel('Prediction (Renewal Probability)', fontsize=11)
    plt.title(f'Segment {segment_id} (Red = Goes UP)', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# FUNCTION 4: Before/After comparison (ONE segment)
# =============================================================================

def plot_before_after(df, segment_id, save_path=None):
    """
    Show before (noisy) and after (smooth) for one segment.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
        segment_id: Which segment (0-9)
        save_path: Optional save path
    """
    
    from scipy.interpolate import UnivariateSpline
    from sklearn.isotonic import IsotonicRegression
    
    # Get data
    data = df[df['segment'] == segment_id].sort_values('margin')
    margins = data['margin'].values
    probs = data['prediction'].values
    
    # Apply smoothing
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    probs_iso = iso.fit_transform(margins, probs)
    
    try:
        spline = UnivariateSpline(margins, probs_iso, s=0.03, k=3)
        probs_smooth = np.clip(spline(margins), 0, 1)
    except:
        probs_smooth = probs_iso
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.plot(margins, probs, 'o', markersize=3, alpha=0.3, 
            color='gray', label='Original')
    plt.plot(margins, probs_smooth, '-', linewidth=2, 
            color='green', label='Smoothed')
    
    plt.xlabel('Margin', fontsize=11)
    plt.ylabel('Prediction (Renewal Probability)', fontsize=11)
    plt.title(f'Segment {segment_id}: Before → After', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# FUNCTION 5: Count violations (FAST - no plotting)
# =============================================================================

def count_violations(df):
    """
    Just count violations per segment. No plotting.
    Super fast.
    
    Args:
        df: DataFrame with columns [segment, margin, prediction]
    
    Returns:
        DataFrame with violation counts
    """
    
    results = []
    
    for seg in sorted(df['segment'].unique()):
        data = df[df['segment'] == seg].sort_values('margin')
        probs = data['prediction'].values
        
        # Count violations
        violations = sum(1 for i in range(len(probs)-1) if probs[i+1] > probs[i])
        pct = 100 * violations / (len(probs) - 1) if len(probs) > 1 else 0
        
        results.append({
            'segment': seg,
            'violations': violations,
            'percent': pct
        })
    
    stats = pd.DataFrame(results).sort_values('violations', ascending=False)
    
    print("\nViolation Counts:")
    print(stats.to_string(index=False))
    
    return stats


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
EXAMPLE 1: Quick plot of all segments
--------------------------------------
import pandas as pd
from simple_viz import plot_segments_simple

df = pd.read_csv('your_data.csv')
# df must have columns: segment, margin, prediction
plot_segments_simple(df, save_path='all_segments.png')


EXAMPLE 2: Look at one segment
-------------------------------
from simple_viz import plot_one_segment

plot_one_segment(df, segment_id=0, save_path='segment_0.png')


EXAMPLE 3: See violations for one segment
------------------------------------------
from simple_viz import plot_with_violations

plot_with_violations(df, segment_id=0, save_path='seg_0_violations.png')


EXAMPLE 4: Before/after for one segment
----------------------------------------
from simple_viz import plot_before_after

plot_before_after(df, segment_id=0, save_path='seg_0_smooth.png')


EXAMPLE 5: Just count violations (no plot)
-------------------------------------------
from simple_viz import count_violations

stats = count_violations(df)
"""


# =============================================================================
# RUN ALL (if you want everything at once)
# =============================================================================

def analyze_all_segments(df, output_dir='.'):
    """
    Run all analyses. Creates multiple files.
    
    Args:
        df: Your dataframe with columns [segment, margin, prediction]
        output_dir: Where to save files
    """
    
    import os
    
    print("Analyzing segments...")
    
    # 1. Count violations (fast - no plotting)
    print("\n1. Counting violations...")
    stats = count_violations(df)
    
    # 2. Plot all segments overview
    print("\n2. Plotting all segments...")
    plot_segments_simple(df, save_path=f'{output_dir}/all_segments.png')
    
    # 3. Detailed plots for worst 3 segments
    print("\n3. Detailed plots for worst segments...")
    worst_3 = stats.head(3)['segment'].tolist()
    
    for seg in worst_3:
        seg = int(seg)
        plot_with_violations(df, seg, f'{output_dir}/segment_{seg}_violations.png')
        plot_before_after(df, seg, f'{output_dir}/segment_{seg}_smooth.png')
    
    print(f"\n✓ Done! Check {output_dir}/ for files")
    
    return stats


# =============================================================================
# SIMPLEST POSSIBLE USAGE
# =============================================================================

if __name__ == '__main__':
    """
    Just run this file:
    
    python simple_viz.py
    
    (Edit the filename below)
    """
    
    import pandas as pd
    import sys
    
    # EDIT THIS LINE WITH YOUR FILENAME:
    filename = 'test_data.csv' if len(sys.argv) == 1 else sys.argv[1]
    
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows, {df['segment'].nunique()} segments")
    
    # Run analysis
    stats = analyze_all_segments(df)
    
    print("\n" + "="*50)
    print("Files created:")
    print("  - all_segments.png")
    print("  - segment_X_violations.png (for worst 3)")
    print("  - segment_X_smooth.png (for worst 3)")
    print("="*50)