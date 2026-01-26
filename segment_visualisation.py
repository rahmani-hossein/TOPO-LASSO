import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline

# ============================================================================
# SEGMENT-LEVEL CURVE SMOOTHER
# ============================================================================

class SegmentCurveSmoother:
    """
    Smooth segment-level renewal probability curves
    
    Input: Segment-level data (already aggregated)
    - Each segment has 200 margin points
    - One probability value per margin
    
    Methods:
    1. isotonic_only: Simple monotonic smoothing
    2. isotonic_anchored: Isotonic + anchors at 0.25 and 1.75 + spline
    """
    
    def isotonic_only(self, margins, probs):
        """
        Method 1: Isotonic regression only
        
        Parameters
        ----------
        margins : array, shape (200,)
            Margin values for this segment
        probs : array, shape (200,)
            Average renewal probability at each margin
            
        Returns
        -------
        smoothed : array, shape (200,)
        """
        iso = IsotonicRegression(out_of_bounds='clip')
        return iso.fit_transform(margins, probs)
    
    def isotonic_anchored(self, margins, probs, 
                         anchor_low_margin=0.25,
                         anchor_high_margin=1.75, 
                         anchor_high_prob=0.05):
        """
        Method 2: Isotonic + anchors + spline
        
        Adds two anchors:
        - Low anchor at margin=0.25 (uses interpolated probability)
        - High anchor at margin=1.75 with prob=0.05
        
        Then applies spline smoothing for realistic start/end behavior
        
        Parameters
        ----------
        margins : array, shape (200,)
        probs : array, shape (200,)
        anchor_low_margin : float
            Low margin anchor point (default: 0.25)
        anchor_high_margin : float
            High margin anchor point (default: 1.75)
        anchor_high_prob : float
            Target probability at high anchor (default: 0.05)
            
        Returns
        -------
        smoothed : array, shape (200,)
        """
        # Add anchors
        m_with_anchors = list(margins)
        p_with_anchors = list(probs)
        
        # Anchor 1: Low margin (use interpolated value)
        if anchor_low_margin not in margins:
            p_low = np.interp(anchor_low_margin, margins, probs)
            idx = np.searchsorted(m_with_anchors, anchor_low_margin)
            m_with_anchors.insert(idx, anchor_low_margin)
            p_with_anchors.insert(idx, p_low)
        
        # Anchor 2: High margin (force specific probability)
        if anchor_high_margin not in margins:
            idx = np.searchsorted(m_with_anchors, anchor_high_margin)
            m_with_anchors.insert(idx, anchor_high_margin)
            p_with_anchors.insert(idx, anchor_high_prob)
        
        # Anchor 3: End point for smooth decay
        if m_with_anchors[-1] < 2.0:
            m_with_anchors.append(2.0)
            p_with_anchors.append(0.02)
        
        m_with_anchors = np.array(m_with_anchors)
        p_with_anchors = np.array(p_with_anchors)
        
        # Apply isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip')
        p_monotonic = iso.fit_transform(m_with_anchors, p_with_anchors)
        
        # Apply spline for smooth start/end
        spline = UnivariateSpline(m_with_anchors, p_monotonic, s=0.01, k=3)
        p_smooth = spline(margins)
        
        # Re-enforce monotonicity (spline might violate slightly)
        for i in range(1, len(p_smooth)):
            p_smooth[i] = min(p_smooth[i], p_smooth[i-1])
        
        # Clip to valid range
        return np.clip(p_smooth, 0.001, 0.999)


# ============================================================================
# EXAMPLE DATA (SEGMENT LEVEL)
# ============================================================================

def create_segment_data():
    """
    Create segment-level data
    
    Returns DataFrame with:
    - segment: 1-5
    - margin: 200 values from 0 to 2
    - prob: average renewal probability at that margin
    """
    np.random.seed(42)
    
    margins = np.linspace(0, 2, 200)
    data = []
    
    for segment in range(1, 6):
        # Segment-level curve (noisy exponential decay)
        decay_rate = 0.9 + 0.4 * (segment / 5)
        base = 0.95 * np.exp(-decay_rate * margins)
        noise = 0.03 * np.sin(8 * margins) + 0.02 * np.random.randn(len(margins))
        probs = base + noise
        probs = np.clip(probs, 0.1, 0.99)
        
        for m, p in zip(margins, probs):
            data.append({
                'segment': segment,
                'margin': m,
                'prob': p
            })
    
    return pd.DataFrame(data)


# ============================================================================
# APPLY TO DATAFRAME
# ============================================================================

def smooth_segment_dataframe(df, smoother, method='isotonic_anchored'):
    """
    Apply smoothing to segment-level dataframe
    
    Parameters
    ----------
    df : DataFrame
        Columns: segment, margin, prob
    smoother : SegmentCurveSmoother instance
    method : str
        'isotonic_only' or 'isotonic_anchored'
        
    Returns
    -------
    df_smoothed : DataFrame
        Original df + 'smoothed_prob' column
    """
    results = []
    
    for segment_id, seg_df in df.groupby('segment'):
        seg_df = seg_df.sort_values('margin').copy()
        
        margins = seg_df['margin'].values
        probs = seg_df['prob'].values
        
        # Apply smoothing
        if method == 'isotonic_only':
            smoothed = smoother.isotonic_only(margins, probs)
        elif method == 'isotonic_anchored':
            smoothed = smoother.isotonic_anchored(margins, probs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        seg_df['smoothed_prob'] = smoothed
        results.append(seg_df)
    
    return pd.concat(results, ignore_index=True)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(df_original, df_method1, df_method2):
    """Compare both methods side by side"""
    
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    
    for idx, segment in enumerate(range(1, 6)):
        ax = axes[idx]
        
        # Get data
        orig = df_original[df_original['segment'] == segment].sort_values('margin')
        m1 = df_method1[df_method1['segment'] == segment].sort_values('margin')
        m2 = df_method2[df_method2['segment'] == segment].sort_values('margin')
        
        # Plot
        ax.plot(orig['margin'], orig['prob'], 
               'gray', linewidth=2, alpha=0.4, label='Original')
        ax.plot(m1['margin'], m1['smoothed_prob'],
               'r--', linewidth=2.5, alpha=0.7, label='Isotonic only')
        ax.plot(m2['margin'], m2['smoothed_prob'],
               'b-', linewidth=3, alpha=0.8, label='Isotonic + Anchors')
        
        # Show anchor locations
        ax.axvline(0.25, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axvline(1.75, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axhline(0.05, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax.set_title(f'Segment {segment}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Margin', fontsize=10)
        ax.set_ylabel('Renewal Probability', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
    
    plt.suptitle('Method Comparison: Isotonic Only vs Isotonic + Anchors (0.25, 1.75) + Spline',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/segment_smoothing_comparison.png', dpi=150)
    plt.show()
    print("✓ Comparison plot saved!")


def plot_detailed_view(df_method2, segment=3):
    """Detailed view of one segment showing the effect"""
    
    seg_df = df_method2[df_method2['segment'] == segment].sort_values('margin')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Before and after
    ax1.scatter(seg_df['margin'], seg_df['prob'], 
               c='gray', s=20, alpha=0.5, label='Original', zorder=1)
    ax1.plot(seg_df['margin'], seg_df['smoothed_prob'],
            'b-', linewidth=3, label='Smoothed', zorder=2)
    
    # Highlight anchor regions
    ax1.axvspan(0.20, 0.30, alpha=0.1, color='orange', label='Low anchor region')
    ax1.axvspan(1.70, 1.80, alpha=0.1, color='red', label='High anchor region')
    ax1.axhline(0.05, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    ax1.set_title(f'Segment {segment}: Before and After', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Margin', fontsize=11)
    ax1.set_ylabel('Renewal Probability', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Derivative (smoothness check)
    derivative = -np.diff(seg_df['smoothed_prob'].values)  # Should be positive (decreasing)
    margins_mid = seg_df['margin'].values[:-1]
    
    ax2.plot(margins_mid, derivative, 'g-', linewidth=2.5)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(margins_mid, 0, derivative, alpha=0.3, color='green')
    
    ax2.set_title('Derivative: -dP/dM (Should be ≥ 0)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Margin', fontsize=11)
    ax2.set_ylabel('-dP/dM', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/detailed_segment_view.png', dpi=150)
    plt.show()
    print("✓ Detailed view saved!")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_smoothing(df_smoothed):
    """Check quality of smoothing"""
    
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    for segment in df_smoothed['segment'].unique():
        seg_df = df_smoothed[df_smoothed['segment'] == segment].sort_values('margin')
        
        probs = seg_df['smoothed_prob'].values
        margins = seg_df['margin'].values
        
        # Check monotonicity
        violations = np.sum(np.diff(probs) > 1e-6)
        
        # Check probability at anchor points
        prob_at_025 = seg_df[np.abs(seg_df['margin'] - 0.25) < 0.01]['smoothed_prob'].iloc[0]
        prob_at_175 = seg_df[np.abs(seg_df['margin'] - 1.75) < 0.01]['smoothed_prob'].iloc[0]
        
        # Check smoothness (average absolute second derivative)
        second_deriv = np.abs(np.diff(probs, n=2))
        avg_wiggle = second_deriv.mean()
        
        status = "✓" if violations == 0 else "⚠"
        
        print(f"{status} Segment {segment}: "
              f"violations={violations}, "
              f"p@0.25={prob_at_025:.3f}, "
              f"p@1.75={prob_at_175:.3f}, "
              f"wiggle={avg_wiggle:.5f}")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete comparison"""
    
    print("\n" + "="*80)
    print("SEGMENT-LEVEL CURVE SMOOTHING")
    print("="*80)
    print()
    print("Input: Segment-level data (each segment has 200 margin points)")
    print("Output: Smoothed segment curves")
    print()
    print("Method 1: Isotonic regression only")
    print("Method 2: Isotonic + Anchors (0.25, 1.75) + Spline")
    print("="*80 + "\n")
    
    # Create segment-level data
    print("Creating segment-level data...")
    df = create_segment_data()
    print(f"✓ {len(df)} rows")
    print(f"  - {df['segment'].nunique()} segments")
    print(f"  - {len(df[df['segment']==1])} margin points per segment\n")
    
    # Initialize smoother
    smoother = SegmentCurveSmoother()
    
    # Method 1: Isotonic only
    print("Applying Method 1: Isotonic only...")
    df_method1 = smooth_segment_dataframe(df, smoother, method='isotonic_only')
    print("✓ Done\n")
    
    # Method 2: Isotonic + anchors
    print("Applying Method 2: Isotonic + Anchors + Spline...")
    df_method2 = smooth_segment_dataframe(df, smoother, method='isotonic_anchored')
    print("✓ Done\n")
    
    # Validate
    print("Validating Method 2 (Isotonic + Anchors)...")
    validate_smoothing(df_method2)
    
    # Visualize
    print("Creating visualizations...")
    plot_comparison(df, df_method1, df_method2)
    plot_detailed_view(df_method2, segment=3)
    
    # Save
    print("Saving results...")
    df_method1.to_csv('/mnt/user-data/outputs/isotonic_only.csv', index=False)
    df_method2.to_csv('/mnt/user-data/outputs/isotonic_anchored.csv', index=False)
    print("✓ Saved to CSV\n")
    
    print("="*80)
    print("RECOMMENDATION: Use Method 2 (Isotonic + Anchors)")
    print("="*80)
    print()
    print("Why?")
    print("✓ Enforces realistic boundary behavior")
    print("✓ Anchor at 0.25: stabilizes high-probability region")
    print("✓ Anchor at 1.75 → 5%: forces realistic decline")
    print("✓ Spline smoothing: natural transitions at start/end")
    print("✓ Perfect monotonicity maintained")
    print()
    print("="*80 + "\n")
    
    return df_method1, df_method2


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Run comparison
    df_m1, df_m2 = main()
    
    print("\n" + "="*80)
    print("USAGE WITH YOUR DATA")
    print("="*80)
    print("""
# Your segment-level data
df_segment = pd.DataFrame({
    'segment': [1, 1, 1, ...],      # Segment ID
    'margin': [0.0, 0.01, 0.02, ...],  # 200 margin values
    'prob': [0.95, 0.94, 0.93, ...]    # Average renewal prob
})

# Initialize smoother
smoother = SegmentCurveSmoother()

# Apply smoothing
df_smoothed = smooth_segment_dataframe(
    df_segment, 
    smoother, 
    method='isotonic_anchored'  # Recommended!
)

# Result: df_smoothed has 'smoothed_prob' column
# Use this for your optimization!
    """)
    print("="*80)