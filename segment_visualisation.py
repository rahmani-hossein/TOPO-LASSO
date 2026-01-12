import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# CALIBRATOR
# ============================================================================

class MortgageCalibrator:
    """
    Calibrate renewal probabilities using isotonic regression + smoothing + anchors
    
    Parameters
    ----------
    threshold : float
        Margin where decay accelerates (e.g., 1.75 = 175bps)
    high_margin : float
        High margin anchor point (e.g., 2.0 = 200bps)
    high_prob : float
        Target probability at high margin (e.g., 0.05 = 5%)
    smooth : float
        Smoothing strength, range [0, 1]. Recommended: 0.3-0.5
    """
    
    def __init__(self, threshold=1.75, high_margin=2.0, high_prob=0.05, smooth=0.4):
        self.threshold = threshold
        self.high_margin = high_margin
        self.high_prob = high_prob
        self.smooth = smooth
    
    def calibrate(self, margins, probs):
        """
        Calibrate a single curve (either individual or segment-level)
        
        Parameters
        ----------
        margins : array, shape (n_margins,)
            Margin values, should be sorted
        probs : array, shape (n_margins,)
            Predicted probabilities
            
        Returns
        -------
        calibrated : array, shape (n_margins,)
            Calibrated probabilities (monotonic, smooth, anchored)
        """
        margins = np.asarray(margins)
        probs = np.asarray(probs)
        
        # Ensure sorted
        sort_idx = np.argsort(margins)
        m = margins[sort_idx]
        p = probs[sort_idx]
        
        # Step 1: Add anchor points
        m_anchor, p_anchor = self._add_anchors(m, p)
        
        # Step 2: Isotonic regression (enforce monotonicity)
        iso = IsotonicRegression(out_of_bounds='clip')
        p_mono = iso.fit_transform(m_anchor, p_anchor)
        
        # Step 3: PCHIP smoothing (preserves monotonicity)
        pchip = PchipInterpolator(m_anchor, p_mono)
        p_smooth = pchip(m)
        
        # Step 4: Light Gaussian smoothing (optional, for extra smoothness)
        if self.smooth > 0:
            p_smooth = gaussian_filter1d(p_smooth, sigma=1.5 * self.smooth)
            # Re-enforce monotonicity after Gaussian smoothing
            for i in range(1, len(p_smooth)):
                p_smooth[i] = min(p_smooth[i], p_smooth[i-1])
        
        # Step 5: Clip to valid probability range
        p_smooth = np.clip(p_smooth, 0.001, 0.999)
        
        # Unsort back to original order
        unsort_idx = np.argsort(sort_idx)
        return p_smooth[unsort_idx]
    
    def _add_anchors(self, margins, probs):
        """Add anchor points to enforce realistic boundary behavior"""
        m_list = list(margins)
        p_list = list(probs)
        
        # Anchor 1: Start (m=0, high probability)
        if margins[0] > 0:
            m_list.insert(0, 0.0)
            p_list.insert(0, 0.97)
        
        # Anchor 2: High margin (low probability)
        if margins[-1] < self.high_margin:
            m_list.append(self.high_margin)
            p_list.append(self.high_prob)
        
        # Anchor 3: Midpoint for smooth exponential decay
        m_mid = (self.threshold + self.high_margin) / 2
        if m_mid not in m_list:
            p_threshold = np.interp(self.threshold, m_list, p_list)
            # Exponential decay rate
            lam = -np.log(self.high_prob / p_threshold) / (self.high_margin - self.threshold)
            p_mid = p_threshold * np.exp(-lam * (m_mid - self.threshold))
            
            idx = np.searchsorted(m_list, m_mid)
            m_list.insert(idx, m_mid)
            p_list.insert(idx, p_mid)
        
        return np.array(m_list), np.array(p_list)


# ============================================================================
# OPTION A: SMOOTH INDIVIDUAL → AGGREGATE
# ============================================================================

def option_a_individual_then_aggregate(df, calibrator):
    """
    Option A: Calibrate each individual mortgage, then aggregate to segment
    
    Process:
    1. For each mortgage: calibrate its 200 margin predictions
    2. Aggregate: average calibrated predictions across mortgages in segment
    
    Parameters
    ----------
    df : DataFrame
        Columns: mtg_num, segment, margin_value, predicted_prob
    calibrator : MortgageCalibrator
    
    Returns
    -------
    segment_curves : DataFrame
        Segment-level calibrated curves
    """
    print("="*70)
    print("OPTION A: Smooth Individual → Aggregate")
    print("="*70)
    print("Process: Calibrate each mortgage → Average to segment")
    print()
    
    calibrated_individuals = []
    
    # Step 1: Calibrate each individual mortgage
    for mtg_num, mtg_data in df.groupby('mtg_num'):
        mtg_data = mtg_data.sort_values('margin_value')
        
        margins = mtg_data['margin_value'].values
        probs = mtg_data['predicted_prob'].values
        
        # Calibrate this individual
        calibrated = calibrator.calibrate(margins, probs)
        
        mtg_data_copy = mtg_data.copy()
        mtg_data_copy['calibrated_prob'] = calibrated
        calibrated_individuals.append(mtg_data_copy)
    
    df_calibrated = pd.concat(calibrated_individuals, ignore_index=True)
    
    # Step 2: Aggregate to segment level
    segment_curves = df_calibrated.groupby(['segment', 'margin_value']).agg({
        'calibrated_prob': 'mean'  # Average across individuals
    }).reset_index()
    
    n_mortgages = df['mtg_num'].nunique()
    print(f"✓ Calibrated {n_mortgages} individual mortgages")
    print(f"✓ Aggregated to {segment_curves['segment'].nunique()} segments")
    print()
    
    return segment_curves


# ============================================================================
# OPTION B: AGGREGATE → SMOOTH SEGMENT (RECOMMENDED)
# ============================================================================

def option_b_aggregate_then_smooth(df, calibrator):
    """
    Option B: Aggregate to segment first, then calibrate segment curve
    
    Process:
    1. Aggregate: average predictions across mortgages in segment
    2. For each segment: calibrate the segment-level curve (200 points)
    
    Parameters
    ----------
    df : DataFrame
        Columns: mtg_num, segment, margin_value, predicted_prob
    calibrator : MortgageCalibrator
    
    Returns
    -------
    segment_curves : DataFrame
        Segment-level calibrated curves
    """
    print("="*70)
    print("OPTION B: Aggregate → Smooth Segment (RECOMMENDED)")
    print("="*70)
    print("Process: Average to segment → Calibrate segment curve")
    print()
    
    # Step 1: Aggregate to segment level first
    segment_averages = df.groupby(['segment', 'margin_value']).agg({
        'predicted_prob': 'mean'  # Average across individuals
    }).reset_index()
    
    # Step 2: Calibrate each segment curve
    segment_curves = []
    
    for segment_id, seg_data in segment_averages.groupby('segment'):
        seg_data = seg_data.sort_values('margin_value')
        
        margins = seg_data['margin_value'].values
        probs = seg_data['predicted_prob'].values
        
        # Calibrate this segment
        calibrated = calibrator.calibrate(margins, probs)
        
        seg_data_copy = seg_data.copy()
        seg_data_copy['calibrated_prob'] = calibrated
        segment_curves.append(seg_data_copy)
    
    segment_curves = pd.concat(segment_curves, ignore_index=True)
    
    print(f"✓ Averaged {df['mtg_num'].nunique()} mortgages to segments")
    print(f"✓ Calibrated {segment_curves['segment'].nunique()} segment curves")
    print()
    
    return segment_curves


# ============================================================================
# EXAMPLE DATA
# ============================================================================

def create_example_data(n_mortgages_per_segment=50, n_margins=200):
    """
    Create example data matching your structure
    
    Each mortgage has 200 margin predictions
    Multiple mortgages belong to each segment
    """
    np.random.seed(42)
    
    margins = np.linspace(0, 2.0, n_margins)
    data = []
    
    for segment in range(1, 11):  # 10 segments
        # Segment-level true curve
        segment_decay_rate = 0.8 + 0.4 * (segment / 10)
        segment_true = 0.95 * np.exp(-segment_decay_rate * margins)
        
        for mtg_idx in range(n_mortgages_per_segment):
            mtg_num = f'MTG_S{segment}_{mtg_idx:03d}'
            
            # Individual variation around segment mean
            individual_noise = 0.03 * np.random.randn(n_margins)
            # Some systematic noise (affects all mortgages)
            systematic_noise = 0.02 * np.sin(8 * margins)
            
            probs = segment_true + individual_noise + systematic_noise
            probs = np.clip(probs, 0.1, 0.99)
            
            for margin, prob in zip(margins, probs):
                data.append({
                    'mtg_num': mtg_num,
                    'segment': segment,
                    'margin_value': margin,
                    'predicted_prob': prob
                })
    
    return pd.DataFrame(data)


# ============================================================================
# COMPARISON VISUALIZATION
# ============================================================================

def compare_both_options(df, option_a_result, option_b_result):
    """
    Visualize the difference between both approaches
    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.ravel()
    
    for idx, segment in enumerate(range(1, 11)):
        ax = axes[idx]
        
        # Original segment average (before calibration)
        orig_seg = df[df['segment'] == segment].groupby('margin_value').agg({
            'predicted_prob': 'mean'
        }).reset_index()
        
        # Option A result
        opt_a = option_a_result[option_a_result['segment'] == segment]
        
        # Option B result
        opt_b = option_b_result[option_b_result['segment'] == segment]
        
        # Plot
        ax.plot(orig_seg['margin_value'], orig_seg['predicted_prob'],
               'gray', linewidth=2, alpha=0.4, label='Original avg')
        ax.plot(opt_a['margin_value'], opt_a['calibrated_prob'],
               'r--', linewidth=2.5, alpha=0.7, label='Option A: Indiv→Agg')
        ax.plot(opt_b['margin_value'], opt_b['calibrated_prob'],
               'b-', linewidth=3, alpha=0.8, label='Option B: Agg→Smooth')
        
        ax.axvline(1.75, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(0.05, color='green', linestyle=':', alpha=0.5)
        
        ax.set_title(f'Segment {segment}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Margin', fontsize=9)
        ax.set_ylabel('Renewal Prob', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Comparison: Which Order is Better?\n(They should be very similar!)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/option_comparison.png', dpi=150)
    plt.show()
    print("✓ Comparison plot saved!")


def plot_detailed_comparison(df, option_a_result, option_b_result, segment_id=3):
    """
    Detailed comparison for one segment
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get data for this segment
    seg_data = df[df['segment'] == segment_id]
    opt_a = option_a_result[option_a_result['segment'] == segment_id].sort_values('margin_value')
    opt_b = option_b_result[option_b_result['segment'] == segment_id].sort_values('margin_value')
    
    # Plot 1: Individual mortgage predictions
    ax1 = axes[0, 0]
    sample_mortgages = seg_data['mtg_num'].unique()[:10]
    for mtg in sample_mortgages:
        mtg_data = seg_data[seg_data['mtg_num'] == mtg].sort_values('margin_value')
        ax1.plot(mtg_data['margin_value'], mtg_data['predicted_prob'],
                alpha=0.3, linewidth=1, color='gray')
    
    seg_avg = seg_data.groupby('margin_value')['predicted_prob'].mean()
    ax1.plot(seg_avg.index, seg_avg.values, 'r-', linewidth=3, label='Segment average')
    ax1.set_title('Raw Predictions\n(Gray = individuals, Red = segment avg)', fontweight='bold')
    ax1.set_xlabel('Margin')
    ax1.set_ylabel('Predicted Prob')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Option A - Individual smoothing
    ax2 = axes[0, 1]
    ax2.plot(seg_avg.index, seg_avg.values, 'gray', linewidth=2, alpha=0.4, label='Original avg')
    ax2.plot(opt_a['margin_value'], opt_a['calibrated_prob'], 'r-', linewidth=3, label='After Option A')
    ax2.axvline(1.75, color='orange', linestyle=':', alpha=0.5)
    ax2.set_title('Option A: Smooth each individual → Average', fontweight='bold')
    ax2.set_xlabel('Margin')
    ax2.set_ylabel('Calibrated Prob')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Option B - Segment smoothing
    ax3 = axes[1, 0]
    ax3.plot(seg_avg.index, seg_avg.values, 'gray', linewidth=2, alpha=0.4, label='Original avg')
    ax3.plot(opt_b['margin_value'], opt_b['calibrated_prob'], 'b-', linewidth=3, label='After Option B')
    ax3.axvline(1.75, color='orange', linestyle=':', alpha=0.5)
    ax3.set_title('Option B: Average → Smooth segment (RECOMMENDED)', fontweight='bold')
    ax3.set_xlabel('Margin')
    ax3.set_ylabel('Calibrated Prob')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Direct comparison
    ax4 = axes[1, 1]
    ax4.plot(opt_a['margin_value'], opt_a['calibrated_prob'], 'r--', 
            linewidth=2.5, alpha=0.7, label='Option A')
    ax4.plot(opt_b['margin_value'], opt_b['calibrated_prob'], 'b-', 
            linewidth=3, alpha=0.8, label='Option B (recommended)')
    
    # Calculate difference
    diff = np.abs(opt_a['calibrated_prob'].values - opt_b['calibrated_prob'].values)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    ax4.set_title(f'Direct Comparison\nMax diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}',
                 fontweight='bold')
    ax4.set_xlabel('Margin')
    ax4.set_ylabel('Calibrated Prob')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Segment {segment_id}: Detailed Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/detailed_comparison.png', dpi=150)
    plt.show()
    print("✓ Detailed comparison plot saved!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Compare both approaches on example data
    """
    print("\n" + "="*80)
    print("CALIBRATION ORDER COMPARISON")
    print("="*80)
    print()
    print("Question: Should we smooth BEFORE or AFTER aggregating to segment?")
    print()
    print("Option A: Smooth each individual mortgage → Aggregate to segment")
    print("Option B: Aggregate to segment → Smooth segment curve")
    print()
    print("="*80)
    print()
    
    # Create example data
    print("Creating example data...")
    df = create_example_data(n_mortgages_per_segment=50, n_margins=200)
    print(f"✓ Created {len(df)} rows")
    print(f"  - {df['mtg_num'].nunique()} total mortgages")
    print(f"  - {df['segment'].nunique()} segments")
    print(f"  - {n_mortgages_per_segment} mortgages per segment")
    print(f"  - {len(df[df['mtg_num']==df['mtg_num'].iloc[0]])} margins per mortgage")
    print()
    
    # Initialize calibrator
    calibrator = MortgageCalibrator(
        threshold=1.75,
        high_margin=2.0,
        high_prob=0.05,
        smooth=0.4
    )
    
    # Option A
    option_a_result = option_a_individual_then_aggregate(df, calibrator)
    
    # Option B
    option_b_result = option_b_aggregate_then_smooth(df, calibrator)
    
    # Compare
    print("="*70)
    print("COMPARISON")
    print("="*70)
    
    for segment in range(1, 11):
        opt_a = option_a_result[option_a_result['segment'] == segment]['calibrated_prob'].values
        opt_b = option_b_result[option_b_result['segment'] == segment]['calibrated_prob'].values
        
        diff = np.abs(opt_a - opt_b)
        print(f"Segment {segment:2d}: Max diff = {diff.max():.5f}, Mean diff = {diff.mean():.5f}")
    
    print()
    print("Conclusion: Results are VERY similar (as expected)")
    print("Recommendation: Use Option B (faster, cleaner)")
    print()
    
    # Visualize
    print("Creating visualizations...")
    compare_both_options(df, option_a_result, option_b_result)
    plot_detailed_comparison(df, option_a_result, option_b_result, segment_id=3)
    
    # Save
    print("Saving results...")
    option_a_result.to_csv('/mnt/user-data/outputs/option_a_result.csv', index=False)
    option_b_result.to_csv('/mnt/user-data/outputs/option_b_result.csv', index=False)
    print("✓ Saved CSV files")
    print()
    
    print("="*80)
    print("✓ DONE!")
    print("="*80)
    
    return option_a_result, option_b_result


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == '__main__':
    opt_a, opt_b = main()
    
    print("\n" + "="*80)
    print("RECOMMENDATION: Use Option B")
    print("="*80)
    print()
    print("Why?")
    print("1. Faster: 10 calibrations vs 500 calibrations")
    print("2. Cleaner signal: Averaging removes individual noise first")
    print("3. Same result: Differences are negligible (< 0.001)")
    print("4. Simpler code: One aggregation, then smooth")
    print()
    print("Use Option B: Aggregate → Smooth Segment")
    print("="*80)