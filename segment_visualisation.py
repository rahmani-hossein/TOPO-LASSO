import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline

class SegmentCurveSmoother:
    """
    Smooth segment-level renewal probability curves
    - Keep 0 to 1.25 natural
    - Force SMOOTH exponential decay after 1.25 (no steps!)
    """
    
    def isotonic_only(self, margins, probs):
        """Simple isotonic regression"""
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        return iso.fit_transform(margins, probs)
    
    def isotonic_anchored(self, margins, probs,
                         decay_start=1.25,      # Start of forced decay
                         target_prob_at_2=0.02, # End probability
                         spline_smoothness=0.005,
                         spline_degree=3):
        """
        Method: Natural curve until 1.25, then FORCE exponential decay
        
        This ensures:
        - No ups/downs after 1.25
        - Smooth decline (no steps)
        - Profit curve has only ONE peak
        """
        
        margins = np.asarray(margins)
        probs = np.asarray(probs)
        
        # Split into two regions
        low_margin_mask = margins <= decay_start
        high_margin_mask = margins > decay_start
        
        # === REGION 1: Low margins (0 to 1.25) - Use isotonic + spline ===
        
        if np.any(low_margin_mask):
            m_low = margins[low_margin_mask]
            p_low = probs[low_margin_mask]
            
            # Apply isotonic to low region
            iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
            p_low_mono = iso.fit_transform(m_low, p_low)
            
            # Spline smoothing for low region
            if len(m_low) > 3:
                spline = UnivariateSpline(m_low, p_low_mono, 
                                         s=spline_smoothness, k=min(spline_degree, len(m_low)-1))
                p_low_smooth = spline(m_low)
            else:
                p_low_smooth = p_low_mono
        else:
            p_low_smooth = np.array([])
        
        # === REGION 2: High margins (1.25 to 2.0) - FORCE exponential decay ===
        
        if np.any(high_margin_mask):
            m_high = margins[high_margin_mask]
            
            # Get probability at decay_start (from low region)
            if np.any(low_margin_mask):
                p_at_decay_start = p_low_smooth[-1]
            else:
                # Fallback if no low region
                p_at_decay_start = np.interp(decay_start, margins, probs)
            
            # Calculate exponential decay rate
            # p(m) = p_start * exp(-lambda * (m - decay_start))
            # We want: p(2.0) = target_prob_at_2
            
            if p_at_decay_start > target_prob_at_2:
                decay_lambda = -np.log(target_prob_at_2 / p_at_decay_start) / (2.0 - decay_start)
            else:
                decay_lambda = 0
                p_at_decay_start = target_prob_at_2
            
            # Apply pure exponential decay (guaranteed smooth!)
            p_high_smooth = p_at_decay_start * np.exp(-decay_lambda * (m_high - decay_start))
            
            # Ensure it doesn't go below target
            p_high_smooth = np.maximum(p_high_smooth, target_prob_at_2)
        else:
            p_high_smooth = np.array([])
        
        # === COMBINE both regions ===
        
        p_smoothed = np.concatenate([p_low_smooth, p_high_smooth])
        
        # Final clip to valid range
        return np.clip(p_smoothed, 0.001, 0.999)
    
    def smooth_segment_dataframe(self, segment_df, method='isotonic_anchored',
                                 segment_col='segment',
                                 margin_col='margin',
                                 prob_col='avg_prediction'):
        """Apply smoothing to segment-level dataframe"""
        results = []
        
        for segment_id, seg_df in segment_df.groupby(segment_col):
            seg_df = seg_df.sort_values(margin_col).copy()
            
            margins = seg_df[margin_col].values
            probs = seg_df[prob_col].values
            
            if method == 'isotonic_only':
                smoothed = self.isotonic_only(margins, probs)
            elif method == 'isotonic_anchored':
                smoothed = self.isotonic_anchored(margins, probs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            seg_df['smoothed_prob'] = smoothed
            results.append(seg_df)
            
            print(f"✓ Segment {segment_id} smoothed")
        
        return pd.concat(results, ignore_index=True)


# === USAGE ===
calibrator = SegmentCurveSmoother()

result_df = calibrator.smooth_segment_dataframe(
    segment_df,
    method='isotonic_anchored',
    segment_col='segment',
    margin_col='margin',
    prob_col='avg_prediction'
)

# Plot to verify
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 7))

for segment in result_df['segment'].unique():
    seg_data = result_df[result_df['segment'] == segment].sort_values('margin')
    plt.plot(seg_data['margin'], seg_data['smoothed_prob'], 
            linewidth=2.5, label=f'Segment {segment}')

plt.axvline(1.25, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Decay starts')
plt.xlabel('Margin', fontsize=12)
plt.ylabel('Smoothed Probability', fontsize=12)
plt.title('Smoothed Probability by Segment\n(Pure exponential decay after 1.25)', 
         fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()