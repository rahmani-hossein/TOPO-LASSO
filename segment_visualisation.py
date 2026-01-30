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
                         decay_start=1.25,
                         target_prob_at_2=0.02):
        """
        Keep everything before 1.25 UNCHANGED
        Force smooth exponential decay after 1.25
        
        Parameters
        ----------
        decay_start : float
            Margin where decay begins (default: 1.25)
        target_prob_at_2 : float
            Target probability at margin=2.0 (default: 0.02)
        """
        
        margins = np.asarray(margins)
        probs = np.asarray(probs)
        
        # Initialize output with original probabilities
        smoothed = probs.copy()
        
        # Find where decay starts
        decay_mask = margins > decay_start
        
        if np.any(decay_mask):
            # Get probability at decay_start (last point before decay)
            idx_at_decay = np.where(margins <= decay_start)[0][-1]
            p_at_decay_start = probs[idx_at_decay]
            
            # Calculate exponential decay rate
            # Formula: p(m) = p_start * exp(-lambda * (m - m_start))
            # We want: p(2.0) = target_prob_at_2
            # Solve: lambda = -log(p_end / p_start) / (m_end - m_start)
            
            if p_at_decay_start > target_prob_at_2:
                decay_lambda = -np.log(target_prob_at_2 / p_at_decay_start) / (2.0 - decay_start)
            else:
                # Already below target
                decay_lambda = 0
            
            # Apply exponential decay to high margin region
            m_high = margins[decay_mask]
            smoothed[decay_mask] = p_at_decay_start * np.exp(-decay_lambda * (m_high - decay_start))
            
            # Ensure doesn't go below target
            smoothed[decay_mask] = np.maximum(smoothed[decay_mask], target_prob_at_2)
        
        # Clip to valid range
        return np.clip(smoothed, 0.001, 0.999)





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