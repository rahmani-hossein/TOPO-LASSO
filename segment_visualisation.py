import numpy as np
import pandas as pd

class SegmentCurveSmoother:
    """
    Smooth segment-level renewal probability curves
    - Before decay_start: UNCHANGED
    - After decay_start: Smooth decay that MATCHES THE GRADIENT at decay_start
    """
    
    def isotonic_only(self, margins, probs):
        """Simple isotonic regression"""
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        return iso.fit_transform(margins, probs)
    
    
    def isotonic_anchored(self, margins, probs,
                         decay_start=1.25,
                         target_prob_at_2=0.02,
                         threshold_at_2=0.1,     # ← New parameter!
                         match_gradient=True):
        """
        Gradient-matched decay with smart end point
        
        Parameters
        ----------
        decay_start : float
            Where decay begins (default: 1.25)
        target_prob_at_2 : float
            Force probability down to this at margin=2.0 (default: 0.02)
        threshold_at_2 : float
            Only force down if actual prob at 2.0 is ABOVE this threshold
            (default: 0.1)
            - If actual_prob(2.0) < threshold_at_2: use actual value
            - If actual_prob(2.0) >= threshold_at_2: force to target_prob_at_2
        match_gradient : bool
            If True, decay starts with same slope as original curve
        
        Example
        -------
        threshold_at_2 = 0.1:
        - Segment has p(2.0) = 0.05 → Keep 0.05 (already low enough)
        - Segment has p(2.0) = 0.25 → Force to 0.02 (too high, needs decay)
        """
        
        margins = np.asarray(margins)
        probs = np.asarray(probs)
        smoothed = probs.copy()
        
        decay_mask = margins > decay_start
        
        if np.any(decay_mask):
            # Get point at decay start
            idx_at_decay = np.where(margins <= decay_start)[0][-1]
            p_at_decay_start = probs[idx_at_decay]
            m_at_decay_start = margins[idx_at_decay]
            
            # === SMART END POINT LOGIC ===
            # Check actual probability at margin=2.0
            p_actual_at_2 = np.interp(2.0, margins, probs)
            
            if p_actual_at_2 < threshold_at_2:
                # Already low enough, use actual value
                p_end = p_actual_at_2
                print(f"  Actual p(2.0)={p_actual_at_2:.4f} < threshold={threshold_at_2:.4f} → Using actual value")
            else:
                # Too high, force down to target
                p_end = target_prob_at_2
                print(f"  Actual p(2.0)={p_actual_at_2:.4f} >= threshold={threshold_at_2:.4f} → Forcing to {target_prob_at_2:.4f}")
            
            if match_gradient:
                # === GRADIENT MATCHING ===
                
                # Estimate gradient at decay_start
                if idx_at_decay >= 2:
                    h1 = margins[idx_at_decay] - margins[idx_at_decay - 1]
                    p1 = probs[idx_at_decay - 1]
                    gradient_at_start = (p_at_decay_start - p1) / h1
                else:
                    gradient_at_start = (p_at_decay_start - probs[idx_at_decay - 1]) / \
                                       (m_at_decay_start - margins[idx_at_decay - 1])
                
                # Hermite cubic interpolation
                m_start = decay_start
                m_end = 2.0
                p_start = p_at_decay_start
                dp_start = gradient_at_start  # Match original gradient
                dp_end = 0.0  # Flatten at end
                
                m_high = margins[decay_mask]
                t = (m_high - m_start) / (m_end - m_start)
                
                # Hermite basis functions
                h00 = 2*t**3 - 3*t**2 + 1
                h10 = t**3 - 2*t**2 + t
                h01 = -2*t**3 + 3*t**2
                h11 = t**3 - t**2
                
                delta_m = m_end - m_start
                
                smoothed[decay_mask] = (
                    h00 * p_start +
                    h10 * delta_m * dp_start +
                    h01 * p_end +
                    h11 * delta_m * dp_end
                )
                
            else:
                # === SIMPLE EXPONENTIAL ===
                if p_at_decay_start > p_end:
                    decay_lambda = -np.log(p_end / p_at_decay_start) / (2.0 - decay_start)
                else:
                    decay_lambda = 0
                
                m_high = margins[decay_mask]
                smoothed[decay_mask] = p_at_decay_start * np.exp(-decay_lambda * (m_high - decay_start))
            
            # Ensure doesn't go below the chosen end point
            smoothed[decay_mask] = np.maximum(smoothed[decay_mask], p_end)
        
        return np.clip(smoothed, 0.001, 0.999)
    
    def smooth_segment_dataframe(self, segment_df, method='isotonic_anchored',
                                 segment_col='segment',
                                 margin_col='margin',
                                 prob_col='avg_prediction',
                                 **kwargs):
        """Apply smoothing"""
        results = []
        
        for segment_id, seg_df in segment_df.groupby(segment_col):
            seg_df = seg_df.sort_values(margin_col).copy()
            
            margins = seg_df[margin_col].values
            probs = seg_df[prob_col].values
            
            if method == 'isotonic_only':
                smoothed = self.isotonic_only(margins, probs)
            elif method == 'isotonic_anchored':
                smoothed = self.isotonic_anchored(margins, probs, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            seg_df['smoothed_prob'] = smoothed
            results.append(seg_df)
            
            print(f"✓ Segment {segment_id}: Gradient-matched decay")
        
        return pd.concat(results, ignore_index=True)


# === USAGE ===
calibrator = SegmentCurveSmoother()

# With gradient matching (smooth transition!)
result_df = calibrator.smooth_segment_dataframe(
    segment_df,
    method='isotonic_anchored',
    decay_start=1.25,
    target_prob_at_2=0.02,
    match_gradient=True  # ← This matches the slope!
)