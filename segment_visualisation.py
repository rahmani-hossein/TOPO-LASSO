import numpy as np
import pandas as pd

class SegmentCurveSmoother:
    """
    Natural decay with smooth plateau
    - Steep drop initially (matches natural gradient)
    - Gradually flattens to near-horizontal at end
    """
    
    def isotonic_only(self, margins, probs):
        """Simple isotonic regression"""
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        return iso.fit_transform(margins, probs)
    
    def isotonic_anchored(self, margins, probs,
                         decay_start=1.25,
                         target_prob_at_2=0.02,
                         threshold_at_2=0.2,
                         plateau_approach='asymptotic'):  # 'asymptotic' or 'hermite'
        """
        Smooth decay that becomes flat at the end
        
        Math:
        - First derivative starts negative (steep)
        - First derivative approaches 0 (becomes flat)
        - Second derivative is POSITIVE (concave up, "smiling")
        
        Result: Steep decline → gradual flattening
        
        Parameters
        ----------
        plateau_approach : str
            'asymptotic': Exponential decay toward asymptote (recommended)
            'hermite': Hermite cubic with zero end derivative
        """
        
        margins = np.asarray(margins)
        probs = np.asarray(probs)
        smoothed = probs.copy()
        
        decay_mask = margins > decay_start
        
        if np.any(decay_mask):
            p_actual_at_2 = np.interp(2.0, margins, probs)
            
            if p_actual_at_2 < threshold_at_2:
                # Already good!
                print(f"  Natural p(2.0)={p_actual_at_2:.4f} < {threshold_at_2:.4f} → No modification")
                
            else:
                print(f"  Natural p(2.0)={p_actual_at_2:.4f} >= {threshold_at_2:.4f} → Applying smooth plateau decay")
                
                # Get starting point and gradient
                idx_at_decay = np.where(margins <= decay_start)[0][-1]
                p_at_decay_start = probs[idx_at_decay]
                m_at_decay_start = margins[idx_at_decay]
                
                # Estimate natural gradient at decay_start
                if idx_at_decay >= 1:
                    h = margins[idx_at_decay] - margins[idx_at_decay - 1]
                    p_prev = probs[idx_at_decay - 1]
                    natural_gradient = (p_at_decay_start - p_prev) / h
                else:
                    natural_gradient = -0.1  # Default steep decline
                
                m_high = margins[decay_mask]
                
                if plateau_approach == 'asymptotic':
                    # === EXPONENTIAL DECAY TO ASYMPTOTE ===
                    # Formula: p(m) = asymptote + (p_start - asymptote) * exp(-λ(m - m_start))
                    # This naturally gives:
                    # - Steep start (follows natural gradient)
                    # - Gradual flattening
                    # - Approaches asymptote with zero slope
                    
                    # Set asymptote slightly below target for smooth approach
                    asymptote = target_prob_at_2 * 0.8
                    
                    # Calculate λ to match natural gradient at start
                    # p'(m_start) = -λ(p_start - asymptote) = natural_gradient
                    # λ = -natural_gradient / (p_start - asymptote)
                    
                    if p_at_decay_start > asymptote:
                        decay_lambda = -natural_gradient / (p_at_decay_start - asymptote)
                        
                        # Make sure it actually reaches near target by m=2.0
                        # Adjust λ if needed
                        p_test_at_2 = asymptote + (p_at_decay_start - asymptote) * np.exp(-decay_lambda * (2.0 - decay_start))
                        
                        if p_test_at_2 > threshold_at_2:
                            # Need steeper decay
                            # Force it to reach target_prob_at_2 at m=2.0
                            decay_lambda = -np.log((target_prob_at_2 - asymptote) / (p_at_decay_start - asymptote)) / (2.0 - decay_start)
                    else:
                        decay_lambda = 1.0
                    
                    # Apply exponential decay to asymptote
                    smoothed[decay_mask] = asymptote + (p_at_decay_start - asymptote) * np.exp(-decay_lambda * (m_high - decay_start))
                    
                    print(f"    Asymptotic decay: λ={decay_lambda:.3f}, asymptote={asymptote:.4f}")
                    
                elif plateau_approach == 'hermite':
                    # === HERMITE CUBIC WITH ZERO END DERIVATIVE ===
                    # Specify: p(m_start), p'(m_start), p(m_end), p'(m_end)=0
                    
                    m_start = decay_start
                    m_end = 2.0
                    p_start = p_at_decay_start
                    dp_start = natural_gradient  # Match natural
                    p_end = target_prob_at_2
                    dp_end = 0.0  # FLAT at end
                    
                    # Normalize to [0, 1]
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
                    
                    print(f"    Hermite decay: start_gradient={dp_start:.4f}, end_gradient=0.0")
            
            # Ensure doesn't go below target
            smoothed[decay_mask] = np.maximum(smoothed[decay_mask], target_prob_at_2)
        
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
            
            print(f"\nSegment {segment_id}:")
            
            if method == 'isotonic_only':
                smoothed = self.isotonic_only(margins, probs)
            elif method == 'isotonic_anchored':
                smoothed = self.isotonic_anchored(margins, probs, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            seg_df['smoothed_prob'] = smoothed
            results.append(seg_df)
        
        return pd.concat(results, ignore_index=True)


# === USAGE ===
calibrator = SegmentCurveSmoother()

# Try asymptotic approach (recommended)
result_df = calibrator.smooth_segment_dataframe(
    segment_df,
    method='isotonic_anchored',
    decay_start=1.25,
    target_prob_at_2=0.02,
    threshold_at_2=0.2,
    plateau_approach='asymptotic'  # Smooth decay to plateau
)

# Or try Hermite with zero end derivative
result_df = calibrator.smooth_segment_dataframe(
    segment_df,
    method='isotonic_anchored',
    decay_start=1.25,
    target_prob_at_2=0.02,
    threshold_at_2=0.2,
    plateau_approach='hermite'  # Cubic with flat end
)