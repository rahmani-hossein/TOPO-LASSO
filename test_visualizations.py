#!/usr/bin/env python3
"""
Test script for stability selection visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.lassonet_selector import LassoNetSelector, StabilityConfig
from src.stability import StabilityVisualizer

def test_stability_visualizations():
    """Test stability selection visualization tools."""
    print("Testing stability selection visualizations...")

    # Create synthetic data with clear causal structure
    np.random.seed(42)
    n_samples = 150

    # X1 -> X2 -> X3, X4 and X5 are noise
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 0.8 * X1 + 0.3 * np.random.normal(0, 1, n_samples)
    X3 = 0.6 * X2 + 0.4 * X1 + 0.2 * np.random.normal(0, 1, n_samples)
    X4 = np.random.normal(0, 1, n_samples)  # Noise
    X5 = np.random.normal(0, 1, n_samples)  # Noise

    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5
    })

    X_candidates = data[['X1', 'X2', 'X4', 'X5']]
    y_target = data['X3']

    # Configure stability selection
    stability_config = StabilityConfig(
        enabled=True,
        n_bootstrap=10,  # Moderate number for good visualization
        threshold=0.4,
        bootstrap_strategy='subsample',
        subsample_size=0.8,
        random_state=42,
        verbose=1
    )

    # Create selector
    selector = LassoNetSelector(
        hidden_dims=(8,),
        cv=2,
        verbose=1,
        random_state=42,
        stability_config=stability_config
    )

    # Fit the model
    print("\n=== Fitting LassoNet with Stability Selection ===")
    result = selector.fit_variable(X_candidates, y_target, 'X3')

    print(f"Selected parents: {result['selected_parents']}")

    if 'stability_results' in result and result['stability_results'] is not None:
        stability_results = result['stability_results']

        # Extract data for visualization
        stability_scores = stability_results.stability_scores
        selection_frequency = stability_results.selection_frequency
        feature_names = list(X_candidates.columns)
        threshold = stability_config.threshold

        print(f"Stability scores: {dict(zip(feature_names, stability_scores))}")

        # Create visualizations
        print("\n=== Creating Visualizations ===")

        # 1. Stability scores bar plot
        print("Creating stability scores plot...")
        fig1 = StabilityVisualizer.plot_stability_scores(
            stability_scores=stability_scores,
            feature_names=feature_names,
            threshold=threshold,
            title="Feature Stability Scores (X3 as Target)",
            figsize=(10, 6)
        )
        fig1.savefig('stability_scores.png', dpi=150, bbox_inches='tight')
        print("  → Saved as 'stability_scores.png'")

        # 2. Selection heatmap
        print("Creating selection heatmap...")
        fig2 = StabilityVisualizer.plot_selection_heatmap(
            selection_frequency=selection_frequency,
            feature_names=feature_names,
            title="Feature Selection Across Bootstrap Samples",
            figsize=(12, 6)
        )
        fig2.savefig('selection_heatmap.png', dpi=150, bbox_inches='tight')
        print("  → Saved as 'selection_heatmap.png'")

        # 3. Stability histogram
        print("Creating stability histogram...")
        fig3 = StabilityVisualizer.plot_stability_histogram(
            stability_scores=stability_scores,
            threshold=threshold,
            title="Distribution of Stability Scores",
            figsize=(8, 6)
        )
        fig3.savefig('stability_histogram.png', dpi=150, bbox_inches='tight')
        print("  → Saved as 'stability_histogram.png'")

        # Print summary statistics
        print(f"\n=== Stability Summary ===")
        print(f"Number of bootstrap samples: {stability_config.n_bootstrap}")
        print(f"Stability threshold: {threshold}")
        print(f"Features above threshold: {np.sum(stability_scores >= threshold)}")
        print(f"Mean stability score: {np.mean(stability_scores):.3f}")
        print(f"Std stability score: {np.std(stability_scores):.3f}")

        # Feature-wise analysis
        print(f"\n=== Feature Analysis ===")
        for i, (feature, score) in enumerate(zip(feature_names, stability_scores)):
            status = "SELECTED" if score >= threshold else "REJECTED"
            selection_freq = f"{score:.1%}"
            print(f"{feature}: {score:.3f} ({selection_freq}) - {status}")

        print(f"\nExpected result: X1 and X2 should have high stability (true causal parents)")
        print(f"                 X4 and X5 should have low stability (noise variables)")

        # Close figures to free memory
        plt.close('all')

    print(f"\nVisualization test completed! ✅")
    print(f"Check the generated PNG files for the stability visualizations.")

if __name__ == "__main__":
    test_stability_visualizations()