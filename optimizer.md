# Comprehensive Mortgage Pricing Optimization Specification

I'll provide a complete specification with working code structure you can give to your copilot.

## **1. Problem Specification**

### **Business Context**
We have a mortgage renewal pricing optimization problem where:
- Each mortgage has 200 possible margin values (treatment levels from 0 to 2)
- A causal inference model predicts renewal probability for each margin
- We need to find optimal pricing across different term lengths (1, 2, 3, 5 years)
- Must maintain minimum portfolio volume while maximizing expected profit

### **Input Data Schema**
```
mortgage_id: str/int          # Unique mortgage identifier
margin: float                 # Treatment value [0, 2], 200 discrete values
renewal_probability: float    # Model prediction [0, 1]
balance: float               # Mortgage principal amount
cost_of_funds: float         # Bank's cost to fund this mortgage
```

### **Output Schema**
```
mortgage_id: str/int
optimal_margin_1yr: float
optimal_margin_2yr: float
optimal_margin_3yr: float
optimal_margin_5yr: float
expected_profit_1yr: float
expected_profit_2yr: float
expected_profit_3yr: float
expected_profit_5yr: float
expected_renewal_prob_1yr: float
expected_renewal_prob_2yr: float
expected_renewal_prob_3yr: float
expected_renewal_prob_5yr: float
```

## **2. Complete Implementation**

```python
"""
Mortgage Pricing Optimization System
=====================================

This module implements a complete pipeline for optimizing mortgage renewal pricing
across different term lengths using causal inference predictions.

Author: [Your Team]
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize, differential_evolution
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline"""
    terms: List[int] = None  # [1, 2, 3, 5] years
    min_volume_threshold: float = 0.0  # Minimum total volume to maintain
    smoothing_method: str = 'isotonic_spline'  # 'isotonic_spline', 'lowess', 'savgol'
    smoothing_strength: float = 0.03  # Spline smoothing parameter
    optimization_method: str = 'continuous'  # 'discrete' or 'continuous'
    max_iterations: int = 1000
    n_restarts: int = 3  # Multiple random starts for robustness
    random_seed: int = 42
    
    def __post_init__(self):
        if self.terms is None:
            self.terms = [1, 2, 3, 5]
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.min_volume_threshold >= 0, "Volume threshold must be non-negative"
        assert self.smoothing_strength > 0, "Smoothing strength must be positive"
        assert self.optimization_method in ['discrete', 'continuous'], "Invalid optimization method"
        assert all(t > 0 for t in self.terms), "Terms must be positive"
        logger.info(f"Configuration validated: {self}")


# ============================================================================
# DATA PROCESSING
# ============================================================================

class MortgageDataProcessor:
    """
    Handles data transformation, validation, and feature engineering.
    
    Responsibilities:
    1. Validate input data schema and quality
    2. Expand data to include multiple term lengths
    3. Calculate profit metrics
    4. Prepare data for optimization
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.required_columns = ['mortgage_id', 'margin', 'renewal_probability', 
                                'balance', 'cost_of_funds']
    
    def validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame meets requirements.
        
        Checks:
        - Required columns exist
        - Data types are correct
        - Value ranges are valid
        - Each mortgage has 200 margin values
        """
        logger.info("Validating input data...")
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if df[self.required_columns].isnull().any().any():
            null_counts = df[self.required_columns].isnull().sum()
            raise ValueError(f"Missing values found:\n{null_counts}")
        
        # Validate renewal probability range
        invalid_probs = (df['renewal_probability'] < 0) | (df['renewal_probability'] > 1)
        if invalid_probs.any():
            raise ValueError(f"Renewal probabilities outside [0,1]: {invalid_probs.sum()} rows")
        
        # Validate margin range
        if df['margin'].min() < 0 or df['margin'].max() > 2:
            warnings.warn(f"Margins outside expected [0,2] range: [{df['margin'].min()}, {df['margin'].max()}]")
        
        # Check each mortgage has 200 margin values
        margin_counts = df.groupby('mortgage_id')['margin'].count()
        if not (margin_counts == 200).all():
            problematic = margin_counts[margin_counts != 200]
            raise ValueError(f"Some mortgages don't have 200 margin values:\n{problematic.head()}")
        
        # Validate balance and cost_of_funds are positive
        if (df['balance'] <= 0).any():
            raise ValueError("Balance must be positive")
        if (df['cost_of_funds'] < 0).any():
            raise ValueError("Cost of funds cannot be negative")
        
        logger.info(f"✓ Data validation passed: {len(df)} rows, {df['mortgage_id'].nunique()} unique mortgages")
    
    def expand_for_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand DataFrame to include all term lengths.
        
        Creates cartesian product: mortgage_id × margin × term
        
        Args:
            df: Input DataFrame with mortgage_id, margin, renewal_probability, balance, cost_of_funds
        
        Returns:
            Expanded DataFrame with 'term' column added
        """
        logger.info(f"Expanding data for terms: {self.config.terms}")
        
        # Create term DataFrame
        terms_df = pd.DataFrame({'term': self.config.terms})
        terms_df['_key'] = 1
        
        # Add key column for cross join
        df_with_key = df.copy()
        df_with_key['_key'] = 1
        
        # Cartesian product
        expanded = df_with_key.merge(terms_df, on='_key', how='outer')
        expanded = expanded.drop('_key', axis=1)
        
        logger.info(f"✓ Expanded to {len(expanded)} rows ({len(df)} × {len(self.config.terms)} terms)")
        
        return expanded
    
    def calculate_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate profit for each row.
        
        Profit Formula:
            profit = (margin - cost_of_funds) × balance × term
        
        This represents the total interest income minus cost over the term period.
        
        Args:
            df: DataFrame with margin, cost_of_funds, balance, term columns
        
        Returns:
            DataFrame with 'profit' column added
        """
        logger.info("Calculating profit metrics...")
        
        df = df.copy()
        
        # Calculate spread (net interest margin)
        df['spread'] = df['margin'] - df['cost_of_funds']
        
        # Calculate total profit over term
        df['profit'] = df['spread'] * df['balance'] * df['term']
        
        # Calculate expected profit (profit × renewal probability)
        df['expected_profit'] = df['profit'] * df['renewal_probability']
        
        # Flag unprofitable combinations
        df['is_profitable'] = df['spread'] > 0
        
        # Log statistics
        total_unprofitable = (~df['is_profitable']).sum()
        if total_unprofitable > 0:
            warnings.warn(f"Found {total_unprofitable} unprofitable combinations (spread ≤ 0)")
        
        logger.info(f"✓ Profit calculation complete")
        logger.info(f"  Average profit: ${df['profit'].mean():,.2f}")
        logger.info(f"  Average expected profit: ${df['expected_profit'].mean():,.2f}")
        
        return df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Steps:
        1. Validate input data
        2. Expand for all terms
        3. Calculate profit metrics
        
        Args:
            df: Raw input DataFrame
        
        Returns:
            Processed DataFrame ready for smoothing and optimization
        """
        self.validate_input_data(df)
        df_expanded = self.expand_for_terms(df)
        df_with_profit = self.calculate_profit(df_expanded)
        
        return df_with_profit


# ============================================================================
# SMOOTHING
# ============================================================================

class RenewalCurveSmoother:
    """
    Smooths renewal probability curves for each mortgage-term combination.
    
    Purpose:
    - Handle noise in predictions, especially at extreme margin values
    - Ensure monotonicity (renewal probability should not increase with higher margins)
    - Provide smooth curves for optimization
    
    Methods supported:
    1. isotonic_spline: Two-stage (isotonic regression + cubic spline)
    2. lowess: Locally weighted scatterplot smoothing
    3. savgol: Savitzky-Golay filter
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def _smooth_isotonic_spline(self, margins: np.ndarray, 
                                 renewal_probs: np.ndarray) -> np.ndarray:
        """
        Two-stage smoothing: isotonic regression + spline.
        
        Stage 1: Isotonic regression enforces monotonicity (decreasing)
        Stage 2: Spline smoothing for visual appeal and continuity
        
        This is the recommended method for mortgage renewal curves.
        """
        # Stage 1: Enforce decreasing monotonicity
        iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
        renewal_monotonic = iso_reg.fit_transform(margins, renewal_probs)
        
        # Stage 2: Spline smoothing
        # k=3 for cubic spline, s controls smoothness
        try:
            spline = UnivariateSpline(
                margins, 
                renewal_monotonic, 
                s=self.config.smoothing_strength,
                k=3
            )
            renewal_smooth = spline(margins)
        except Exception as e:
            logger.warning(f"Spline smoothing failed, using isotonic only: {e}")
            renewal_smooth = renewal_monotonic
        
        # Clip to valid probability range
        renewal_smooth = np.clip(renewal_smooth, 0, 1)
        
        return renewal_smooth
    
    def _smooth_lowess(self, margins: np.ndarray, 
                       renewal_probs: np.ndarray) -> np.ndarray:
        """
        LOWESS (Locally Weighted Scatterplot Smoothing).
        
        Good for adaptive smoothing based on local data density.
        """
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # frac controls the fraction of data used for each local regression
            frac = min(0.3, 50 / len(margins))  # Use at least 50 points or 30%
            smoothed = lowess(renewal_probs, margins, frac=frac, return_sorted=False)
            
            # Ensure monotonicity as post-processing
            iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
            smoothed = iso_reg.fit_transform(margins, smoothed)
            
            return np.clip(smoothed, 0, 1)
        
        except ImportError:
            logger.warning("statsmodels not available, falling back to isotonic_spline")
            return self._smooth_isotonic_spline(margins, renewal_probs)
    
    def _smooth_savgol(self, margins: np.ndarray, 
                       renewal_probs: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay filter for smoothing.
        
        Preserves peak shapes better than moving averages.
        """
        from scipy.signal import savgol_filter
        
        # Window length should be odd
        window_length = min(51, len(margins) // 4)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 5)
        
        # Polynomial order
        polyorder = 3
        
        try:
            smoothed = savgol_filter(renewal_probs, window_length, polyorder)
            
            # Ensure monotonicity
            iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
            smoothed = iso_reg.fit_transform(margins, smoothed)
            
            return np.clip(smoothed, 0, 1)
        
        except Exception as e:
            logger.warning(f"Savgol filter failed, using isotonic_spline: {e}")
            return self._smooth_isotonic_spline(margins, renewal_probs)
    
    def smooth_single_curve(self, margins: np.ndarray, 
                           renewal_probs: np.ndarray) -> np.ndarray:
        """
        Smooth a single renewal probability curve.
        
        Args:
            margins: Array of margin values (should be sorted)
            renewal_probs: Corresponding renewal probabilities
        
        Returns:
            Smoothed renewal probabilities
        """
        # Sort by margin if not already sorted
        if not np.all(margins[:-1] <= margins[1:]):
            sort_idx = np.argsort(margins)
            margins = margins[sort_idx]
            renewal_probs = renewal_probs[sort_idx]
        
        # Select smoothing method
        if self.config.smoothing_method == 'isotonic_spline':
            return self._smooth_isotonic_spline(margins, renewal_probs)
        elif self.config.smoothing_method == 'lowess':
            return self._smooth_lowess(margins, renewal_probs)
        elif self.config.smoothing_method == 'savgol':
            return self._smooth_savgol(margins, renewal_probs)
        else:
            raise ValueError(f"Unknown smoothing method: {self.config.smoothing_method}")
    
    def smooth_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothing to all mortgage-term combinations in DataFrame.
        
        Args:
            df: DataFrame with columns [mortgage_id, term, margin, renewal_probability, ...]
        
        Returns:
            DataFrame with added column 'renewal_probability_smooth'
        """
        logger.info("Smoothing renewal probability curves...")
        
        df = df.copy()
        df['renewal_probability_smooth'] = np.nan
        
        # Group by mortgage and term
        groups = df.groupby(['mortgage_id', 'term'])
        total_groups = len(groups)
        
        for i, ((mort_id, term), group) in enumerate(groups):
            if (i + 1) % 100 == 0:
                logger.info(f"  Smoothing progress: {i+1}/{total_groups} mortgage-term combinations")
            
            # Sort by margin
            group = group.sort_values('margin')
            
            # Extract arrays
            margins = group['margin'].values
            renewal_probs = group['renewal_probability'].values
            
            # Smooth
            renewal_smooth = self.smooth_single_curve(margins, renewal_probs)
            
            # Assign back to dataframe
            df.loc[group.index, 'renewal_probability_smooth'] = renewal_smooth
        
        # Recalculate expected profit with smoothed probabilities
        df['expected_profit_smooth'] = df['profit'] * df['renewal_probability_smooth']
        
        logger.info(f"✓ Smoothing complete for {total_groups} mortgage-term combinations")
        
        # Log smoothing statistics
        diff = df['renewal_probability_smooth'] - df['renewal_probability']
        logger.info(f"  Mean absolute change: {abs(diff).mean():.4f}")
        logger.info(f"  Max absolute change: {abs(diff).max():.4f}")
        
        return df


# ============================================================================
# OPTIMIZATION
# ============================================================================

class MortgagePortfolioOptimizer:
    """
    Optimizes margin selection across the mortgage portfolio.
    
    For each term separately:
    - Decision variables: optimal margin for each mortgage
    - Objective: Maximize total expected profit
    - Constraint: Maintain minimum portfolio volume
    
    Supports two optimization approaches:
    1. Discrete: Select from 200 predefined margin values
    2. Continuous: Interpolate between margin values for smooth optimization
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results = {}
    
    def _prepare_term_data(self, df: pd.DataFrame, term: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare data structures for optimization of a specific term.
        
        Returns:
            df_term: Filtered DataFrame for this term
            metadata: Dictionary with mortgage-level information
        """
        df_term = df[df['term'] == term].copy()
        
        mortgages = df_term['mortgage_id'].unique()
        
        metadata = {
            'mortgages': mortgages,
            'n_mortgages': len(mortgages),
            'total_balance': df_term.groupby('mortgage_id')['balance'].first().sum()
        }
        
        return df_term, metadata
    
    def _optimize_discrete(self, df_term: pd.DataFrame, 
                          metadata: Dict, 
                          term: int) -> Dict:
        """
        Discrete optimization: select from 200 predefined margin values.
        
        Uses differential evolution for global optimization with discrete variables.
        
        Decision variables: margin_index[i] ∈ {0, 1, ..., 199} for each mortgage i
        """
        mortgages = metadata['mortgages']
        n_mortgages = metadata['n_mortgages']
        min_volume = self.config.min_volume_threshold
        
        logger.info(f"  Running discrete optimization for {n_mortgages} mortgages")
        
        # Create lookup structure for fast access
        # Structure: {mortgage_id: DataFrame of 200 rows sorted by margin}
        mort_data_lookup = {}
        for mort_id in mortgages:
            mort_df = df_term[df_term['mortgage_id'] == mort_id].sort_values('margin')
            mort_data_lookup[mort_id] = mort_df.reset_index(drop=True)
        
        def objective(margin_indices):
            """
            Objective function for optimization.
            
            Args:
                margin_indices: Array of shape (n_mortgages,) with values in [0, 199]
            
            Returns:
                Negative expected profit (for minimization)
            """
            margin_indices = margin_indices.astype(int)
            
            total_profit = 0.0
            total_volume = 0.0
            
            for i, mort_id in enumerate(mortgages):
                idx = margin_indices[i]
                row = mort_data_lookup[mort_id].iloc[idx]
                
                renewal_prob = row['renewal_probability_smooth']
                profit = row['profit']
                balance = row['balance']
                
                total_profit += profit * renewal_prob
                total_volume += renewal_prob * balance
            
            # Penalty for volume constraint violation
            volume_shortfall = max(0, min_volume - total_volume)
            penalty = volume_shortfall * 1e8  # Large penalty
            
            return -(total_profit - penalty)
        
        # Set up bounds: each mortgage can select index 0-199
        bounds = [(0, 199) for _ in range(n_mortgages)]
        
        # Run optimization with multiple restarts
        best_result = None
        best_value = float('inf')
        
        for restart in range(self.config.n_restarts):
            seed = self.config.random_seed + restart
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.max_iterations,
                seed=seed,
                workers=-1,  # Use all CPU cores
                polish=True,  # Local refinement
                atol=1e-6,
                tol=1e-6
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        
        # Extract optimal solutions
        optimal_indices = best_result.x.astype(int)
        
        optimal_margins = []
        optimal_renewal_probs = []
        optimal_profits = []
        
        for i, mort_id in enumerate(mortgages):
            idx = optimal_indices[i]
            row = mort_data_lookup[mort_id].iloc[idx]
            
            optimal_margins.append(row['margin'])
            optimal_renewal_probs.append(row['renewal_probability_smooth'])
            optimal_profits.append(row['expected_profit_smooth'])
        
        # Calculate final statistics
        total_expected_profit = sum(optimal_profits)
        total_expected_volume = sum(
            optimal_renewal_probs[i] * mort_data_lookup[mortgages[i]].iloc[0]['balance']
            for i in range(n_mortgages)
        )
        
        return {
            'term': term,
            'method': 'discrete',
            'mortgage_ids': mortgages.tolist(),
            'optimal_margins': optimal_margins,
            'optimal_renewal_probs': optimal_renewal_probs,
            'optimal_expected_profits': optimal_profits,
            'total_expected_profit': total_expected_profit,
            'total_expected_volume': total_expected_volume,
            'optimization_success': best_result.success,
            'n_iterations': best_result.nit,
            'volume_constraint_met': total_expected_volume >= min_volume
        }
    
    def _optimize_continuous(self, df_term: pd.DataFrame, 
                            metadata: Dict, 
                            term: int) -> Dict:
        """
        Continuous optimization with interpolation.
        
        Creates interpolation functions for renewal_prob(margin) and profit(margin),
        then optimizes over continuous margin space [0, 2].
        
        Uses SLSQP (Sequential Least Squares Programming) for constrained optimization.
        """
        mortgages = metadata['mortgages']
        n_mortgages = metadata['n_mortgages']
        min_volume = self.config.min_volume_threshold
        
        logger.info(f"  Running continuous optimization for {n_mortgages} mortgages")
        
        # Create interpolators for each mortgage
        interpolators = {}
        
        for mort_id in mortgages:
            mort_df = df_term[df_term['mortgage_id'] == mort_id].sort_values('margin')
            
            margins = mort_df['margin'].values
            renewal_probs = mort_df['renewal_probability_smooth'].values
            profits = mort_df['profit'].values
            balance = mort_df['balance'].iloc[0]
            
            # Create cubic interpolators
            interpolators[mort_id] = {
                'renewal_prob_interp': interp1d(
                    margins, 
                    renewal_probs, 
                    kind='cubic',
                    bounds_error=False,
                    fill_value=(renewal_probs[0], renewal_probs[-1])
                ),
                'profit_interp': interp1d(
                    margins,
                    profits,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(profits[0], profits[-1])
                ),
                'balance': balance
            }
        
        def objective(margins):
            """Total expected profit (negative for minimization)"""
            total_profit = 0.0
            
            for i, mort_id in enumerate(mortgages):
                margin = margins[i]
                interp = interpolators[mort_id]
                
                renewal_prob = float(interp['renewal_prob_interp'](margin))
                profit = float(interp['profit_interp'](margin))
                
                total_profit += profit * renewal_prob
            
            return -total_profit
        
        def volume_constraint(margins):
            """Volume constraint: total_volume >= min_volume"""
            total_volume = 0.0
            
            for i, mort_id in enumerate(mortgages):
                margin = margins[i]
                interp = interpolators[mort_id]
                
                renewal_prob = float(interp['renewal_prob_interp'](margin))
                balance = interp['balance']
                
                total_volume += renewal_prob * balance
            
            return total_volume - min_volume
        
        # Set up optimization
        bounds = [(0, 2) for _ in range(n_mortgages)]
        constraints = {'type': 'ineq', 'fun': volume_constraint}
        
        # Initial guess: try current mid-point pricing
        x0 = np.ones(n_mortgages) * 1.0
        
        # Run optimization with multiple restarts
        best_result = None
        best_value = float('inf')
        
        for restart in range(self.config.n_restarts):
            # Random perturbation for restarts
            if restart > 0:
                x0 = np.random.uniform(0.5, 1.5, n_mortgages)
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': 1e-8}
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        
        # Extract optimal solutions
        optimal_margins = best_result.x
        
        optimal_renewal_probs = []
        optimal_profits = []
        
        for i, mort_id in enumerate(mortgages):
            margin = optimal_margins[i]
            interp = interpolators[mort_id]
            
            renewal_prob = float(interp['renewal_prob_interp'](margin))
            profit = float(interp['profit_interp'](margin))
            
            optimal_renewal_probs.append(renewal_prob)
            optimal_profits.append(profit * renewal_prob)
        
        # Calculate final statistics
        total_expected_profit = sum(optimal_profits)
        total_expected_volume = sum(
            optimal_renewal_probs[i] * interpolators[mortgages[i]]['balance']
            for i in range(n_mortgages)
        )
        
        return {
            'term': term,
            'method': 'continuous',
            'mortgage_ids': mortgages.tolist(),
            'optimal_margins': optimal_margins.tolist(),
            'optimal_renewal_probs': optimal_renewal_probs,
            'optimal_expected_profits': optimal_profits,
            'total_expected_profit': total_expected_profit,
            'total_expected_volume': total_expected_volume,
            'optimization_success': best_result.success,
            'n_iterations': best_result.nit,
            'volume_constraint_met': total_expected_volume >= min_volume
        }
    
    def optimize_single_term(self, df: pd.DataFrame, term: int) -> Dict:
        """
        Run optimization for a single term.
        
        Args:
            df: Full DataFrame with all terms (will be filtered)
            term: Term length to optimize (e.g., 1, 2, 3, 5)
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"\nOptimizing term: {term} years")
        
        # Prepare data
        df_term, metadata = self._prepare_term_data(df, term)
        
        # Run optimization based on method
        if self.config.optimization_method == 'discrete':
            result = self._optimize_discrete(df_term, metadata, term)
        else:
            result = self._optimize_continuous(df_term, metadata, term)
        
        # Log results
        logger.info(f"✓ Optimization complete for {term}-year term")
        logger.info(f"  Total expected profit: ${result['total_expected_profit']:,.2f}")
        logger.info(f"  Total expected volume: ${result['total_expected_volume']:,.2f}")
        logger.info(f"  Volume constraint met: {result['volume_constraint_met']}")
        logger.info(f"  Average optimal margin: {np.mean(result['optimal_margins']):.4f}")
        
        self.results[term] = result
        return result
    
    def optimize_all_terms(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Run optimization for all configured terms.
        
        Args:
            df: Full DataFrame with all data
        
        Returns:
            Dictionary mapping term -> optimization results
        """
        logger.info(f"\n{'='*70}")
        logger.info("Starting optimization for all terms")
        logger.info(f"{'='*70}")
        
        for term in self.config.terms:
            self.optimize_single_term(df, term)
        
        logger.info(f"\n{'='*70}")
        logger.info("All optimizations complete")
        logger.info(f"{'='*70}")
        
        return self.results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

class ResultsFormatter:
    """
    Formats optimization results into various output formats.
    """
    
    @staticmethod
    def create_recommendations_dataframe(results: Dict[int, Dict]) -> pd.DataFrame:
        """
        Create a DataFrame with optimal pricing recommendations.
        
        Output format:
        mortgage_id | optimal_margin_1yr | optimal_margin_2yr | ... | expected_profit_1yr | ...
        
        Args:
            results: Dictionary mapping term -> optimization results
        
        Returns:
            DataFrame with recommendations for all mortgages and terms
        """
        # Get all mortgage IDs (should be same across all terms)
        first_term = list(results.keys())[0]
        mortgage_ids = results[first_term]['mortgage_ids']
        
        # Initialize output dictionary
        output_data = {'mortgage_id': mortgage_ids}
        
        # Add columns for each term
        for term, result in sorted(results.items()):
            # Optimal margins
            output_data[f'optimal_margin_{term}yr'] = result['optimal_margins']
            
            # Expected renewal probabilities
            output_data[f'expected_renewal_prob_{term}yr'] = result['optimal_renewal_probs']
            
            # Expected profits
            output_data[f'expected_profit_{term}yr'] = result['optimal_expected_profits']
        
        df_recommendations = pd.DataFrame(output_data)
        
        return df_recommendations
    
    @staticmethod
    def create_summary_statistics(results: Dict[int, Dict]) -> pd.DataFrame:
        """
        Create summary statistics across all terms.
        
        Returns DataFrame with aggregate metrics for each term.
        """
        summary_data = []
        
        for term, result in sorted(results.items()):
            summary_data.append({
                'term': term,
                'n_mortgages': len(result['mortgage_ids']),
                'total_expected_profit': result['total_expected_profit'],
                'total_expected_volume': result['total_expected_volume'],
                'avg_optimal_margin': np.mean(result['optimal_margins']),
                'min_optimal_margin': np.min(result['optimal_margins']),
                'max_optimal_margin': np.max(result['optimal_margins']),
                'avg_renewal_prob': np.mean(result['optimal_renewal_probs']),
                'volume_constraint_met': result['volume_constraint_met'],
                'optimization_success': result['optimization_success']
            })
        
        return pd.DataFrame(summary_data)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MortgagePricingPipeline:
    """
    End-to-end pipeline orchestrating all components.
    
    Usage:
        config = OptimizationConfig(terms=[1, 2, 3, 5], min_volume_threshold=1e9)
        pipeline = MortgagePricingPipeline(config)
        recommendations = pipeline.run(input_df)
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.config.validate()
        
        self.processor = MortgageDataProcessor(config)
        self.smoother = RenewalCurveSmoother(config)
        self.optimizer = MortgagePortfolioOptimizer(config)
        self.formatter = ResultsFormatter()
        
        logger.info(f"\n{'='*70}")
        logger.info("Mortgage Pricing Optimization Pipeline Initialized")
        logger.info(f"{'='*70}")
        logger.info(f"Configuration: {config}")
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute complete optimization pipeline.
        
        Args:
            df: Input DataFrame with columns [mortgage_id, margin, renewal_probability, 
                                             balance, cost_of_funds]
        
        Returns:
            recommendations_df: Optimal pricing recommendations
            summary_df: Summary statistics
            processed_df: Full processed data with smoothed values
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Process data
        logger.info("\n[1/4] Data Processing")
        processed_df = self.processor.process(df)
        
        # Step 2: Smooth renewal curves
        logger.info("\n[2/4] Curve Smoothing")
        smoothed_df = self.smoother.smooth_dataframe(processed_df)
        
        # Step 3: Optimize
        logger.info("\n[3/4] Portfolio Optimization")
        results = self.optimizer.optimize_all_terms(smoothed_df)
        
        # Step 4: Format output
        logger.info("\n[4/4] Formatting Results")
        recommendations_df = self.formatter.create_recommendations_dataframe(results)
        summary_df = self.formatter.create_summary_statistics(results)
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"\nRecommendations generated for {len(recommendations_df)} mortgages")
        logger.info(f"Across {len(self.config.terms)} term lengths")
        
        return recommendations_df, summary_df, smoothed_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use the pipeline.
    """
    
    # Step 1: Load your data
    # df = pd.read_csv('mortgage_predictions.csv')
    
    # For demonstration, create synthetic data
    np.random.seed(42)
    
    n_mortgages = 100
    n_margins = 200
    
    data = []
    for mort_id in range(n_mortgages):
        balance = np.random.uniform(100000, 500000)
        cof = np.random.uniform(0.02, 0.05)
        
        for margin in np.linspace(0, 2, n_margins):
            # Simulate declining renewal probability with increasing margin
            base_prob = 0.9 * np.exp(-1.5 * margin)
            noise = np.random.normal(0, 0.05)
            renewal_prob = np.clip(base_prob + noise, 0, 1)
            
            data.append({
                'mortgage_id': f'MORT_{mort_id:04d}',
                'margin': margin,
                'renewal_probability': renewal_prob,
                'balance': balance,
                'cost_of_funds': cof
            })
    
    df = pd.DataFrame(data)
    
    # Step 2: Configure optimization
    config = OptimizationConfig(
        terms=[1, 2, 3, 5],
        min_volume_threshold=10_000_000,  # $10M minimum volume
        smoothing_method='isotonic_spline',
        smoothing_strength=0.03,
        optimization_method='continuous',
        max_iterations=1000,
        n_restarts=3
    )
    
    # Step 3: Run pipeline
    pipeline = MortgagePricingPipeline(config)
    recommendations, summary, processed_data = pipeline.run(df)
    
    # Step 4: Save results
    recommendations.to_csv('optimal_pricing_recommendations.csv', index=False)
    summary.to_csv('optimization_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("RESULTS PREVIEW")
    print("="*70)
    print("\nRecommendations (first 5 mortgages):")
    print(recommendations.head())
    print("\nSummary Statistics:")
    print(summary)
    
    return recommendations, summary, processed_data


if __name__ == '__main__':
    recommendations, summary, processed_data = example_usage()
```

## **3. Key Implementation Notes**

### **For Your Copilot Prompt:**

```
Please implement the following mortgage pricing optimization system based on this specification:

CONTEXT:
We have mortgage renewal data where a causal inference model predicts renewal probability 
for 200 different margin values (0 to 2) for each mortgage. We need to find optimal pricing 
across multiple term lengths (1, 2, 3, 5 years) while maintaining minimum portfolio volume.

REQUIRED COMPONENTS:

1. MortgageDataProcessor class:
   - validate_input_data(): Check schema, ranges, completeness
   - expand_for_terms(): Create cartesian product with term lengths
   - calculate_profit(): profit = (margin - cost_of_funds) × balance × term

2. RenewalCurveSmoother class:
   - smooth_single_curve(): Two-stage smoothing (isotonic + spline)
   - smooth_dataframe(): Apply to all mortgage-term combinations
   - Handle sparse data at extreme margins

3. MortgagePortfolioOptimizer class:
   - optimize_single_term(): Optimize for one term length
   - optimize_all_terms(): Run across all terms
   - Support both discrete and continuous optimization
   - Objective: maximize Σ(profit × renewal_probability)
   - Constraint: Σ(renewal_probability × balance) ≥ min_volume

4. ResultsFormatter class:
   - create_recommendations_dataframe(): Wide format output
   - create_summary_statistics(): Aggregate metrics

5. MortgagePricingPipeline class:
   - Orchestrate all components
   - Handle logging and error checking

REQUIREMENTS:
- Use scipy.optimize for optimization
- Use sklearn.isotonic for monotonicity
- Include comprehensive logging
- Add docstrings to all methods
- Handle edge cases gracefully

Please implement with proper type hints and error handling.
```

This specification provides everything needed for implementation. Would you like me to add visualization code or testing utilities as well?