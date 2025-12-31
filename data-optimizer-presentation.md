
## **1. Why Optimization After Causal Modeling?**

### **The Core Problem**

**Causal models provide:** $P(\text{renew} | \text{price}, X)$ for 200 discrete price points

**Causal models DON'T provide:**
- Optimal decisions under portfolio constraints
- Segment-level pricing (operational requirement)
- Multi-term coordination
- Volume guarantees

**Solution:** Two-stage framework

```
Stage 1: Causal Model → Predictions
Stage 2: Optimization → Decisions under constraints
```

---

## **2. Mathematical Formulation**

### **Notation**

- $\mathcal{S}$: Segments (indexed by $s$)
- $\mathcal{T}$: Terms (indexed by $t$, e.g., $\{1,2,3,5\}$ years)
- $\mathcal{P}$: Price options (200 discrete margins)
- $b_i$: Balance of mortgage $i$
- $c_i$: Cost of funds for mortgage $i$
- $P_{i,t,p}$: Renewal probability (from causal model)
- $V_{\min}$: Minimum required volume

### **Decision Variables**

$$x_{s,t} \in \mathcal{P} \quad \text{(margin for segment } s \text{, term } t \text{)}$$

### **Objective Function**

Expected profit for mortgage $i$:
$$\pi_{i,t,p} = (p - c_i) \times b_i \times t \times P_{i,t,p}$$

Total portfolio profit:
$$\max_{x_{s,t}} \sum_{s \in \mathcal{S}} \sum_{t \in \mathcal{T}} \sum_{i \in s} \pi_{i,t,x_{s,t}}$$

### **Constraints**

Volume constraint:
$$\sum_{s,t} \sum_{i \in s} P_{i,t,x_{s,t}} \times b_i \geq V_{\min}$$

Price bounds:
$$p_{\min} \leq x_{s,t} \leq p_{\max}$$

### **Segment-Level Aggregation**

Segment average expected profit:
$$\bar{\pi}_{s,t,p} = \frac{1}{|s|} \sum_{i \in s} (p - c_i) \times b_i \times t \times P_{i,t,p}$$

Reformulated objective:
$$\max_{x_{s,t}} \sum_{s,t} |s| \times \bar{\pi}_{s,t,x_{s,t}}$$

---

## **3. Numerical Example**

### **Setup**

**Portfolio:** 3,500 mortgages, 3 segments, 3-year term, 5 price options

| Segment | # | Avg Balance | Avg CoF |
|---------|---|-------------|---------|
| S1 (Shoppers) | 1,000 | $300K | 1.5% |
| S2 (Aware) | 2,000 | $250K | 1.6% |
| S3 (Loyal) | 500 | $400K | 1.4% |

**Renewal Probabilities:**

| Price | S1 | S2 | S3 |
|-------|----|----|-------|
| 3.29% | 0.92 | 0.91 | 0.94 |
| 3.49% | 0.88 | 0.89 | 0.93 |
| 3.69% | 0.75 | 0.85 | 0.92 |
| 3.89% | 0.60 | 0.78 | 0.91 |
| 3.99% | 0.50 | 0.73 | 0.90 |

### **Expected Profit Calculation**

**S1 @ 3.49%:** 
\[(0.0349 - 0.015) \times 300K \times 3 \times 0.88 = \$15,761\]
per mortgage  → Segment total: $\$15.8M$

**S2 @ 3.89%:** \[(0.0389 - 0.016) \times 250K \times 3 \times 0.78 = \$13,397\]
→ Segment total: $\$26.8M$

**S3 @ 3.99%:** \[(0.0399 - 0.014) \times 400K \times 3 \times 0.90 = \$27,972\] 
→ Segment total: $\$14.0M$

**Unconstrained optimal:** $\$56.6M$ profit, $834M$ volume (80% retention)

**Problem:** Violates 85% volume constraint ($886M$ required)

**Constrained solution:** Lower S2 to 3.49% → $\$55.0M$ profit, $889M$ volume ✓

**Trade-off:** Sacrifice $\$1.6M$ profit to meet volume target

---

## **4. Implementation Pipeline**

```python
# Pseudocode
def optimize_portfolio(df_predictions, config):
    # Step 1: Expand for terms
    df = expand_terms(df_predictions, terms=[1,2,3,5])
    
    # Step 2: Calculate profit
    df['profit'] = (df['margin'] - df['cof']) * df['balance'] * df['term']
    df['expected_profit'] = df['profit'] * df['renewal_prob']
    
    # Step 3: Segment by elasticity
    df['segment'] = assign_segments(df, method='elasticity_quantiles')
    
    # Step 4: Smooth curves (isotonic + spline)
    df['renewal_prob_smooth'] = smooth_curves(df, group_by=['segment','term'])
    
    # Step 5: Aggregate to segment level
    segment_data = df.groupby(['segment','term','margin']).agg({
        'expected_profit': 'mean',
        'renewal_prob_smooth': 'mean'
    })
    
    # Step 6: Optimize per term
    results = {}
    for term in [1,2,3,5]:
        results[term] = optimize_lp(segment_data[term], config.min_volume)
    
    return format_recommendations(results)
```

---

## **5. Key Technical Components**

### **Smoothing: Two-Stage Approach**

```python
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression

def smooth_curve(margins, renewal_probs, s=0.03):
    # Stage 1: Enforce monotonicity
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    probs_mono = iso.fit_transform(margins, renewal_probs)
    
    # Stage 2: Spline smoothing
    spline = UnivariateSpline(margins, probs_mono, s=s, k=3)
    return np.clip(spline(margins), 0, 1)
```

**Why:** Causal predictions are noisy at extremes; smoothing ensures monotonicity and stability

### **Segmentation: Elasticity-Based**

Price elasticity: $\varepsilon = \frac{\Delta P(\text{renew}) / P(\text{renew})}{\Delta \text{price} / \text{price}}$

```python
def create_segments(df, n_segments=20):
    # Calculate elasticity for each mortgage
    elasticities = calculate_arc_elasticity(df)
    
    # Quantile-based segmentation
    df['segment'] = pd.qcut(elasticities, q=n_segments, labels=False)
    
    return df
```

**Benefit:** Reduces from O(millions) to O(thousands) of variables

### **Optimization: Three Approaches**

**1. Discrete (Differential Evolution):**
- Select from exact 200 price points
- Global search over discrete grid
- Use for non-smooth objectives

**2. Continuous (SLSQP):**
- Interpolate between prices (cubic splines)
- Smooth optimization landscape
- Faster convergence

**3. Linear Programming (Recommended):**
- Formulate as discrete choice (MIP)
- Provably optimal solution
- Fastest for large-scale problems

```python
from ortools.linear_solver import pywraplp

def optimize_lp(segment_data, min_volume):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    # Binary variables: y[s,p] = 1 if segment s chooses price p
    y = {(s,p): solver.BoolVar(f'y_{s}_{p}') 
         for s in segments for p in prices}
    
    # Each segment chooses exactly one price
    for s in segments:
        solver.Add(sum(y[s,p] for p in prices) == 1)
    
    # Objective: maximize profit
    solver.Maximize(sum(y[s,p] * profit[s,p] for s,p in y))
    
    # Volume constraint
    solver.Add(sum(y[s,p] * volume[s,p] for s,p in y) >= min_volume)
    
    solver.Solve()
    return extract_solution(y)
```

---

## **6. Performance Metrics**

**Computational Complexity:**
- Without segmentation: O(millions of mortgages) - infeasible
- With segmentation: O(10 segments × 4 terms × 200 prices) ≈ 8K variables
- **Solve time:** <1 minute

**Typical Results:**
- **Profit lift:** 8-15% vs one-size-fits-all
- **Volume control:** Guaranteed 85-95% retention
- **Segmentation efficiency:** 90% of personalization value at 1% computational cost

---

## **7. Validation Strategy**

### **Backtesting**
```python
for month in historical_months:
    train = data[data.month < month]
    test = data[data.month == month]
    
    recommendations = optimizer.run(train)
    actuals = test.merge(recommendations)
    
    metrics[month] = {
        'predicted_profit': recommendations.profit.sum(),
        'actual_profit': actuals.profit.sum(),
        'error': abs(predicted - actual) / actual
    }
```

## **8. Key Takeaways**

| Aspect | Causal Model Alone | With Optimization |
|--------|-------------------|-------------------|
| Predictions | ✓ | ✓ |
| Portfolio constraints | ✗ | ✓ |
| Segment pricing | ✗ | ✓ |
| Multi-term coordination | ✗ | ✓ |
| Volume guarantees | ✗ | ✓ |
| Actionable decisions | ✗ | ✓ |

**Bottom Line:** Causal models predict elasticity to the price/rate; optimization prescribes strategy under real-world constraints.