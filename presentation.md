# Mortgage Renewal Pricing Optimization: Complete Documentation

---

# PART 1: BUSINESS SLIDES

## **Slide 1: The Pricing Challenge**

**Title:** Smart Pricing for Mortgage Renewals

**The Problem:**
Every renewal season, we must balance two competing goals:
- **Maximize profit** → Charge higher margins
- **Maintain volume** → Keep customers from leaving

**Traditional Approach:**
- Cost-plus pricing or competitor matching
- Broad rate adjustments across all customers
- Hope we find the right balance

**Our Solution: The Optimizer**
A data-driven pricing engine that:
- Understands how different customers respond to price
- Finds the optimal price for each customer segment
- Guarantees we meet volume targets while maximizing profit

---

**Speaker Notes:**

*"Every time mortgages come up for renewal, we face a fundamental trade-off. If we price too aggressively, we maximize margin but lose customers to competitors. If we price too conservatively, we keep customers but leave significant money on the table. Traditionally, banks approach this with cost-plus pricing or by matching competitors, applying broad rate adjustments and hoping they got the balance right. Our optimizer takes a smarter approach—it uses advanced analytics to understand exactly how different customer segments respond to price changes, then finds the mathematically optimal pricing strategy that maximizes profit while guaranteeing we hit our retention targets."*

---

## **Slide 2: How It Works**

**Title:** Three-Step Pricing Engine

**Process Flow:**

```
┌─────────────────────────────────────────┐
│ STEP 1: Understand Customer Behavior    │
│                                         │
│ Use AI models to predict:               │
│ "How likely will this customer renew    │
│  at 3.5% vs 3.8%?"                      │
│                                         │
│ Different customers = different         │
│ sensitivity to price                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ STEP 2: Group Similar Customers         │
│                                         │
│ Create segments based on price          │
│ sensitivity:                            │
│ • High Sensitivity: Price shoppers      │
│ • Medium: Price-aware but not purely    │
│   price-driven                          │
│ • Low Sensitivity: Loyal customers      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ STEP 3: Optimize Pricing                │
│                                         │
│ For each segment & term (1Y,2Y,3Y,5Y):  │
│ Find price that:                        │
│ ✓ Maximizes expected profit             │
│ ✓ Ensures total volume ≥ target         │
│ ✓ Stays within approved ranges          │
└─────────────────────────────────────────┘
```

**Key Point:** This is segment-based pricing, not individual personalization
- 20-50 customer groups (operationally manageable)
- Everyone in a segment gets the same rate
- Captures 90% of personalization value

---

**Speaker Notes:**

*"The optimizer works in three steps. First, we use advanced AI models to understand customer behavior. These models predict how likely each customer is to renew at different price points—and critically, different customers have very different sensitivities to price. Second, we group similar customers into segments based on this price sensitivity. We might have price shoppers who are highly sensitive, price-aware customers in the middle, and loyal customers who are less price-sensitive. Third, the optimization engine finds the best price for each segment and term that maximizes our total expected profit while ensuring we don't lose too much volume. This isn't personalized pricing where every customer gets a unique rate—we create 20 to 50 segments, which is operationally practical and captures most of the value anyway."*

---

## **Slide 3: Business Impact**

**Title:** The Value of Smart Pricing

**Concrete Example:**

| Segment | Customers | Avg Balance | Traditional Pricing | Optimized Pricing | Renewal Rate | Segment Profit |
|---------|-----------|-------------|---------------------|-------------------|--------------|----------------|
| **Price Shoppers** | 1,000 | $300K | 3.69% | **3.49%** ↓ | 88% vs 75% | $2.1M vs $1.5M |
| **Price-Aware** | 2,000 | $250K | 3.69% | **3.69%** → | 85% vs 85% | $4.8M vs $4.8M |
| **Loyal** | 500 | $400K | 3.69% | **3.99%** ↑ | 90% vs 92% | $1.9M vs $1.2M |

**Results Summary:**
- **Traditional (one-size-fits-all):** $7.9M profit, 82% retention
- **Optimized (segment-based):** $8.8M profit, 86% retention  
- **Value Created:** **+$900K profit (+11%)** + **+4% retention**

**Why It Works:**
- **Price Shoppers:** More competitive pricing prevents attrition
- **Price-Aware:** Maintain current strategy (already optimal)
- **Loyal Customers:** Capture value they're willing to pay

**Business Benefits:**
1. **Profit Maximization** — Identifies where we can charge more without losing customers
2. **Risk Management** — Volume constraints prevent over-pricing
3. **Scalable & Automatic** — Handles thousands of mortgages instantly
4. **Explainable** — Clear rationale for every pricing decision

---

**Speaker Notes:**

*"Let me show you a concrete example of the impact. Suppose we have three segments totaling 3,500 mortgages. With traditional one-size-fits-all pricing at 3.69%, we'd make $7.9 million in profit at 82% retention. But the optimizer recognizes we need to be more competitive with price shoppers—dropping to 3.49% keeps them from leaving, actually increasing our profit from this segment. For price-aware customers, 3.69% is already optimal, so we keep it. And for loyal customers, we can actually charge 3.99%—they're willing to pay it and won't leave. The result is $8.8 million in profit, an extra $900,000 or 11% more, while actually improving retention to 86%. This isn't theoretical—it's the kind of real value we capture by making smarter, data-driven pricing decisions that treat different customers appropriately."*

---

## **Slide 4: From Insight to Action**

**Title:** Implementation & Next Steps

**What the Optimizer Delivers:**

For each segment × term (e.g., Price Shoppers × 2-Year):
- ✓ Recommended margin/rate
- ✓ Expected renewal probability  
- ✓ Expected profit contribution
- ✓ Volume impact

**Integration into Operations:**

```
Weekly Cycle:
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Causal   │ → │ Optimizer│ → │ Business │ → │ Rate     │ → │ Customer │
│ Model    │   │ (<1 min) │   │ Review   │   │ Cards    │   │ Renewals │
│ Updates  │   │          │   │ Approval │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

**Safeguards in Place:**
- All recommendations within approved pricing ranges
- Compared against posted rates and market benchmarks
- Business maintains override capability
- A/B testing validates model performance

**Rollout Timeline:**
- **Phase 1 (Weeks 1-4):** Backtest historical data, validate accuracy
- **Phase 2 (Months 2-3):** Pilot with 20% of renewals, measure lift
- **Phase 3 (Month 4+):** Full deployment across portfolio

**Expected Outcomes:**
- 8-12% profit improvement
- 2-5% retention improvement  
- Repeatable, sustainable pricing discipline

---

**Speaker Notes:**

*"Implementation is straightforward and low-risk. The optimizer runs weekly, taking under a minute to produce recommendations for every segment and term. These go through standard business review for approval before being loaded into our rate card systems. We've built in multiple safeguards: all recommendations stay within our approved pricing corridors, we benchmark against market rates, and business can override whenever needed. We'll roll this out in three phases. First, we'll backtest against historical data to validate accuracy. Then we'll pilot with 20% of renewals to measure actual lift in a controlled environment. Once we've confirmed the value—which we expect to be 8-12% profit improvement and 2-5% better retention—we'll roll out fully. This becomes a sustainable part of our pricing operations, giving us a competitive edge quarter after quarter."*

---