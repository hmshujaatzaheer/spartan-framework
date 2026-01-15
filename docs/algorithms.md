# SPARTAN Algorithms

This document describes the core algorithms implemented in SPARTAN.

## Algorithm 1: Mechanistic Privacy Leakage Quantification (MPLQ)

MPLQ quantifies privacy leakage through three TTC mechanism channels.

### Mathematical Formulation

**Total Leakage Score:**
```
L_total = σ(ψ · (α·L_prm + β·L_vote + γ·L_mcts))
```

Where:
- `ψ` = importance weight
- `α, β, γ` = learned component weights
- `σ` = sigmoid function

### PRM Leakage (Equation 1)

```
L_prm(x) = D_KL(P_θ(s|x) || P_ref(s)) + λ·Var(s)
```

- `P_θ(s|x)` = PRM score distribution for input x
- `P_ref(s)` = reference distribution from non-member data
- `λ` = variance weight (default: 0.1)

### Vote Leakage (Equation 2)

```
L_vote(x) = 1 - H(v) / log(K)
```

- `H(v)` = Shannon entropy of vote distribution
- `K` = number of candidates
- Low entropy indicates memorization

### MCTS Leakage (Equation 3)

```
L_mcts(x) = max_{n ∈ T} |V_θ(n) - V_baseline(n)|
```

- `V_θ(n)` = value network output for node n
- `V_baseline(n)` = expected baseline value
- Large deviations indicate memorization

---

## Algorithm 2: Reasoning-Aware Adaptive Sanitization (RAAS)

RAAS applies defense mechanisms proportional to detected risk.

### Defense Intensity (Equation 4)

```
ε = ε_min + (ε_max - ε_min) · L · ψ
```

Where:
- `L` = total leakage score
- `ψ` = importance weight
- `ε_min, ε_max` = defense intensity bounds

### PRM Defense: Feature-Selective Noise

Exploits NL-blindness (Ma et al.):
- High noise on natural language content
- Minimal noise on mathematical expressions

### Vote Defense: Temperature Scaling

```
v'_k = softmax(v_k / T)
```

Where `T = 1 + ε · (L_vote - τ_vote)` for `L_vote > τ_vote`

### MCTS Defense: Value Perturbation

```
V'(n) = V(n) + N(0, ε · depth(n))
```

Depth-scaled noise preserves search quality while obscuring memorization.

---

## Algorithm 3: Reasoning-Privacy Pareto Optimization (RPPO)

RPPO optimizes defense parameters for best privacy-utility tradeoff.

### Multi-Objective Reward

```
R = ω_1·R_acc + ω_2·R_priv + ω_3·R_comp
```

Where:
- `R_acc` = accuracy retention
- `R_priv` = privacy protection (1 - attack success)
- `R_comp` = computational efficiency

### UCB Bandit Selection

```
UCB(a) = R̄_a + c · √(log(t) / N_a)
```

Where:
- `R̄_a` = average reward for arm a
- `c` = exploration constant
- `t` = total pulls
- `N_a` = pulls for arm a

### Pareto Front Tracking

Maintains set of non-dominated configurations where no single configuration dominates another in all objectives.

---

## Theoretical Guarantees

### Privacy Bound (Theorem 1)

Under RAAS with defense intensity ε:
```
Adv_MIA ≤ O(e^{-ε·L})
```

The membership inference advantage decreases exponentially with defense intensity.

### Utility Preservation (Theorem 2)

With importance-weighted defense:
```
E[Acc_defended] ≥ (1 - ε_max·L_max) · E[Acc_original]
```

Accuracy retention is bounded by maximum defense intensity.

---

## Implementation Notes

1. **Numerical Stability**: All sigmoid computations use numerically stable implementations
2. **Normalization**: Component weights are automatically normalized to sum to 1
3. **Clipping**: Defense intensity is clipped to [ε_min, ε_max]
4. **Reference Distribution**: PRM reference is computed from held-out non-member data

<!-- Last updated: 2026-01-15 -->
