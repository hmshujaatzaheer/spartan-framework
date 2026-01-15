# API Reference

## Main Classes

### SPARTAN

```python
class SPARTAN:
    """Main SPARTAN framework class."""
    
    def __init__(
        self,
        model: BaseReasoningLLM,
        config: Optional[SPARTANConfig] = None,
        enable_optimization: bool = True,
    ):
        """Initialize SPARTAN.
        
        Args:
            model: Reasoning LLM to protect
            config: Configuration options
            enable_optimization: Enable RPPO optimization
        """
    
    def process(
        self,
        query: str,
        num_samples: int = 5,
        return_trace: bool = False,
    ) -> SPARTANResult:
        """Process query through SPARTAN pipeline."""
    
    def batch_process(
        self,
        queries: List[str],
        num_samples: int = 5,
    ) -> List[SPARTANResult]:
        """Process multiple queries."""
```

### SPARTANConfig

```python
@dataclass
class SPARTANConfig:
    """Configuration for SPARTAN framework."""
    
    # MPLQ settings
    prm_threshold: float = 0.3
    vote_threshold: float = 0.4
    mcts_threshold: float = 0.5
    
    # RAAS settings
    epsilon_min: float = 0.01
    epsilon_max: float = 0.5
    use_feature_selective: bool = True
    use_implicit_rewards: bool = True
    
    # RPPO settings
    learning_rate: float = 0.01
    num_arms: int = 10
    accuracy_weight: float = 0.4
    privacy_weight: float = 0.4
    compute_weight: float = 0.2
```

### SPARTANResult

```python
@dataclass
class SPARTANResult:
    """Result from SPARTAN processing."""
    
    output: str              # Sanitized output
    original_output: str     # Original output
    risk_score: float        # Privacy risk (0-1)
    risk_analysis: MPLQResult
    defense_result: RAASResult
    optimization_result: Optional[RPPOResult]
    defense_applied: bool
    metadata: Dict[str, Any]
```

---

## MPLQ Module

### MPLQ

```python
class MPLQ:
    """Mechanistic Privacy Leakage Quantification."""
    
    def analyze(
        self,
        query: str,
        reasoning_steps: Optional[List[str]] = None,
        prm_scores: Optional[List[float]] = None,
        vote_distribution: Optional[List[float]] = None,
        mcts_values: Optional[List[float]] = None,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> MPLQResult:
        """Analyze privacy leakage."""
    
    def update_weights(
        self,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> None:
        """Update component weights."""
```

### MPLQResult

```python
@dataclass
class MPLQResult:
    total_risk: float
    prm_leakage: float
    vote_leakage: float
    mcts_leakage: float
    importance_weight: float
    component_weights: Tuple[float, float, float]
    details: Dict[str, Any]
```

---

## RAAS Module

### RAAS

```python
class RAAS:
    """Reasoning-Aware Adaptive Sanitization."""
    
    def sanitize(
        self,
        output: str,
        risk_analysis: MPLQResult,
        reasoning_steps: Optional[List[str]] = None,
        vote_distribution: Optional[List[float]] = None,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> RAASResult:
        """Apply adaptive defense."""
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update defense parameters."""
```

### RAASResult

```python
@dataclass
class RAASResult:
    sanitized_output: str
    original_output: str
    defense_applied: bool
    epsilon_used: float
    prm_defense_applied: bool
    vote_defense_applied: bool
    mcts_defense_applied: bool
    details: Dict[str, Any]
```

---

## RPPO Module

### RPPO

```python
class RPPO:
    """Reasoning-Privacy Pareto Optimization."""
    
    def update(self, observation: Dict[str, Any]) -> None:
        """Update with new observation."""
    
    def get_optimal_params(self) -> Optional[RPPOResult]:
        """Get optimized parameters."""
    
    def get_pareto_front(self) -> List[Tuple[np.ndarray, Dict]]:
        """Get Pareto-optimal configurations."""
```

---

## Attack Module

### NLBAAttack

```python
class NLBAAttack(BaseAttack):
    """Natural Language Blindness Attack."""
    
    def execute(
        self,
        target_model: Any,
        query: str,
        prm_scores: Optional[List[float]] = None,
        reasoning_steps: Optional[List[str]] = None,
    ) -> AttackResult:
        """Execute NLBA attack."""
```

### SMVAAttack

```python
class SMVAAttack(BaseAttack):
    """Single-Model Voting Attack."""
    
    def execute(
        self,
        target_model: Any,
        query: str,
        vote_distribution: Optional[List[float]] = None,
    ) -> AttackResult:
        """Execute SMVA attack."""
```

### MVNAAttack

```python
class MVNAAttack(BaseAttack):
    """MCTS Value Network Attack."""
    
    def execute(
        self,
        target_model: Any,
        query: str,
        mcts_values: Optional[List[float]] = None,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> AttackResult:
        """Execute MVNA attack."""
```

---

## Utility Functions

### Metrics

```python
def compute_auc_roc(labels: List[int], scores: List[float]) -> float
def compute_accuracy(labels: List[int], predictions: List[int]) -> float
def compute_tpr_at_fpr(labels: List[int], scores: List[float], fpr_threshold: float) -> float
def compute_f1_score(labels: List[int], predictions: List[int]) -> float
```

### Distributions

```python
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float
def js_divergence(p: np.ndarray, q: np.ndarray) -> float
def entropy(distribution: np.ndarray) -> float
```

### Noise

```python
def gaussian_noise(shape, scale: float, seed: Optional[int] = None) -> np.ndarray
def laplace_noise(shape, scale: float, seed: Optional[int] = None) -> np.ndarray
def calibrated_noise(shape, sensitivity: float, epsilon: float, mechanism: str) -> np.ndarray
```

<!-- Last updated: 2026-01-15 -->
