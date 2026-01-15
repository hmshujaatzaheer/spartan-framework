"""
Tests for RPPO (Reasoning-Privacy Pareto Optimization) module.
"""

import numpy as np
import pytest

from spartan.config import SPARTANConfig
from spartan.rppo import RPPO, RPPOResult
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.pareto import ParetoFront


class TestRPPO:
    """Tests for main RPPO class."""

    def test_init_default(self):
        """Test default initialization."""
        rppo = RPPO()
        assert rppo is not None
        assert rppo.config is not None

    def test_init_with_config(self, config):
        """Test initialization with config."""
        rppo = RPPO(config=config)
        assert rppo.config == config

    def test_update_single(self):
        """Test single update."""
        rppo = RPPO()
        rppo.update(
            {
                "risk_score": 0.3,
                "accuracy": 0.9,
                "compute": 0.4,
            }
        )

        stats = rppo.get_statistics()
        assert stats["total_episodes"] == 1

    def test_update_multiple(self):
        """Test multiple updates."""
        rppo = RPPO()

        for i in range(15):
            rppo.update(
                {
                    "risk_score": 0.3 + 0.02 * i,
                    "accuracy": 0.9 - 0.01 * i,
                    "compute": 0.4,
                }
            )

        stats = rppo.get_statistics()
        assert stats["total_episodes"] == 15

    def test_get_optimal_params_insufficient_data(self):
        """Test with insufficient data."""
        rppo = RPPO()
        rppo.update({"risk_score": 0.3})

        result = rppo.get_optimal_params()
        assert result is None  # Not enough data

    def test_get_optimal_params_sufficient_data(self):
        """Test with sufficient data."""
        rppo = RPPO()

        for i in range(12):
            rppo.update(
                {
                    "risk_score": 0.3 + 0.01 * i,
                    "accuracy": 0.85,
                    "compute": 0.4,
                }
            )

        result = rppo.get_optimal_params()
        assert isinstance(result, RPPOResult)
        assert "epsilon_min" in result.params
        assert "epsilon_max" in result.params

    def test_pareto_front(self):
        """Test Pareto front tracking."""
        rppo = RPPO()

        for i in range(15):
            rppo.update(
                {
                    "risk_score": 0.2 + 0.05 * (i % 5),
                    "accuracy": 0.9 - 0.02 * (i % 4),
                    "compute": 0.3,
                }
            )

        front = rppo.get_pareto_front()
        assert isinstance(front, list)

    def test_reset(self):
        """Test reset functionality."""
        rppo = RPPO()

        for i in range(5):
            rppo.update({"risk_score": 0.3})

        rppo.reset()
        stats = rppo.get_statistics()
        assert stats["total_episodes"] == 0

    def test_objective_weights(self):
        """Test objective weight setting."""
        rppo = RPPO()
        rppo.set_objective_weights(0.5, 0.3, 0.2)

        stats = rppo.get_statistics()
        weights = stats["objective_weights"]
        assert sum(weights) == pytest.approx(1.0)

    def test_result_to_dict(self):
        """Test result serialization."""
        rppo = RPPO()

        for i in range(12):
            rppo.update({"risk_score": 0.3})

        result = rppo.get_optimal_params()
        result_dict = result.to_dict()

        assert "params" in result_dict
        assert "reward" in result_dict
        assert "is_pareto_optimal" in result_dict


class TestUCBBandit:
    """Tests for UCB bandit."""

    def test_init(self):
        """Test initialization."""
        bandit = UCBBandit(num_arms=10)
        assert bandit.num_arms == 10

    def test_initial_exploration(self):
        """Test initial exploration phase."""
        bandit = UCBBandit(num_arms=5)

        selected_arms = []
        for _ in range(5):
            arm = bandit.select_arm()
            bandit.update(arm, 0.5)
            selected_arms.append(arm)

        # Should explore all arms once
        assert len(set(selected_arms)) == 5

    def test_exploitation(self):
        """Test exploitation after exploration."""
        bandit = UCBBandit(num_arms=3, exploration_constant=0.1)

        # Explore all arms
        for i in range(3):
            bandit.update(i, 0.3 if i != 1 else 0.9)

        # Further pulls should favor arm 1
        arm_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(100):
            arm = bandit.select_arm()
            bandit.update(arm, 0.3 if arm != 1 else 0.9)
            arm_counts[arm] += 1

        assert arm_counts[1] >= arm_counts[0]
        assert arm_counts[1] >= arm_counts[2]

    def test_update(self):
        """Test update functionality."""
        bandit = UCBBandit(num_arms=3)

        bandit.update(0, 0.8)
        bandit.update(0, 0.7)
        bandit.update(1, 0.5)

        stats = bandit.get_arm_stats()
        assert stats[0]["count"] == 2
        assert stats[1]["count"] == 1

    def test_get_best_arm(self):
        """Test best arm selection."""
        bandit = UCBBandit(num_arms=3)

        bandit.update(0, 0.3)
        bandit.update(1, 0.9)
        bandit.update(2, 0.5)

        best = bandit.get_best_arm()
        assert best == 1

    def test_reset(self):
        """Test reset functionality."""
        bandit = UCBBandit(num_arms=3)

        bandit.update(0, 0.5)
        bandit.update(1, 0.6)
        bandit.reset()

        stats = bandit.get_arm_stats()
        assert all(s["count"] == 0 for s in stats)

    def test_invalid_arm(self):
        """Test invalid arm handling."""
        bandit = UCBBandit(num_arms=3)

        with pytest.raises(ValueError):
            bandit.update(5, 0.5)


class TestParetoFront:
    """Tests for Pareto front utilities."""

    def test_init(self):
        """Test initialization."""
        pf = ParetoFront()
        assert pf.size() == 0

    def test_add_point(self):
        """Test adding points."""
        pf = ParetoFront()

        added = pf.add_point(
            point=np.array([0.8, 0.7, 0.6]),
            params={"config": "A"},
        )

        assert added
        assert pf.size() == 1

    def test_dominance(self):
        """Test dominance checking."""
        pf = ParetoFront()

        pf.add_point(np.array([0.9, 0.9, 0.9]), {"id": "A"})

        # This point is dominated by A
        added = pf.add_point(np.array([0.8, 0.8, 0.8]), {"id": "B"})
        assert not added

        # This point is not dominated (better in one dimension)
        added = pf.add_point(np.array([0.95, 0.85, 0.85]), {"id": "C"})
        assert added

    def test_pareto_optimal_check(self):
        """Test Pareto optimality checking."""
        pf = ParetoFront()
        pf.add_point(np.array([0.8, 0.7]), {"id": "A"})
        pf.add_point(np.array([0.7, 0.8]), {"id": "B"})

        # Point dominated by both
        assert not pf.is_pareto_optimal(np.array([0.6, 0.6]))

        # Point not dominated
        assert pf.is_pareto_optimal(np.array([0.85, 0.85]))

    def test_get_front(self):
        """Test getting Pareto front."""
        pf = ParetoFront()

        pf.add_point(np.array([0.8, 0.7]), {"id": "A"})
        pf.add_point(np.array([0.7, 0.8]), {"id": "B"})

        front = pf.get_front()
        assert len(front) == 2

    def test_hypervolume(self):
        """Test hypervolume computation."""
        pf = ParetoFront()

        pf.add_point(np.array([0.8, 0.7]), {"id": "A"})
        pf.add_point(np.array([0.7, 0.8]), {"id": "B"})

        hv = pf.get_hypervolume()
        assert hv > 0

    def test_closest_to_ideal(self):
        """Test closest to ideal point."""
        pf = ParetoFront()

        pf.add_point(np.array([0.9, 0.7]), {"id": "A"})
        pf.add_point(np.array([0.8, 0.85]), {"id": "B"})
        pf.add_point(np.array([0.7, 0.9]), {"id": "C"})

        closest = pf.get_closest_to_ideal(np.array([1.0, 1.0]))
        assert closest is not None
        assert closest[1]["id"] == "B"  # Most balanced

    def test_clear(self):
        """Test clearing the front."""
        pf = ParetoFront()

        pf.add_point(np.array([0.8, 0.7]), {"id": "A"})
        assert pf.size() == 1

        pf.clear()
        assert pf.size() == 0
