"""
Tests for utility modules.
"""

import pytest
import numpy as np

from spartan.utils.metrics import (
    compute_auc_roc,
    compute_accuracy,
    compute_tpr_at_fpr,
    compute_f1_score,
)
from spartan.utils.distributions import (
    kl_divergence,
    js_divergence,
    entropy,
    total_variation_distance,
    hellinger_distance,
)
from spartan.utils.noise import (
    gaussian_noise,
    laplace_noise,
    calibrated_noise,
    adaptive_noise,
)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_auc_roc_perfect(self):
        """Test AUC-ROC with perfect predictions."""
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        
        auc = compute_auc_roc(labels, scores)
        assert auc == 1.0
    
    def test_auc_roc_random(self):
        """Test AUC-ROC with random predictions."""
        np.random.seed(42)
        labels = [0] * 50 + [1] * 50
        scores = np.random.uniform(0, 1, 100).tolist()
        
        auc = compute_auc_roc(labels, scores)
        assert 0.3 <= auc <= 0.7  # Should be around 0.5
    
    def test_auc_roc_empty(self):
        """Test AUC-ROC with empty input."""
        auc = compute_auc_roc([], [])
        assert auc == 0.5
    
    def test_accuracy(self):
        """Test accuracy computation."""
        labels = [0, 1, 1, 0]
        predictions = [0, 1, 0, 0]
        
        acc = compute_accuracy(labels, predictions)
        assert acc == 0.75
    
    def test_accuracy_perfect(self):
        """Test perfect accuracy."""
        labels = [0, 1, 1, 0]
        predictions = [0, 1, 1, 0]
        
        acc = compute_accuracy(labels, predictions)
        assert acc == 1.0
    
    def test_tpr_at_fpr(self):
        """Test TPR at FPR computation."""
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        
        tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.25)
        assert 0 <= tpr <= 1
    
    def test_f1_score(self):
        """Test F1 score computation."""
        labels = [1, 1, 0, 0, 1]
        predictions = [1, 0, 0, 0, 1]
        
        f1 = compute_f1_score(labels, predictions)
        assert 0 <= f1 <= 1


class TestDistributions:
    """Tests for distribution utilities."""
    
    def test_kl_divergence_same(self):
        """Test KL divergence of same distribution."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        
        kl = kl_divergence(p, p)
        assert kl < 0.01  # Should be ~0
    
    def test_kl_divergence_different(self):
        """Test KL divergence of different distributions."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        
        kl = kl_divergence(p, q)
        assert kl > 1.0  # Should be large
    
    def test_js_divergence_bounded(self):
        """Test JS divergence is bounded."""
        p = np.array([0.8, 0.2])
        q = np.array([0.2, 0.8])
        
        js = js_divergence(p, q)
        assert 0 <= js <= np.log(2)
    
    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        
        h = entropy(uniform)
        assert abs(h - np.log(4)) < 0.01
    
    def test_entropy_concentrated(self):
        """Test entropy of concentrated distribution."""
        concentrated = np.array([0.97, 0.01, 0.01, 0.01])
        
        h = entropy(concentrated)
        assert h < np.log(4)  # Less than uniform
    
    def test_total_variation(self):
        """Test total variation distance."""
        p = np.array([0.8, 0.2])
        q = np.array([0.2, 0.8])
        
        tv = total_variation_distance(p, q)
        assert 0 <= tv <= 1
        assert abs(tv - 0.6) < 0.01
    
    def test_hellinger_distance(self):
        """Test Hellinger distance."""
        p = np.array([0.8, 0.2])
        q = np.array([0.2, 0.8])
        
        h = hellinger_distance(p, q)
        assert 0 <= h <= 1


class TestNoise:
    """Tests for noise generation."""
    
    def test_gaussian_noise_shape(self):
        """Test Gaussian noise shape."""
        noise = gaussian_noise((10, 5), scale=1.0)
        
        assert noise.shape == (10, 5)
    
    def test_gaussian_noise_stats(self):
        """Test Gaussian noise statistics."""
        np.random.seed(42)
        noise = gaussian_noise(10000, scale=2.0)
        
        assert abs(np.mean(noise)) < 0.1
        assert abs(np.std(noise) - 2.0) < 0.1
    
    def test_laplace_noise_shape(self):
        """Test Laplace noise shape."""
        noise = laplace_noise((10, 5), scale=1.0)
        
        assert noise.shape == (10, 5)
    
    def test_calibrated_noise_laplace(self):
        """Test calibrated Laplace noise."""
        noise = calibrated_noise(
            shape=1000,
            sensitivity=1.0,
            epsilon=1.0,
            mechanism="laplace",
        )
        
        # Laplace scale should be sensitivity/epsilon = 1
        assert abs(np.std(noise) - np.sqrt(2)) < 0.2
    
    def test_calibrated_noise_gaussian(self):
        """Test calibrated Gaussian noise."""
        noise = calibrated_noise(
            shape=1000,
            sensitivity=1.0,
            epsilon=1.0,
            mechanism="gaussian",
            delta=1e-5,
        )
        
        assert len(noise) == 1000
    
    def test_adaptive_noise(self):
        """Test adaptive noise generation."""
        high_risk = adaptive_noise(
            shape=1000,
            risk_score=0.9,
            epsilon_min=0.01,
            epsilon_max=0.5,
        )
        
        low_risk = adaptive_noise(
            shape=1000,
            risk_score=0.1,
            epsilon_min=0.01,
            epsilon_max=0.5,
        )
        
        # High risk should have more noise (higher std)
        assert np.std(high_risk) >= np.std(low_risk)
    
    def test_reproducibility(self):
        """Test noise reproducibility with seed."""
        noise1 = gaussian_noise(100, scale=1.0, seed=42)
        noise2 = gaussian_noise(100, scale=1.0, seed=42)
        
        np.testing.assert_array_equal(noise1, noise2)
