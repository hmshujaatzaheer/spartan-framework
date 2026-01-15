"""
Evaluation Metrics

Metrics for evaluating attack and defense performance.
"""

from typing import List, Optional, Tuple

import numpy as np


def compute_auc_roc(
    labels: List[int],
    scores: List[float],
) -> float:
    """Compute Area Under ROC Curve.

    Args:
        labels: Ground truth labels (0 or 1)
        scores: Prediction scores

    Returns:
        AUC-ROC value
    """
    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have same length")

    if len(labels) == 0:
        return 0.5

    labels = np.array(labels)
    scores = np.array(scores)

    # Check for edge cases
    if len(np.unique(labels)) < 2:
        return 0.5

    # Sort by scores descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    # Compute TPR and FPR at each threshold
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)

    if num_pos == 0 or num_neg == 0:
        return 0.5

    tpr = np.cumsum(sorted_labels == 1) / num_pos
    fpr = np.cumsum(sorted_labels == 0) / num_neg

    # Add origin point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Compute AUC using trapezoidal rule
    # Use trapezoid (NumPy 2.0+) or trapz (older versions)
    try:
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc = np.trapz(tpr, fpr)

    return float(auc)


def compute_accuracy(
    labels: List[int],
    predictions: List[int],
) -> float:
    """Compute classification accuracy.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels

    Returns:
        Accuracy value
    """
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have same length")

    if len(labels) == 0:
        return 0.0

    labels = np.array(labels)
    predictions = np.array(predictions)

    correct = np.sum(labels == predictions)
    accuracy = correct / len(labels)

    return float(accuracy)


def compute_tpr_at_fpr(
    labels: List[int],
    scores: List[float],
    fpr_threshold: float = 0.01,
) -> float:
    """Compute True Positive Rate at given False Positive Rate.

    Args:
        labels: Ground truth labels
        scores: Prediction scores
        fpr_threshold: Target FPR threshold

    Returns:
        TPR at specified FPR
    """
    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have same length")

    if len(labels) == 0:
        return 0.0

    labels = np.array(labels)
    scores = np.array(scores)

    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)

    if num_pos == 0 or num_neg == 0:
        return 0.0

    # Find threshold that achieves target FPR
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    fp_count = 0
    tp_count = 0

    for i, label in enumerate(sorted_labels):
        if label == 0:
            fp_count += 1
        else:
            tp_count += 1

        current_fpr = fp_count / num_neg

        if current_fpr > fpr_threshold:
            # Went past threshold, use previous counts
            if label == 0:
                fp_count -= 1
            else:
                tp_count -= 1
            break

    tpr = tp_count / num_pos

    return float(tpr)


def compute_f1_score(
    labels: List[int],
    predictions: List[int],
) -> float:
    """Compute F1 score.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels

    Returns:
        F1 score
    """
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have same length")

    if len(labels) == 0:
        return 0.0

    labels = np.array(labels)
    predictions = np.array(predictions)

    # Compute TP, FP, FN
    tp = np.sum((labels == 1) & (predictions == 1))
    fp = np.sum((labels == 0) & (predictions == 1))
    fn = np.sum((labels == 1) & (predictions == 0))

    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return float(f1)


def compute_precision_recall_curve(
    labels: List[int],
    scores: List[float],
    num_thresholds: int = 100,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute precision-recall curve.

    Args:
        labels: Ground truth labels
        scores: Prediction scores
        num_thresholds: Number of threshold points

    Returns:
        Tuple of (precisions, recalls, thresholds)
    """
    labels = np.array(labels)
    scores = np.array(scores)

    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []
    recalls = []

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)

        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls, thresholds.tolist()

