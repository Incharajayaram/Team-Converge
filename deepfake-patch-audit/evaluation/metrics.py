"""Evaluation metrics: AUC, Accuracy@τ, and other metrics."""

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve


def compute_roc_curve(y_true, y_scores):
    """
    Compute ROC curve.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities for positive class

    Returns:
        fpr, tpr, thresholds, auc_score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def compute_accuracy_at_threshold(y_true, y_scores, threshold):
    """
    Compute accuracy at specific threshold.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        threshold: Decision threshold

    Returns:
        Accuracy at threshold
    """
    predictions = (y_scores > threshold).astype(int)
    accuracy = np.mean(predictions == y_true)
    return accuracy


def compute_metrics_at_threshold(y_true, y_scores, threshold):
    """
    Compute multiple metrics at specific threshold.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        threshold: Decision threshold

    Returns:
        dict with accuracy, precision, recall, f1
    """
    predictions = (y_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_optimal_threshold(y_true, y_scores, metric="f1"):
    """
    Find optimal decision threshold based on metric.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        metric: 'f1', 'accuracy', 'precision', 'recall'

    Returns:
        Optimal threshold and metric value
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_scores, threshold)
        score = metrics[metric]

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def compute_pr_auc(y_true, y_scores):
    """
    Compute Precision-Recall AUC.
    
    More informative than ROC-AUC when class balance is skewed.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Predicted probabilities for positive class
    
    Returns:
        pr_auc: Precision-Recall AUC score
        precision: Precision values
        recall: Recall values
        thresholds: Thresholds used
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    return pr_auc, precision, recall, thresholds


def compute_tpr_at_fpr(y_true, y_scores, target_fpr=0.01):
    """
    Compute True Positive Rate (TPR) at a fixed False Positive Rate (FPR).
    
    This is critical for deployment: "At 1% false alarm rate, what fraction
    of real deepfakes do we catch?"
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Predicted probabilities for positive class
        target_fpr: Target false positive rate (default: 0.01 = 1%)
    
    Returns:
        tpr_at_target: TPR at the target FPR
        threshold_at_target: Threshold that achieves this FPR
        actual_fpr: Actual FPR achieved (may differ slightly from target)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the threshold where FPR is closest to target
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    tpr_at_target = tpr[idx]
    threshold_at_target = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    actual_fpr = fpr[idx]
    
    return tpr_at_target, threshold_at_target, actual_fpr


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    """
    Compute False Positive Rate (FPR) at a fixed True Positive Rate (TPR).
    
    This answers: "To catch 95% of deepfakes, what false alarm rate do we accept?"
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Predicted probabilities for positive class
        target_tpr: Target true positive rate (default: 0.95 = 95%)
    
    Returns:
        fpr_at_target: FPR at the target TPR
        threshold_at_target: Threshold that achieves this TPR
        actual_tpr: Actual TPR achieved
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the threshold where TPR is closest to target
    idx = np.argmin(np.abs(tpr - target_tpr))
    
    fpr_at_target = fpr[idx]
    threshold_at_target = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    actual_tpr = tpr[idx]
    
    return fpr_at_target, threshold_at_target, actual_tpr


def compute_brier_score(y_true, y_scores):
    """
    Compute Brier score (mean squared error between predictions and outcomes).
    
    Measures calibration quality: how well predicted probabilities match
    actual outcomes. Lower is better.
    
    Brier = (1/N) * Σ(p_i - y_i)²
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Predicted probabilities for positive class
    
    Returns:
        brier_score: Brier score (range [0, 1], lower is better)
    """
    return np.mean((y_scores - y_true) ** 2)


def compute_ece(y_true, y_scores, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Measures calibration by binning predictions and comparing average
    predicted probability to actual frequency in each bin.
    Lower is better.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve (default: 10)
    
    Returns:
        ece: Expected Calibration Error (range [0, 1], lower is better)
        bin_stats: Dict with per-bin statistics for debugging
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
        
        if np.sum(in_bin) > 0:
            # Average predicted probability in bin
            bin_confidence = np.mean(y_scores[in_bin])
            
            # Actual accuracy (fraction of positives) in bin
            bin_accuracy = np.mean(y_true[in_bin])
            
            # Count of samples in bin
            bin_count = np.sum(in_bin)
            
            # Weighted contribution to ECE
            ece += (bin_count / len(y_true)) * np.abs(bin_confidence - bin_accuracy)
            
            bin_stats.append({
                "bin_range": (bin_lower, bin_upper),
                "count": int(bin_count),
                "avg_confidence": float(bin_confidence),
                "avg_accuracy": float(bin_accuracy),
                "calibration_error": float(np.abs(bin_confidence - bin_accuracy)),
            })
        else:
            bin_stats.append({
                "bin_range": (bin_lower, bin_upper),
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
                "calibration_error": 0.0,
            })
    
    return ece, bin_stats


def compute_comprehensive_metrics(y_true, y_scores, threshold=0.5):
    """
    Compute a comprehensive suite of metrics for model evaluation.
    
    This is the one-stop function for getting all metrics needed for
    a defensible demo/paper.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Predicted probabilities for positive class
        threshold: Decision threshold (default: 0.5)
    
    Returns:
        dict with all metrics:
        - Ranking metrics: ROC-AUC, PR-AUC
        - Operating point metrics: TPR@FPR=1%, FPR@TPR=95%, Accuracy@threshold
        - Calibration metrics: Brier score, ECE
        - Confusion matrix: TP, TN, FP, FN, Precision, Recall, F1
    """
    # Compute ranking metrics
    fpr, tpr, _ , roc_auc = compute_roc_curve(y_true, y_scores)
    pr_auc, precision_curve, recall_curve, _ = compute_pr_auc(y_true, y_scores)
    
    # Compute operating point metrics
    tpr_at_1fpr, thresh_at_1fpr, actual_fpr = compute_tpr_at_fpr(y_true, y_scores, target_fpr=0.01)
    fpr_at_95tpr, thresh_at_95tpr, actual_tpr = compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)
    
    # Compute calibration metrics
    brier = compute_brier_score(y_true, y_scores)
    ece, _ = compute_ece(y_true, y_scores)
    
    # Compute confusion matrix metrics at threshold
    threshold_metrics = compute_metrics_at_threshold(y_true, y_scores, threshold)
    
    return {
        # Ranking metrics
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        
        # Operating point metrics
        "tpr_at_1pct_fpr": tpr_at_1fpr,
        "threshold_at_1pct_fpr": thresh_at_1fpr,
        "actual_fpr_at_1pct": actual_fpr,
        "fpr_at_95pct_tpr": fpr_at_95tpr,
        "threshold_at_95pct_tpr": thresh_at_95tpr,
        "actual_tpr_at_95pct": actual_tpr,
        
        # Calibration metrics
        "brier_score": brier,
        "ece": ece,
        
        # Threshold-based metrics
        "threshold": threshold,
        "accuracy": threshold_metrics["accuracy"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "tp": threshold_metrics["tp"],
        "tn": threshold_metrics["tn"],
        "fp": threshold_metrics["fp"],
        "fn": threshold_metrics["fn"],
    }
