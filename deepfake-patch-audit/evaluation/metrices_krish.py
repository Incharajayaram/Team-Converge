"""Minimal evaluation metrics for deepfake detection."""

import logging
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score, log_loss
)

logger = logging.getLogger(__name__)


class MetricsError(Exception):
    pass


def validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate inputs are binary labels and scores."""
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    if len(y_true) != len(y_scores):
        raise MetricsError(f"Length mismatch: {len(y_true)} vs {len(y_scores)}")
    if not np.all(np.isin(y_true, [0, 1])):
        raise MetricsError("y_true must be binary (0, 1)")
    if len(np.unique(y_true)) < 2:
        raise MetricsError("y_true must contain both classes")


@dataclass
class MetricsResult:
    """Evaluation metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    threshold: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    mcc: float = 0.0
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, filepath: Path) -> None:
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2))
    
    def __str__(self) -> str:
        s = (f"Threshold: {self.threshold:.3f}\n"
             f"Accuracy: {self.accuracy:.4f} | Balanced: {self.balanced_accuracy:.4f}\n"
             f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | F1: {self.f1_score:.4f}\n"
             f"MCC: {self.mcc:.4f} | Specificity: {self.specificity:.4f}\n"
             f"TP={self.true_positives} TN={self.true_negatives} FP={self.false_positives} FN={self.false_negatives}")
        if self.auc_roc:
            s += f"\nAUC-ROC: {self.auc_roc:.4f} | AUC-PR: {self.auc_pr:.4f}"
        return s


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    def __init__(self, validate: bool = True):
        self.validate = validate
    
    def compute_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple:
        """Compute ROC curve."""
        if self.validate:
            validate_inputs(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return fpr, tpr, thresholds, auc(fpr, tpr)
    
    def compute_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple:
        """Compute Precision-Recall curve."""
        if self.validate:
            validate_inputs(y_true, y_scores)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        return precision, recall, thresholds, average_precision_score(y_true, y_scores)
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
        include_auc: bool = True
    ) -> MetricsResult:
        """Compute metrics at threshold."""
        if self.validate:
            validate_inputs(y_true, y_scores)
        
        y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
        predictions = (y_scores >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result = MetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            threshold=threshold,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            specificity=specificity,
            balanced_accuracy=balanced_accuracy_score(y_true, predictions),
            mcc=matthews_corrcoef(y_true, predictions)
        )
        
        if include_auc:
            try:
                _, _, _, result.auc_roc = self.compute_roc_curve(y_true, y_scores)
                _, _, _, result.auc_pr = self.compute_pr_curve(y_true, y_scores)
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}")
        
        return result
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = "f1",
        num_points: int = 101
    ) -> Tuple[float, float]:
        """Find optimal threshold for metric."""
        if self.validate:
            validate_inputs(y_true, y_scores)
        
        valid_metrics = {'f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'mcc'}
        if metric not in valid_metrics:
            raise MetricsError(f"Invalid metric '{metric}'. Use: {valid_metrics}")
        
        thresholds = np.linspace(0.0, 1.0, num_points)
        best_threshold, best_score = 0.5, -np.inf
        
        for threshold in thresholds:
            result = self.compute_metrics(y_true, y_scores, threshold, include_auc=False)
            score = getattr(result, metric)
            if score > best_score:
                best_score, best_threshold = score, threshold
        
        logger.info(f"Optimal: {best_threshold:.4f} ({metric}={best_score:.4f})")
        return best_threshold, best_score


def evaluate_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True
) -> MetricsResult:
    """Quick model evaluation."""
    calc = MetricsCalculator()
    result = calc.compute_metrics(y_true, y_scores, threshold)
    if verbose:
        print(result)
    return result


def find_best_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """Find best threshold."""
    return MetricsCalculator().find_optimal_threshold(y_true, y_scores, metric)
