"""Minimal threshold tuning for deepfake detection."""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json

import torch
import numpy as np
from tqdm import tqdm

from .metrics_minimal import MetricsCalculator, MetricsResult, validate_inputs, MetricsError

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSearchResult:
    """Threshold search results."""
    optimal_threshold: float
    optimal_score: float
    metric_name: str
    optimal_metrics: Optional[MetricsResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.optimal_metrics:
            data['optimal_metrics'] = self.optimal_metrics.to_dict()
        return data
    
    def save(self, filepath: Path) -> None:
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2, default=str))


@dataclass
class ModelComparisonResult:
    """Model comparison results."""
    max_score_difference: float
    mean_score_difference: float
    num_mismatches: int
    mismatch_rate: float
    agreement_score: float
    correlation: float
    within_tolerance: bool
    tolerance_threshold: float
    num_samples: int = 0
    
    def __str__(self) -> str:
        return (f"Model Comparison:\n"
                f"  Max Diff: {self.max_score_difference:.6f} | Mean: {self.mean_score_difference:.6f}\n"
                f"  Mismatches: {self.num_mismatches}/{self.num_samples} ({self.mismatch_rate:.2%})\n"
                f"  Agreement: {self.agreement_score:.4f} | Correlation: {self.correlation:.4f}\n"
                f"  Within Tolerance ({self.tolerance_threshold:.6f}): {self.within_tolerance}")


class ThresholdTuner:
    """Threshold tuning with model comparison."""
    
    def __init__(
        self,
        validation_loader: torch.utils.data.DataLoader,
        device: str = "cuda",
        use_amp: bool = False,
        show_progress: bool = True
    ):
        self.validation_loader = validation_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.show_progress = show_progress
        self._cache = {}
        self.calc = MetricsCalculator()
        logger.info(f"ThresholdTuner on {self.device}" + (" with AMP" if self.use_amp else ""))
    
    def get_predictions(
        self,
        model: torch.nn.Module,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions on validation set."""
        model_id = id(model)
        if use_cache and model_id in self._cache:
            return self._cache[model_id]
        
        all_scores, all_labels = [], []
        model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(self.validation_loader, desc="Predictions", disable=not self.show_progress):
                # Handle batch format
                if isinstance(batch, dict):
                    images, labels = batch["image"], batch["label"]
                elif isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(images)
                else:
                    logits = model(images)
                
                # Get probabilities
                if logits.dim() == 1:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)
                    probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                
                all_scores.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        
        self._cache[model_id] = (labels, scores)
        logger.info(f"Predictions cached for {len(scores)} samples")
        return labels, scores
    
    def find_best_threshold(
        self,
        model: torch.nn.Module,
        metric: str = "f1",
        num_points: int = 101
    ) -> ThresholdSearchResult:
        """Find best threshold."""
        labels, scores = self.get_predictions(model)
        validate_inputs(labels, scores)
        
        thresholds = np.linspace(0.0, 1.0, num_points)
        best_threshold, best_score = 0.5, -np.inf
        
        for threshold in thresholds:
            result = self.calc.compute_metrics(labels, scores, threshold, include_auc=False)
            score = getattr(result, metric)
            if score > best_score:
                best_score, best_threshold = score, threshold
        
        # Get full metrics at optimal
        optimal_metrics = self.calc.compute_metrics(labels, scores, best_threshold, include_auc=True)
        
        logger.info(f"Optimal: {best_threshold:.4f} ({metric}={best_score:.4f})")
        return ThresholdSearchResult(
            optimal_threshold=best_threshold,
            optimal_score=best_score,
            metric_name=metric,
            optimal_metrics=optimal_metrics
        )
    
    def compare_models(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        tolerance: float = 0.01,
        threshold: float = 0.5
    ) -> ModelComparisonResult:
        """Compare two models (e.g., float vs quantized)."""
        labels1, scores1 = self.get_predictions(model1)
        labels2, scores2 = self.get_predictions(model2)
        
        if not np.array_equal(labels1, labels2):
            raise MetricsError("Label mismatch between models")
        
        # Score differences
        score_diff = np.abs(scores1 - scores2)
        max_diff = float(np.max(score_diff))
        mean_diff = float(np.mean(score_diff))
        
        # Prediction agreement
        preds1 = (scores1 >= threshold).astype(int)
        preds2 = (scores2 >= threshold).astype(int)
        num_mismatches = int(np.sum(preds1 != preds2))
        mismatch_rate = num_mismatches / len(preds1)
        agreement = float(np.mean(preds1 == preds2))
        
        # Stats
        correlation = float(np.corrcoef(scores1, scores2)[0, 1])
        within_tolerance = max_diff <= tolerance
        
        result = ModelComparisonResult(
            max_score_difference=max_diff,
            mean_score_difference=mean_diff,
            num_mismatches=num_mismatches,
            mismatch_rate=mismatch_rate,
            agreement_score=agreement,
            correlation=correlation,
            within_tolerance=within_tolerance,
            tolerance_threshold=tolerance,
            num_samples=len(labels1)
        )
        
        logger.info(f"Comparison: max_diff={max_diff:.6f}, within_tol={within_tolerance}")
        return result
    
    def clear_cache(self):
        """Clear prediction cache."""
        self._cache.clear()
        logger.info("Cache cleared")
