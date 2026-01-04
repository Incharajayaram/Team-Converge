#!/usr/bin/env python3
"""
Training Diagnostics Script - Analyze training history and detect issues.

This script analyzes training logs to diagnose common problems:
1. Distillation loss imbalance (distill_loss >> task_loss)
2. Training stagnation (loss not decreasing)
3. Overfitting (train loss << val loss)
4. AUC progression issues

Usage:
    python scripts/diagnose_training.py \
        --history outputs/checkpoints/training_history.json \
        --output outputs/diagnostics/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")


def load_training_history(history_path: str) -> Dict:
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        return json.load(f)


def analyze_loss_balance(history: Dict) -> Dict:
    """
    Analyze balance between distillation and task loss.
    
    Returns:
        Dict with analysis results
    """
    distill_losses = history.get("train_distill_loss", [])
    task_losses = history.get("train_task_loss", [])
    
    if not distill_losses or not task_losses:
        return {"error": "No distillation/task loss data found"}
    
    # Calculate ratios
    ratios = [d / (t + 1e-8) for d, t in zip(distill_losses, task_losses)]
    avg_ratio = sum(ratios) / len(ratios)
    max_ratio = max(ratios)
    
    # Diagnose
    if avg_ratio > 10:
        severity = "CRITICAL"
        diagnosis = "Distillation loss dominates task loss by 10x+"
        recommendation = "Reduce alpha_distill to 0.1 or lower"
    elif avg_ratio > 3:
        severity = "WARNING"
        diagnosis = "Distillation loss is 3x+ higher than task loss"
        recommendation = "Consider reducing alpha_distill to 0.3"
    elif avg_ratio < 0.1:
        severity = "WARNING"
        diagnosis = "Task loss dominates - distillation may not be effective"
        recommendation = "Consider increasing alpha_distill"
    else:
        severity = "OK"
        diagnosis = "Loss balance looks reasonable"
        recommendation = "No changes needed"
    
    return {
        "avg_ratio": avg_ratio,
        "max_ratio": max_ratio,
        "severity": severity,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "distill_losses": distill_losses,
        "task_losses": task_losses,
    }


def analyze_convergence(history: Dict) -> Dict:
    """
    Analyze if training is converging properly.
    """
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    val_aucs = history.get("val_auc", [])
    
    if not train_losses:
        return {"error": "No training loss data found"}
    
    # Check if loss decreased
    first_loss = train_losses[0]
    last_loss = train_losses[-1]
    loss_decrease = (first_loss - last_loss) / (first_loss + 1e-8)
    
    # Check AUC progression
    if val_aucs:
        first_auc = val_aucs[0]
        last_auc = val_aucs[-1]
        best_auc = max(val_aucs)
        auc_improvement = last_auc - first_auc
    else:
        first_auc = last_auc = best_auc = auc_improvement = None
    
    # Diagnose
    if loss_decrease < 0.1:
        severity = "CRITICAL"
        diagnosis = "Training loss barely decreased"
        recommendation = "Learning rate may be too low, or model is stuck"
    elif last_auc and last_auc < 0.55:
        severity = "CRITICAL"
        diagnosis = f"Final AUC ({last_auc:.3f}) is near random"
        recommendation = "Model learned nothing - check data loading and labels"
    elif auc_improvement and auc_improvement < 0:
        severity = "WARNING"
        diagnosis = "AUC decreased during training"
        recommendation = "Possible overfitting or learning rate too high"
    else:
        severity = "OK"
        diagnosis = "Training appears to be converging"
        recommendation = "Continue monitoring"
    
    return {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_decrease_pct": loss_decrease * 100,
        "first_auc": first_auc,
        "last_auc": last_auc,
        "best_auc": best_auc,
        "auc_improvement": auc_improvement,
        "severity": severity,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def analyze_overfitting(history: Dict) -> Dict:
    """
    Check for overfitting (train loss << val loss).
    """
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    
    if not train_losses or not val_losses:
        return {"error": "Missing train/val loss data"}
    
    # Calculate gap at end of training
    train_final = train_losses[-1]
    val_final = val_losses[-1]
    gap = val_final - train_final
    gap_ratio = gap / (train_final + 1e-8)
    
    # Check if gap increased over time
    early_gap = val_losses[0] - train_losses[0] if len(train_losses) > 0 else 0
    gap_increase = gap - early_gap
    
    if gap_ratio > 0.5:
        severity = "WARNING"
        diagnosis = "Significant train/val gap (possible overfitting)"
        recommendation = "Consider more regularization or early stopping"
    else:
        severity = "OK"
        diagnosis = "No significant overfitting detected"
        recommendation = "Generalization looks reasonable"
    
    return {
        "train_final": train_final,
        "val_final": val_final,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "severity": severity,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def analyze_two_stage(history: Dict) -> Dict:
    """
    Analyze two-stage training (if applicable).
    """
    stages = history.get("stage", [])
    val_aucs = history.get("val_auc", [])
    
    if not stages:
        return {"error": "No stage data found (not two-stage training?)"}
    
    # Find stage transition
    stage1_aucs = [auc for s, auc in zip(stages, val_aucs) if s == 1]
    stage2_aucs = [auc for s, auc in zip(stages, val_aucs) if s == 2]
    
    if not stage1_aucs or not stage2_aucs:
        return {"error": "Could not find both stages"}
    
    stage1_best = max(stage1_aucs)
    stage2_best = max(stage2_aucs)
    improvement = stage2_best - stage1_best
    
    if improvement < 0:
        severity = "WARNING"
        diagnosis = f"Stage 2 AUC ({stage2_best:.3f}) is worse than Stage 1 ({stage1_best:.3f})"
        recommendation = "Stage 2 may be overfitting. Try fewer epochs or smaller LR"
    elif improvement < 0.01:
        severity = "INFO"
        diagnosis = "Minimal improvement from Stage 2"
        recommendation = "Stage 1 may be sufficient"
    else:
        severity = "OK"
        diagnosis = f"Stage 2 improved AUC by {improvement:.3f}"
        recommendation = "Two-stage training working as expected"
    
    return {
        "stage1_epochs": len(stage1_aucs),
        "stage2_epochs": len(stage2_aucs),
        "stage1_best_auc": stage1_best,
        "stage2_best_auc": stage2_best,
        "improvement": improvement,
        "severity": severity,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def plot_training_curves(history: Dict, output_dir: Path):
    """Generate training visualization plots."""
    if not HAS_MATPLOTLIB:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training vs Validation Loss
    if "train_loss" in history and "val_loss" in history:
        ax = axes[0, 0]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train", marker='o', markersize=3)
        ax.plot(epochs, history["val_loss"], label="Val", marker='s', markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Distillation vs Task Loss
    if "train_distill_loss" in history and "train_task_loss" in history:
        ax = axes[0, 1]
        epochs = range(1, len(history["train_distill_loss"]) + 1)
        ax.plot(epochs, history["train_distill_loss"], label="Distill", marker='o', markersize=3)
        ax.plot(epochs, history["train_task_loss"], label="Task", marker='s', markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Distillation vs Task Loss (KEY DIAGNOSTIC)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # AUC progression
    if "val_auc" in history:
        ax = axes[1, 0]
        epochs = range(1, len(history["val_auc"]) + 1)
        ax.plot(epochs, history["val_auc"], label="Val AUC", marker='o', markersize=3, color='green')
        ax.axhline(y=0.5, color='red', linestyle='--', label='Random')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("Validation AUC Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    # Loss ratio
    if "train_distill_loss" in history and "train_task_loss" in history:
        ax = axes[1, 1]
        epochs = range(1, len(history["train_distill_loss"]) + 1)
        ratios = [d / (t + 1e-8) for d, t in zip(history["train_distill_loss"], history["train_task_loss"])]
        ax.plot(epochs, ratios, marker='o', markersize=3, color='purple')
        ax.axhline(y=1.0, color='green', linestyle='--', label='Balanced')
        ax.axhline(y=10.0, color='red', linestyle='--', label='Imbalanced (10x)')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Ratio (Distill / Task)")
        ax.set_title("Loss Ratio (should be ~1.0)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"✓ Saved training curves to {output_dir / 'training_curves.png'}")


def generate_report(history: Dict, output_dir: Path) -> str:
    """Generate a text diagnostic report."""
    
    report = []
    report.append("=" * 70)
    report.append("TRAINING DIAGNOSTICS REPORT")
    report.append("=" * 70)
    
    # Loss balance analysis
    report.append("\n## 1. LOSS BALANCE ANALYSIS")
    report.append("-" * 40)
    balance = analyze_loss_balance(history)
    if "error" not in balance:
        report.append(f"Average Distill/Task Ratio: {balance['avg_ratio']:.2f}")
        report.append(f"Max Ratio: {balance['max_ratio']:.2f}")
        report.append(f"Severity: {balance['severity']}")
        report.append(f"Diagnosis: {balance['diagnosis']}")
        report.append(f"Recommendation: {balance['recommendation']}")
    else:
        report.append(f"Error: {balance['error']}")
    
    # Convergence analysis
    report.append("\n## 2. CONVERGENCE ANALYSIS")
    report.append("-" * 40)
    convergence = analyze_convergence(history)
    if "error" not in convergence:
        report.append(f"Loss Decrease: {convergence['loss_decrease_pct']:.1f}%")
        if convergence['last_auc']:
            report.append(f"Final AUC: {convergence['last_auc']:.4f}")
            report.append(f"Best AUC: {convergence['best_auc']:.4f}")
        report.append(f"Severity: {convergence['severity']}")
        report.append(f"Diagnosis: {convergence['diagnosis']}")
        report.append(f"Recommendation: {convergence['recommendation']}")
    else:
        report.append(f"Error: {convergence['error']}")
    
    # Overfitting analysis
    report.append("\n## 3. OVERFITTING ANALYSIS")
    report.append("-" * 40)
    overfit = analyze_overfitting(history)
    if "error" not in overfit:
        report.append(f"Train/Val Gap: {overfit['gap']:.4f}")
        report.append(f"Gap Ratio: {overfit['gap_ratio']:.2f}")
        report.append(f"Severity: {overfit['severity']}")
        report.append(f"Diagnosis: {overfit['diagnosis']}")
    else:
        report.append(f"Error: {overfit['error']}")
    
    # Two-stage analysis
    report.append("\n## 4. TWO-STAGE ANALYSIS")
    report.append("-" * 40)
    twostage = analyze_two_stage(history)
    if "error" not in twostage:
        report.append(f"Stage 1 Best AUC: {twostage['stage1_best_auc']:.4f}")
        report.append(f"Stage 2 Best AUC: {twostage['stage2_best_auc']:.4f}")
        report.append(f"Improvement: {twostage['improvement']:.4f}")
        report.append(f"Severity: {twostage['severity']}")
        report.append(f"Diagnosis: {twostage['diagnosis']}")
    else:
        report.append(f"Note: {twostage['error']}")
    
    # Summary
    report.append("\n" + "=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    
    severities = []
    if "severity" in balance: severities.append(balance["severity"])
    if "severity" in convergence: severities.append(convergence["severity"])
    if "severity" in overfit: severities.append(overfit["severity"])
    
    if "CRITICAL" in severities:
        report.append("\n⚠️  CRITICAL ISSUES DETECTED")
        report.append("Training has serious problems that need to be addressed.")
    elif "WARNING" in severities:
        report.append("\n⚡ WARNINGS DETECTED")
        report.append("Some issues found, but may be acceptable.")
    else:
        report.append("\n✅ TRAINING LOOKS HEALTHY")
        report.append("No major issues detected.")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs and diagnose issues")
    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Path to training_history.json or training_history_two_stage.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/diagnostics",
        help="Output directory for report and plots"
    )
    
    args = parser.parse_args()
    
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    print(f"Loading training history from: {history_path}")
    history = load_training_history(str(history_path))
    
    # Generate report
    report = generate_report(history, output_dir)
    print(report)
    
    # Save report
    report_path = output_dir / "diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Generate plots
    plot_training_curves(history, output_dir)
    
    print(f"\n✓ All diagnostics saved to: {output_dir}")


if __name__ == "__main__":
    main()
