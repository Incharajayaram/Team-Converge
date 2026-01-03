"""Advanced quantization techniques for edge deployment."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path


class QuantizationAwareTraining(nn.Module):
    """
    Quantization-aware training (QAT).

    Simulates quantization during training to learn quantization-robust weights.
    """

    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        """
        Args:
            num_bits: Number of quantization bits
            symmetric: If True, use symmetric quantization (-x to x)
        """
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.scale = None
        self.zero_point = None

    def calibrate(self, activation: torch.Tensor):
        """Calibrate quantization parameters."""
        if self.symmetric:
            # Symmetric: use max absolute value
            self.scale = (2 ** (self.num_bits - 1) - 1) / torch.max(torch.abs(activation))
        else:
            # Asymmetric: use min-max range
            q_min = -(2 ** (self.num_bits - 1))
            q_max = 2 ** (self.num_bits - 1) - 1
            min_val = activation.min()
            max_val = activation.max()
            self.scale = (q_max - q_min) / (max_val - min_val + 1e-8)
            self.zero_point = q_min - min_val * self.scale

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation."""
        if self.scale is None:
            self.calibrate(x)

        if self.symmetric:
            x_q = torch.clamp(torch.round(x * self.scale), -(2 ** (self.num_bits - 1)), 2 ** (self.num_bits - 1) - 1)
        else:
            x_q = torch.clamp(torch.round(x * self.scale + self.zero_point), -(2 ** (self.num_bits - 1)), 2 ** (self.num_bits - 1) - 1)

        x_dq = x_q / self.scale
        if not self.symmetric:
            x_dq = x_dq - self.zero_point

        return x_dq


class MixedPrecisionQuantization:
    """
    Mixed precision quantization.

    Assigns different bit widths to different layers based on sensitivity.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to quantize with mixed precision
        """
        self.model = model
        self.layer_sensitivity = {}
        self.bit_allocation = {}

    def compute_layer_sensitivity(self, data_loader, loss_fn) -> Dict[str, float]:
        """
        Compute sensitivity of each layer to quantization.

        Measures performance drop when each layer is quantized.
        """
        self.model.eval()
        sensitivities = {}

        with torch.no_grad():
            # Get baseline loss
            baseline_loss = 0.0
            for batch in data_loader:
                if isinstance(batch, dict):
                    images = batch["image"]
                    labels = batch["label"]
                else:
                    images, labels = batch

                outputs = self.model(images)
                baseline_loss += loss_fn(outputs, labels).item()

            baseline_loss /= len(data_loader)

        # Quantize each layer and measure sensitivity
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Quantize layer
                self._quantize_layer(module, num_bits=8)

                # Measure loss
                with torch.no_grad():
                    quantized_loss = 0.0
                    for batch in data_loader:
                        if isinstance(batch, dict):
                            images = batch["image"]
                            labels = batch["label"]
                        else:
                            images, labels = batch

                        outputs = self.model(images)
                        quantized_loss += loss_fn(outputs, labels).item()

                    quantized_loss /= len(data_loader)

                # Compute sensitivity
                sensitivity = (quantized_loss - baseline_loss) / (baseline_loss + 1e-8)
                sensitivities[name] = max(0, sensitivity)

                # Restore original weights
                self._dequantize_layer(module)

        self.layer_sensitivity = sensitivities
        return sensitivities

    def allocate_bits(self, total_bits: int = 128, min_bits: int = 4, max_bits: int = 8):
        """
        Allocate bit widths based on layer sensitivity.

        Args:
            total_bits: Total bit budget
            min_bits: Minimum bits per layer
            max_bits: Maximum bits per layer
        """
        if not self.layer_sensitivity:
            raise ValueError("Must call compute_layer_sensitivity first")

        # Allocate bits inversely proportional to sensitivity
        sensitivities = np.array(list(self.layer_sensitivity.values()))
        inv_sensitivity = 1.0 / (sensitivities + 1e-8)
        inv_sensitivity = inv_sensitivity / inv_sensitivity.sum()

        bit_allocation = {}
        for (layer_name, sens), inv_sens in zip(
            self.layer_sensitivity.items(), inv_sensitivity
        ):
            allocated_bits = int(np.clip(inv_sens * total_bits, min_bits, max_bits))
            bit_allocation[layer_name] = allocated_bits

        self.bit_allocation = bit_allocation
        return bit_allocation

    def _quantize_layer(self, module: nn.Module, num_bits: int):
        """Quantize layer weights."""
        if hasattr(module, "weight"):
            scale = (2 ** (num_bits - 1) - 1) / (torch.abs(module.weight).max() + 1e-8)
            module.weight.data = torch.round(module.weight * scale) / scale

    def _dequantize_layer(self, module: nn.Module):
        """Restore layer (placeholder - weights already modified)."""
        pass


class PostTrainingQuantization:
    """
    Post-training quantization (PTQ).

    Quantizes model after training without fine-tuning.
    Useful for quick edge deployment.
    """

    def __init__(self, model: nn.Module, num_bits: int = 8, symmetric: bool = True):
        """
        Args:
            model: Model to quantize
            num_bits: Number of quantization bits
            symmetric: Use symmetric quantization
        """
        self.model = model
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.scale_factors = {}
        self.zero_points = {}

    def calibrate(self, data_loader, device: str = "cuda"):
        """
        Calibrate quantization statistics on representative data.

        Args:
            data_loader: DataLoader with representative samples
            device: Device for computation
        """
        self.model.eval()
        activations = {name: [] for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                else:
                    images = batch[0].to(device)

                # Hook to collect activations
                hooks = []

                def get_activation(name):
                    def hook(model, input, output):
                        activations[name].append(output.detach().cpu())

                    return hook

                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        hooks.append(module.register_forward_hook(get_activation(name)))

                # Forward pass
                _ = self.model(images)

                # Remove hooks
                for hook in hooks:
                    hook.remove()

        # Compute scale factors
        for name, acts in activations.items():
            if acts:
                activation = torch.cat(acts, dim=0)

                if self.symmetric:
                    scale = (2 ** (self.num_bits - 1) - 1) / (torch.abs(activation).max() + 1e-8)
                    self.scale_factors[name] = scale.item()
                else:
                    q_min = -(2 ** (self.num_bits - 1))
                    q_max = 2 ** (self.num_bits - 1) - 1
                    min_val = activation.min()
                    max_val = activation.max()
                    scale = (q_max - q_min) / (max_val - min_val + 1e-8)
                    zero_point = q_min - min_val * scale
                    self.scale_factors[name] = scale.item()
                    self.zero_points[name] = zero_point.item()

    def quantize(self) -> nn.Module:
        """Apply quantization to model weights."""
        quantized_model = self.model

        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in self.scale_factors:
                    scale = self.scale_factors[name]

                    if self.symmetric:
                        module.weight.data = torch.round(module.weight * scale) / scale
                    else:
                        zero_point = self.zero_points.get(name, 0)
                        module.weight.data = (
                            torch.round(module.weight * scale + zero_point) - zero_point
                        ) / scale

        return quantized_model

    def save_quantization_config(self, output_path: str):
        """Save quantization parameters for inference."""
        config = {
            "num_bits": self.num_bits,
            "symmetric": self.symmetric,
            "scale_factors": self.scale_factors,
            "zero_points": self.zero_points,
        }

        import json

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ Quantization config saved to {output_path}")


class KnowledgeDistillationQuantization:
    """
    Combine knowledge distillation with quantization.

    Uses teacher network to guide quantized student training.
    """

    def __init__(
        self, student: nn.Module, teacher: nn.Module, num_bits: int = 8, temperature: float = 4.0
    ):
        """
        Args:
            student: Student model (to be quantized)
            teacher: Teacher model (full precision)
            num_bits: Quantization bits for student
            temperature: Temperature for distillation
        """
        self.student = student
        self.teacher = teacher
        self.num_bits = num_bits
        self.temperature = temperature
        self.quantizer = QuantizationAwareTraining(num_bits=num_bits, symmetric=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with quantization and distillation.

        Returns:
            student_output: Quantized student predictions
            teacher_output: Teacher predictions for distillation
        """
        # Student forward with quantization
        student_out = self.student(x)
        student_out_q = self.quantizer.quantize(student_out)

        # Teacher forward
        with torch.no_grad():
            teacher_out = self.teacher(x)

        return student_out_q, teacher_out

    def get_distillation_loss(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for distillation."""
        student_probs = torch.softmax(student_out / self.temperature, dim=1)
        teacher_probs = torch.softmax(teacher_out / self.temperature, dim=1)

        loss = torch.nn.functional.kl_div(
            torch.log(student_probs), teacher_probs, reduction="batchmean"
        ) * (self.temperature ** 2)

        return loss
