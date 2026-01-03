"""Minimal benchmarking for deepfake detection models."""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import json
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Benchmark results container."""
    mean_latency_ms: float
    std_latency_ms: float
    median_latency_ms: float
    throughput_fps: float
    peak_memory_mb: float
    avg_memory_mb: float
    num_parameters: int
    model_size_mb: float
    device: str
    batch_size: int
    num_samples: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    def save(self, filepath: Path) -> None:
        Path(filepath).write_text(json.dumps(asdict(self), indent=2))
    
    def __str__(self) -> str:
        s = (f"Latency: {self.mean_latency_ms:.2f}±{self.std_latency_ms:.2f}ms "
             f"(median: {self.median_latency_ms:.2f}ms)\n"
             f"Throughput: {self.throughput_fps:.1f} fps | Memory: {self.peak_memory_mb:.1f}MB peak\n"
             f"Model: {self.num_parameters:,} params, {self.model_size_mb:.2f}MB")
        if self.accuracy:
            s += (f"\nAccuracy: {self.accuracy:.4f} | Precision: {self.precision:.4f} | "
                  f"Recall: {self.recall:.4f} | F1: {self.f1_score:.4f}")
            if self.auc_roc:
                s += f" | AUC: {self.auc_roc:.4f}"
        return s


class ModelBenchmark:
    """Model benchmarking with latency, memory, and metrics."""
    
    def __init__(self, device="cuda", use_amp=False, warmup=10, show_progress=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.warmup = warmup
        self.show_progress = show_progress
    
    def _model_info(self, model: torch.nn.Module) -> Tuple[int, float]:
        """Get model parameters and size."""
        params = sum(p.numel() for p in model.parameters())
        size = sum(p.nelement() * p.element_size() for p in model.parameters())
        size += sum(b.nelement() * b.element_size() for b in model.buffers())
        return params, size / (1024 ** 2)
    
    def _warmup(self, model, dataloader):
        """Warmup iterations."""
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.warmup:
                    break
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                inputs = inputs.to(self.device)
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    _ = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    def benchmark(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_iterations: Optional[int] = None,
        compute_metrics: bool = False
    ) -> BenchmarkResult:
        """Benchmark model performance."""
        model.eval()
        model.to(self.device)
        
        num_params, model_size = self._model_info(model)
        batch_size = dataloader.batch_size
        
        self._warmup(model, dataloader)
        
        latencies, memory_usage = [], []
        all_preds, all_labels = [], []
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        total = num_iterations or len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=total, disable=not self.show_progress, desc="Benchmarking")
        
        for i, batch in pbar:
            if num_iterations and i >= num_iterations:
                break
            
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            labels = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
            
            inputs = inputs.to(self.device)
            
            # Measure inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    outputs = model(inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            
            # Collect predictions
            if compute_metrics and labels is not None:
                if outputs.dim() > 2:
                    outputs = outputs.mean(dim=(2, 3))
                preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
            
            # Memory
            mem = (torch.cuda.memory_allocated() / (1024 ** 2) if self.device.type == "cuda" 
                   else psutil.Process().memory_info().rss / (1024 ** 2))
            memory_usage.append(mem)
            
            if self.show_progress:
                pbar.set_postfix({'lat': f'{latencies[-1]:.1f}ms', 'mem': f'{mem:.0f}MB'})
        
        pbar.close()
        
        # Compute stats
        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        std_lat = float(np.std(latencies))
        median_lat = float(np.median(latencies))
        throughput = batch_size * 1000 / mean_lat
        peak_mem = float(np.max(memory_usage))
        avg_mem = float(np.mean(memory_usage))
        
        # Metrics
        acc = prec = rec = f1 = auc = None
        if compute_metrics and all_preds:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            preds = np.concatenate(all_preds)
            labels_arr = np.concatenate(all_labels)
            binary_preds = (preds >= 0.5).astype(int)
            
            acc = float(accuracy_score(labels_arr, binary_preds))
            prec = float(precision_score(labels_arr, binary_preds, zero_division=0))
            rec = float(recall_score(labels_arr, binary_preds, zero_division=0))
            f1 = float(f1_score(labels_arr, binary_preds, zero_division=0))
            try:
                auc = float(roc_auc_score(labels_arr, preds))
            except:
                pass
        
        return BenchmarkResult(
            mean_latency_ms=mean_lat,
            std_latency_ms=std_lat,
            median_latency_ms=median_lat,
            throughput_fps=throughput,
            peak_memory_mb=peak_mem,
            avg_memory_mb=avg_mem,
            num_parameters=num_params,
            model_size_mb=model_size,
            device=str(self.device),
            batch_size=batch_size,
            num_samples=len(latencies) * batch_size,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            auc_roc=auc
        )
    
    def compare(
        self,
        models: Dict[str, torch.nn.Module],
        dataloader: torch.utils.data.DataLoader,
        num_iterations: Optional[int] = None
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple models."""
        results = {}
        for name, model in models.items():
            print(f"\nBenchmarking: {name}")
            results[name] = self.benchmark(model, dataloader, num_iterations, compute_metrics=True)
        
        # Comparison table
        print("\n" + "="*100)
        print(f"{'Model':<20} {'Latency(ms)':<16} {'FPS':<10} {'Memory(MB)':<12} {'Params':<14} {'Accuracy'}")
        print("-"*100)
        for name, r in results.items():
            acc_str = f"{r.accuracy:.4f}" if r.accuracy else "N/A"
            print(f"{name:<20} {r.mean_latency_ms:>5.2f}±{r.std_latency_ms:<6.2f} "
                  f"{r.throughput_fps:>8.1f}  {r.peak_memory_mb:>10.1f}  "
                  f"{r.num_parameters:>12,}  {acc_str}")
        print("="*100)
        return results
    
    def benchmark_batch_sizes(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_iterations: int = 100
    ) -> Dict[int, BenchmarkResult]:
        """Test different batch sizes."""
        results = {}
        for bs in batch_sizes:
            print(f"\nBatch size: {bs}")
            dataset = torch.utils.data.TensorDataset(
                torch.randn(bs * num_iterations, *input_shape),
                torch.randint(0, 2, (bs * num_iterations,))
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            try:
                results[bs] = self.benchmark(model, dataloader, num_iterations)
            except RuntimeError as e:
                print(f"Failed: {e}")
        
        # Summary
        print("\n" + "="*75)
        print(f"{'Batch Size':<12} {'Latency(ms)':<18} {'Throughput(fps)':<18} {'Memory(MB)'}")
        print("-"*75)
        for bs, r in results.items():
            print(f"{bs:<12} {r.mean_latency_ms:>7.2f}±{r.std_latency_ms:<6.2f} "
                  f"{r.throughput_fps:>14.1f}     {r.peak_memory_mb:>10.1f}")
        print("="*75)
        return results


def compare_quantized(
    float_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Compare float vs quantized models."""
    benchmarker = ModelBenchmark(device=device, use_amp=False)
    results = benchmarker.compare({'Float32': float_model, 'Quantized': quantized_model}, dataloader)
    
    float_r = results['Float32']
    quant_r = results['Quantized']
    
    speedup = float_r.mean_latency_ms / quant_r.mean_latency_ms
    mem_reduction = (float_r.peak_memory_mb - quant_r.peak_memory_mb) / float_r.peak_memory_mb * 100
    size_reduction = (float_r.model_size_mb - quant_r.model_size_mb) / float_r.model_size_mb * 100
    acc_drop = (float_r.accuracy - quant_r.accuracy) * 100 if float_r.accuracy else None
    
    print("\n" + "="*60)
    print("QUANTIZATION IMPACT")
    print("="*60)
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Memory Reduction: {mem_reduction:.1f}%")
    print(f"Size Reduction:   {size_reduction:.1f}%")
    if acc_drop is not None:
        print(f"Accuracy Drop:    {acc_drop:.2f}%")
    print("="*60)
    
    return {
        'float_result': float_r,
        'quantized_result': quant_r,
        'speedup': speedup,
        'memory_reduction_percent': mem_reduction,
        'size_reduction_percent': size_reduction,
        'accuracy_drop': acc_drop
    }
