"""Latency and performance measurement utilities for model evaluation."""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LatencyBenchmark:
    """
    Measure model inference latency and throughput.
    
    Provides accurate measurements for:
    - Per-image latency (mean, median, p95, p99)
    - Throughput (FPS)
    - Peak memory usage
    - Warm-up handling for GPU kernels
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ):
        """
        Args:
            model: Model to benchmark
            device: Device to run on ('cuda' or 'cpu')
            warmup_iterations: Number of warm-up iterations (for GPU kernel compilation)
            benchmark_iterations: Number of timing iterations
        """
        self.model = model
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        self.model.eval()
        self.model.to(device)
        
        # CUDA timing utilities
        self.use_cuda_events = (device == "cuda" and torch.cuda.is_available())
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def measure_single_image_latency(
        self,
        image: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measure latency for a single image.
        
        Args:
            image: Input tensor (C, H, W) or (1, C, H, W)
        
        Returns:
            dict with latency statistics (ms)
        """
        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Warm-up phase
        print(f"Warming up ({self.warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = self.model(image)
                if self.use_cuda_events:
                    torch.cuda.synchronize()
        
        # Benchmark phase
        print(f"Benchmarking ({self.benchmark_iterations} iterations)...")
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                if self.use_cuda_events:
                    # Use CUDA events for precise GPU timing
                    self.start_event.record()
                    _ = self.model(image)
                    self.end_event.record()
                    torch.cuda.synchronize()
                    latency_ms = self.start_event.elapsed_time(self.end_event)
                else:
                    # Use CPU timing
                    start = time.perf_counter()
                    _ = self.model(image)
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                
                latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "fps": float(1000.0 / np.mean(latencies)),
        }
    
    def measure_batch_throughput(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int] = (3, 256, 256),
    ) -> Dict[str, float]:
        """
        Measure throughput for batch processing.
        
        Args:
            batch_size: Number of images in batch
            image_shape: Shape of single image (C, H, W)
        
        Returns:
            dict with throughput statistics
        """
        # Create dummy batch
        batch = torch.randn(batch_size, *image_shape).to(self.device)
        
        # Warm-up
        print(f"Warming up batch processing (batch_size={batch_size})...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = self.model(batch)
                if self.use_cuda_events:
                    torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking batch throughput...")
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                if self.use_cuda_events:
                    self.start_event.record()
                    _ = self.model(batch)
                    self.end_event.record()
                    torch.cuda.synchronize()
                    latency_ms = self.start_event.elapsed_time(self.end_event)
                else:
                    start = time.perf_counter()
                    _ = self.model(batch)
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                
                latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        
        return {
            "batch_size": batch_size,
            "mean_batch_latency_ms": float(np.mean(latencies)),
            "mean_per_image_latency_ms": float(np.mean(latencies) / batch_size),
            "throughput_fps": float(batch_size * 1000.0 / np.mean(latencies)),
            "total_images_processed": batch_size * self.benchmark_iterations,
        }
    
    def measure_memory_usage(
        self,
        image_shape: Tuple[int, int, int] = (3, 256, 256),
    ) -> Dict[str, float]:
        """
        Measure peak GPU/CPU memory usage during inference.
        
        Args:
            image_shape: Shape of single image (C, H, W)
        
        Returns:
            dict with memory statistics (MB)
        """
        image = torch.randn(1, *image_shape).to(self.device)
        
        if self.use_cuda_events:
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
            
            # Run inference
            with torch.no_grad():
                _ = self.model(image)
                torch.cuda.synchronize()
            
            # Get peak memory
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_memory_mb = peak_memory_bytes / (1024 ** 2)
            
            return {
                "peak_memory_mb": float(peak_memory_mb),
                "device": self.device,
            }
        else:
            # CPU memory measurement is more complex, skip for now
            return {
                "peak_memory_mb": None,
                "device": self.device,
                "note": "CPU memory tracking not implemented",
            }
    
    def run_full_benchmark(
        self,
        image_shape: Tuple[int, int, int] = (3, 256, 256),
        batch_sizes: Optional[List[int]] = None,
    ) -> Dict:
        """
        Run comprehensive benchmark suite.
        
        Args:
            image_shape: Shape of single image (C, H, W)
            batch_sizes: List of batch sizes to test (default: [1, 4, 8, 16])
        
        Returns:
            Complete benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
        
        print("\n" + "=" * 80)
        print("LATENCY BENCHMARK")
        print("=" * 80)
        
        # Single image latency
        dummy_image = torch.randn(*image_shape).to(self.device)
        single_latency = self.measure_single_image_latency(dummy_image)
        
        print("\n✓ Single Image Latency:")
        print(f"  Mean:   {single_latency['mean_ms']:.2f} ms")
        print(f"  Median: {single_latency['median_ms']:.2f} ms")
        print(f"  P95:    {single_latency['p95_ms']:.2f} ms")
        print(f"  P99:    {single_latency['p99_ms']:.2f} ms")
        print(f"  FPS:    {single_latency['fps']:.1f}")
        
        # Batch throughput
        batch_results = []
        print("\n✓ Batch Throughput:")
        for batch_size in batch_sizes:
            batch_stats = self.measure_batch_throughput(batch_size, image_shape)
            batch_results.append(batch_stats)
            print(f"  Batch {batch_size:2d}: {batch_stats['throughput_fps']:6.1f} FPS "
                  f"({batch_stats['mean_per_image_latency_ms']:.2f} ms/image)")
        
        # Memory usage
        memory_stats = self.measure_memory_usage(image_shape)
        if memory_stats["peak_memory_mb"] is not None:
            print(f"\n✓ Peak Memory: {memory_stats['peak_memory_mb']:.2f} MB")
        
        print("\n" + "=" * 80)
        
        return {
            "single_image_latency": single_latency,
            "batch_throughput": batch_results,
            "memory": memory_stats,
            "config": {
                "device": self.device,
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations,
                "image_shape": image_shape,
            },
        }


def compare_model_latencies(
    models: Dict[str, torch.nn.Module],
    device: str = "cuda",
    image_shape: Tuple[int, int, int] = (3, 256, 256),
) -> Dict:
    """
    Compare latency across multiple models.
    
    Useful for comparing teacher vs student, float vs quantized, etc.
    
    Args:
        models: Dict mapping model names to models
        device: Device to run on
        image_shape: Input image shape
    
    Returns:
        Comparison results
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nBenchmarking: {name}")
        benchmark = LatencyBenchmark(model, device=device, warmup_iterations=10, benchmark_iterations=100)
        results[name] = benchmark.run_full_benchmark(image_shape=image_shape)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean Latency (ms)':<20} {'FPS':<10} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_fps = None
    for name, result in results.items():
        latency = result["single_image_latency"]["mean_ms"]
        fps = result["single_image_latency"]["fps"]
        
        if baseline_fps is None:
            baseline_fps = fps
            speedup_str = "1.00x (baseline)"
        else:
            speedup = fps / baseline_fps
            speedup_str = f"{speedup:.2f}x"
        
        print(f"{name:<20} {latency:<20.2f} {fps:<10.1f} {speedup_str:<10}")
    
    print("=" * 80)
    
    return results
