"""Minimal benchmark script for deepfake detection models."""

import argparse
import sys
from pathlib import Path
import torch
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_minimal import ModelBenchmark, compare_quantized
from datasets.frame_dataset import FrameDataset
from models.student.tiny_ladeda import TinyLaDeDa
from models.teacher.ladeda_wrapper import LaDeDaWrapper


def load_model(model_type: str, checkpoint_path: str):
    """Load model from checkpoint."""
    model = TinyLaDeDa(pretrained=False) if model_type == 'student' else LaDeDaWrapper(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    return model


def main():
    parser = argparse.ArgumentParser(description='Benchmark deepfake detection models')
    
    # Model & Data
    parser.add_argument('--model', type=str, choices=['student', 'teacher'], required=True)
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Benchmark
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--num-iterations', type=int, default=None, help='Limit iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    
    # Comparison modes
    parser.add_argument('--compare', type=str, help='Compare with another checkpoint')
    parser.add_argument('--compare-quantized', action='store_true', help='Compare float vs quantized')
    parser.add_argument('--float-model', type=str, help='Float model checkpoint')
    parser.add_argument('--quant-model', type=str, help='Quantized model checkpoint')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=None, 
                        help='Test multiple batch sizes')
    
    # Output
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    parser.add_argument('--quiet', action='store_true', help='Disable progress bars')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data: {args.data}")
    dataset = FrameDataset(data_dir=args.data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Samples: {len(dataset)}\n")
    
    # Init benchmarker
    benchmarker = ModelBenchmark(
        device=args.device,
        use_amp=args.use_amp,
        warmup=args.warmup,
        show_progress=not args.quiet
    )
    
    # Quantization comparison
    if args.compare_quantized:
        print("Float vs Quantized Comparison\n")
        float_model = load_model(args.model, args.float_model)
        quant_model = load_model(args.model, args.quant_model)
        
        comparison = compare_quantized(float_model, quant_model, dataloader, args.device)
        
        output = {
            'float': comparison['float_result'].__dict__,
            'quantized': comparison['quantized_result'].__dict__,
            'speedup': comparison['speedup'],
            'memory_reduction_percent': comparison['memory_reduction_percent'],
            'size_reduction_percent': comparison['size_reduction_percent'],
            'accuracy_drop': comparison['accuracy_drop']
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved: {args.output}")
    
    # Compare two checkpoints
    elif args.compare:
        print("Model Comparison\n")
        model1 = load_model(args.model, args.checkpoint)
        model2 = load_model(args.model, args.compare)
        
        results = benchmarker.compare(
            {'Checkpoint_1': model1, 'Checkpoint_2': model2},
            dataloader, args.num_iterations
        )
        
        output = {k: v.__dict__ for k, v in results.items()}
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved: {args.output}")
    
    # Batch size sweep
    elif args.batch_sizes:
        print("Batch Size Analysis\n")
        model = load_model(args.model, args.checkpoint)
        results = benchmarker.benchmark_batch_sizes(model, (3, 256, 256), args.batch_sizes)
        
        output = {str(bs): r.__dict__ for bs, r in results.items()}
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved: {args.output}")
    
    # Single model benchmark
    else:
        print(f"Benchmarking {args.model.title()} Model\n")
        model = load_model(args.model, args.checkpoint)
        
        result = benchmarker.benchmark(
            model, dataloader, args.num_iterations, compute_metrics=True
        )
        
        print(f"\n{result}")
        result.save(Path(args.output))
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
