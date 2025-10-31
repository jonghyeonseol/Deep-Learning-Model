"""
Performance profiling utilities for deep learning models.

This module provides tools for:
- Timing training operations
- Memory usage tracking (CPU and GPU)
- Model complexity analysis (FLOPs, parameters)
- Inference speed benchmarking
- Bottleneck identification

Example Usage:
    >>> from utils.profiler import Profiler, profile_model
    >>>
    >>> # Profile training
    >>> profiler = Profiler()
    >>> with profiler.profile('forward_pass'):
    ...     output = model(inputs)
    >>> profiler.print_stats()
    >>>
    >>> # Profile model complexity
    >>> stats = profile_model(model, input_shape=(3, 32, 32))
    >>> print(f"FLOPs: {stats['flops'] / 1e9:.2f}G")
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)


class Profiler:
    """
    Performance profiler for tracking execution time and resource usage.

    Features:
    - Context manager for timing code blocks
    - Cumulative statistics across multiple runs
    - Memory usage tracking (CPU and GPU)
    - Pretty-printed summary reports

    Example:
        >>> profiler = Profiler()
        >>> with profiler.profile('data_loading'):
        ...     data = loader.load_batch()
        >>> with profiler.profile('forward_pass'):
        ...     output = model(data)
        >>> profiler.print_stats()
    """

    def __init__(self):
        """Initialize profiler with empty statistics."""
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.current_section = None
        self.start_time = None
        self.start_memory = None

    @contextmanager
    def profile(self, section_name: str, track_memory: bool = True):
        """
        Context manager for profiling a code section.

        Args:
            section_name: Name of the section being profiled
            track_memory: Whether to track memory usage

        Example:
            >>> with profiler.profile('training_step'):
            ...     loss = model(inputs)
            ...     loss.backward()
        """
        self.start_section(section_name, track_memory)
        try:
            yield self
        finally:
            self.end_section(section_name, track_memory)

    def start_section(self, section_name: str, track_memory: bool = True):
        """Start profiling a section."""
        self.current_section = section_name
        self.start_time = time.perf_counter()

        if track_memory:
            self.start_memory = self._get_memory_usage()

    def end_section(self, section_name: str, track_memory: bool = True):
        """End profiling a section and record statistics."""
        if self.current_section != section_name:
            logger.warning(
                f"Section mismatch: expected {self.current_section}, got {section_name}"
            )

        # Record timing
        elapsed_time = time.perf_counter() - self.start_time
        self.timings[section_name].append(elapsed_time)

        # Record memory usage
        if track_memory and self.start_memory is not None:
            current_memory = self._get_memory_usage()
            memory_delta = {
                'cpu': current_memory['cpu'] - self.start_memory['cpu'],
                'gpu': current_memory['gpu'] - self.start_memory['gpu']
            }
            self.memory_usage[section_name].append(memory_delta)

        self.current_section = None
        self.start_time = None
        self.start_memory = None

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory = {
            'cpu': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }

        if torch.cuda.is_available():
            memory['gpu'] = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            memory['gpu'] = 0.0

        return memory

    def get_stats(self, section_name: Optional[str] = None) -> Dict:
        """
        Get statistics for a section or all sections.

        Args:
            section_name: Specific section name, or None for all sections

        Returns:
            Dictionary with timing and memory statistics
        """
        if section_name is not None:
            return self._compute_section_stats(section_name)

        # Return stats for all sections
        all_stats = {}
        for name in self.timings.keys():
            all_stats[name] = self._compute_section_stats(name)
        return all_stats

    def _compute_section_stats(self, section_name: str) -> Dict:
        """Compute statistics for a specific section."""
        timings = self.timings[section_name]

        stats = {
            'count': len(timings),
            'total_time': sum(timings),
            'mean_time': np.mean(timings),
            'std_time': np.std(timings),
            'min_time': min(timings),
            'max_time': max(timings)
        }

        # Add memory statistics if available
        if section_name in self.memory_usage:
            cpu_memory = [m['cpu'] for m in self.memory_usage[section_name]]
            gpu_memory = [m['gpu'] for m in self.memory_usage[section_name]]

            stats['mean_cpu_memory'] = np.mean(cpu_memory)
            stats['mean_gpu_memory'] = np.mean(gpu_memory)
            stats['max_cpu_memory'] = max(cpu_memory)
            stats['max_gpu_memory'] = max(gpu_memory)

        return stats

    def print_stats(self, sort_by: str = 'total_time'):
        """
        Print formatted profiling statistics.

        Args:
            sort_by: Sort sections by this metric
                     ('total_time', 'mean_time', 'count')
        """
        all_stats = self.get_stats()

        if not all_stats:
            logger.info("No profiling data collected")
            return

        # Sort sections
        if sort_by in ['total_time', 'mean_time']:
            sorted_sections = sorted(
                all_stats.items(),
                key=lambda x: x[1][sort_by],
                reverse=True
            )
        else:
            sorted_sections = sorted(all_stats.items(), key=lambda x: x[0])

        # Print header
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE PROFILING RESULTS")
        logger.info("=" * 80)
        logger.info(
            f"{'Section':<30} {'Count':>8} {'Total (s)':>12} "
            f"{'Mean (s)':>12} {'Std (s)':>12}"
        )
        logger.info("-" * 80)

        # Print section stats
        for section_name, stats in sorted_sections:
            logger.info(
                f"{section_name:<30} {stats['count']:>8} "
                f"{stats['total_time']:>12.4f} {stats['mean_time']:>12.4f} "
                f"{stats['std_time']:>12.4f}"
            )

        # Print memory usage if available
        if any('mean_cpu_memory' in stats for _, stats in sorted_sections):
            logger.info("\n" + "-" * 80)
            logger.info(
                f"{'Section':<30} {'CPU Mem (MB)':>15} {'GPU Mem (MB)':>15}"
            )
            logger.info("-" * 80)

            for section_name, stats in sorted_sections:
                if 'mean_cpu_memory' in stats:
                    logger.info(
                        f"{section_name:<30} "
                        f"{stats['mean_cpu_memory']:>15.2f} "
                        f"{stats['mean_gpu_memory']:>15.2f}"
                    )

        logger.info("=" * 80)

    def reset(self):
        """Reset all profiling statistics."""
        self.timings.clear()
        self.memory_usage.clear()
        self.current_section = None
        self.start_time = None
        self.start_memory = None


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (3, 32, 32),
    batch_size: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Profile model complexity and inference speed.

    Args:
        model: PyTorch model to profile
        input_shape: Input tensor shape (C, H, W)
        batch_size: Batch size for profiling
        device: Device to run profiling on

    Returns:
        Dictionary with model statistics:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - flops: Estimated FLOPs for forward pass
        - inference_time: Mean inference time (ms)
        - throughput: Images per second

    Example:
        >>> from models.resnet import ResNet18
        >>> model = ResNet18(num_classes=10)
        >>> stats = profile_model(model, input_shape=(3, 32, 32))
        >>> print(f"Parameters: {stats['total_params']:,}")
        >>> print(f"FLOPs: {stats['flops'] / 1e9:.2f}G")
    """
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape).to(device)

    # Estimate FLOPs
    flops = estimate_flops(model, dummy_input)

    # Benchmark inference speed
    inference_time, throughput = benchmark_inference(
        model, input_shape, batch_size, device, num_iterations=100
    )

    # Get memory footprint
    memory_footprint = get_model_memory(model)

    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'inference_time_ms': inference_time,
        'throughput_fps': throughput,
        'memory_mb': memory_footprint
    }

    return stats


def estimate_flops(model: nn.Module, dummy_input: torch.Tensor) -> int:
    """
    Estimate FLOPs for model forward pass.

    Args:
        model: PyTorch model
        dummy_input: Example input tensor

    Returns:
        Estimated FLOPs (floating point operations)
    """
    flops = 0

    def count_conv2d(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        output_height, output_width = output.size(2), output.size(3)
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        # FLOPs = batch_size * output_size * kernel_ops * out_channels
        kernel_ops = kernel_height * kernel_width * (in_channels / groups)
        output_size = output_height * output_width
        flops += batch_size * output_size * kernel_ops * out_channels

    def count_linear(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        flops += batch_size * module.in_features * module.out_features

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv2d))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))

    # Run forward pass
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return int(flops)


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    device: str = 'cuda',
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Tuple[float, float]:
    """
    Benchmark model inference speed.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        batch_size: Batch size
        device: Device for benchmarking
        num_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Tuple of (mean_time_ms, throughput_fps)
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed_time = time.perf_counter() - start_time
            timings.append(elapsed_time)

    # Calculate statistics
    mean_time_s = np.mean(timings)
    mean_time_ms = mean_time_s * 1000
    throughput = batch_size / mean_time_s

    return mean_time_ms, throughput


def get_model_memory(model: nn.Module) -> float:
    """
    Calculate model memory footprint in MB.

    Args:
        model: PyTorch model

    Returns:
        Memory footprint in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024 / 1024
    return total_size_mb


def compare_models(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...] = (3, 32, 32),
    batch_size: int = 1
) -> Dict:
    """
    Compare multiple models' complexity and speed.

    Args:
        models: Dictionary mapping model names to model instances
        input_shape: Input tensor shape
        batch_size: Batch size for benchmarking

    Returns:
        Dictionary with comparison results

    Example:
        >>> from models.resnet import ResNet18, ResNet50
        >>> models = {
        ...     'ResNet-18': ResNet18(num_classes=10),
        ...     'ResNet-50': ResNet50(num_classes=10)
        ... }
        >>> results = compare_models(models)
        >>> print_model_comparison(results)
    """
    results = {}

    for name, model in models.items():
        logger.info(f"Profiling {name}...")
        stats = profile_model(model, input_shape, batch_size)
        results[name] = stats

    return results


def print_model_comparison(results: Dict):
    """
    Print formatted model comparison table.

    Args:
        results: Dictionary from compare_models()
    """
    logger.info("\n" + "=" * 100)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 100)
    logger.info(
        f"{'Model':<20} {'Params (M)':>12} {'FLOPs (G)':>12} "
        f"{'Latency (ms)':>15} {'Throughput (fps)':>18}"
    )
    logger.info("-" * 100)

    for name, stats in results.items():
        logger.info(
            f"{name:<20} "
            f"{stats['total_params'] / 1e6:>12.2f} "
            f"{stats['flops'] / 1e9:>12.2f} "
            f"{stats['inference_time_ms']:>15.2f} "
            f"{stats['throughput_fps']:>18.2f}"
        )

    logger.info("=" * 100)


def profile_training_step(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: str = 'cuda',
    num_batches: int = 10
):
    """
    Profile a complete training step including forward, backward, and optimizer step.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device for training
        num_batches: Number of batches to profile

    Returns:
        Profiler with timing statistics
    """
    model.train()
    profiler = Profiler()

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        with profiler.profile('data_transfer'):
            pass  # Already transferred above

        optimizer.zero_grad()

        with profiler.profile('forward_pass'):
            outputs = model(inputs)

        with profiler.profile('loss_computation'):
            loss = criterion(outputs, targets)

        with profiler.profile('backward_pass'):
            loss.backward()

        with profiler.profile('optimizer_step'):
            optimizer.step()

    return profiler


if __name__ == '__main__':
    # Example usage
    from models.resnet import ResNet18

    print("Profiling ResNet-18...")
    model = ResNet18(num_classes=10)
    stats = profile_model(model, input_shape=(3, 32, 32), batch_size=32)

    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {stats['total_params']:,}")
    print(f"  Trainable Parameters: {stats['trainable_params']:,}")
    print(f"  FLOPs: {stats['flops'] / 1e9:.2f}G")
    print(f"  Inference Time: {stats['inference_time_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_fps']:.2f} fps")
    print(f"  Memory Footprint: {stats['memory_mb']:.2f} MB")
