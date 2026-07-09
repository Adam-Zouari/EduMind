"""
GPU Monitoring Utilities for MLflow Experiments

Provides functions for tracking GPU memory usage, utilization, and performance.
Gracefully handles cases where CUDA is not available.
"""

import logging
from typing import Dict, Optional, Callable, Any
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU monitoring will be limited.")

try:
    import pynvml
    PYNVML_AVAILABLE = True
    # Initialize NVML
    try:
        pynvml.nvmlInit()
        logger.info("NVML initialized successfully for GPU monitoring")
    except Exception as e:
        PYNVML_AVAILABLE = False
        logger.warning(f"Could not initialize NVML: {e}")
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. Install with: pip install pynvml")


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_gpu_memory_usage(device_id: int = 0) -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device_id: GPU device ID (default: 0)
        
    Returns:
        Dictionary with memory metrics in MB
    """
    metrics = {
        'allocated_mb': 0.0,
        'reserved_mb': 0.0,
        'free_mb': 0.0,
        'total_mb': 0.0
    }
    
    if not is_cuda_available():
        logger.debug("CUDA not available, returning zero GPU memory")
        return metrics
    
    try:
        # PyTorch memory stats
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        
        metrics['allocated_mb'] = allocated
        metrics['reserved_mb'] = reserved
        
        # Get total memory using pynvml if available
        if PYNVML_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics['total_mb'] = mem_info.total / (1024 ** 2)
            metrics['free_mb'] = mem_info.free / (1024 ** 2)
        
    except Exception as e:
        logger.warning(f"Error getting GPU memory usage: {e}")
    
    return metrics


def get_gpu_utilization(device_id: int = 0) -> Dict[str, float]:
    """
    Get GPU utilization percentage.
    
    Args:
        device_id: GPU device ID (default: 0)
        
    Returns:
        Dictionary with utilization metrics
    """
    metrics = {
        'gpu_utilization_percent': 0.0,
        'memory_utilization_percent': 0.0
    }
    
    if not PYNVML_AVAILABLE:
        logger.debug("NVML not available, returning zero utilization")
        return metrics
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # Get utilization rates
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics['gpu_utilization_percent'] = utilization.gpu
        metrics['memory_utilization_percent'] = utilization.memory
        
    except Exception as e:
        logger.warning(f"Error getting GPU utilization: {e}")
    
    return metrics


def get_gpu_info(device_id: int = 0) -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Args:
        device_id: GPU device ID (default: 0)
        
    Returns:
        Dictionary with GPU information
    """
    info = {
        'cuda_available': is_cuda_available(),
        'device_id': device_id,
        'device_name': 'N/A',
        'driver_version': 'N/A',
        'cuda_version': 'N/A'
    }
    
    if not is_cuda_available():
        return info
    
    try:
        # Get device name
        if TORCH_AVAILABLE:
            info['device_name'] = torch.cuda.get_device_name(device_id)
            info['cuda_version'] = torch.version.cuda
        
        # Get driver version
        if PYNVML_AVAILABLE:
            info['driver_version'] = pynvml.nvmlSystemGetDriverVersion()
            
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
    
    return info


def monitor_gpu_during_execution(func: Callable) -> Callable:
    """
    Decorator to monitor GPU usage during function execution.
    
    Usage:
        @monitor_gpu_during_execution
        def my_gpu_function():
            # GPU-intensive code
            pass
        
        result = my_gpu_function()
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function that returns (result, gpu_metrics)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_cuda_available():
            result = func(*args, **kwargs)
            return result, {'cuda_available': False}
        
        # Get initial state
        initial_memory = get_gpu_memory_usage()
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get final state
        final_memory = get_gpu_memory_usage()
        utilization = get_gpu_utilization()
        
        # Compile metrics
        gpu_metrics = {
            'execution_time_sec': execution_time,
            'peak_allocated_mb': final_memory['allocated_mb'],
            'peak_reserved_mb': final_memory['reserved_mb'],
            'memory_delta_mb': final_memory['allocated_mb'] - initial_memory['allocated_mb'],
            'gpu_utilization_percent': utilization['gpu_utilization_percent'],
            'memory_utilization_percent': utilization['memory_utilization_percent']
        }
        
        return result, gpu_metrics
    
    return wrapper


def measure_throughput(
    func: Callable,
    num_items: int,
    *args,
    **kwargs
) -> Dict[str, float]:
    """
    Measure throughput (items per second) for a batch processing function.
    
    Args:
        func: Function to measure
        num_items: Number of items processed
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Dictionary with throughput metrics
    """
    start_time = time.time()
    
    # Get initial GPU state if available
    if is_cuda_available():
        initial_memory = get_gpu_memory_usage()
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Calculate metrics
    execution_time = time.time() - start_time
    throughput = num_items / execution_time if execution_time > 0 else 0
    
    metrics = {
        'num_items': num_items,
        'execution_time_sec': execution_time,
        'throughput_items_per_sec': throughput,
        'latency_per_item_ms': (execution_time * 1000) / num_items if num_items > 0 else 0
    }
    
    # Add GPU metrics if available
    if is_cuda_available():
        final_memory = get_gpu_memory_usage()
        utilization = get_gpu_utilization()
        
        metrics['gpu_memory_mb'] = final_memory['allocated_mb']
        metrics['gpu_utilization_percent'] = utilization['gpu_utilization_percent']
    
    return metrics


def reset_peak_memory_stats(device_id: int = 0):
    """
    Reset PyTorch's peak memory tracking.
    
    Args:
        device_id: GPU device ID (default: 0)
    """
    if is_cuda_available() and TORCH_AVAILABLE:
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.empty_cache()


def get_peak_memory_stats(device_id: int = 0) -> Dict[str, float]:
    """
    Get peak memory statistics.
    
    Args:
        device_id: GPU device ID (default: 0)
        
    Returns:
        Dictionary with peak memory metrics in MB
    """
    metrics = {
        'peak_allocated_mb': 0.0,
        'peak_reserved_mb': 0.0
    }
    
    if not is_cuda_available():
        return metrics
    
    try:
        metrics['peak_allocated_mb'] = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
        metrics['peak_reserved_mb'] = torch.cuda.max_memory_reserved(device_id) / (1024 ** 2)
    except Exception as e:
        logger.warning(f"Error getting peak memory stats: {e}")
    
    return metrics


if __name__ == "__main__":
    # Test GPU utilities
    print("=== GPU Utilities Test ===\n")
    
    print(f"CUDA Available: {is_cuda_available()}")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    print(f"NVML Available: {PYNVML_AVAILABLE}")
    
    if is_cuda_available():
        print("\n--- GPU Info ---")
        info = get_gpu_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("\n--- GPU Memory ---")
        memory = get_gpu_memory_usage()
        for key, value in memory.items():
            print(f"{key}: {value:.2f} MB")
        
        print("\n--- GPU Utilization ---")
        utilization = get_gpu_utilization()
        for key, value in utilization.items():
            print(f"{key}: {value:.2f}%")
        
        # Test decorator
        @monitor_gpu_during_execution
        def dummy_gpu_function():
            import torch
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            return y
        
        print("\n--- Testing GPU Monitoring Decorator ---")
        result, metrics = dummy_gpu_function()
        print("GPU Metrics during execution:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo CUDA available. GPU monitoring features disabled.")
    
    print("\n✓ GPU utilities test complete")
