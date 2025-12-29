"""
Advanced-level topic content and explanations.

Provides deep technical explanations for experienced users.
"""

from typing import Dict, Any, List


def get_torch_compile_explanation() -> Dict[str, Any]:
    """Get torch.compile explanation."""
    return {
        "summary": """
**torch.compile (PyTorch 2.0+)**

JIT compilation for automatic optimization:

**Basic usage:**
```python
model = torch.compile(model)
# or with options
model = torch.compile(model, mode="reduce-overhead")
```

**Compilation modes:**
- `default`: Balanced compilation time and performance
- `reduce-overhead`: Minimize CPU overhead (good for small models)
- `max-autotune`: Maximum optimization (long compile, best performance)

**Backends:**
- `inductor` (default): TorchInductor, generates Triton/C++ kernels
- `eager`: No compilation (debugging)
- `aot_eager`: AOT compilation to eager (debugging)

**What gets optimized:**
- Operator fusion (reduce memory bandwidth)
- Memory format optimization
- Kernel selection and tuning
""",
        "tips": [
            "First run is slow (compilation), subsequent runs are fast",
            "Use torch._dynamo.explain(model)(x) to understand compilation",
            "Avoid graph breaks for best performance",
        ],
        "common_issues": [
            "Graph breaks: Dynamic control flow, data-dependent shapes",
            "Fallback to eager: Check torch._dynamo.list_backends()",
            "Long compile times: Use mode='reduce-overhead' for development",
        ],
        "requirements": [
            "PyTorch 2.0+",
            "Triton for GPU kernels (auto-installed on Linux)",
        ],
    }


def get_profiler_explanation() -> Dict[str, Any]:
    """Get PyTorch Profiler explanation."""
    return {
        "summary": """
**PyTorch Profiler**

Identify performance bottlenecks:

**Basic usage:**
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Key metrics:**
- `cpu_time`: Time on CPU
- `cuda_time`: Time on GPU
- `self_cpu_time`: Excluding child ops
- `cpu_memory_usage`: Memory allocated

**TensorBoard integration:**
```python
with torch.profiler.profile(
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as prof:
    for step, data in enumerate(loader):
        prof.step()
        train_step(data)
```
""",
        "tips": [
            "Use warmup iterations before profiling",
            "Profile representative workload",
            "Check both CPU and GPU time for bottlenecks",
        ],
        "common_bottlenecks": [
            "Data loading: Profile shows gaps between steps",
            "Memory transfers: High time in to() or copy_",
            "Small operations: CPU overhead dominates (use compile)",
        ],
    }


def get_dataloader_performance_explanation() -> Dict[str, Any]:
    """Get DataLoader performance explanation."""
    return {
        "summary": """
**DataLoader Performance Tuning**

Optimize data loading to prevent GPU starvation:

**Key parameters:**
```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,      # Parallel workers (0 = main process)
    pin_memory=True,    # Faster CPU->GPU transfer (CUDA only)
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2,  # Batches to prefetch per worker
)
```

**num_workers guidelines:**
- Start with 2-4
- More isn't always better (memory overhead)
- Profile to find optimal value
- Windows: Often needs 0 due to multiprocessing issues

**pin_memory:**
- Allocates in pinned (page-locked) memory
- Faster transfer to GPU
- Only useful with CUDA

**prefetch_factor:**
- Higher = more memory, potentially faster
- Default is 2
""",
        "platform_notes": {
            "Linux": "Best multiprocessing support, use num_workers > 0",
            "macOS": "Works well, may need spawn start method",
            "Windows": "Often set num_workers=0 due to spawn overhead",
        },
        "tips": [
            "Profile data loading separately from training",
            "Consider memory-mapped datasets for large data",
            "Use IterableDataset for streaming data",
        ],
    }


def get_mps_explanation() -> Dict[str, Any]:
    """Get MPS (Metal Performance Shaders) explanation."""
    return {
        "summary": """
**MPS Backend (Apple Silicon)**

GPU acceleration on Apple M1/M2/M3 chips:

**Setup:**
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    data = data.to(device)
```

**What works well:**
- Standard neural network operations
- Convolutions, linear layers, activations
- Most torchvision operations

**Known limitations:**
- Some operations fall back to CPU
- Certain dtype combinations unsupported
- Performance varies by operation

**Debugging:**
```python
# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Force sync for accurate timing
torch.mps.synchronize()
```
""",
        "tips": [
            "MPS is evolving rapidly - update PyTorch for best support",
            "Use torch.mps.synchronize() before timing",
            "Fall back to CPU for unsupported operations",
        ],
        "official_docs": "https://docs.pytorch.org/docs/stable/notes/mps.html",
    }


def get_quantization_explanation() -> Dict[str, Any]:
    """Get quantization explanation."""
    return {
        "summary": """
**Quantization**

Reduce model size and increase inference speed:

**Types:**
- **Dynamic quantization**: Weights quantized ahead of time, activations on-the-fly
- **Static quantization**: Both weights and activations quantized (requires calibration)
- **Quantization-aware training (QAT)**: Train with fake quantization for best accuracy

**Dynamic quantization (simplest):**
```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)
```

**Benefits:**
- 2-4x model size reduction
- 2-4x speedup on CPU
- Minimal accuracy loss with proper technique

**Considerations:**
- CPU inference only (for most methods)
- Not all layers support quantization
- Accuracy may degrade - validate carefully
""",
        "tips": [
            "Start with dynamic quantization for simplicity",
            "Use QAT for best accuracy",
            "Profile to verify speedup on your hardware",
        ],
        "official_docs": "https://pytorch.org/docs/stable/quantization.html",
    }


def get_distributed_explanation() -> Dict[str, Any]:
    """Get distributed training explanation."""
    return {
        "summary": """
**Distributed Training**

Scale training across multiple GPUs/nodes:

**Approaches:**

**DataParallel (DP) - Simple but limited:**
```python
model = nn.DataParallel(model)  # Wraps model
```
- Single process, easy to use
- Limited scaling due to GIL and GPU-GPU communication

**DistributedDataParallel (DDP) - Recommended:**
```python
# Initialize process group
dist.init_process_group("nccl")

model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank]
)
```
- One process per GPU
- Efficient gradient synchronization
- Near-linear scaling

**FSDP - For large models:**
- Shards model across GPUs
- Enables training models larger than GPU memory
- More complex setup

**Key concepts:**
- `world_size`: Total number of processes
- `rank`: This process's ID (0 to world_size-1)
- `local_rank`: GPU ID on this machine
""",
        "tips": [
            "Use torchrun for launching distributed training",
            "DDP is almost always better than DP",
            "Adjust batch size and LR for effective batch size",
        ],
        "launch_command": "torchrun --nproc_per_node=4 train.py",
    }


def get_memory_optimization_explanation() -> Dict[str, Any]:
    """Get memory optimization explanation."""
    return {
        "summary": """
**Memory Optimization**

Reduce GPU memory usage for larger models/batches:

**Gradient Checkpointing:**
```python
from torch.utils.checkpoint import checkpoint

# Trade compute for memory
output = checkpoint(layer, input)
```
- Recomputes activations during backward
- Saves memory at cost of ~20% more compute

**Mixed Precision (AMP):**
- FP16 uses half the memory of FP32
- Combine with gradient scaling

**Gradient Accumulation:**
```python
for i, (data, target) in enumerate(loader):
    loss = model(data, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
- Simulates larger batch sizes
- Reduces memory per step

**Memory profiling:**
```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
torch.cuda.empty_cache()  # Free cached memory
```
""",
        "tips": [
            "Combine multiple techniques for best results",
            "Profile to understand where memory goes",
            "Consider model parallelism for very large models",
        ],
    }


def get_all_advanced_explanations() -> Dict[str, Dict[str, Any]]:
    """Get all advanced-level explanations."""
    return {
        "torch_compile": get_torch_compile_explanation(),
        "profiler": get_profiler_explanation(),
        "dataloader_performance": get_dataloader_performance_explanation(),
        "mps": get_mps_explanation(),
        "quantization": get_quantization_explanation(),
        "distributed": get_distributed_explanation(),
        "memory_optimization": get_memory_optimization_explanation(),
    }
