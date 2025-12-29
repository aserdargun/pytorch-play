"""
Beginner-level topic content and explanations.

Provides simplified explanations and guided examples for beginners.
"""

from typing import Dict, Any


def get_tensor_explanation(level: str = "beginner") -> Dict[str, Any]:
    """Get tensor topic explanation based on level."""
    if level == "beginner":
        return {
            "summary": """
**What are Tensors?**

Tensors are the basic building blocks in PyTorch - think of them as containers for numbers.

- A **scalar** is a single number (0-dimensional tensor)
- A **vector** is a list of numbers (1-dimensional tensor)
- A **matrix** is a grid of numbers (2-dimensional tensor)
- **Higher-dimensional tensors** are like stacks of matrices

**Key things to know:**
1. **Shape**: The dimensions of your tensor (e.g., 3x4 means 3 rows, 4 columns)
2. **dtype**: The type of numbers (float32, int64, etc.)
3. **device**: Where the tensor lives (CPU or GPU)
""",
            "tips": [
                "Start with simple 1D and 2D tensors",
                "Use .shape to check dimensions",
                "Use .to(device) to move between CPU and GPU",
            ],
            "common_errors": [
                "Shape mismatch: Check tensor dimensions match for operations",
                "Device mismatch: All tensors in an operation must be on the same device",
            ],
        }
    elif level == "intermediate":
        return {
            "summary": """
**Tensors Deep Dive**

Tensors are multi-dimensional arrays with:
- Automatic differentiation support via `requires_grad`
- Device-agnostic computation (CPU, CUDA, MPS)
- Broadcasting for shape-compatible operations
- Memory-efficient views and slicing

**Important concepts:**
- **Contiguity**: Memory layout affects performance
- **Views vs Copies**: `.view()` shares memory, `.clone()` copies
- **In-place operations**: Use `_` suffix (e.g., `add_()`) carefully with autograd
""",
            "tips": [
                "Use torch.no_grad() or inference_mode() for inference",
                "Prefer contiguous() before view() for safety",
                "Check is_contiguous() when debugging performance",
            ],
            "common_errors": [
                "Modifying tensors in-place can break gradient computation",
                "Broadcasting can silently create large tensors",
            ],
        }
    else:  # advanced
        return {
            "summary": """
**Tensor Internals**

Understanding tensor memory and computation:

- **Strided Tensors**: Shape + strides define how to traverse memory
- **Storage**: Underlying contiguous memory buffer shared by views
- **Tensor Subclasses**: Custom tensors for sparse, quantized, meta tensors

**Performance considerations:**
- Channel-last format (`memory_format=torch.channels_last`) for vision
- Contiguous memory access patterns
- Avoiding unnecessary copies in data pipelines
""",
            "tips": [
                "Profile with torch.profiler for memory and compute bottlenecks",
                "Use .storage() to inspect underlying data sharing",
                "Consider torch.compile for optimized tensor operations",
            ],
            "advanced_topics": [
                "Custom tensor subclasses",
                "Sparse tensor formats",
                "Quantized tensors",
            ],
        }


def get_autograd_explanation(level: str = "beginner") -> Dict[str, Any]:
    """Get autograd topic explanation based on level."""
    if level == "beginner":
        return {
            "summary": """
**What is Autograd?**

Autograd automatically calculates gradients - the numbers that tell us how to update our model.

**The basics:**
1. Create tensors with `requires_grad=True`
2. Do math operations (this builds a computation graph)
3. Call `.backward()` on the result
4. Check `.grad` for the gradients

**Simple example:**
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4
y.backward()  # Compute gradient
print(x.grad)  # tensor([4.]) because d(x²)/dx = 2x = 4
```
""",
            "tips": [
                "Remember to zero gradients before each backward pass",
                "Use .detach() to stop gradient tracking",
            ],
        }
    elif level == "intermediate":
        return {
            "summary": """
**Autograd Mechanics**

PyTorch builds a dynamic computation graph during the forward pass:
- Each operation creates a node with a `grad_fn`
- `backward()` traverses this graph in reverse
- Gradients accumulate in `.grad` (additive!)

**Key patterns:**
- `torch.no_grad()`: Disable gradient tracking (inference)
- `tensor.detach()`: Remove from computation graph
- `retain_graph=True`: Keep graph for multiple backward passes
- `create_graph=True`: For higher-order gradients
""",
            "tips": [
                "Use optimizer.zero_grad() before each training step",
                "Check for gradient explosions with grad.norm()",
                "Use hooks for gradient debugging",
            ],
        }
    else:
        return {
            "summary": """
**Advanced Autograd**

Customization and optimization:

- **Custom backward**: Implement `torch.autograd.Function` for custom ops
- **Gradient checkpointing**: Trade compute for memory
- **Gradient hooks**: Register hooks for gradient manipulation

**Memory management:**
- Graph is freed after backward() by default
- Intermediate tensors can be saved/recomputed
- `checkpoint_sequential` for activation checkpointing
""",
            "advanced_topics": [
                "Implementing custom autograd.Function",
                "Higher-order gradients (Hessian, Jacobian)",
                "Gradient accumulation strategies",
            ],
        }


def get_dataset_explanation(level: str = "beginner") -> Dict[str, Any]:
    """Get dataset topic explanation based on level."""
    if level == "beginner":
        return {
            "summary": """
**Datasets and DataLoaders**

**Dataset**: Stores your data samples and labels
**DataLoader**: Loads data in batches for training

**Simple workflow:**
1. Create or load a Dataset
2. Wrap it with DataLoader
3. Iterate over batches in your training loop

```python
dataset = MyDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # Train on this batch
```
""",
            "tips": [
                "Use shuffle=True for training, False for validation",
                "Start with small batch sizes (16-64)",
            ],
        }
    else:
        return {
            "summary": """
**Advanced Data Loading**

**Performance tuning:**
- `num_workers`: Parallel data loading (start with 2-4)
- `pin_memory`: Faster CPU-to-GPU transfer (CUDA only)
- `persistent_workers`: Keep workers alive between epochs
- `prefetch_factor`: Control prefetching

**Custom datasets:**
- Implement `__len__` and `__getitem__`
- Lazy loading for large datasets
- Memory mapping for efficiency
""",
            "tips": [
                "Profile data loading to find bottlenecks",
                "Use IterableDataset for streaming data",
                "Consider TorchData for complex pipelines",
            ],
        }


def get_training_explanation(level: str = "beginner") -> Dict[str, Any]:
    """Get training loop explanation based on level."""
    if level == "beginner":
        return {
            "summary": """
**The Training Loop**

Training follows these steps each iteration:
1. **Forward pass**: Run data through the model
2. **Compute loss**: Measure how wrong the predictions are
3. **Backward pass**: Calculate gradients
4. **Update**: Adjust model weights using the optimizer

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()      # Clear old gradients
        pred = model(x)            # Forward pass
        loss = loss_fn(pred, y)    # Compute loss
        loss.backward()            # Backward pass
        optimizer.step()           # Update weights
```
""",
            "tips": [
                "Always zero gradients before backward",
                "Monitor loss to see if training is working",
                "Start with small learning rates (0.001)",
            ],
        }
    else:
        return {
            "summary": """
**Advanced Training**

**Stability techniques:**
- Gradient clipping: `torch.nn.utils.clip_grad_norm_`
- Learning rate scheduling: Warmup, cosine decay
- Regularization: Dropout, weight decay, batch norm

**Performance optimization:**
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- torch.compile for model optimization
""",
            "advanced_topics": [
                "Custom training loops for complex scenarios",
                "Multi-task learning",
                "Curriculum learning",
            ],
        }
