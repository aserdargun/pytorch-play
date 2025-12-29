"""
Intermediate-level topic content and explanations.

Provides deeper explanations for users with basic PyTorch knowledge.
"""

from typing import Dict, Any, List


def get_cnn_explanation() -> Dict[str, Any]:
    """Get CNN topic explanation."""
    return {
        "summary": """
**Convolutional Neural Networks (CNNs)**

CNNs are designed for spatial data like images:

**Key layers:**
- `Conv2d`: Learns local patterns (edges, textures, shapes)
- `MaxPool2d`: Reduces spatial dimensions, keeps important features
- `BatchNorm2d`: Stabilizes training, allows higher learning rates
- `Linear`: Final classification layers

**Architecture pattern:**
```
Conv -> BatchNorm -> ReLU -> Pool -> ... -> Flatten -> Linear -> Output
```

**Important parameters:**
- `in_channels`, `out_channels`: Feature dimensions
- `kernel_size`: Receptive field size (3x3 is common)
- `stride`, `padding`: Control output size
""",
        "tips": [
            "Start with pretrained models for most tasks",
            "Use BatchNorm for stable training",
            "Channel dimensions: input → small → medium → large",
        ],
        "common_architectures": [
            "LeNet: Simple, great for learning",
            "ResNet: Skip connections prevent degradation",
            "VGG: Deep but simple (3x3 convs only)",
        ],
    }


def get_regularization_explanation() -> Dict[str, Any]:
    """Get regularization techniques explanation."""
    return {
        "summary": """
**Regularization Techniques**

Prevent overfitting and improve generalization:

**Dropout** (`nn.Dropout`)
- Randomly zeros elements during training
- Forces network to learn redundant representations
- Typical rates: 0.1-0.5

**Weight Decay** (L2 regularization)
- Penalizes large weights in loss function
- Set in optimizer: `Adam(..., weight_decay=1e-4)`
- Prevents any single weight from dominating

**Batch Normalization** (`nn.BatchNorm2d`)
- Normalizes layer inputs
- Provides implicit regularization
- Enables higher learning rates

**Data Augmentation**
- Random crops, flips, color jitter
- Creates more training examples
- `torchvision.transforms` provides common augmentations
""",
        "tips": [
            "Start with dropout=0.2 and weight_decay=1e-4",
            "Disable dropout during evaluation (model.eval())",
            "Combine multiple regularization techniques",
        ],
        "when_to_use": {
            "overfitting": "High training acc, low validation acc",
            "solution": "Increase regularization, add data augmentation",
        },
    }


def get_lr_scheduler_explanation() -> Dict[str, Any]:
    """Get learning rate scheduler explanation."""
    return {
        "summary": """
**Learning Rate Schedulers**

Adjust learning rate during training for better convergence:

**Common schedulers:**
- `StepLR`: Decay by factor every N epochs
- `ExponentialLR`: Continuous exponential decay
- `CosineAnnealingLR`: Smooth decay following cosine curve
- `ReduceLROnPlateau`: Reduce when metric stops improving
- `OneCycleLR`: Warmup then decay (often best results)

**Usage pattern:**
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    train(...)
    scheduler.step()  # Update LR
```

**Warmup:**
- Start with small LR, increase gradually
- Helps stability early in training
- LinearLR or custom warmup schedule
""",
        "tips": [
            "ReduceLROnPlateau is great when you don't know optimal schedule",
            "OneCycleLR often gives best results with less tuning",
            "Log LR alongside loss for debugging",
        ],
        "scheduler_comparison": [
            {"name": "StepLR", "use_case": "Simple, predictable decay"},
            {"name": "CosineAnnealingLR", "use_case": "Smooth convergence"},
            {"name": "OneCycleLR", "use_case": "Fast training, often best results"},
        ],
    }


def get_mixed_precision_explanation() -> Dict[str, Any]:
    """Get mixed precision training explanation."""
    return {
        "summary": """
**Automatic Mixed Precision (AMP)**

Train faster with lower precision while maintaining accuracy:

**How it works:**
- Forward pass uses FP16 (half precision)
- Backward pass and optimizer use FP32
- GradScaler prevents underflow in gradients

**Usage:**
```python
scaler = torch.amp.GradScaler('cuda')

for data, target in loader:
    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 2-3x speedup on modern GPUs (Volta+)
- Reduced memory usage (larger batch sizes)
- Minimal accuracy impact
""",
        "requirements": [
            "CUDA-capable GPU (compute capability 7.0+)",
            "PyTorch 1.6+ (native AMP)",
        ],
        "tips": [
            "Always use GradScaler with autocast",
            "Some ops stay in FP32 automatically (softmax, loss)",
            "BFloat16 (BF16) is an alternative on newer hardware",
        ],
    }


def get_transfer_learning_explanation() -> Dict[str, Any]:
    """Get transfer learning explanation."""
    return {
        "summary": """
**Transfer Learning**

Leverage pretrained models for your tasks:

**Strategy 1: Feature Extraction**
- Freeze pretrained backbone
- Train only new classification head
- Fast, works well with small datasets

**Strategy 2: Fine-tuning**
- Unfreeze some/all pretrained layers
- Train with small learning rate
- Better results, needs more data

**Implementation:**
```python
# Load pretrained model
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Replace classification head
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Optionally freeze backbone
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)  # Unfreeze head
```
""",
        "tips": [
            "Start with frozen backbone, then fine-tune if needed",
            "Use smaller LR for fine-tuning (1/10 to 1/100 of normal)",
            "Normalize inputs to match pretrained model's training",
        ],
        "common_backbones": [
            "ResNet: Good balance of speed and accuracy",
            "EfficientNet: Best accuracy per FLOP",
            "Vision Transformer (ViT): State-of-the-art with enough data",
        ],
    }


def get_debugging_explanation() -> Dict[str, Any]:
    """Get debugging techniques explanation."""
    return {
        "summary": """
**Debugging PyTorch Models**

**Common errors and fixes:**

**Shape Mismatch:**
```python
# Always check shapes
print(f"Input: {x.shape}, Output: {model(x).shape}")
# Use torchinfo for model summary
from torchinfo import summary
summary(model, input_size=(batch, channels, height, width))
```

**Device Mismatch:**
```python
# Ensure all tensors on same device
x = x.to(device)
model = model.to(device)
```

**Gradient Issues:**
```python
# Check for NaN/Inf
torch.autograd.set_detect_anomaly(True)

# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

**Training not converging:**
- Check loss is decreasing
- Verify data loading (visualize samples)
- Try smaller learning rate
- Check for data leakage
""",
        "debugging_checklist": [
            "Verify data shapes and types",
            "Check tensors are on correct device",
            "Ensure model in correct mode (train/eval)",
            "Monitor loss and gradients",
            "Visualize predictions vs targets",
        ],
    }


def get_metrics_explanation() -> Dict[str, Any]:
    """Get evaluation metrics explanation."""
    return {
        "summary": """
**Evaluation Metrics**

**Classification:**
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows all prediction outcomes

**When to use what:**
- Balanced classes: Accuracy is fine
- Imbalanced classes: Use F1, precision, recall
- Multi-class: Macro/Micro/Weighted averaging

**Implementation:**
```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model(x).argmax(dim=1)
print(classification_report(y_true, y_pred))
```
""",
        "tips": [
            "Always use validation/test set for metrics",
            "For imbalanced data, accuracy is misleading",
            "Track multiple metrics to get full picture",
        ],
    }


def get_all_intermediate_explanations() -> Dict[str, Dict[str, Any]]:
    """Get all intermediate-level explanations."""
    return {
        "cnn": get_cnn_explanation(),
        "regularization": get_regularization_explanation(),
        "lr_scheduler": get_lr_scheduler_explanation(),
        "mixed_precision": get_mixed_precision_explanation(),
        "transfer_learning": get_transfer_learning_explanation(),
        "debugging": get_debugging_explanation(),
        "metrics": get_metrics_explanation(),
    }
