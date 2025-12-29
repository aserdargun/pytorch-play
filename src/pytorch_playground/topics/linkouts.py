"""
External link-out resources for PyTorch Playground.

Curated lists of tutorials, videos, and webinars with official links.
These are reference resources, not dynamically scraped.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ExternalResource:
    """An external learning resource."""

    id: str
    title: str
    url: str
    category: str
    description: str
    level: str  # Beginner, Intermediate, Advanced
    tags: List[str]


# Official PyTorch Tutorial Links
TUTORIAL_LINKS: List[ExternalResource] = [
    # Beginner Tutorials
    ExternalResource(
        id="tut_60min_blitz",
        title="Deep Learning with PyTorch: A 60 Minute Blitz",
        url="https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html",
        category="Tutorials",
        description="A quick introduction to PyTorch's tensor library and neural networks.",
        level="Beginner",
        tags=["introduction", "tensors", "neural-networks"],
    ),
    ExternalResource(
        id="tut_learn_basics",
        title="Learn the Basics",
        url="https://docs.pytorch.org/tutorials/beginner/basics/intro.html",
        category="Tutorials",
        description="A step-by-step guide to building a neural network in PyTorch.",
        level="Beginner",
        tags=["basics", "neural-networks", "training"],
    ),
    ExternalResource(
        id="tut_quickstart",
        title="Quickstart",
        url="https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html",
        category="Tutorials",
        description="Build a neural network from scratch in this quickstart guide.",
        level="Beginner",
        tags=["quickstart", "neural-networks"],
    ),
    # Vision Tutorials
    ExternalResource(
        id="tut_cifar10",
        title="Training a Classifier (CIFAR-10)",
        url="https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html",
        category="Tutorials",
        description="Train an image classifier on the CIFAR-10 dataset.",
        level="Intermediate",
        tags=["vision", "classification", "cnn"],
    ),
    ExternalResource(
        id="tut_transfer_learning",
        title="Transfer Learning for Computer Vision",
        url="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
        category="Tutorials",
        description="Fine-tune a pretrained model for custom image classification.",
        level="Intermediate",
        tags=["vision", "transfer-learning", "pretrained"],
    ),
    ExternalResource(
        id="tut_finetuning_torchvision",
        title="TorchVision Fine-tuning",
        url="https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html",
        category="Tutorials",
        description="Fine-tune detection and segmentation models from torchvision.",
        level="Advanced",
        tags=["vision", "detection", "segmentation"],
    ),
    # NLP Tutorials
    ExternalResource(
        id="tut_transformer",
        title="Language Modeling with Transformer",
        url="https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html",
        category="Tutorials",
        description="Build a Transformer model for language modeling.",
        level="Advanced",
        tags=["nlp", "transformer", "language-model"],
    ),
    ExternalResource(
        id="tut_text_classification",
        title="Text Classification with TorchText",
        url="https://docs.pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html",
        category="Tutorials",
        description="Build a text classification model using TorchText.",
        level="Intermediate",
        tags=["nlp", "text", "classification"],
    ),
    # Audio Tutorials
    ExternalResource(
        id="tut_audio_data",
        title="Audio Data Loading and Preprocessing",
        url="https://docs.pytorch.org/tutorials/beginner/audio_io_tutorial.html",
        category="Tutorials",
        description="Load and preprocess audio data with torchaudio.",
        level="Beginner",
        tags=["audio", "data", "torchaudio"],
    ),
    # Deployment Tutorials
    ExternalResource(
        id="tut_torchscript",
        title="Introduction to TorchScript",
        url="https://docs.pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html",
        category="Tutorials",
        description="Export PyTorch models to TorchScript for deployment.",
        level="Advanced",
        tags=["deployment", "torchscript", "production"],
    ),
    ExternalResource(
        id="tut_onnx",
        title="Export to ONNX",
        url="https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html",
        category="Tutorials",
        description="Export models to ONNX format for cross-platform inference.",
        level="Advanced",
        tags=["deployment", "onnx", "inference"],
    ),
    # Distributed Training
    ExternalResource(
        id="tut_ddp",
        title="Distributed Data Parallel Tutorial",
        url="https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html",
        category="Tutorials",
        description="Scale training across multiple GPUs with DDP.",
        level="Advanced",
        tags=["distributed", "multi-gpu", "scaling"],
    ),
]

# YouTube Series Links
YOUTUBE_LINKS: List[ExternalResource] = [
    ExternalResource(
        id="yt_series_index",
        title="Intro to PyTorch - YouTube Series Index",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html",
        category="YouTube Series",
        description="Complete index of the Introduction to PyTorch video series.",
        level="Beginner",
        tags=["video", "series", "introduction"],
    ),
    ExternalResource(
        id="yt_intro",
        title="Introduction to PyTorch (Video)",
        url="https://www.youtube.com/watch?v=IC0_FRiX-sw",
        category="YouTube Series",
        description="Introduction to PyTorch and its ecosystem.",
        level="Beginner",
        tags=["video", "introduction"],
    ),
    ExternalResource(
        id="yt_tensors",
        title="Tensors in PyTorch (Video)",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html",
        category="YouTube Series",
        description="Deep dive into PyTorch tensors.",
        level="Beginner",
        tags=["video", "tensors"],
    ),
    ExternalResource(
        id="yt_autograd",
        title="The Fundamentals of Autograd (Video)",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html",
        category="YouTube Series",
        description="Understanding automatic differentiation in PyTorch.",
        level="Beginner",
        tags=["video", "autograd", "gradients"],
    ),
    ExternalResource(
        id="yt_building_models",
        title="Building Models with PyTorch (Video)",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html",
        category="YouTube Series",
        description="Learn to build neural network architectures.",
        level="Beginner",
        tags=["video", "models", "nn.Module"],
    ),
    ExternalResource(
        id="yt_tensorboard",
        title="PyTorch TensorBoard Support (Video)",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html",
        category="YouTube Series",
        description="Visualize training with TensorBoard.",
        level="Intermediate",
        tags=["video", "tensorboard", "visualization"],
    ),
    ExternalResource(
        id="yt_training",
        title="Training with PyTorch (Video)",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html",
        category="YouTube Series",
        description="Complete training loop walkthrough.",
        level="Beginner",
        tags=["video", "training", "loop"],
    ),
    ExternalResource(
        id="yt_captum",
        title="Model Understanding with Captum (Video)",
        url="https://docs.pytorch.org/tutorials/beginner/introyt/captumyt.html",
        category="YouTube Series",
        description="Interpret your models with Captum.",
        level="Advanced",
        tags=["video", "captum", "interpretability"],
    ),
]

# Webinar Links (curated, not scraped)
WEBINAR_LINKS: List[ExternalResource] = [
    ExternalResource(
        id="web_pytorch2_release",
        title="PyTorch 2.0 Release Blog",
        url="https://pytorch.org/blog/pytorch-2.0-release/",
        category="Webinars",
        description="Overview of PyTorch 2.0 features and improvements.",
        level="Intermediate",
        tags=["pytorch2", "release", "compile"],
    ),
    ExternalResource(
        id="web_distributed_overview",
        title="Distributed Training Overview",
        url="https://pytorch.org/tutorials/intermediate/dist_overview.html",
        category="Webinars",
        description="Comprehensive guide to distributed training in PyTorch.",
        level="Advanced",
        tags=["distributed", "ddp", "scaling"],
    ),
    ExternalResource(
        id="web_performance_tuning",
        title="Performance Tuning Guide",
        url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
        category="Webinars",
        description="Best practices for PyTorch performance optimization.",
        level="Advanced",
        tags=["performance", "optimization", "tuning"],
    ),
    ExternalResource(
        id="web_pytorch_blog",
        title="PyTorch Blog",
        url="https://pytorch.org/blog/",
        category="Webinars",
        description="Latest news, tutorials, and announcements from PyTorch.",
        level="Beginner",
        tags=["blog", "news", "updates"],
    ),
]

# Official Documentation Links
DOCUMENTATION_LINKS: List[ExternalResource] = [
    ExternalResource(
        id="doc_main",
        title="PyTorch Documentation",
        url="https://pytorch.org/docs/stable/index.html",
        category="Documentation",
        description="Official PyTorch API documentation.",
        level="Beginner",
        tags=["docs", "api", "reference"],
    ),
    ExternalResource(
        id="doc_mps",
        title="MPS Backend Documentation",
        url="https://docs.pytorch.org/docs/stable/notes/mps.html",
        category="Documentation",
        description="Apple Silicon MPS backend documentation.",
        level="Advanced",
        tags=["docs", "mps", "apple-silicon"],
    ),
    ExternalResource(
        id="doc_cuda",
        title="CUDA Semantics",
        url="https://pytorch.org/docs/stable/notes/cuda.html",
        category="Documentation",
        description="CUDA usage and best practices.",
        level="Intermediate",
        tags=["docs", "cuda", "gpu"],
    ),
    ExternalResource(
        id="doc_amp",
        title="Automatic Mixed Precision",
        url="https://pytorch.org/docs/stable/amp.html",
        category="Documentation",
        description="AMP documentation for mixed-precision training.",
        level="Intermediate",
        tags=["docs", "amp", "mixed-precision"],
    ),
    ExternalResource(
        id="doc_quantization",
        title="Quantization Documentation",
        url="https://pytorch.org/docs/stable/quantization.html",
        category="Documentation",
        description="Model quantization guide and API.",
        level="Advanced",
        tags=["docs", "quantization", "inference"],
    ),
]


def get_all_tutorials() -> List[ExternalResource]:
    """Get all tutorial links."""
    return TUTORIAL_LINKS


def get_all_youtube() -> List[ExternalResource]:
    """Get all YouTube series links."""
    return YOUTUBE_LINKS


def get_all_webinars() -> List[ExternalResource]:
    """Get all webinar links."""
    return WEBINAR_LINKS


def get_all_documentation() -> List[ExternalResource]:
    """Get all documentation links."""
    return DOCUMENTATION_LINKS


def get_resources_by_level(level: str) -> Dict[str, List[ExternalResource]]:
    """Get all resources filtered by level."""
    all_resources = (
        TUTORIAL_LINKS + YOUTUBE_LINKS + WEBINAR_LINKS + DOCUMENTATION_LINKS
    )

    level_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    target_level = level_order.get(level, 0)

    filtered = [
        r for r in all_resources if level_order.get(r.level, 0) <= target_level
    ]

    # Group by category
    by_category: Dict[str, List[ExternalResource]] = {}
    for r in filtered:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)

    return by_category


def search_resources(query: str) -> List[ExternalResource]:
    """Search all resources by query."""
    query = query.lower()
    all_resources = (
        TUTORIAL_LINKS + YOUTUBE_LINKS + WEBINAR_LINKS + DOCUMENTATION_LINKS
    )

    results = []
    for r in all_resources:
        if (
            query in r.title.lower()
            or query in r.description.lower()
            or any(query in tag.lower() for tag in r.tags)
        ):
            results.append(r)

    return results
