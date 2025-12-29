"""
Topic Catalog for PyTorch Playground.

Contains the master catalog of all learning topics with:
- Level gating (Beginner/Intermediate/Advanced)
- Category organization
- Official links
- Runnable demo flags
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from enum import Enum

from pytorch_playground.state import Level


class TopicCategory(Enum):
    """Categories matching PyTorch's Learn navigation."""

    TUTORIALS = "Tutorials"
    BASICS = "Learn the Basics"
    RECIPES = "Recipes"
    YOUTUBE = "Intro to PyTorch (YouTube)"
    WEBINARS = "Webinars"


@dataclass
class Topic:
    """A learning topic with metadata and optional runnable demo."""

    id: str
    title: str
    category: TopicCategory
    level_min: Level
    description: str
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    official_link: Optional[str] = None
    runnable: bool = False
    demo_id: Optional[str] = None  # References demo module
    explanation_beginner: str = ""
    explanation_intermediate: str = ""
    explanation_advanced: str = ""

    def get_explanation(self, level: Level) -> str:
        """Get level-appropriate explanation."""
        if level == Level.BEGINNER:
            return self.explanation_beginner or self.description
        elif level == Level.INTERMEDIATE:
            return self.explanation_intermediate or self.explanation_beginner or self.description
        else:
            return (
                self.explanation_advanced
                or self.explanation_intermediate
                or self.explanation_beginner
                or self.description
            )

    def is_available_for_level(self, level: Level) -> bool:
        """Check if topic is available for given level."""
        level_order = [Level.BEGINNER, Level.INTERMEDIATE, Level.ADVANCED]
        return level_order.index(level) >= level_order.index(self.level_min)


class TopicCatalog:
    """Master catalog of all learning topics."""

    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self._load_all_topics()

    def _load_all_topics(self):
        """Load all topics into the catalog."""
        all_topics = (
            self._get_basics_topics()
            + self._get_tutorials_topics()
            + self._get_recipes_topics()
            + self._get_youtube_topics()
            + self._get_webinar_topics()
        )

        for topic in all_topics:
            self.topics[topic.id] = topic

    def _get_basics_topics(self) -> List[Topic]:
        """Get Learn the Basics topics - these map to core concepts."""
        return [
            Topic(
                id="basics_tensors",
                title="Tensors",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Learn about PyTorch tensors - the fundamental data structure.",
                tags=["tensors", "basics", "data-structures"],
                outcomes=["Create tensors", "Understand shapes and dtypes", "Move tensors between devices"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
                runnable=True,
                demo_id="tensor_lab",
                explanation_beginner="Tensors are like NumPy arrays but can run on GPUs. They're the basic building blocks of PyTorch.",
                explanation_intermediate="Tensors are multi-dimensional arrays with automatic differentiation support. They can be created on various devices (CPU, CUDA, MPS) and support broadcasting, indexing, and mathematical operations.",
                explanation_advanced="Tensors are the fundamental data abstraction in PyTorch, implementing the autograd computation graph. They support strided memory layouts, views vs copies semantics, and device-specific optimizations. Understanding memory contiguity and stride patterns is crucial for performance.",
            ),
            Topic(
                id="basics_datasets",
                title="Datasets & DataLoaders",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Learn to load and preprocess data using PyTorch's data utilities.",
                tags=["data", "datasets", "dataloaders", "basics"],
                prerequisites=["basics_tensors"],
                outcomes=["Use built-in datasets", "Create custom datasets", "Configure DataLoaders"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html",
                runnable=True,
                demo_id="data_lab",
                explanation_beginner="Datasets hold your data, and DataLoaders help you iterate through it in batches during training.",
                explanation_intermediate="The Dataset class provides a uniform interface for accessing data samples. DataLoader handles batching, shuffling, and parallel loading with worker processes.",
                explanation_advanced="Understanding Dataset/DataLoader architecture enables optimization via prefetching, pinned memory, persistent workers, and custom collate functions. Memory-mapped datasets and streaming approaches handle datasets larger than RAM.",
            ),
            Topic(
                id="basics_transforms",
                title="Transforms",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Apply transformations to data for preprocessing and augmentation.",
                tags=["transforms", "preprocessing", "augmentation", "basics"],
                prerequisites=["basics_datasets"],
                outcomes=["Apply image transforms", "Compose transform pipelines", "Understand common augmentations"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html",
                runnable=False,
                explanation_beginner="Transforms change your data, like resizing images or converting them to tensors.",
                explanation_intermediate="Transform pipelines allow composable preprocessing. torchvision.transforms provides common image operations while custom transforms enable domain-specific processing.",
                explanation_advanced="Transform design affects training dynamics through augmentation policies. Modern approaches like RandAugment, AutoAugment, and TrivialAugment learn or define augmentation strategies. GPU-accelerated transforms via torch.compile offer performance benefits.",
            ),
            Topic(
                id="basics_neural_networks",
                title="Build the Neural Network",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Learn to define neural network architectures using nn.Module.",
                tags=["neural-networks", "nn.Module", "layers", "basics"],
                prerequisites=["basics_tensors"],
                outcomes=["Create nn.Module subclasses", "Define forward pass", "Understand layer types"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
                runnable=True,
                demo_id="model_builder",
                explanation_beginner="Neural networks are built from layers. In PyTorch, you create a class that defines these layers and how data flows through them.",
                explanation_intermediate="nn.Module provides a container for parameters and submodules with hooks for training mode, device placement, and state serialization. The forward() method defines the computation graph.",
                explanation_advanced="Advanced architectures leverage nn.ModuleList, nn.ModuleDict, and nn.Sequential for dynamic architectures. Understanding parameter registration, buffer management, and hook systems enables custom training behaviors and model surgery.",
            ),
            Topic(
                id="basics_autograd",
                title="Automatic Differentiation",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Understand PyTorch's autograd engine for automatic gradient computation.",
                tags=["autograd", "gradients", "backpropagation", "basics"],
                prerequisites=["basics_tensors"],
                outcomes=["Compute gradients", "Understand requires_grad", "Use backward()"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html",
                runnable=True,
                demo_id="autograd_lab",
                explanation_beginner="Autograd automatically calculates gradients for you - the numbers needed to update your model during training.",
                explanation_intermediate="Autograd builds a dynamic computation graph during forward pass and uses reverse-mode automatic differentiation during backward(). Gradient accumulation and zeroing are critical for correct training.",
                explanation_advanced="The autograd engine implements reverse-mode AD with support for higher-order gradients, custom backward functions via torch.autograd.Function, and gradient checkpointing for memory efficiency. Understanding the computation graph lifecycle helps debug memory issues.",
            ),
            Topic(
                id="basics_optimization",
                title="Optimization Loop",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Learn the training loop: forward pass, loss, backward pass, optimizer step.",
                tags=["training", "optimization", "loss", "basics"],
                prerequisites=["basics_autograd", "basics_neural_networks"],
                outcomes=["Implement training loop", "Use optimizers", "Compute loss"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
                runnable=True,
                demo_id="train_eval",
                explanation_beginner="Training follows a loop: feed data in, measure error (loss), calculate how to improve (gradients), then update the model.",
                explanation_intermediate="The optimization loop coordinates data loading, forward pass, loss computation, backward pass, and parameter updates. Understanding optimizer state, learning rate schedules, and gradient clipping improves training stability.",
                explanation_advanced="Advanced training involves gradient accumulation for large effective batch sizes, mixed precision training, gradient scaling, and distributed synchronization. Profiling identifies bottlenecks in the training pipeline.",
            ),
            Topic(
                id="basics_save_load",
                title="Save and Load Models",
                category=TopicCategory.BASICS,
                level_min=Level.BEGINNER,
                description="Persist and restore model state for inference or continued training.",
                tags=["save", "load", "checkpoint", "state_dict", "basics"],
                prerequisites=["basics_neural_networks"],
                outcomes=["Save model state", "Load for inference", "Create training checkpoints"],
                official_link="https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html",
                runnable=True,
                demo_id="save_load",
                explanation_beginner="Save your trained model to a file so you can use it later without retraining.",
                explanation_intermediate="state_dict() captures learnable parameters. For training resumption, also save optimizer state, epoch, and metrics. torch.save/load handle serialization with pickle.",
                explanation_advanced="Production deployments use TorchScript or torch.export for portable formats. Checkpoint sharding handles large models. Understanding the save format enables model surgery and architecture migration.",
            ),
        ]

    def _get_tutorials_topics(self) -> List[Topic]:
        """Get Tutorials topics - comprehensive guides."""
        return [
            # Image/Vision tutorials
            Topic(
                id="tutorial_image_classification",
                title="Image Classification with CNNs",
                category=TopicCategory.TUTORIALS,
                level_min=Level.INTERMEDIATE,
                description="Train a convolutional neural network for image classification.",
                tags=["vision", "cnn", "classification", "torchvision"],
                prerequisites=["basics_neural_networks", "basics_optimization"],
                outcomes=["Build CNN architectures", "Train on image datasets", "Evaluate accuracy"],
                official_link="https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html",
                runnable=True,
                demo_id="train_eval",
                explanation_intermediate="CNNs use convolutional layers to learn spatial hierarchies in images. This tutorial covers building and training a classifier on CIFAR-10.",
            ),
            Topic(
                id="tutorial_transfer_learning",
                title="Transfer Learning for Computer Vision",
                category=TopicCategory.TUTORIALS,
                level_min=Level.INTERMEDIATE,
                description="Fine-tune pretrained models for custom image classification tasks.",
                tags=["vision", "transfer-learning", "fine-tuning", "pretrained"],
                prerequisites=["tutorial_image_classification"],
                outcomes=["Load pretrained models", "Replace classification head", "Fine-tune effectively"],
                official_link="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
                runnable=False,
                explanation_intermediate="Transfer learning leverages features learned on large datasets (ImageNet) for new tasks. Fine-tuning the last layers is faster and often more effective than training from scratch.",
            ),
            Topic(
                id="tutorial_object_detection",
                title="TorchVision Object Detection Fine-tuning",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Fine-tune object detection models from torchvision.",
                tags=["vision", "object-detection", "fine-tuning"],
                prerequisites=["tutorial_transfer_learning"],
                outcomes=["Understand detection architectures", "Prepare detection datasets", "Fine-tune detectors"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="tutorial_semantic_segmentation",
                title="Semantic Segmentation",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Pixel-wise classification for scene understanding.",
                tags=["vision", "segmentation", "dense-prediction"],
                prerequisites=["tutorial_image_classification"],
                outcomes=["Understand segmentation architectures", "Train segmentation models"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html",
                runnable=False,
            ),
            # NLP tutorials
            Topic(
                id="tutorial_text_classification",
                title="Text Classification with Transformers",
                category=TopicCategory.TUTORIALS,
                level_min=Level.INTERMEDIATE,
                description="Classify text using modern transformer architectures.",
                tags=["nlp", "transformers", "classification", "text"],
                prerequisites=["basics_optimization"],
                outcomes=["Tokenize text data", "Use transformer models", "Train text classifiers"],
                official_link="https://docs.pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="tutorial_seq2seq",
                title="Sequence-to-Sequence with Attention",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Build translation models with attention mechanisms.",
                tags=["nlp", "seq2seq", "attention", "translation"],
                prerequisites=["tutorial_text_classification"],
                outcomes=["Implement attention", "Build encoder-decoder models", "Train translation systems"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="tutorial_transformer",
                title="Transformer from Scratch",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Implement the Transformer architecture from the ground up.",
                tags=["nlp", "transformers", "attention", "architecture"],
                prerequisites=["tutorial_seq2seq"],
                outcomes=["Understand multi-head attention", "Build transformer layers", "Train language models"],
                official_link="https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html",
                runnable=False,
            ),
            # Audio tutorials
            Topic(
                id="tutorial_audio_classification",
                title="Audio Classification with torchaudio",
                category=TopicCategory.TUTORIALS,
                level_min=Level.INTERMEDIATE,
                description="Classify audio samples using spectrograms and CNNs.",
                tags=["audio", "classification", "torchaudio"],
                prerequisites=["basics_datasets"],
                outcomes=["Load audio data", "Extract spectrograms", "Train audio classifiers"],
                official_link="https://docs.pytorch.org/tutorials/beginner/audio_classifier_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="tutorial_speech_recognition",
                title="Speech Recognition with Wav2Vec2",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Use pretrained Wav2Vec2 for automatic speech recognition.",
                tags=["audio", "speech", "asr", "pretrained"],
                prerequisites=["tutorial_audio_classification"],
                outcomes=["Use pretrained audio models", "Transcribe speech", "Fine-tune ASR models"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html",
                runnable=False,
            ),
            # Reinforcement Learning
            Topic(
                id="tutorial_rl_dqn",
                title="Deep Q-Learning (DQN)",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Implement Deep Q-Networks for reinforcement learning.",
                tags=["rl", "dqn", "reinforcement-learning"],
                prerequisites=["basics_optimization"],
                outcomes=["Understand Q-learning", "Implement experience replay", "Train DQN agents"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html",
                runnable=False,
            ),
            # Deployment
            Topic(
                id="tutorial_torchscript",
                title="TorchScript for Deployment",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Export models to TorchScript for production deployment.",
                tags=["deployment", "torchscript", "production"],
                prerequisites=["basics_save_load"],
                outcomes=["Trace and script models", "Optimize for inference", "Deploy to C++"],
                official_link="https://docs.pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="tutorial_onnx_export",
                title="Export to ONNX",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Export PyTorch models to ONNX format for cross-platform deployment.",
                tags=["deployment", "onnx", "export"],
                prerequisites=["basics_save_load"],
                outcomes=["Export models to ONNX", "Verify exports", "Use ONNX Runtime"],
                official_link="https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html",
                runnable=False,
            ),
            # Distributed Training
            Topic(
                id="tutorial_ddp",
                title="Distributed Data Parallel (DDP)",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Scale training across multiple GPUs using DDP.",
                tags=["distributed", "ddp", "multi-gpu", "scaling"],
                prerequisites=["basics_optimization"],
                outcomes=["Understand DDP concepts", "Set up multi-GPU training", "Handle synchronization"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html",
                runnable=False,
                explanation_advanced="DDP replicates the model across processes and synchronizes gradients during backward pass. Understanding process groups, gradient buckets, and communication backends is essential for efficient distributed training.",
            ),
            Topic(
                id="tutorial_fsdp",
                title="Fully Sharded Data Parallel (FSDP)",
                category=TopicCategory.TUTORIALS,
                level_min=Level.ADVANCED,
                description="Train very large models by sharding across GPUs.",
                tags=["distributed", "fsdp", "large-models", "memory"],
                prerequisites=["tutorial_ddp"],
                outcomes=["Understand FSDP sharding", "Configure FSDP policies", "Train large models"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html",
                runnable=False,
            ),
        ]

    def _get_recipes_topics(self) -> List[Topic]:
        """Get Recipes topics - focused how-to guides."""
        return [
            Topic(
                id="recipe_profiling",
                title="PyTorch Profiler",
                category=TopicCategory.RECIPES,
                level_min=Level.ADVANCED,
                description="Profile training to identify performance bottlenecks.",
                tags=["profiling", "performance", "debugging"],
                prerequisites=["basics_optimization"],
                outcomes=["Use torch.profiler", "Read trace files", "Identify bottlenecks"],
                official_link="https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html",
                runnable=True,
                demo_id="performance",
            ),
            Topic(
                id="recipe_mixed_precision",
                title="Automatic Mixed Precision (AMP)",
                category=TopicCategory.RECIPES,
                level_min=Level.INTERMEDIATE,
                description="Speed up training with mixed precision on CUDA.",
                tags=["amp", "mixed-precision", "performance", "cuda"],
                prerequisites=["basics_optimization"],
                outcomes=["Use autocast and GradScaler", "Understand FP16/BF16", "Handle loss scaling"],
                official_link="https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html",
                runnable=True,
                demo_id="performance",
            ),
            Topic(
                id="recipe_learning_rate_scheduler",
                title="Learning Rate Schedulers",
                category=TopicCategory.RECIPES,
                level_min=Level.INTERMEDIATE,
                description="Adjust learning rate during training for better convergence.",
                tags=["lr-scheduler", "training", "optimization"],
                prerequisites=["basics_optimization"],
                outcomes=["Use built-in schedulers", "Create custom schedules", "Combine with warmup"],
                official_link="https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                runnable=True,
                demo_id="train_eval",
            ),
            Topic(
                id="recipe_gradient_clipping",
                title="Gradient Clipping",
                category=TopicCategory.RECIPES,
                level_min=Level.INTERMEDIATE,
                description="Prevent exploding gradients during training.",
                tags=["gradients", "stability", "training"],
                prerequisites=["basics_autograd"],
                outcomes=["Clip gradient norms", "Debug gradient issues", "Stabilize training"],
                official_link="https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
                runnable=False,
            ),
            Topic(
                id="recipe_checkpoint",
                title="Saving and Loading Checkpoints",
                category=TopicCategory.RECIPES,
                level_min=Level.BEGINNER,
                description="Save training state for resumption and best model tracking.",
                tags=["checkpoint", "save", "training"],
                prerequisites=["basics_save_load"],
                outcomes=["Save full training state", "Resume training", "Track best models"],
                official_link="https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html",
                runnable=True,
                demo_id="save_load",
            ),
            Topic(
                id="recipe_reproducibility",
                title="Reproducibility",
                category=TopicCategory.RECIPES,
                level_min=Level.INTERMEDIATE,
                description="Ensure reproducible results across runs.",
                tags=["reproducibility", "seeding", "deterministic"],
                prerequisites=["basics_tensors"],
                outcomes=["Set random seeds", "Enable deterministic mode", "Understand sources of randomness"],
                official_link="https://docs.pytorch.org/docs/stable/notes/randomness.html",
                runnable=True,
                demo_id="tensor_lab",
            ),
            Topic(
                id="recipe_inference_mode",
                title="Inference Mode",
                category=TopicCategory.RECIPES,
                level_min=Level.INTERMEDIATE,
                description="Optimize inference with torch.inference_mode.",
                tags=["inference", "performance", "no_grad"],
                prerequisites=["basics_autograd"],
                outcomes=["Use inference_mode", "Understand vs no_grad", "Optimize inference"],
                official_link="https://docs.pytorch.org/docs/stable/generated/torch.inference_mode.html",
                runnable=False,
            ),
            Topic(
                id="recipe_torch_compile",
                title="torch.compile Optimization",
                category=TopicCategory.RECIPES,
                level_min=Level.ADVANCED,
                description="Accelerate models with torch.compile (PyTorch 2.0+).",
                tags=["compile", "performance", "pytorch2"],
                prerequisites=["basics_optimization"],
                outcomes=["Use torch.compile", "Choose backends", "Debug compilation"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
                runnable=True,
                demo_id="performance",
            ),
            Topic(
                id="recipe_quantization",
                title="Quantization",
                category=TopicCategory.RECIPES,
                level_min=Level.ADVANCED,
                description="Reduce model size and increase inference speed with quantization.",
                tags=["quantization", "inference", "optimization"],
                prerequisites=["basics_save_load"],
                outcomes=["Apply dynamic quantization", "Understand quantization types", "Benchmark quantized models"],
                official_link="https://docs.pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html",
                runnable=False,
            ),
            Topic(
                id="recipe_memory_format",
                title="Memory Formats (Channels Last)",
                category=TopicCategory.RECIPES,
                level_min=Level.ADVANCED,
                description="Optimize vision models with channels-last memory format.",
                tags=["memory", "performance", "vision"],
                prerequisites=["tutorial_image_classification"],
                outcomes=["Convert to channels_last", "Understand memory layouts", "Benchmark improvements"],
                official_link="https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="recipe_custom_dataset",
                title="Writing Custom Datasets",
                category=TopicCategory.RECIPES,
                level_min=Level.BEGINNER,
                description="Create custom Dataset classes for your data.",
                tags=["data", "datasets", "custom"],
                prerequisites=["basics_datasets"],
                outcomes=["Implement __getitem__", "Handle different data types", "Add transforms"],
                official_link="https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html",
                runnable=True,
                demo_id="data_lab",
            ),
            Topic(
                id="recipe_debugging_nan",
                title="Debugging NaN and Inf",
                category=TopicCategory.RECIPES,
                level_min=Level.INTERMEDIATE,
                description="Detect and fix numerical instabilities in training.",
                tags=["debugging", "nan", "numerical-stability"],
                prerequisites=["basics_autograd"],
                outcomes=["Detect NaN/Inf", "Enable anomaly detection", "Fix common causes"],
                official_link="https://docs.pytorch.org/docs/stable/autograd.html#anomaly-detection",
                runnable=False,
            ),
        ]

    def _get_youtube_topics(self) -> List[Topic]:
        """Get Intro to PyTorch YouTube Series topics."""
        return [
            Topic(
                id="youtube_intro",
                title="Introduction to PyTorch",
                category=TopicCategory.YOUTUBE,
                level_min=Level.BEGINNER,
                description="Video introduction to PyTorch fundamentals.",
                tags=["video", "introduction", "basics"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html",
                runnable=False,
            ),
            Topic(
                id="youtube_tensors",
                title="Tensors (Video)",
                category=TopicCategory.YOUTUBE,
                level_min=Level.BEGINNER,
                description="Video tutorial on PyTorch tensors.",
                tags=["video", "tensors", "basics"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="youtube_autograd",
                title="Autograd (Video)",
                category=TopicCategory.YOUTUBE,
                level_min=Level.BEGINNER,
                description="Video tutorial on automatic differentiation.",
                tags=["video", "autograd", "basics"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="youtube_models",
                title="Building Models (Video)",
                category=TopicCategory.YOUTUBE,
                level_min=Level.BEGINNER,
                description="Video tutorial on building neural networks.",
                tags=["video", "models", "nn.Module"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="youtube_tensorboard",
                title="TensorBoard Support (Video)",
                category=TopicCategory.YOUTUBE,
                level_min=Level.INTERMEDIATE,
                description="Video tutorial on using TensorBoard with PyTorch.",
                tags=["video", "tensorboard", "visualization"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html",
                runnable=False,
            ),
            Topic(
                id="youtube_training",
                title="Training with PyTorch (Video)",
                category=TopicCategory.YOUTUBE,
                level_min=Level.BEGINNER,
                description="Video tutorial on the complete training loop.",
                tags=["video", "training", "optimization"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html",
                runnable=False,
            ),
            Topic(
                id="youtube_captum",
                title="Model Understanding with Captum (Video)",
                category=TopicCategory.YOUTUBE,
                level_min=Level.ADVANCED,
                description="Video tutorial on model interpretability.",
                tags=["video", "captum", "interpretability"],
                official_link="https://docs.pytorch.org/tutorials/beginner/introyt/captumyt.html",
                runnable=False,
            ),
        ]

    def _get_webinar_topics(self) -> List[Topic]:
        """Get Webinar topics - curated list of PyTorch webinars."""
        return [
            Topic(
                id="webinar_pytorch2",
                title="PyTorch 2.0 Overview",
                category=TopicCategory.WEBINARS,
                level_min=Level.INTERMEDIATE,
                description="Overview of PyTorch 2.0 features including torch.compile.",
                tags=["webinar", "pytorch2", "compile"],
                official_link="https://pytorch.org/blog/pytorch-2.0-release/",
                runnable=False,
            ),
            Topic(
                id="webinar_production",
                title="PyTorch in Production",
                category=TopicCategory.WEBINARS,
                level_min=Level.ADVANCED,
                description="Best practices for deploying PyTorch models.",
                tags=["webinar", "production", "deployment"],
                official_link="https://pytorch.org/blog/",
                runnable=False,
            ),
            Topic(
                id="webinar_distributed",
                title="Distributed Training Best Practices",
                category=TopicCategory.WEBINARS,
                level_min=Level.ADVANCED,
                description="Scaling PyTorch training across multiple nodes.",
                tags=["webinar", "distributed", "scaling"],
                official_link="https://pytorch.org/tutorials/intermediate/dist_overview.html",
                runnable=False,
            ),
            Topic(
                id="webinar_performance",
                title="Performance Tuning Guide",
                category=TopicCategory.WEBINARS,
                level_min=Level.ADVANCED,
                description="Optimizing PyTorch for maximum performance.",
                tags=["webinar", "performance", "optimization"],
                official_link="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
                runnable=False,
            ),
        ]

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Get a topic by ID."""
        return self.topics.get(topic_id)

    def get_all_topics(self) -> List[Topic]:
        """Get all topics."""
        return list(self.topics.values())

    def get_topics_by_category(self, category: TopicCategory) -> List[Topic]:
        """Get topics filtered by category."""
        return [t for t in self.topics.values() if t.category == category]

    def get_topics_for_level(self, level: Level) -> List[Topic]:
        """Get topics available for a given level."""
        return [t for t in self.topics.values() if t.is_available_for_level(level)]

    def get_runnable_topics(self, level: Optional[Level] = None) -> List[Topic]:
        """Get topics that have runnable demos."""
        topics = [t for t in self.topics.values() if t.runnable]
        if level:
            topics = [t for t in topics if t.is_available_for_level(level)]
        return topics

    def search(self, query: str, level: Optional[Level] = None) -> List[Topic]:
        """Search topics by query string."""
        query = query.lower()
        results = []

        for topic in self.topics.values():
            # Search in title, description, and tags
            if (
                query in topic.title.lower()
                or query in topic.description.lower()
                or any(query in tag.lower() for tag in topic.tags)
            ):
                if level is None or topic.is_available_for_level(level):
                    results.append(topic)

        return results


# Singleton catalog instance
_catalog: Optional[TopicCatalog] = None


def get_catalog() -> TopicCatalog:
    """Get the singleton catalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = TopicCatalog()
    return _catalog


def filter_by_level(topics: List[Topic], level: Level) -> List[Topic]:
    """Filter topics by level availability."""
    return [t for t in topics if t.is_available_for_level(level)]


def filter_by_category(topics: List[Topic], category: TopicCategory) -> List[Topic]:
    """Filter topics by category."""
    return [t for t in topics if t.category == category]


def search_topics(query: str, level: Optional[Level] = None) -> List[Topic]:
    """Search the catalog."""
    return get_catalog().search(query, level)
