"""
Data Lab Demo - Interactive exploration of Datasets and DataLoaders.

Covers:
- Built-in datasets (FashionMNIST)
- Synthetic datasets (Power, Retail)
- CSV upload and parsing
- DataLoader configuration
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import pandas as pd

from pytorch_playground.state import Level
from pytorch_playground.devices import get_device
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, CodeBuilder
from pytorch_playground.utils.guards import check_dataloader_workers, cap_dataset_size
import matplotlib.pyplot as plt


class DataLabDemo:
    """Interactive data loading demo."""

    def __init__(self):
        self.name = "Data Lab"
        self.description = "Explore PyTorch Datasets and DataLoaders"

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "max_samples": 1000,
                "show_num_workers": False,
                "show_pin_memory": False,
                "show_prefetch": False,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "max_samples": 10000,
                "show_num_workers": True,
                "show_pin_memory": True,
                "show_prefetch": False,
            }
        else:
            return {
                "max_samples": 60000,
                "show_num_workers": True,
                "show_pin_memory": True,
                "show_prefetch": True,
            }

    def generate_synthetic_power_data(
        self, n_samples: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic power industry sensor data."""
        np.random.seed(seed)

        # Time features
        t = np.linspace(0, 24, n_samples)  # 24 hours

        # Sensor readings with daily patterns
        temperature = 20 + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_samples) * 2
        pressure = 100 + 5 * np.cos(2 * np.pi * t / 24) + np.random.randn(n_samples)
        vibration = 0.5 + 0.2 * np.abs(np.sin(4 * np.pi * t / 24)) + np.random.randn(n_samples) * 0.1
        power_output = 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_samples) * 5

        X = np.column_stack([temperature, pressure, vibration, power_output])

        # Binary classification: anomaly detection
        # Anomaly if any sensor is outside 2 std from mean
        anomaly = (
            (np.abs(temperature - 20) > 15) |
            (np.abs(pressure - 100) > 8) |
            (vibration > 0.9)
        ).astype(np.float32)

        return X.astype(np.float32), anomaly

    def generate_synthetic_retail_data(
        self, n_samples: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic retail sales data."""
        np.random.seed(seed)

        # Features
        day_of_week = np.random.randint(0, 7, n_samples)
        month = np.random.randint(1, 13, n_samples)
        is_holiday = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        is_promo = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        price = 10 + np.random.randn(n_samples) * 3
        competitor_price = price + np.random.randn(n_samples) * 2

        X = np.column_stack([
            day_of_week, month, is_holiday, is_promo, price, competitor_price
        ])

        # Sales uplift (binary: above/below average)
        base_sales = 100 + 20 * is_promo + 30 * is_holiday - 5 * (price - competitor_price)
        base_sales += np.random.randn(n_samples) * 20
        y = (base_sales > base_sales.mean()).astype(np.float32)

        return X.astype(np.float32), y

    def run(
        self,
        dataset_source: str,
        n_samples: int,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        device_selection: str,
        seed: int,
        csv_file: Optional[Any] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Run the data loading demo."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        logger = DemoLogger(self.name)
        logger.start()

        try:
            set_seed(seed)
            logger.info(f"Seed set to {seed}")

            # Get device
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)
            logger.info(f"Device: {device_label}")

            # Check num_workers for platform
            num_workers, worker_warning = check_dataloader_workers(num_workers)
            if worker_warning:
                logger.warning(worker_warning)

            # Cap dataset size based on level
            level = Level.INTERMEDIATE  # Default
            n_samples, size_warning = cap_dataset_size(n_samples, level)
            if size_warning:
                logger.warning(size_warning)

            # Load or generate data
            X, y = None, None
            feature_names = None

            if dataset_source == "FashionMNIST (subset)":
                logger.info("Loading FashionMNIST dataset...")
                try:
                    from torchvision import datasets, transforms

                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])

                    train_data = datasets.FashionMNIST(
                        root='./data',
                        train=True,
                        download=True,
                        transform=transform
                    )

                    # Subset
                    n_samples = min(n_samples, len(train_data))
                    indices = list(range(n_samples))
                    train_data = torch.utils.data.Subset(train_data, indices)

                    dataset = train_data
                    feature_names = ["image"]
                    class_names = [
                        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                    ]
                    logger.info(f"Loaded {n_samples} FashionMNIST samples")

                except Exception as e:
                    logger.warning(f"Could not load FashionMNIST: {e}")
                    logger.info("Falling back to synthetic data")
                    X, y = self.generate_synthetic_power_data(n_samples, seed)
                    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                    feature_names = ["temperature", "pressure", "vibration", "power"]

            elif dataset_source == "Synthetic: Power Industry":
                logger.info("Generating synthetic power industry data...")
                X, y = self.generate_synthetic_power_data(n_samples, seed)
                dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                feature_names = ["temperature", "pressure", "vibration", "power"]
                logger.info(f"Generated {n_samples} samples with {X.shape[1]} features")

            elif dataset_source == "Synthetic: Retail":
                logger.info("Generating synthetic retail data...")
                X, y = self.generate_synthetic_retail_data(n_samples, seed)
                dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                feature_names = ["day_of_week", "month", "is_holiday", "is_promo", "price", "competitor_price"]
                logger.info(f"Generated {n_samples} samples with {X.shape[1]} features")

            elif dataset_source == "CSV Upload" and csv_file is not None:
                logger.info("Loading CSV file...")
                try:
                    df = pd.read_csv(csv_file.name)
                    logger.info(f"Loaded CSV with shape {df.shape}")

                    if target_column and target_column in df.columns:
                        y = df[target_column].values.astype(np.float32)
                        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).values.astype(np.float32)
                        feature_names = list(df.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns)
                    else:
                        # Use last column as target
                        y = df.iloc[:, -1].values.astype(np.float32)
                        X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values.astype(np.float32)
                        feature_names = list(df.iloc[:, :-1].select_dtypes(include=[np.number]).columns)

                    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                    n_samples = len(dataset)

                except Exception as e:
                    logger.error(f"Error loading CSV: {e}")
                    X, y = self.generate_synthetic_power_data(100, seed)
                    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                    feature_names = ["f1", "f2", "f3", "f4"]
            else:
                # Default
                X, y = self.generate_synthetic_power_data(n_samples, seed)
                dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                feature_names = ["temperature", "pressure", "vibration", "power"]

            # Create DataLoader
            loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
            }

            if pin_memory and torch.cuda.is_available():
                loader_kwargs["pin_memory"] = True
                logger.info("pin_memory enabled (CUDA available)")

            loader = DataLoader(dataset, **loader_kwargs)

            logger.info(f"Created DataLoader with batch_size={batch_size}, shuffle={shuffle}")
            logger.info(f"Number of batches: {len(loader)}")

            # Iterate one batch as demo
            logger.info("")
            logger.info("--- Sample Batch ---")

            for batch_idx, batch in enumerate(loader):
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        batch_x, batch_y = batch
                        if hasattr(batch_x, 'shape'):
                            logger.info(f"Batch X shape: {tuple(batch_x.shape)}")
                            logger.info(f"Batch Y shape: {tuple(batch_y.shape)}")
                else:
                    logger.info(f"Batch shape: {tuple(batch.shape)}")
                break

            # Metrics
            logger.metric("dataset_size", n_samples)
            logger.metric("batch_size", batch_size)
            logger.metric("num_batches", len(loader))
            logger.metric("num_workers", num_workers)
            if feature_names:
                logger.metric("features", feature_names)

            # Create visualization
            fig = self._create_data_plot(dataset, dataset_source, feature_names)

            # Build code
            code = self._build_code(
                dataset_source, n_samples, batch_size, shuffle, num_workers, pin_memory, seed
            )

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _create_data_plot(self, dataset, source: str, feature_names: Optional[List[str]]) -> Any:
        """Create data visualization."""
        import torch

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        try:
            # Get first batch
            if hasattr(dataset, '__getitem__'):
                sample = dataset[0]
                if isinstance(sample, (list, tuple)):
                    data, label = sample
                else:
                    data = sample
                    label = None

                # Plot 1: Data sample
                if "FashionMNIST" in source:
                    # Image data
                    if hasattr(data, 'squeeze'):
                        img = data.squeeze().numpy()
                    else:
                        img = np.array(data).squeeze()
                    axes[0].imshow(img, cmap='gray')
                    axes[0].set_title(f"Sample Image (label: {label})")
                    axes[0].axis('off')
                else:
                    # Tabular data - show feature distribution
                    if hasattr(data, 'numpy'):
                        data = data.numpy()
                    if len(data.shape) == 1:
                        bars = axes[0].bar(range(len(data)), data)
                        if feature_names:
                            axes[0].set_xticks(range(len(feature_names)))
                            axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
                        axes[0].set_title("Sample Features")
                        axes[0].set_ylabel("Value")

                # Plot 2: Class distribution
                labels = []
                for i in range(min(1000, len(dataset))):
                    sample = dataset[i]
                    if isinstance(sample, (list, tuple)) and len(sample) == 2:
                        if hasattr(sample[1], 'item'):
                            labels.append(sample[1].item())
                        else:
                            labels.append(sample[1])

                if labels:
                    unique, counts = np.unique(labels, return_counts=True)
                    axes[1].bar(unique, counts, color='steelblue')
                    axes[1].set_xlabel("Class")
                    axes[1].set_ylabel("Count")
                    axes[1].set_title("Label Distribution")

        except Exception as e:
            axes[0].text(0.5, 0.5, f"Visualization error:\n{str(e)}", ha='center', va='center')
            axes[0].axis('off')

        plt.tight_layout()
        return fig

    def _build_code(
        self,
        source: str,
        n_samples: int,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        seed: int,
    ) -> str:
        """Build the code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add("from torch.utils.data import DataLoader, TensorDataset")

        if "FashionMNIST" in source:
            builder.add("from torchvision import datasets, transforms")

        builder.add("import numpy as np")
        builder.add_blank()

        builder.add(f"torch.manual_seed({seed})")
        builder.add(f"np.random.seed({seed})")
        builder.add_blank()

        if "FashionMNIST" in source:
            builder.add_comment("Load FashionMNIST dataset")
            builder.add("transform = transforms.Compose([")
            builder.indent()
            builder.add("transforms.ToTensor(),")
            builder.add("transforms.Normalize((0.5,), (0.5,))")
            builder.dedent()
            builder.add("])")
            builder.add_blank()
            builder.add("dataset = datasets.FashionMNIST(")
            builder.indent()
            builder.add("root='./data',")
            builder.add("train=True,")
            builder.add("download=True,")
            builder.add("transform=transform")
            builder.dedent()
            builder.add(")")
            builder.add_blank()
            builder.add_comment(f"Use subset of {n_samples} samples")
            builder.add(f"dataset = torch.utils.data.Subset(dataset, range({n_samples}))")
        else:
            builder.add_comment("Create synthetic dataset")
            builder.add(f"n_samples = {n_samples}")
            builder.add("X = np.random.randn(n_samples, 4).astype(np.float32)")
            builder.add("y = np.random.randint(0, 2, n_samples).astype(np.float32)")
            builder.add("dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))")

        builder.add_blank()
        builder.add_comment("Create DataLoader")
        builder.add("loader = DataLoader(")
        builder.indent()
        builder.add("dataset,")
        builder.add(f"batch_size={batch_size},")
        builder.add(f"shuffle={shuffle},")
        builder.add(f"num_workers={num_workers},")
        if pin_memory:
            builder.add("pin_memory=True,")
        builder.dedent()
        builder.add(")")
        builder.add_blank()

        builder.add_comment("Iterate through batches")
        builder.add("for batch_idx, (data, labels) in enumerate(loader):")
        builder.indent()
        builder.add("print(f'Batch {batch_idx}: data shape {data.shape}, labels shape {labels.shape}')")
        builder.add("if batch_idx >= 2:")
        builder.indent()
        builder.add("break")
        builder.dedent()
        builder.dedent()

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Data Source")

                    source_dropdown = gr.Dropdown(
                        choices=[
                            "FashionMNIST (subset)",
                            "Synthetic: Power Industry",
                            "Synthetic: Retail",
                            "CSV Upload",
                        ],
                        value="Synthetic: Power Industry",
                        label="Dataset Source",
                    )

                    n_samples_slider = gr.Slider(
                        minimum=100,
                        maximum=params["max_samples"],
                        value=min(1000, params["max_samples"]),
                        step=100,
                        label="Number of Samples",
                    )

                    csv_upload = gr.File(
                        label="CSV File (for CSV Upload)",
                        file_types=[".csv"],
                        visible=True,
                    )

                    target_column = gr.Textbox(
                        label="Target Column Name (optional)",
                        placeholder="Leave empty to use last column",
                    )

                    gr.Markdown("### DataLoader Settings")

                    batch_size_slider = gr.Slider(
                        minimum=1,
                        maximum=256,
                        value=32,
                        step=1,
                        label="Batch Size",
                    )

                    shuffle_check = gr.Checkbox(
                        label="Shuffle",
                        value=True,
                    )

                    if params["show_num_workers"]:
                        num_workers_slider = gr.Slider(
                            minimum=0,
                            maximum=8,
                            value=0,
                            step=1,
                            label="num_workers",
                            info="Parallel data loading workers",
                        )
                    else:
                        num_workers_slider = gr.Slider(value=0, visible=False)

                    if params["show_pin_memory"]:
                        pin_memory_check = gr.Checkbox(
                            label="pin_memory",
                            value=False,
                            info="Faster GPU transfer (CUDA only)",
                        )
                    else:
                        pin_memory_check = gr.Checkbox(value=False, visible=False)

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cpu", "cuda", "mps"],
                        value="auto",
                        label="Device",
                    )

                    seed_input = gr.Number(label="Seed", value=42, precision=0)

                    run_btn = gr.Button("Load Data", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Output"):
                            logs_output = gr.Textbox(label="Logs", lines=20, interactive=False)
                            metrics_output = gr.JSON(label="Metrics")

                        with gr.Tab("Visualization"):
                            plot_output = gr.Plot(label="Data Visualization")

                        with gr.Tab("Code"):
                            code_output = gr.Code(language="python", lines=35)

            run_btn.click(
                fn=self.run,
                inputs=[
                    source_dropdown,
                    n_samples_slider,
                    batch_size_slider,
                    shuffle_check,
                    num_workers_slider,
                    pin_memory_check,
                    device_dropdown,
                    seed_input,
                    csv_upload,
                    target_column,
                ],
                outputs=[logs_output, metrics_output, plot_output, code_output],
            )

        return demo


def create_data_lab_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Data Lab demo UI."""
    demo = DataLabDemo()
    return demo.create_ui(level)
