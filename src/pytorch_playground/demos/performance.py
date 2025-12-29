"""
Performance Demo - Timing, profiling, and optimization.

Covers:
- Mixed precision training (AMP)
- torch.compile optimization
- Basic profiling
- Performance comparison
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict, List
import time

from pytorch_playground.state import Level
from pytorch_playground.devices import get_device, get_available_devices
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, CodeBuilder
from pytorch_playground.utils.plotting import create_timing_comparison_plot
import matplotlib.pyplot as plt
import numpy as np


class PerformanceDemo:
    """Interactive performance testing demo."""

    def __init__(self):
        self.name = "Performance Lab"
        self.description = "Explore PyTorch performance optimization techniques"

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "show_amp": False,
                "show_compile": False,
                "show_profiler": False,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "show_amp": True,
                "show_compile": False,
                "show_profiler": False,
            }
        else:
            return {
                "show_amp": True,
                "show_compile": True,
                "show_profiler": True,
            }

    def run_timing_benchmark(
        self,
        model_size: str,
        batch_size: int,
        warmup_runs: int,
        benchmark_runs: int,
        test_amp: bool,
        test_compile: bool,
        device_selection: str,
        seed: int,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Run timing benchmarks."""
        import torch
        import torch.nn as nn

        logger = DemoLogger("Timing Benchmark")
        logger.start()

        results = {}

        try:
            set_seed(seed)

            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)
            logger.info(f"Device: {device_label}")

            # Parse model size
            size_map = {
                "Small (64-32)": [64, 32],
                "Medium (256-128-64)": [256, 128, 64],
                "Large (512-256-128)": [512, 256, 128],
                "XL (1024-512-256-128)": [1024, 512, 256, 128],
            }
            hidden_sizes = size_map.get(model_size, [64, 32])

            # Build model
            layers = []
            layer_sizes = [1000] + hidden_sizes + [100]
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU())

            model = nn.Sequential(*layers).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model: {layer_sizes}")
            logger.info(f"Parameters: {total_params:,}")
            logger.info(f"Batch size: {batch_size}")
            logger.info("")

            # Create test data
            x = torch.randn(batch_size, 1000, device=device)
            target = torch.randint(0, 100, (batch_size,), device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())

            def train_step():
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                return loss.item()

            # Benchmark baseline
            logger.info("--- Baseline (FP32) ---")
            results["Baseline"] = self._benchmark(
                train_step, warmup_runs, benchmark_runs, device, logger
            )

            # Test AMP (CUDA only)
            if test_amp and device.type == "cuda":
                logger.info("")
                logger.info("--- Mixed Precision (AMP) ---")

                scaler = torch.amp.GradScaler('cuda')

                def train_step_amp():
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        out = model(x)
                        loss = criterion(out, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    return loss.item()

                results["AMP (FP16)"] = self._benchmark(
                    train_step_amp, warmup_runs, benchmark_runs, device, logger
                )
            elif test_amp:
                logger.warning("AMP requires CUDA, skipping")

            # Test torch.compile (PyTorch 2.0+)
            if test_compile:
                logger.info("")
                logger.info("--- torch.compile ---")

                try:
                    # Check if torch.compile is available
                    if hasattr(torch, "compile"):
                        compiled_model = torch.compile(model, mode="reduce-overhead")

                        def train_step_compiled():
                            optimizer.zero_grad()
                            out = compiled_model(x)
                            loss = criterion(out, target)
                            loss.backward()
                            optimizer.step()
                            return loss.item()

                        # Extra warmup for compilation
                        logger.info("Compiling (this may take a moment)...")
                        for _ in range(3):
                            train_step_compiled()

                        results["torch.compile"] = self._benchmark(
                            train_step_compiled, warmup_runs, benchmark_runs, device, logger
                        )
                    else:
                        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

            # Summary
            logger.info("")
            logger.info("--- Summary ---")
            baseline_time = results.get("Baseline", 1.0)

            for name, time_ms in results.items():
                speedup = baseline_time / time_ms if time_ms > 0 else 0
                logger.info(f"{name}: {time_ms:.3f}ms (speedup: {speedup:.2f}x)")

            # Metrics
            logger.metric("model_params", total_params)
            logger.metric("batch_size", batch_size)
            for name, time_ms in results.items():
                logger.metric(f"time_{name.lower().replace(' ', '_')}", time_ms)

            # Create plot
            fig = create_timing_comparison_plot(results, "Training Step Time")

            # Build code
            code = self._build_code(test_amp, test_compile)

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _benchmark(
        self,
        fn,
        warmup: int,
        runs: int,
        device,
        logger: DemoLogger,
    ) -> float:
        """Run a benchmark and return average time in ms."""
        import torch

        # Warmup
        for _ in range(warmup):
            fn()

        # Synchronize before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        std_time = np.std(times)

        logger.info(f"Average: {avg_time:.3f}ms (+/- {std_time:.3f}ms)")

        return avg_time

    def run_memory_analysis(
        self,
        model_size: str,
        batch_size: int,
        device_selection: str,
        seed: int,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Analyze memory usage."""
        import torch
        import torch.nn as nn

        logger = DemoLogger("Memory Analysis")
        logger.start()

        try:
            set_seed(seed)

            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)

            if device.type != "cuda":
                logger.warning("Detailed memory analysis is only available for CUDA devices")
                logger.info("Running basic analysis...")

            # Parse model size
            size_map = {
                "Small (64-32)": [64, 32],
                "Medium (256-128-64)": [256, 128, 64],
                "Large (512-256-128)": [512, 256, 128],
                "XL (1024-512-256-128)": [1024, 512, 256, 128],
            }
            hidden_sizes = size_map.get(model_size, [64, 32])

            # Build model
            layers = []
            layer_sizes = [1000] + hidden_sizes + [100]
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU())

            model = nn.Sequential(*layers)

            # Calculate model memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            logger.info(f"Model parameter memory: {param_memory / 1024:.2f} KB")

            # Move to device and measure
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                initial_mem = torch.cuda.memory_allocated()

            model = model.to(device)

            if device.type == "cuda":
                after_model = torch.cuda.memory_allocated()
                logger.info(f"Memory after model: {(after_model - initial_mem) / 1024:.2f} KB")

            # Create batch
            x = torch.randn(batch_size, 1000, device=device)
            target = torch.randint(0, 100, (batch_size,), device=device)

            if device.type == "cuda":
                after_data = torch.cuda.memory_allocated()
                logger.info(f"Memory after data: {(after_data - initial_mem) / 1024:.2f} KB")

            # Forward pass
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)

            if device.type == "cuda":
                after_forward = torch.cuda.memory_allocated()
                logger.info(f"Memory after forward: {(after_forward - initial_mem) / 1024:.2f} KB")

            # Backward pass
            loss.backward()

            if device.type == "cuda":
                after_backward = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                logger.info(f"Memory after backward: {(after_backward - initial_mem) / 1024:.2f} KB")
                logger.info(f"Peak memory: {peak_memory / 1024 / 1024:.2f} MB")

            # Metrics
            logger.metric("param_memory_kb", param_memory / 1024)
            logger.metric("batch_size", batch_size)
            if device.type == "cuda":
                logger.metric("peak_memory_mb", peak_memory / 1024 / 1024)

            # Create visualization
            fig = self._create_memory_plot(device, model, batch_size)

            # Build code
            code = self._build_memory_code()

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _create_memory_plot(self, device, model, batch_size) -> Any:
        """Create memory usage visualization."""
        import torch

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Parameter distribution by layer
        layer_params = []
        layer_names = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                params = sum(p.numel() for p in module.parameters())
                layer_params.append(params)
                layer_names.append(name if name else 'root')

        if layer_params:
            ax1.barh(range(len(layer_params)), layer_params, color='steelblue')
            ax1.set_yticks(range(len(layer_names)))
            ax1.set_yticklabels(layer_names)
            ax1.set_xlabel('Parameters')
            ax1.set_title('Parameters by Layer')

        # Right: Memory breakdown (if CUDA)
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            cached = torch.cuda.memory_reserved() / 1024 / 1024

            labels = ['Allocated', 'Cached (Reserved)']
            sizes = [allocated, cached - allocated]
            colors = ['#ff9999', '#66b3ff']

            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax2.set_title('GPU Memory Usage (MB)')
        else:
            ax2.text(0.5, 0.5, 'Detailed memory analysis\nrequires CUDA',
                    ha='center', va='center', fontsize=12)
            ax2.axis('off')

        plt.tight_layout()
        return fig

    def _build_code(self, test_amp: bool, test_compile: bool) -> str:
        """Build benchmark code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add("import torch.nn as nn")
        builder.add("import time")
        builder.add_blank()

        builder.add_comment("Create model and data")
        builder.add("model = nn.Sequential(...).to(device)")
        builder.add("x = torch.randn(batch_size, input_size, device=device)")
        builder.add("criterion = nn.CrossEntropyLoss()")
        builder.add("optimizer = torch.optim.Adam(model.parameters())")
        builder.add_blank()

        builder.add_comment("Benchmark function")
        builder.add("def benchmark(fn, warmup=5, runs=20):")
        builder.indent()
        builder.add("for _ in range(warmup):")
        builder.indent()
        builder.add("fn()")
        builder.dedent()
        builder.add_blank()
        builder.add("torch.cuda.synchronize()  # For CUDA")
        builder.add("times = []")
        builder.add("for _ in range(runs):")
        builder.indent()
        builder.add("start = time.perf_counter()")
        builder.add("fn()")
        builder.add("torch.cuda.synchronize()")
        builder.add("times.append(time.perf_counter() - start)")
        builder.dedent()
        builder.add("return sum(times) / len(times) * 1000  # ms")
        builder.dedent()
        builder.add_blank()

        builder.add_comment("Baseline")
        builder.add("def train_step():")
        builder.indent()
        builder.add("optimizer.zero_grad()")
        builder.add("out = model(x)")
        builder.add("loss = criterion(out, target)")
        builder.add("loss.backward()")
        builder.add("optimizer.step()")
        builder.dedent()
        builder.add_blank()
        builder.add("baseline_time = benchmark(train_step)")
        builder.add("print(f'Baseline: {baseline_time:.3f}ms')")

        if test_amp:
            builder.add_blank()
            builder.add_comment("Mixed Precision")
            builder.add("scaler = torch.amp.GradScaler('cuda')")
            builder.add_blank()
            builder.add("def train_step_amp():")
            builder.indent()
            builder.add("optimizer.zero_grad()")
            builder.add("with torch.amp.autocast('cuda'):")
            builder.indent()
            builder.add("out = model(x)")
            builder.add("loss = criterion(out, target)")
            builder.dedent()
            builder.add("scaler.scale(loss).backward()")
            builder.add("scaler.step(optimizer)")
            builder.add("scaler.update()")
            builder.dedent()
            builder.add_blank()
            builder.add("amp_time = benchmark(train_step_amp)")
            builder.add("print(f'AMP: {amp_time:.3f}ms')")

        if test_compile:
            builder.add_blank()
            builder.add_comment("torch.compile (PyTorch 2.0+)")
            builder.add("compiled_model = torch.compile(model)")
            builder.add_blank()
            builder.add("def train_step_compiled():")
            builder.indent()
            builder.add("optimizer.zero_grad()")
            builder.add("out = compiled_model(x)")
            builder.add("loss = criterion(out, target)")
            builder.add("loss.backward()")
            builder.add("optimizer.step()")
            builder.dedent()
            builder.add_blank()
            builder.add("compile_time = benchmark(train_step_compiled)")
            builder.add("print(f'Compiled: {compile_time:.3f}ms')")

        return builder.build()

    def _build_memory_code(self) -> str:
        """Build memory analysis code."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add_blank()

        builder.add_comment("Memory monitoring utilities")
        builder.add("def print_memory():")
        builder.indent()
        builder.add("if torch.cuda.is_available():")
        builder.indent()
        builder.add("allocated = torch.cuda.memory_allocated() / 1024**2")
        builder.add("cached = torch.cuda.memory_reserved() / 1024**2")
        builder.add("print(f'Allocated: {allocated:.2f} MB')")
        builder.add("print(f'Cached: {cached:.2f} MB')")
        builder.dedent()
        builder.dedent()
        builder.add_blank()

        builder.add_comment("Reset memory stats")
        builder.add("torch.cuda.reset_peak_memory_stats()")
        builder.add("torch.cuda.empty_cache()")
        builder.add_blank()

        builder.add_comment("Track memory at each step")
        builder.add("print('Initial:'); print_memory()")
        builder.add("model = model.to(device)")
        builder.add("print('After model:'); print_memory()")
        builder.add_blank()

        builder.add_comment("Check peak memory after training step")
        builder.add("# ... training code ...")
        builder.add("peak = torch.cuda.max_memory_allocated() / 1024**2")
        builder.add("print(f'Peak memory: {peak:.2f} MB')")

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            with gr.Tabs():
                with gr.Tab("Timing Benchmark"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Configuration")

                            model_size = gr.Dropdown(
                                choices=["Small (64-32)", "Medium (256-128-64)",
                                        "Large (512-256-128)", "XL (1024-512-256-128)"],
                                value="Medium (256-128-64)",
                                label="Model Size",
                            )

                            batch_size = gr.Slider(1, 256, value=32, step=1, label="Batch Size")
                            warmup_runs = gr.Slider(1, 20, value=5, step=1, label="Warmup Runs")
                            benchmark_runs = gr.Slider(5, 100, value=20, step=5, label="Benchmark Runs")

                            gr.Markdown("### Optimizations to Test")

                            if params.get("show_amp"):
                                test_amp = gr.Checkbox(label="Test Mixed Precision (CUDA only)", value=True)
                            else:
                                test_amp = gr.Checkbox(value=False, visible=False)

                            if params.get("show_compile"):
                                test_compile = gr.Checkbox(label="Test torch.compile (PyTorch 2.0+)", value=False)
                            else:
                                test_compile = gr.Checkbox(value=False, visible=False)

                            device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                            seed = gr.Number(value=42, precision=0, label="Seed")

                            timing_btn = gr.Button("Run Benchmark", variant="primary")

                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.Tab("Output"):
                                    timing_logs = gr.Textbox(label="Logs", lines=25, interactive=False)
                                    timing_metrics = gr.JSON(label="Metrics")

                                with gr.Tab("Visualization"):
                                    timing_plot = gr.Plot(label="Timing Comparison")

                                with gr.Tab("Code"):
                                    timing_code = gr.Code(language="python", lines=40)

                    timing_btn.click(
                        fn=self.run_timing_benchmark,
                        inputs=[model_size, batch_size, warmup_runs, benchmark_runs,
                               test_amp, test_compile, device, seed],
                        outputs=[timing_logs, timing_metrics, timing_plot, timing_code],
                    )

                with gr.Tab("Memory Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Configuration")

                            mem_model_size = gr.Dropdown(
                                choices=["Small (64-32)", "Medium (256-128-64)",
                                        "Large (512-256-128)", "XL (1024-512-256-128)"],
                                value="Medium (256-128-64)",
                                label="Model Size",
                            )

                            mem_batch_size = gr.Slider(1, 256, value=32, step=1, label="Batch Size")
                            mem_device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                            mem_seed = gr.Number(value=42, precision=0, label="Seed")

                            mem_btn = gr.Button("Analyze Memory", variant="primary")

                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.Tab("Output"):
                                    mem_logs = gr.Textbox(label="Logs", lines=20, interactive=False)
                                    mem_metrics = gr.JSON(label="Metrics")

                                with gr.Tab("Visualization"):
                                    mem_plot = gr.Plot(label="Memory Breakdown")

                                with gr.Tab("Code"):
                                    mem_code = gr.Code(language="python", lines=30)

                    mem_btn.click(
                        fn=self.run_memory_analysis,
                        inputs=[mem_model_size, mem_batch_size, mem_device, mem_seed],
                        outputs=[mem_logs, mem_metrics, mem_plot, mem_code],
                    )

        return demo


def create_performance_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Performance Lab demo UI."""
    demo = PerformanceDemo()
    return demo.create_ui(level)
