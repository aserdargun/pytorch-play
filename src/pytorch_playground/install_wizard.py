"""
Install Wizard UI for PyTorch Playground.

Replicates the PyTorch "Start Locally" selector experience using Gradio components.
"""

import gradio as gr
from typing import Tuple, Optional

from pytorch_playground.install_matrix import (
    BUILDS,
    OPERATING_SYSTEMS,
    get_compute_options,
    generate_pip_command,
    generate_verification_code,
    validate_combination,
    get_conda_guidance,
    get_source_guidance,
    get_libtorch_guidance,
    InstallConfig,
)


def create_install_wizard_ui() -> gr.Blocks:
    """Create the Install Wizard UI component."""

    def get_compute_choices(os_name: str) -> list:
        """Get compute choices for dropdown based on OS."""
        options = get_compute_options(os_name)
        return [(opt["name"], opt["id"]) for opt in options if opt.get("available", True)]

    def update_compute_dropdown(os_name: str):
        """Update compute dropdown when OS changes."""
        choices = get_compute_choices(os_name)
        # Default to CPU
        default = "cpu"
        return gr.Dropdown(choices=choices, value=default)

    def generate_command(
        build: str,
        os_name: str,
        compute: str,
        include_vision: bool,
        include_audio: bool,
    ) -> Tuple[str, str, str, str]:
        """Generate install command and related outputs."""

        config = InstallConfig(
            build=build,
            os=os_name,
            package_manager="pip",
            compute=compute,
            include_vision=include_vision,
            include_audio=include_audio,
        )

        # Validate combination
        is_valid, warnings = validate_combination(config)

        # Generate command
        result = generate_pip_command(config)

        # Build command output
        command_output = result.command

        # Build notes output
        notes_parts = []

        if not result.verified:
            notes_parts.append("**Verification needed:** This combination may not be fully supported.")
            notes_parts.append("Please confirm on the official PyTorch selector.\n")

        if warnings:
            notes_parts.append("**Warnings:**")
            for warning in warnings:
                notes_parts.append(f"- {warning}")
            notes_parts.append("")

        if result.notes:
            notes_parts.append("**Notes:**")
            for note in result.notes:
                notes_parts.append(f"- {note}")

        notes_output = "\n".join(notes_parts) if notes_parts else "No additional notes."

        # Verification code
        verification_output = generate_verification_code()

        # Status message
        if not is_valid:
            status = "This combination has issues. Please check the warnings above."
        elif not result.verified:
            status = "Command generated. Please verify on the official selector."
        else:
            status = "Command generated successfully!"

        return command_output, notes_output, verification_output, status

    # Build the UI
    with gr.Blocks() as wizard:
        gr.Markdown("""
        ## PyTorch Installation Wizard

        This wizard helps you generate the correct PyTorch installation command.

        **Important:** Always verify your command on the [official PyTorch selector](https://pytorch.org/get-started/locally/).
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                build_dropdown = gr.Dropdown(
                    choices=BUILDS,
                    value="Stable",
                    label="Build",
                    info="Stable for production, Nightly for latest features",
                )

                os_dropdown = gr.Dropdown(
                    choices=OPERATING_SYSTEMS,
                    value="Linux",
                    label="Operating System",
                )

                compute_dropdown = gr.Dropdown(
                    choices=get_compute_choices("Linux"),
                    value="cpu",
                    label="Compute Platform",
                    info="Select your GPU/compute backend",
                )

                gr.Markdown("### Optional Packages")

                include_vision = gr.Checkbox(
                    value=True,
                    label="Include torchvision",
                    info="Computer vision models and datasets",
                )

                include_audio = gr.Checkbox(
                    value=True,
                    label="Include torchaudio",
                    info="Audio processing and models",
                )

                generate_btn = gr.Button("Generate Command", variant="primary")

                gr.Markdown("---")

                official_link = gr.Markdown(
                    "[Open Official PyTorch Selector](https://pytorch.org/get-started/locally/)"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Generated Command")

                status_output = gr.Markdown("Configure your options and click 'Generate Command'.")

                command_output = gr.Textbox(
                    label="Install Command",
                    lines=2,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Your install command will appear here...",
                )

                notes_output = gr.Markdown("Notes will appear here after generation.")

                with gr.Accordion("Verification Code", open=False):
                    verification_output = gr.Code(
                        label="Run this after installation to verify",
                        language="python",
                        lines=25,
                    )

        # Alternative package managers section
        with gr.Accordion("Other Installation Methods", open=False):
            with gr.Tabs():
                with gr.Tab("Conda"):
                    gr.Markdown(get_conda_guidance())

                with gr.Tab("Build from Source"):
                    gr.Markdown(get_source_guidance())

                with gr.Tab("LibTorch (C++)"):
                    gr.Markdown(get_libtorch_guidance())

        # Event handlers
        os_dropdown.change(
            fn=update_compute_dropdown,
            inputs=[os_dropdown],
            outputs=[compute_dropdown],
        )

        generate_btn.click(
            fn=generate_command,
            inputs=[
                build_dropdown,
                os_dropdown,
                compute_dropdown,
                include_vision,
                include_audio,
            ],
            outputs=[command_output, notes_output, verification_output, status_output],
        )

        # Auto-generate on load
        wizard.load(
            fn=generate_command,
            inputs=[
                build_dropdown,
                os_dropdown,
                compute_dropdown,
                include_vision,
                include_audio,
            ],
            outputs=[command_output, notes_output, verification_output, status_output],
        )

    return wizard


def create_quick_install_display() -> gr.Blocks:
    """Create a compact install guidance display for the home page."""
    from pytorch_playground.devices import get_torch_info

    with gr.Blocks() as display:
        torch_info = get_torch_info()

        if torch_info["installed"]:
            gr.Markdown(f"""
            **PyTorch Status:** Installed

            - Version: `{torch_info['version']}`
            - CUDA: `{torch_info.get('cuda_version') or 'N/A'}`
            - torchvision: `{torch_info.get('torchvision_version') or 'Not installed'}`
            - torchaudio: `{torch_info.get('torchaudio_version') or 'Not installed'}`
            """)
        else:
            gr.Markdown("""
            **PyTorch Status:** Not installed

            Use the Install Wizard tab to get installation instructions.
            """)

    return display
