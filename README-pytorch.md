# PyTorch setup (macOS)

This project has PyTorch installed in the local virtual environment `.venv` (Python 3.12).

Quick steps to reproduce locally:

1. Create and activate venv (if you don't have it):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install packages (we used PyTorch's macOS wheels):

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Run the quick test to verify MPS/CUDA availability:

```bash
python test_torch.py
```

Troubleshooting:

- If `torch.backends.mps.is_available()` is False on Apple silicon, ensure you are running macOS 12.3+ and Xcode command line tools are installed.
- For CUDA support, you need an NVIDIA GPU on Linux/Windows. On macOS, use MPS where available.

Notes:

- The venv python is at `.venv/bin/python` in this workspace.
- We installed torch 2.8.0 which supports Python 3.12 on macOS arm64.
