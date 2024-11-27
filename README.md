# flux-schnell-edge-inference

This holds the baseline for the FLUX Schnel NVIDIA GeForce RTX 4090 contest, which can be forked freely and optimized

Some recommendations are as follows:
- Installing dependencies should be done in `pyproject.toml`, including git dependencies
- HuggingFace models should be specified in the `models` array in the `pyproject.toml` file, and will be downloaded before benchmarking
    - The pipeline does **not** have internet access so all dependencies and models must be included in the `pyproject.toml`
    - Compiled models should be hosted on HuggingFace and included in the `models` array in the `pyproject.toml` (rather than compiling during loading). Loading time matters far more than file sizes
- Avoid changing `src/main.py`, as that includes mostly protocol logic. Most changes should be in `models` and `src/pipeline.py`
- Ensure the entire repository (excluding dependencies and HuggingFace models) is under 16MB

For testing, you need a docker container with pytorch and ubuntu 22.04.
You can download your listed dependencies with `uv`, installed with:
```bash
pipx ensurepath
pipx install uv
```
You can then relock with `uv lock`, and then run with `uv run start_inference`
