# flux-schnell-edge-inference

A FLUX.1-schnell text-to-image inference pipeline with FP8 weight quantization, built as a submission for the WOMBO edge-maxxing FLUX.1-schnell optimization contest on an NVIDIA GeForce RTX 4090.

## Overview

The edge-maxxing contest ([womboai/edge-maxxing](https://github.com/womboai/edge-maxxing)) provides a baseline repository that participants fork and optimize. The contest harness starts the pipeline, sends text-to-image requests over a Unix socket, and benchmarks the pipeline.

This repository follows that template. `src/main.py` is the contest's protocol code and is close to the upstream baseline. The changes are in `src/pipeline.py` and `pyproject.toml`:

- **FP8 transformer checkpoint.** The FLUX transformer is loaded from a single-file FP8 checkpoint hosted at [`mhussainahmad/flux-fp8`](https://huggingface.co/mhussainahmad/flux-fp8) (`flux1-schnell-fp8.safetensors`) and cast to bfloat16.
- **Weight quantization with optimum-quanto.** Both the transformer and the T5 text encoder (`text_encoder_2`) are quantized to `qfloat8_e5m2` and frozen before they are attached to the pipeline.
- **Component assembly.** The tokenizer and T5 encoder come from `black-forest-labs/FLUX.1-schnell`. The pipeline is built with `transformer=None` and `text_encoder_2=None`, and the quantized modules are attached afterwards, so `from_pretrained` does not load a second, unquantized copy of these components from the base checkpoint.
- **Warm-up.** Two warm-up generations run at load time so that the first timed request does not include one-time CUDA initialization.
- **Inference settings.** 4 denoising steps, `guidance_scale=0.0` (the setting used for the distilled schnell model), default resolution 1024x1024, and an optional seeded generator for each request.

No benchmark results are stored in this repository.

## How it works

1. `start_inference` (mapped to `main:main`) calls `load_pipeline()` and opens a Unix socket at `inferences.sock` in the repository root.
2. Each request arrives as a JSON-encoded `TextToImageRequest` (prompt, optional seed, width and height), defined in the contest's `edge-maxxing-pipelines` package.
3. `infer()` clears the CUDA cache, runs the pipeline, and the image is returned over the socket as JPEG bytes.

## Repository layout

```
src/main.py       Socket server and request loop (contest protocol code)
src/pipeline.py   Model loading, FP8 quantization, warm-up and inference
pyproject.toml    Pinned dependencies and the [tool.edge-maxxing] models list
uv.lock           Locked dependency versions
```

## Getting started

You need an NVIDIA GPU (the contest target was an RTX 4090) and a Docker container with PyTorch on Ubuntu 22.04.

```bash
pipx ensurepath
pipx install uv
uv lock          # relock if you change dependencies
uv run start_inference
```

The process loads the models, then waits for a client to connect to `inferences.sock`. The client is the contest's benchmarking harness, which is not included here.

Contest constraints from the upstream template:

- All dependencies, including git dependencies, must be declared in `pyproject.toml`.
- Hugging Face models must be listed in the `models` array under `[tool.edge-maxxing]`. The harness downloads them before benchmarking, and the pipeline has no internet access at run time.
- Precompiled or converted models should be hosted on Hugging Face rather than built at load time.
- The repository (excluding dependencies and models) must stay under 16 MB.

## Tech stack

Python 3.10 to 3.12, PyTorch 2.5.1, Hugging Face Diffusers, Accelerate and PEFT (pinned git commits), Transformers 4.43.2, optimum-quanto, uv.

## Credits

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by Black Forest Labs, released under the Apache 2.0 license.
- Baseline template and request protocol from [womboai/edge-maxxing](https://github.com/womboai/edge-maxxing).
