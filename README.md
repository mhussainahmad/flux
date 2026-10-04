# flux-schnell-fp8-inference

FLUX.1-schnell text-to-image inference with FP8 weight quantization, served from a single GPU over a local socket.

## Overview

FLUX.1-schnell is a 12B-parameter distilled text-to-image model that generates in a few denoising steps. This project loads it with FP8 weights to cut GPU memory and keeps the pipeline resident behind a small socket server, so each request pays only for generation, not for model loading. It was built and tuned for a 24 GB consumer GPU (RTX 4090).

What `src/pipeline.py` does:

- **FP8 transformer checkpoint.** The FLUX transformer is loaded from a single-file FP8 checkpoint hosted at [`mhussainahmad/flux-fp8`](https://huggingface.co/mhussainahmad/flux-fp8) (`flux1-schnell-fp8.safetensors`).
- **Weight quantization with optimum-quanto.** Both the transformer and the T5 text encoder (`text_encoder_2`) are quantized to `qfloat8_e5m2` and frozen.
- **Single copy of the large modules.** The pipeline is assembled with `transformer=None` and `text_encoder_2=None`, and the quantized modules are attached afterwards, so `from_pretrained` never loads a second, full-precision copy.
- **Warm-up.** Two generations run at load time, so the first real request does not include one-time CUDA initialization.
- **Inference settings.** 4 denoising steps, `guidance_scale=0.0` (as intended for the distilled schnell model), 1024x1024 by default, optional per-request seed.

No benchmark results are stored in this repository.

## How it works

1. `start_inference` (`src/main.py`) loads the pipeline and listens on a Unix socket, `inferences.sock`, in the repository root.
2. A client sends a JSON-encoded `TextToImageRequest` (`src/request.py`): `prompt`, and optional `seed`, `width`, `height`.
3. The server runs `infer()` and replies with the image as JPEG bytes. It keeps serving the same client until it disconnects.

## Repository layout

```
src/pipeline.py   Model loading, FP8 quantization, warm-up and inference
src/main.py       Socket server
src/request.py    Request schema (pydantic)
src/client.py     Minimal client: send one prompt, save the JPEG
pyproject.toml    Pinned dependencies
uv.lock           Locked dependency versions
```

## Getting started

Requires an NVIDIA GPU with about 24 GB of memory, CUDA, and Python 3.10 to 3.12.

```bash
pip install uv
uv run start_inference        # loads the models, then waits on inferences.sock
```

In a second terminal:

```bash
uv run python src/client.py "a red tractor in a wheat field" out.jpg
```

The first run downloads FLUX.1-schnell from Hugging Face; accept the model's terms there first if prompted.

## Tech stack

Python, PyTorch 2.5.1, Hugging Face Diffusers, Transformers 4.43.2, Accelerate, optimum-quanto, pydantic, uv.

## Credits

[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by Black Forest Labs, released under the Apache 2.0 license.
