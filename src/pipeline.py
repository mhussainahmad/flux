from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, T5TokenizerFast
import torch
import gc
from PIL.Image import Image
from pipelines.models import TextToImageRequest
from torch import Generator
from optimum.quanto import quantize, freeze, qfloat8_e5m2

Pipeline = FluxPipeline
CHECKPOINT = "black-forest-labs/FLUX.1-schnell"
TRANSFORMER = "https://huggingface.co/mhussainahmad/flux-fp8/blob/main/flux1-schnell-fp8.safetensors"


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def load_pipeline() -> Pipeline:
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        CHECKPOINT, subfolder="tokenizer_2"
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        CHECKPOINT, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )

    transformer = FluxTransformer2DModel.from_single_file(
        TRANSFORMER, torch_dtype=torch.bfloat16
    )
    quantize(transformer, qfloat8_e5m2)
    freeze(transformer)
    quantize(text_encoder_2, qfloat8_e5m2)
    freeze(text_encoder_2)

    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        tokenizer_2=tokenizer_2,
        transformer=None,
        text_encoder_2=None,
        torch_dtype=torch.bfloat16,
    )

    pipeline.text_encoder_2 = text_encoder_2
    pipeline.transformer = transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    # Run a warm-up step
    for _ in range(2):
        _ = pipeline(
            "A beautiful sunset over the mountains",
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            output_type="pil",
            generator=Generator(pipeline.device).manual_seed(6118176),
        ).images[0]

    return pipeline


def infer(request: TextToImageRequest, pipeline: Pipeline) -> Image:
    empty_cache()

    if request.seed is None:
        generator = None
    else:
        generator = Generator(pipeline.device.type).manual_seed(request.seed)

    print("Received prompt:", request.prompt)
    print("Generating image...")

    image = pipeline(
        prompt=request.prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        width=request.width or 1024,  # Default width
        height=request.height or 1024,  # Default height
        output_type="pil",
        generator=generator,
    ).images[0]

    return image
