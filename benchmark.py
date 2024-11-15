import sys
from dataclasses import dataclass
from time import perf_counter
from PIL import Image

import torch

from torch import Generator, Tensor
from transformers import T5EncoderModel
from optimum.quanto import freeze, quantize, qfloat8_e4m3fnuz, qfloat8_e5m2

from diffusers import FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast, CLIPProcessor, CLIPVisionModelWithProjection



from io import BytesIO
from os import urandom
from random import sample, shuffle

import nltk

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk import pos_tag


SAMPLE_COUNT = 5
BASELINE_AVERAGE = 45.0


AVAILABLE_WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]


def generate_random_prompt():
    # sampled_words = sample(AVAILABLE_WORDS, k=min(len(AVAILABLE_WORDS), min(urandom(1)[0] % 32, 8)))
    # shuffle(sampled_words)

    prompts = [
        "A beautiful sunset over the mountains",
        "A futuristic cityscape at night",
        "A serene beach with crystal clear water",
        "A dense forest with rays of sunlight",
        "A bustling market in a small village"
    ]

    return sample(prompts, 1)[0]


@dataclass
class CheckpointBenchmark:
    baseline_average: float
    average_time: float
    average_similarity: float
    failed: bool


@dataclass
class GenerationOutput:
    prompt: str
    seed: int
    output: Image
    generation_time: float


def compare(baseline: bytes, optimized: bytes, device: str = "cpu") -> float:
        from torch import manual_seed
        from torch.nn.functional import cosine_similarity

        from PIL import Image

        from skimage import metrics
        import cv2

        import numpy

        manual_seed(0)
        clip = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device)
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        def load_image(data: bytes):
            with BytesIO(data) as fp:
                return numpy.array(Image.open(fp).convert("RGB"))

        def clip_embeddings(image: numpy.ndarray):
            processed_input = processor(images=image, return_tensors="pt").to(device)

            return clip(**processed_input).image_embeds.to(device)

        baseline_array = load_image(baseline)
        optimized_array = load_image(optimized)

        grayscale_baseline = cv2.cvtColor(baseline_array, cv2.COLOR_RGB2GRAY)
        grayscale_optimized = cv2.cvtColor(optimized_array, cv2.COLOR_RGB2GRAY)

        structural_similarity = metrics.structural_similarity(grayscale_baseline, grayscale_optimized, full=True)[0]

        del grayscale_baseline
        del grayscale_optimized

        baseline_embeddings = clip_embeddings(baseline_array)
        optimized_embeddings = clip_embeddings(optimized_array)

        clip_similarity = cosine_similarity(baseline_embeddings, optimized_embeddings)[0].item()

        return clip_similarity * 0.35 + structural_similarity * 0.65



def calculate_score(model_average: float, similarity: float) -> float:
    return max(
        0.0,
        BASELINE_AVERAGE - model_average
    ) * similarity


def generate(pipeline: FluxPipeline, prompt: str, seed: int):
    start = perf_counter()
    output = pipeline(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        output_type="pil",
        generator=Generator(pipeline.device).manual_seed(seed)
    ).images[0]

    generation_time = perf_counter() - start


    output.save(f"original_{seed}.png")
    
    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
    )

def efficient_generate(pipeline: FluxPipeline, prompt: str, seed: int):

    start = perf_counter()
    
    output = pipeline(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        output_type="pil",
        generator=Generator(pipeline.device).manual_seed(seed)
    ).images[0]

    generation_time = perf_counter() - start

    output.save(f"optimized_{seed}.png")

    return GenerationOutput(
        prompt,
        seed,
        output,
        generation_time,
    )


def compare_checkpoints():
    baseline_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    baseline_pipeline.enable_model_cpu_offload()
    print("Generating baseline samples to compare")

    baseline_outputs: list[GenerationOutput] = [
        generate(
            baseline_pipeline,
            generate_random_prompt(),
            int.from_bytes(urandom(4), "little"),
        )
        for _ in range(SAMPLE_COUNT)
    ]

    del baseline_pipeline

    torch.cuda.empty_cache()

    baseline_average = sum([output.generation_time for output in baseline_outputs]) / len(baseline_outputs)

    average_time = float("inf")
    average_similarity = 1.0

    # Optimized Model
    
    bfl_repo = "black-forest-labs/FLUX.1-schnell"
    revision = "refs/pr/1"
    dtype = torch.bfloat16
    # scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
    # tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
    # vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = FluxTransformer2DModel.from_single_file("/workspace/original_model/flux1-schnell-fp8-e4m3fn.safetensors", torch_dtype=dtype)
    pipeline = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)

    quantize(transformer, weights=qfloat8_e5m2)
    freeze(transformer)

    quantize(text_encoder_2, weights=qfloat8_e5m2)
    freeze(text_encoder_2)

    # pipeline = FluxPipeline(
    #     scheduler=scheduler,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     text_encoder_2=None,
    #     tokenizer_2=tokenizer_2,
    #     vae=vae,
    #     transformer=None,
    # )
    
    pipeline.text_encoder_2 = text_encoder_2
    pipeline.transformer = transformer

    pipeline.to("cuda")
    

    i = 0

    # Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been
    for i, baseline in enumerate(baseline_outputs):
        print(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}, took baseline time {baseline.generation_time}")

        generated = i
        remaining = SAMPLE_COUNT - generated

        generation = efficient_generate(
            pipeline,
            baseline.prompt,
            baseline.seed,
        )
        
        baseline_byte_stream = BytesIO()
        baseline.output.save(baseline_byte_stream, format="PNG")  # Specify format if needed, e.g., "PNG", "JPEG"
        baseline_bytes = baseline_byte_stream.getvalue()

        generation_byte_stream = BytesIO()
        generation.output.save(generation_byte_stream, format="PNG")  # Specify format if needed, e.g., "PNG", "JPEG"
        generation_bytes = generation_byte_stream.getvalue()

        
        similarity = compare(
            baseline_bytes,
            generation_bytes,
            device="cpu"
        )

        print(
            f"Sample {i} generated "
            f"with generation time of {generation.generation_time} "
            f"and similarity {similarity}"
        )

        if generated:
            average_time = (average_time * generated + generation.generation_time) / (generated + 1)
        else:
            average_time = generation.generation_time

        average_similarity = (average_similarity * generated + similarity) / (generated + 1)

        if average_time < baseline_average * 1.0625:
            # So far, the average time is better than the baseline, so we can continue
            continue

        needed_time = (baseline_average * SAMPLE_COUNT - generated * average_time) / remaining

        if needed_time < average_time * 0.75:
            # Needs %33 faster than current performance to beat the baseline,
            # thus we shouldn't waste compute testing farther
            print("Too different from baseline, failing", file=sys.stderr)
            break

        if average_similarity < 0.85:
            # Deviating too much from original quality
            print("Too different from baseline, failing", file=sys.stderr)
            break

    print(
        f"Tested {i + 1} samples, "
        f"average similarity of {average_similarity}, "
        f"and speed of {average_time}"
        f"with a final score of {calculate_score(average_time, average_similarity)}"
    )


if __name__ == '__main__':
    compare_checkpoints()
