import sys
from dataclasses import dataclass
from time import perf_counter
from PIL import Image

import torch

from torch import Generator
from transformers import T5EncoderModel
from optimum.quanto import freeze, quantize, qfloat8_e5m2

from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import T5EncoderModel, CLIPProcessor, CLIPVisionModelWithProjection

from io import BytesIO
from os import urandom
from random import sample

SAMPLE_COUNT = 5
BASELINE_AVERAGE = 20



def generate_random_prompt():
    PROMPTS = [
        "painting of King Henry VIII carrying an umbrella",
        "Fox Mulder and a chinchilla walking down a road in the style of an old fashioned photograph",
        "photo of a gas burner by a soft pretzel",
        "photo of Shyster standing street lights on at night",
        "cute young man eating a plant over a fence in the style of an old fashioned photograph",
        "Krusty the Clown with a cutter playing in sand in the style of a vector drawing",
        "Judge Joe Dredd doing a fashion show at a town",
        "man standing in batters box, while a pitcher pitches in the style of a 3D render",
        "elastic old man saying something",
        "photo of a frightening father throwing a white Frisbee in a factory",
        "Eddard Stark selling fresh fruits and vegetables",
        "bumpy clock with a cyprinodont",
        "tenement in a cabin in the style of an installation art piece",
        "green wild apple in a village in the style of a claymation figure",
        "group of daring boys sitting next to a very tall building on a spaceship in the style of a DSLR photograph",
        "a barbershop near an umbrella by a train in the style of a Rembrandt painting",
        "dangerous Riesling at a bar",
        "four average young men tilting around a curve in the style of a fisheye lens photograph",
        "drawing of a hoopoe leaning on a laptop at a big city",
        "crayon drawing of the Tower Bridge",
        "an imaginary Unknown Soldier",
        "seahorse walking in the woods with a pair of skis in the style of a classic photograph",
        "Pianist nearby a English foxhound flying in the sky",
        "masculine teddy bear next to a hotchpotch",
        "flawed mother preparing food on a stove",
        "painting of an enraged birdhouse downtown",
        "lovely digital art of a jaboticaba",
        "grandiose bottle",
        "pass with a department store standing behind a fence on a motorcycle in the style of a Rubens painting",
        "candid cutworm and also a bur marigold in the style of a line drawing",
        "redheaded woodpecker with an organ-grinder by a bus",
        "superb steel drum",
        "painting of Jimmy Carter leaning on a boar",
        "painting of a scary man holding a green lacewing",
        "Wolfgang Amadeus Mozart sitting inside of a space bedroom in the style of a Vincent van Gogh painting",
        "origami art of a restaurant beside a popinjay",
        "screen-print t-shirt of a cuddly printer cable and also a mountain sheep",
        "3D render of Newgrange on a tricorn",
        "Judas Iscariot sitting on a childrens toilet",
        "majestic oxeye daisy",
        "paper art of a soda water in a restaurant",
        "Victor Frankenstein showing its tools",
        "traffic light in the style of a CCTV image",
        "formal father bouncing on a bed near a laptop by a bicycle in the style of a caricature",
        "majestic auditorium beside a teriyaki",
        "lego project of a refrigerator in a river",
        "striped petticoat",
        "revolting ivory carving of an insignificant dog with a churn",
        "flimsy Malay"
    ]

    return sample(PROMPTS, 1)[0]



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
    output: Image.Image | BytesIO
    generation_time: float


def compare(baseline: bytes, optimized: bytes, device: str = "cpu") -> float:
        from torch import manual_seed
        from torch.nn.functional import cosine_similarity

        from PIL import Image

        from skimage import metrics
        import cv2

        import numpy
        import random

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


    prompt_with_underscores = prompt.replace(" ", "_")
    output.save(f"original_{prompt_with_underscores}__{seed}.png")
    
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

    prompt_with_underscores = prompt.replace(" ", "_")
    output.save(f"optimized_{prompt_with_underscores}__{seed}.png")

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
    dtype = torch.bfloat16
    print("Loading optimized model")
    print("Loading Text encoder 2")
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
    print("Loading transformer")
    transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/mhussainahmad/flux-fp8/blob/main/flux1-schnell-fp8.safetensors", torch_dtype=dtype)

    quantize(transformer, weights=qfloat8_e5m2)
    freeze(transformer)
    quantize(text_encoder_2, weights=qfloat8_e5m2)
    freeze(text_encoder_2)
    
    
    print("Loading pipeline")
    pipeline: FluxPipeline = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
    
    pipeline.text_encoder_2 = text_encoder_2
    pipeline.transformer = transformer

    pipeline.to("cuda") 
    print("Warming up...")
    
    for _ in range(2):
        _ = pipeline(
            "A beautiful sunset over the mountains",
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            output_type="pil",
            generator=Generator(pipeline.device).manual_seed(6118176)
        ).images[0]


    i = 0

    for i, baseline in enumerate(baseline_outputs):
        print(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}")

        generated = i
        remaining = SAMPLE_COUNT - generated

        generation = efficient_generate(
            pipeline,
            baseline.prompt,
            baseline.seed,
        )
        
        baseline_byte_stream = BytesIO()
        baseline.output.save(baseline_byte_stream, format="PNG")  
        baseline_bytes = baseline_byte_stream.getvalue()

        generation_byte_stream = BytesIO()
        generation.output.save(generation_byte_stream, format="PNG") 
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
            continue

        needed_time = (baseline_average * SAMPLE_COUNT - generated * average_time) / remaining

        if needed_time < average_time * 0.75:
            print("Too different from baseline, failing", file=sys.stderr)
            break

        if average_similarity < 0.85:
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
