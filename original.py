# import torch
# from diffusers import FluxPipeline
# import time

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float8_e4m3fn)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A cat holding a sign that says hello world"
# start_time = time.time()

# # Measure VRAM usage before generation
# torch.cuda.synchronize()
# vram_before = torch.cuda.memory_allocated()

# image = pipe(
#     prompt,
#     guidance_scale=0.0,
#     num_inference_steps=4,
#     max_sequence_length=256,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# # Measure VRAM usage after generation
# torch.cuda.synchronize()
# vram_after = torch.cuda.memory_allocated()
# vram_after = torch.cuda.memory_allocated()

# end_time = time.time()

# # Calculate metrics
# generation_time = end_time - start_time
# vram_used = vram_after - vram_before

# # Save the image
# image.save("flux-schnell.png")

# # Print metrics
# print(f"Generation time: {generation_time:.2f} seconds")
# print(f"VRAM used: {vram_used / (1024 ** 2):.2f} MB")

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize, qfloat8_e4m3fn, qfloat8_e4m3fnuz, qfloat8_e5m2
import time
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast


bfl_repo = "black-forest-labs/FLUX.1-schnell"
dtype = torch.bfloat16

# Load and quantize transformer
bfl_repo = "black-forest-labs/FLUX.1-schnell"
revision = "refs/pr/1"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_single_file("/workspace/original_model/flux1-schnell-fp8.safetensors", torch_dtype=dtype)
# Load pipeline
# pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype)
# pipe.transformer = transformer
# pipe.text_encoder_2 = text_encoder_2
quantize(transformer, weights=qfloat8_e4m3fn)
freeze(transformer)

quantize(text_encoder_2, weights=qfloat8_e4m3fn)
freeze(text_encoder_2)
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)

pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer

pipe.to("cuda")

# Track VRAM usage
torch.cuda.reset_max_memory_allocated()
print("Starting generation...")
start_time = time.time()

# Generate the image
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=3.5,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]

end_time = time.time()
generation_time = end_time - start_time

# Get VRAM stats
max_vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
current_vram_usage = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB

# Display results
print(f"Generation time: {generation_time:.2f} seconds")
print(f"Maximum VRAM usage during generation: {max_vram_usage:.2f} MB")
print(f"Current VRAM usage: {current_vram_usage:.2f} MB")

# Save the image
image.save("flux-fp8-schnell.png")
