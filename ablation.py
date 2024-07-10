from models import CLIP, UNet, VAE
import torch
import onnx
import onnx_graphsurgeon as gs
from demo_diffusion import DemoDiffusion
import argparse
import tensorrt as trt
from cuda import cudart
import json
from utilities import Engine, DPMScheduler, LMSDiscreteScheduler, save_image, TRT_LOGGER


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='*')
    parser.add_argument('--optimization-level', type=int, default=0)
    parser.add_argument('--negative-prompt', nargs = '*', default=[''], help="The negative prompt(s) to guide the image generation.")
    parser.add_argument('--repeat-prompt', type=int, default=1, choices=[1, 2, 4, 8, 16], help="Number of times to repeat the prompt (batch size multiplier)")
    parser.add_argument('--height', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--hf-token', default='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',type=str)
    parser.add_argument('--engine-dir', default='engine', type=str)
    parser.add_argument('--onnx-dir', default='onnx', type=str)
    parser.add_argument('--seed', type=int, default=None, help="Seed for random generator to get consistent results")
    parser.add_argument('--nvtx-profile', action='store_true', help="Enable NVTX markers for performance profiling")
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    prompt = args.prompt

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    demo = DemoDiffusion(denoising_steps=50,
                         denoising_fp16=True,
                         output_dir="output",
                         scheduler="LMSD",
                         hf_token=args.hf_token,
                         verbose=True,
                         nvtx_profile=True,
                         max_batch_size=1,
                         optimization_level=args.optimization_level)

    demo.loadEngines(args.engine_dir, args.onnx_dir, 16, 
        opt_batch_size=len(prompt), opt_image_height=args.height, opt_image_width=args.width, \
        force_export=True, force_optimize=True, \
        force_build=True, minimal_optimization=False, \
        static_batch=False, static_shape=True, \
        enable_preview=False)
    
    demo.loadModules()

    print("[I] Warming up ..")
    for _ in range(10):
        images = demo.infer(prompt, args.negative_prompt, args.height, args.width, warmup=True, verbose=False)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images = demo.infer(prompt, args.negative_prompt, args.height, args.width, verbose=False, save_to_json=True, seed=args.seed)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    demo.teardown()