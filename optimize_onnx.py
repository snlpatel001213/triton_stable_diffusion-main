from models import CLIP, UNet, VAE
import torch
import onnx
import onnx_graphsurgeon as gs
from enum import Enum

DUMMY_BATCH_SIZE = 1
ONNX_OPSET = 16

class OnnxPath(Enum):
    CLIP_ONNX_PATH = "clip.onnx"
    UNET_ONNX_PATH = "unet.onnx"
    VAE_ONNX_PATH =  "vae.onnx"

class Models(Enum):
    clip_model = CLIP()
    unet_model = UNet(fp16=True)
    vae_model = VAE(fp16=True)

def export_onnx(model, onnx_path):
    inputs = model.get_sample_input(DUMMY_BATCH_SIZE)
    with torch.inference_mode(), torch.autocast("cuda"):
        torch.onnx.export(model.get_model(),
                          inputs,
                          onnx_path,
                          export_params=True,
                          opset_version=ONNX_OPSET,
                          do_constant_folding=True,
                          input_names=model.get_input_names(),
                          output_names=model.get_output_names(),
                          dynamic_axes=model.get_dynamic_axes())

def expandTimestep(unet_path, save_path):
    model = onnx.load(unet_path)
    graph = gs.import_onnx(model)

    tmap = graph.tensors()
    timestep =tmap["timestep"]
    timestep.outputs.clear()
    timestep.shape=['2B', 1]

    tmap["onnx::Mul_710"].inputs[0].inputs[0]=timestep
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), save_path)


def main():
    for m, o in zip(Models, OnnxPath):
        export_onnx(m.value, o.value)
        onnx_model = onnx.load(o.value)
        optimized_model = m.value.optimize(onnx_model)
        model_save_path = "optimized_"+o.value
        onnx.save(optimized_model, model_save_path)

        if o.value == "unet.onnx":
            expandTimestep(model_save_path, model_save_path)

if __name__=="__main__":
    main()