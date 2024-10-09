import hydra
from omegaconf import OmegaConf
import os
import torch
import math
import time
import numpy as np

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    data = torch.load("./state_5000.pt", weights_only=True)

    model.load_state_dict(data["model"])
    print("Load Model successfully!")

    model.eval()
    # torch.manual_seed(0)
    torch_input = torch.zeros(1, 1, 23).to("cuda:0")
    # input = {"state": torch.randn(1,1,23).to("cuda:0")}
    for i in range(10):
        t1 = time.time()
        output = model(torch_input)
        print(time.time()-t1)
    print(output)
    
    # onnx_program = torch.onnx.dynamo_export(model, torch_input)
    # onnx_program.save("state_5000.onnx")
    torch.onnx.export( 
        model, 
        torch_input, 
        "state_5000_export.onnx", 
        opset_version=18, 
        input_names=['input'], 
        output_names=['output'])

    import onnx
    onnx_model = onnx.load("state_5000_export.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime 
    ort_session = onnxruntime.InferenceSession("state_5000_export.onnx") 
    ort_inputs = {'input': np.zeros((1,1,23),dtype=np.float32)} 
    for i in range(10):
        t2 = time.time()
        ort_output = ort_session.run(['output'], ort_inputs)[0] 
        print(time.time()-t2)
    
    print(ort_output)

if __name__ == "__main__":
    main()