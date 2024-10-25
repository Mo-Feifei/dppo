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
    data = torch.load("/home/qxy/dppo/state_8000.pt", weights_only=True)

    model.load_state_dict(data["model"])
    print("Load Model successfully!")

    model.eval()
    # torch.manual_seed(0)
    torch_input = torch.zeros(1, 1170).to("cuda:0")
    torch_input[0,-39:] = torch.tensor([-0.01817463,  0.00316524, -0.99982983, -0.001551,    0.00625625, -0.02259382,
  0.00605177,  0.03229075, -0.01595549, -0.00646622, -0.01030582, -0.03389338,
  0.00265833, -0.0066327,  -0.02148847, -0.00350176, -0.04536879,  0.08309361,
  0.00468482, -0.06002165,  0.10744572, -0.0034748,  -0.05103638,  0.08126458,
  0.00258044, -0.03308586,  0.05241467,  0.,          0.       ,   0.,
  0.    ,      0.   ,       0. ,         0.  ,        0.,          0.,
  0.   ,       0.    ,      0.        ]
)
    # input = {"state": torch.randn(1,1,23).to("cuda:0")}
    # for i in range(10):
    #     t1 = time.time()
    #     output = model(torch_input)
    #     print(time.time()-t1)
    # print(output)
    output = model(torch_input)
    print(output)
    
    # onnx_program = torch.onnx.dynamo_export(model, torch_input)
    # onnx_program.save("state_5000.onnx")
    # torch.onnx.export( 
    #     model, 
    #     torch_input, 
    #     "trotting_export.onnx", 
    #     opset_version=18, 
    #     input_names=['input'], 
    #     output_names=['output'])

    import onnx
    onnx_model = onnx.load("trotting_export.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime 
    ort_session = onnxruntime.InferenceSession("trotting_export.onnx") 
    inputs = np.zeros((1,1170))
    inputs[0,-39:] = np.array([-0.01817463,  0.00316524, -0.99982983, -0.001551,    0.00625625, -0.02259382,
  0.00605177,  0.03229075, -0.01595549, -0.00646622, -0.01030582, -0.03389338,
  0.00265833, -0.0066327,  -0.02148847, -0.00350176, -0.04536879,  0.08309361,
  0.00468482, -0.06002165,  0.10744572, -0.0034748,  -0.05103638,  0.08126458,
  0.00258044, -0.03308586,  0.05241467,  0.,          0.       ,   0.,
  0.    ,      0.   ,       0. ,         0.  ,        0.,          0.,
  0.   ,       0.    ,      0.        ]
)
    ort_inputs = {'input': np.zeros((1,1170),dtype=np.float32)} 
    for i in range(10):
        t2 = time.time()
        ort_output = ort_session.run(['output'], ort_inputs)[0] 
        print(time.time()-t2)
    
    print(ort_output)

if __name__ == "__main__":
    main()