import numpy as np
import hydra
from omegaconf import OmegaConf
import os
import torch
import math
import time

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
    # model = hydra.utils.instantiate(cfg.model)
    # data = torch.load("./log/aliengo-pretrain/trotting_straight_dim_pre_diffusion_unet_ta8_td100/2024-11-01_16-10-46_42/checkpoint/state_8000.pt", weights_only=True)

    # model.load_state_dict(data["model"])
    # print("Load Model successfully!")
    # model.eval()

    data = np.load("/home/qxy/dppo/data/aliengo/trotting_straight/trotting_delay_2_obs.npz")
    states = data["states"]
    action = data["actions"]

    print(states[0])
    print(action[0])
    print(states[1])
    print(action[1])
    print(states[2])
    print(action[2])

    # torch_input = torch.from_numpy(states[3,:]).to("cuda:0").unsqueeze(0).to(torch.float32)
    # cond = {}
    # cond['state'] = torch_input
    # for i in range(10):
    #     output = model(cond)
    #     print(output.trajectories[:,0,:])
    


if __name__ == "__main__":
    main()

