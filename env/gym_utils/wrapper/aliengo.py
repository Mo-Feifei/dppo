import gym
import torch
import numpy as np

class AliengoRLSimEnvMultiStepWrapper(gym.Wrapper):
    def __init__(self, env, n_action_steps, num_obs, obs_history_length):
        super().__init__(env)
        self.env = env

        self.obs_history_length = obs_history_length
        self.num_obs = num_obs
        self.n_action_steps = n_action_steps

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)

    def step(self, action):
        # privileged information and observation history are stored in info
        action = torch.from_numpy(action).to(self.env.device)
        total_rew = torch.zeros(action.shape[0],device=self.env.device)
        for i in range(self.n_action_steps):
            obs, rew, done, info = self.env.step(action[:,i,:])
            obs = obs[0:self.num_obs]
            total_rew += rew
            self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'state': self.obs_history.cpu().numpy()}, total_rew.cpu().numpy(), torch.zeros_like(done).cpu().numpy(), done.cpu().numpy(), info

    def get_observations(self):
        obs = self.env.get_observations()
        obs = obs[0:self.num_obs]
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'state': self.obs_history.clone().cpu().numpy()}

    def reset_one_arg(self, env_ind=None, options=None):  # it might be a problem that this isn't getting called!!
        if env_ind is not None:
            env_ind = torch.tensor([env_ind], device=self.device)
        ret = super().reset_idx(env_ind)
        ret = ret[0:self.num_obs]
        self.obs_history[env_ind, :] = 0
        return {'state': ret.clone().cpu().numpy()}

    def reset_arg(self, options_list=None):
        ret = super().reset()
        ret = ret[:, 0:self.num_obs]
        self.obs_history[:, :] = 0
        return {'state': self.obs_history.clone().cpu().numpy()}