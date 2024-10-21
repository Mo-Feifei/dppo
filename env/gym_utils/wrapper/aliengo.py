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
            obs = self.convert_obs(obs)
            total_rew += rew
            env_ids = done.nonzero(as_tuple=False).flatten()
            self.reset_one_arg(env_ids)
            self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        return {'state': self.obs_history.cpu().numpy()}, total_rew.cpu().numpy(), torch.zeros_like(done).cpu().numpy(), done.cpu().numpy(), info

    def convert_observations(self):
        obs = self.env.convert_observations()
        obs = self.convert_obs(obs)
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        return {'state': self.obs_history.clone().cpu().numpy()}

    def reset_one_arg(self, env_ind):  # it might be a problem that this isn't getting called!!
        self.obs_history[env_ind, :] = 0


    def reset_arg(self, options_list=None):
        ret = super().reset()
        obs = self.convert_obs(ret)
        print(obs.shape)
        self.obs_history[:, :] = 0
        print(self.obs_history.shape)
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        print(self.obs_history.shape)
        return {'state': self.obs_history.clone().cpu().numpy()}
    
    def convert_obs(self, obs):
        o = torch.zeros((self.env.num_envs, 39),device=self.env.device)
        o[:,0:3] = obs[:,0:3]
        o[:,3:39] = obs[:,18:54]
        return o