import os
import json

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)


def make_async(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    **kwargs,
):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment.

    asynchronous : bool
        If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses
        `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.

    wrappers : dictionary, optional
        Each key is a wrapper class, and each value is a dictionary of arguments

    Returns
    -------
    :class:`gym.vector.VectorEnv`
        The vectorized environment.

    Example
    -------
    >>> env = gym.vector.make('CartPole-v1', num_envs=3)
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """

    if env_type == "furniture":
        from furniture_bench.envs.observation import DEFAULT_STATE_OBS
        from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
        from env.gym_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper

        env = FurnitureRLSimEnv(
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture=furniture,
            gpu_id=gpu_id,
            headless=headless,
            num_envs=num_envs,
            observation_space="state",
            randomness=randomness,
            max_env_steps=max_episode_steps,
            record=record,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env = FurnitureRLSimEnvMultiStepWrapper(
            env,
            n_obs_steps=obs_steps,
            n_action_steps=act_steps,
            prev_action=False,
            reset_within_step=False,
            pass_full_observations=False,
            normalization_path=normalization_path,
            sparse_reward=sparse_reward,
        )
        return env
    
    if env_type == "aliengo":
        from go1_gym.envs.base.legged_robot_config import Cfg
        from go1_gym.envs.go1.aliengo_config import config_aliengo
        from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
        from env.gym_utils.wrapper.aliengo import AliengoRLSimEnvMultiStepWrapper

        config_aliengo(Cfg)
        Cfg.commands.num_lin_vel_bins = 30
        Cfg.commands.num_ang_vel_bins = 30
        Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
        Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
        Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
        Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

        Cfg.commands.distributional_commands = True

        Cfg.domain_rand.lag_timesteps = 6
        Cfg.domain_rand.randomize_lag_timesteps = True
        Cfg.control.control_type = "P"

        Cfg.domain_rand.randomize_rigids_after_start = False
        Cfg.env.priv_observe_motion = False
        Cfg.env.priv_observe_gravity_transformed_motion = False
        Cfg.domain_rand.randomize_friction_indep = False
        Cfg.env.priv_observe_friction_indep = False
        Cfg.domain_rand.randomize_friction = True
        Cfg.env.priv_observe_friction = True
        Cfg.domain_rand.friction_range = [0.1, 4.5]
        Cfg.domain_rand.randomize_restitution = True
        Cfg.env.priv_observe_restitution = True
        Cfg.domain_rand.restitution_range = [0.0, 0.4]
        Cfg.domain_rand.randomize_base_mass = True
        Cfg.env.priv_observe_base_mass = False
        Cfg.domain_rand.added_mass_range = [-1.0, 3.5]
        Cfg.domain_rand.randomize_gravity = True
        Cfg.domain_rand.gravity_range = [-1.0, 1.0]
        Cfg.domain_rand.gravity_rand_interval_s = 8.0
        Cfg.domain_rand.gravity_impulse_duration = 0.99
        Cfg.env.priv_observe_gravity = False
        Cfg.domain_rand.randomize_com_displacement = True
        Cfg.domain_rand.com_displacement_range = [-0.1, 0.1]
        Cfg.env.priv_observe_com_displacement = False
        Cfg.domain_rand.randomize_ground_friction = True
        Cfg.env.priv_observe_ground_friction = False
        Cfg.env.priv_observe_ground_friction_per_foot = False
        Cfg.domain_rand.ground_friction_range = [0.1, 4.5]
        Cfg.domain_rand.randomize_motor_strength = True
        Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
        Cfg.env.priv_observe_motor_strength = False
        Cfg.domain_rand.randomize_motor_offset = True
        Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
        Cfg.env.priv_observe_motor_offset = False
        Cfg.domain_rand.push_robots = False
        Cfg.domain_rand.max_push_vel_xy = 0.5
        Cfg.domain_rand.randomize_Kp_factor = True
        Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
        Cfg.env.priv_observe_Kp_factor = False
        Cfg.domain_rand.randomize_Kd_factor = True
        Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
        Cfg.env.priv_observe_Kd_factor = False
        Cfg.env.priv_observe_body_velocity = True
        Cfg.env.priv_observe_body_height = True
        Cfg.env.priv_observe_desired_contact_states = False
        Cfg.env.priv_observe_contact_forces = False
        Cfg.env.priv_observe_foot_displacement = False
        Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

        Cfg.env.num_privileged_obs = 6
        Cfg.env.num_observation_history = 30
        Cfg.reward_scales.feet_contact_forces = 0.0

        Cfg.domain_rand.rand_interval_s = 6
        Cfg.commands.num_commands = 15
        Cfg.env.observe_two_prev_actions = True
        Cfg.env.observe_yaw = False
        Cfg.env.num_observations = 70
        Cfg.env.num_scalar_observations = 70
        Cfg.env.observe_gait_commands = True
        Cfg.env.observe_timing_parameter = False
        Cfg.env.observe_clock_inputs = True

        Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
        Cfg.domain_rand.tile_height_curriculum = False
        Cfg.domain_rand.tile_height_update_interval = 1000000
        Cfg.domain_rand.tile_height_curriculum_step = 0.01
        Cfg.terrain.border_size = 0.0
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.num_cols = 30
        Cfg.terrain.num_rows = 30
        Cfg.terrain.terrain_width = 5.0
        Cfg.terrain.terrain_length = 5.0
        Cfg.terrain.x_init_range = 0.2
        Cfg.terrain.y_init_range = 0.2
        Cfg.terrain.teleport_thresh = 0.3
        Cfg.terrain.teleport_robots = False
        Cfg.terrain.center_robots = True
        Cfg.terrain.center_span = 4
        Cfg.terrain.horizontal_scale = 0.10
        Cfg.rewards.use_terminal_foot_height = False
        Cfg.rewards.use_terminal_body_height = True
        Cfg.rewards.terminal_body_height = 0.05
        Cfg.rewards.use_terminal_roll_pitch = True
        Cfg.rewards.terminal_body_ori = 1.6

        Cfg.commands.resampling_time = 10

        Cfg.reward_scales.feet_slip = -0.04
        Cfg.reward_scales.action_smoothness_1 = -0.1
        Cfg.reward_scales.action_smoothness_2 = -0.1
        Cfg.reward_scales.dof_vel = -1e-4
        Cfg.reward_scales.dof_pos = -0.0
        Cfg.reward_scales.jump = 10.0
        Cfg.reward_scales.base_height = 0.0
        # Cfg.rewards.base_height_target = 0.38
        Cfg.reward_scales.estimation_bonus = 0.0
        Cfg.reward_scales.raibert_heuristic = -10.0
        Cfg.reward_scales.feet_impact_vel = -0.0
        Cfg.reward_scales.feet_clearance = -0.0
        Cfg.reward_scales.feet_clearance_cmd = -0.0
        Cfg.reward_scales.feet_clearance_cmd_linear = -30.0
        Cfg.reward_scales.orientation = 0.0
        Cfg.reward_scales.orientation_control = -5.0
        Cfg.reward_scales.tracking_stance_width = -0.0
        Cfg.reward_scales.tracking_stance_length = -0.0
        Cfg.reward_scales.lin_vel_z = -0.02
        Cfg.reward_scales.ang_vel_xy = -0.001
        Cfg.reward_scales.feet_air_time = 0.0
        Cfg.reward_scales.hop_symmetry = 0.0
        Cfg.rewards.kappa_gait_probs = 0.07
        Cfg.rewards.gait_force_sigma = 100.
        Cfg.rewards.gait_vel_sigma = 10.
        Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
        Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
        Cfg.reward_scales.collision = -5.0

        Cfg.rewards.reward_container_name = "CoRLRewards"
        Cfg.rewards.only_positive_rewards = False
        Cfg.rewards.only_positive_rewards_ji22_style = True
        Cfg.rewards.sigma_rew_neg = 0.02



        Cfg.commands.lin_vel_x = [-1.0, 1.0]
        Cfg.commands.lin_vel_y = [-0.6, 0.6]
        Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
        Cfg.commands.body_height_cmd = [-0.15, 0.05]
        Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
        Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
        Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
        Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
        Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
        Cfg.commands.footswing_height_range = [0.03, 0.35]
        Cfg.commands.body_pitch_range = [-0.4, 0.4]
        Cfg.commands.body_roll_range = [-0.0, 0.0]
        Cfg.commands.stance_width_range = [0.10, 0.45]
        Cfg.commands.stance_length_range = [0.35, 0.45]

        Cfg.commands.limit_vel_x = [-5.0, 5.0]
        Cfg.commands.limit_vel_y = [-0.6, 0.6]
        Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
        Cfg.commands.limit_body_height = [-0.25, 0.15]
        Cfg.commands.limit_gait_frequency = [2.0, 4.0]
        Cfg.commands.limit_gait_phase = [0.0, 1.0]
        Cfg.commands.limit_gait_offset = [0.0, 1.0]
        Cfg.commands.limit_gait_bound = [0.0, 1.0]
        Cfg.commands.limit_gait_duration = [0.5, 0.5]
        Cfg.commands.limit_footswing_height = [0.03, 0.35]
        Cfg.commands.limit_body_pitch = [-0.4, 0.4]
        Cfg.commands.limit_body_roll = [-0.0, 0.0]
        Cfg.commands.limit_stance_width = [0.10, 0.45]
        Cfg.commands.limit_stance_length = [0.35, 0.45]

        Cfg.commands.num_bins_vel_x = 21
        Cfg.commands.num_bins_vel_y = 1
        Cfg.commands.num_bins_vel_yaw = 21
        Cfg.commands.num_bins_body_height = 1
        Cfg.commands.num_bins_gait_frequency = 1
        Cfg.commands.num_bins_gait_phase = 1
        Cfg.commands.num_bins_gait_offset = 1
        Cfg.commands.num_bins_gait_bound = 1
        Cfg.commands.num_bins_gait_duration = 1
        Cfg.commands.num_bins_footswing_height = 1
        Cfg.commands.num_bins_body_roll = 1
        Cfg.commands.num_bins_body_pitch = 1
        Cfg.commands.num_bins_stance_width = 1

        Cfg.normalization.friction_range = [0, 1]
        Cfg.normalization.ground_friction_range = [0, 1]
        Cfg.terrain.yaw_init_range = 3.14
        Cfg.normalization.clip_actions = 10.0

        Cfg.commands.exclusive_phase_offset = False
        Cfg.commands.pacing_offset = False
        Cfg.commands.binary_phases = True
        Cfg.commands.gaitwise_curricula = True

        Cfg.control.control_type = "P"
        Cfg.control.action_scale = kwargs.get("action_scale", 2.5)

        Cfg.domain_rand.randomize_rigids_after_start = False
        Cfg.domain_rand.push_robots = False
        Cfg.domain_rand.randomize_friction = False
        Cfg.domain_rand.randomize_gravity = False
        Cfg.domain_rand.randomize_restitution = False
        Cfg.domain_rand.randomize_motor_offset = False
        Cfg.domain_rand.randomize_motor_strength = False
        Cfg.domain_rand.randomize_friction_indep = False
        Cfg.domain_rand.randomize_ground_friction = False
        Cfg.domain_rand.randomize_base_mass = True
        Cfg.domain_rand.randomize_Kd_factor = False
        Cfg.domain_rand.randomize_Kp_factor = False
        Cfg.domain_rand.randomize_joint_friction = False
        Cfg.domain_rand.randomize_com_displacement = False

        Cfg.env.num_envs = 1
        Cfg.terrain.num_rows = 5
        Cfg.terrain.num_cols = 5
        Cfg.terrain.border_size = 0
        Cfg.terrain.center_robots = True
        Cfg.terrain.center_span = 1
        Cfg.terrain.teleport_robots = True
        Cfg.commands.command_curriculum = False
        Cfg.env.num_privileged_obs = 6

        Cfg.domain_rand.lag_timesteps = 6
        Cfg.domain_rand.randomize_lag_timesteps = True

        env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
        env = AliengoRLSimEnvMultiStepWrapper(
            env, 
            n_action_steps = act_steps,
            num_obs=kwargs.get('num_obs', 39), 
            obs_history_length=kwargs.get('history_len', 30))

        return env

    # avoid import error due incompatible gym versions
    from gym import spaces
    from env.gym_utils.async_vector_env import AsyncVectorEnv
    from env.gym_utils.sync_vector_env import SyncVectorEnv
    from env.gym_utils.wrapper import wrapper_dict

    __all__ = [
        "AsyncVectorEnv",
        "SyncVectorEnv",
        "VectorEnv",
        "VectorEnvWrapper",
        "make",
    ]

    # import the envs
    if robomimic_env_cfg_path is not None:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
    elif "avoiding" in id:
        import gym_avoiding
    else:
        import d4rl.gym_mujoco
    from gym.envs import make as make_

    def _make_env():
        if robomimic_env_cfg_path is not None:
            obs_modality_dict = {
                "low_dim": (
                    wrappers.robomimic_image.low_dim_keys
                    if "robomimic_image" in wrappers
                    else wrappers.robomimic_lowdim.low_dim_keys
                ),
                "rgb": (
                    wrappers.robomimic_image.image_keys
                    if "robomimic_image" in wrappers
                    else None
                ),
            }
            if obs_modality_dict["rgb"] is None:
                obs_modality_dict.pop("rgb")
            ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
            if render_offscreen or use_image_obs:
                os.environ["MUJOCO_GL"] = "egl"
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            env_meta["reward_shaping"] = reward_shaping
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,
                # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
                render_offscreen=render_offscreen,
                use_image_obs=use_image_obs,
                # render_gpu_device_id=0,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            env.env.hard_reset = False
        else:  # d3il, gym
            env = make_(id, render=render, **kwargs)

        # add wrappers
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env

    def dummy_env_fn():
        """TODO(allenzren): does this dummy env allow camera obs for other envs besides robomimic?"""
        import gym
        import numpy as np
        from env.gym_utils.wrapper.multi_step import MultiStep

        # Avoid importing or using env in the main process
        # to prevent OpenGL context issue with fork.
        # Create a fake env whose sole purpose is to provide
        # obs/action spaces and metadata.
        env = gym.Env()
        observation_space = spaces.Dict()
        if shape_meta is not None:  # rn only for images
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = -1, 1
                elif key.endswith("state"):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        else:
            observation_space["state"] = gym.spaces.Box(
                -1,
                1,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        env.observation_space = observation_space
        env.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.int64)
        env.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": 12,
        }
        return MultiStep(env=env, n_obs_steps=wrappers.multi_step.n_obs_steps)

    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(
            env_fns,
            dummy_env_fn=(
                dummy_env_fn if render or render_offscreen or use_image_obs else None
            ),
            delay_init="avoiding" in id,  # add delay for D3IL initialization
        )
        if asynchronous
        else SyncVectorEnv(env_fns)
    )
