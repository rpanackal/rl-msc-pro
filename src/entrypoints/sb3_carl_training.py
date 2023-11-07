from datetime import datetime
from pathlib import PurePath

import gym
import gymnasium as gymz
import stable_baselines3 as sb3
from carl.envs import CARLCartPole, CARLBraxHalfcheetah, CARLDmcQuadrupedEnv, CARLDmcWalkerEnv, CARLMountainCarContinuous
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv
from shimmy.openai_gym_compatibility import _convert_space
from stable_baselines3.common.env_util import make_vec_env



def train(
    env_name, algorithm="SAC", total_timesteps=1e6, n_envs=1, checkpoint_dir=PurePath("src/checkpoints/")
):
    """
    Train an environment using the SAC, PPO etc. algorithm.
    """

    # Experiment logging
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{env_name}_sb3.{algorithm}_{current_datetime}"
    log_dir = checkpoint_dir / exp_name

    def make_env():
        EnvCls: CARLEnv = eval(env_name)
        default_context = EnvCls.get_default_context()
        env = EnvCls(obs_context_as_dict=True, contexts={0: default_context})

        print(f"Action space: type={type(env.action_space)}")

        # Correcting for action space type
        if isinstance(env.action_space, gym.Space):
            if isinstance(env, CARLDmcEnv):
                action_spec = env.env.env.action_spec()
                env.action_space = gymz.spaces.Box(
                    action_spec.minimum, action_spec.maximum, dtype=action_spec.dtype
                )
            else:
                print(type(env.action_space))
                env.action_space = _convert_space(env.action_space)

        return env

    # Create the vectorized environment
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=60)

    # Initialize the SAC model
    algorithm_cls = "sb3."+algorithm
    model = eval(algorithm_cls)("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=str(checkpoint_dir))

    # Train the model
    model.learn(total_timesteps=total_timesteps, log_interval=1, tb_log_name=exp_name)

    # Save the model
    model.save(str(log_dir / "checkpoint"))

    print(f"Training of {env_name} using {algorithm} completed!")


if __name__ == "__main__":
    env_name = "CARLDmcWalkerEnv"  # You can change this to the environment you want
    algorithm = "PPO"
    train(env_name, algorithm)
