# Create a gym vector and non-vector environment
# Check shape of space
# Look for single to batch outward env and batch to single for inward env when non-vector env is involved
import gym
import gymnasium as gymz
from carl.envs import CARLBraxHalfcheetah, CARLCartPole
from shimmy.openai_gym_compatibility import _convert_space
from ..envs.wrappers.compatibility import (
    VecEnvProjectCompatibility,
    EnvProjectCompatibility,
)
from ..envs.wrappers.batch_step import EnvVectorResponse
from ..envs.wrappers.normalization import RMVNormalizeVecObservation
gym_hc_env = gym.make("HalfCheetah-v2")
# gym_hc_env = gymz.make("GymV26Environment-v0", env_id="HalfCheetah-v2")
# gym_hc_env = gymz.make("GymV26Environment-v0", env=gym_hc_env, apply_api_compatibility=True)
gymz_hc_env = gymz.make(id="HalfCheetah-v2")
carl_hc_env = CARLCartPole(obs_context_as_dict=False)

# Wrapper
gym_hc_env = EnvProjectCompatibility(gym_hc_env)
gymz_hc_env = EnvProjectCompatibility(gymz_hc_env)
carl_hc_env = EnvProjectCompatibility(carl_hc_env)

gymz_hc_env = gymz.wrappers.AutoResetWrapper(gymz_hc_env)
carl_hc_env = gymz.wrappers.AutoResetWrapper(carl_hc_env)

gymz_hc_env = EnvVectorResponse(gymz_hc_env)
carl_hc_env = EnvVectorResponse(carl_hc_env)

gymz_hc_env = RMVNormalizeVecObservation(gymz_hc_env)
carl_hc_env = RMVNormalizeVecObservation(carl_hc_env)

gymz_hc_env = gymz.wrappers.StepAPICompatibility(gymz_hc_env)
carl_hc_env = gymz.wrappers.StepAPICompatibility(carl_hc_env)

# assert (
#     gym_hc_env.observation_space.shape
#     == gymz_hc_env.observation_space.shape
#     == carl_hc_env.observation_space["obs"].shape
# ), "Heterogeneous environments"

# Reset the environments
gym_obs = gym_hc_env.reset()
gymz_obs, _ = gymz_hc_env.reset()
carl_obs, _ = carl_hc_env.reset()

# Take random action
gym_obs, gym_reward, gym_done, info = gym_hc_env.step(
    gym_hc_env.action_space.sample()
)
gymz_obs, gymz_reward, gymz_terminated, truncated, info = gymz_hc_env.step(
    gymz_hc_env.action_space.sample()
)
carl_obs, carl_reward, carl_terminated, truncated, info = carl_hc_env.step(
    carl_hc_env.action_space.sample()
)

print(
    f"Observation shapes: Gym={gym_obs.shape} Gymnasium={gymz_obs.shape} carl={carl_obs['obs'].shape}"
)
print(
    f"Reward shapes: Gym={gym_reward.shape} Gymnasium={gymz_reward.shape} carl={carl_reward.shape}"
)

print(
    f"Terminated shapes: Gym={None} Gymnasium={gymz_terminated.shape} carl={carl_terminated.shape}"
)

#* Second Half

gym_hc_env = gym.make("HalfCheetah-v2")
# gym_hc_env = gymz.make("GymV26Environment-v0", env_id="HalfCheetah-v2")
# gym_hc_env = gymz.make("GymV26Environment-v0", env=gym_hc_env, apply_api_compatibility=True)
gymz_hc_env = gymz.make(id="HalfCheetah-v2")
carl_hc_env = CARLCartPole(obs_context_as_dict=False)

# Vectorizing Environments

vec_gym_hc_env = gym.vector.SyncVectorEnv([lambda: gym_hc_env for _ in range(2)])
vec_gymz_hc_env = gymz.vector.SyncVectorEnv([lambda: gymz_hc_env for _ in range(2)])
vec_carl_hc_env = gymz.vector.SyncVectorEnv([lambda: carl_hc_env for _ in range(2)])

# Vector Wrappers

vec_gym_hc_env = VecEnvProjectCompatibility(vec_gym_hc_env)
vec_gymz_hc_env = VecEnvProjectCompatibility(vec_gymz_hc_env)
vec_carl_hc_env = VecEnvProjectCompatibility(vec_carl_hc_env)

# assert (
#     vec_gym_hc_env.single_observation_space.shape
#     == vec_gymz_hc_env.single_observation_space.shape
#     == vec_carl_hc_env.single_observation_space["obs"].shape
# ), "Heterogeneous environments"


# Reset the vectorized environments
gym_vec_obs, _ = vec_gym_hc_env.reset()
gymz_vec_obs, _ = vec_gymz_hc_env.reset()
carl_vec_obs, _ = vec_carl_hc_env.reset()

# Take random action
gym_vec_obs, gym_vec_reward, gym_vec_done, info = vec_gym_hc_env.step(
    vec_gym_hc_env.action_space.sample()
)
gymz_vec_obs, gymz_vec_reward, gymz_vec_terminated, truncated, info = vec_gymz_hc_env.step(
    vec_gymz_hc_env.action_space.sample()
)
carl_vec_obs, carl_vec_reward, carl_vec_terminated, truncated, info = vec_carl_hc_env.step(
    vec_carl_hc_env.action_space.sample()
)

print(
    f"Vectorized Observation shapes: Gym={gym_vec_obs.shape} Gymnasium={gymz_vec_obs.shape} carl={carl_vec_obs['obs'].shape}"
)
print(
    f"Reward shapes: Gym={gym_vec_reward.shape} Gymnasium={gymz_vec_reward.shape} carl={carl_vec_reward.shape}"
)

print(
    f"Terminated shapes: Gym={None} Gymnasium={gymz_vec_terminated.shape} carl={carl_vec_terminated.shape}"
)