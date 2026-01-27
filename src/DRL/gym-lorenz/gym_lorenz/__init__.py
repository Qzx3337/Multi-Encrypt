from gym.envs.registration import register

register(
    id='lorenz_transient-v0',
    entry_point='gym_lorenz.envs:lorenzEnv_transient',
    max_episode_steps = 4000,
    reward_threshold  = 1e50
)
