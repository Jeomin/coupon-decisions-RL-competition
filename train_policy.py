from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from virtual_env import get_env_instance


if __name__ == '__main__':
    # *****改了env的路径*****
    env = get_env_instance('./user_states_by_day.npy', './venv.pkl')
    model = PPO("MlpPolicy", env, n_steps=840, batch_size=420, verbose=1, tensorboard_log='./data/logs')
    checkpoint_callback = CheckpointCallback(save_freq=8e4, save_path='./data/model_checkpoints')
    model.learn(total_timesteps=int(8e6), callback=[checkpoint_callback])


