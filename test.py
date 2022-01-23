import importlib
from stable_baselines3 import PPO
import sys
sys.path.append("")
import baseline.virtual_env
from baseline.virtual_env import get_env_instance

importlib.reload(baseline.virtual_env)

env = get_env_instance("user_states_by_day.npy", "venv.pkl")
policy = PPO.load("rl_model.zip")
validation_length = 14

total_gmv = 0.0
total_cost = 0.0
obs = env.reset()
for day_index in range(validation_length):
    coupon_action, _ = policy.predict(obs, deterministic=True) # Some randomness will be added to action if deterministic=False
    obs, reward, done, info = env.step(coupon_action)
    if reward != 0:
        info["Reward"] = reward
    print(f"Day {day_index+1}: {info}")