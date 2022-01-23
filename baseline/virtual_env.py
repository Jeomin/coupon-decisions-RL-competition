import torch
import numpy as np
import pickle as pk
from typing import List
from gym import Env
from gym.utils.seeding import np_random
from gym.spaces import Box, MultiDiscrete


class VirtualMarketEnv(Env):
    """A very simple example of virtual marketing environment
    """

    MAX_ENV_STEP = 14 # Number of test days in the current phase
    DISCOUNT_COUPON_LIST = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    ROI_THRESHOLD = 8.0
    # In real validation environment, if we do not send any coupons in 14 days, we can get this gmv value
    ZERO_GMV = 81840.0763705537

    def __init__(self,
                 initial_user_states: np.ndarray,
                 venv_model: object,
                 act_num_size: List[int] = [6, 8],
                 obs_size: int = 14,
                 device: torch.device = torch.device('cpu'),
                 seed_number: int = 0):
        """
        Args:
            initial_user_states: The initial states set from the user states of every day
            venv_model: The virtual environment model is trained by default with the revive algorithm package
            act_num_size: The size of the action for each dimension
            obs_size: The size of the community state
            device: Computation device
            seed_number: Random seed
        """
        self.rng = self.seed(seed_number)
        self.initial_user_states = initial_user_states
        self.venv_model = venv_model
        self.current_env_step = None
        self.states = None
        self.done = None
        self.device = device
        self._set_action_space(act_num_size)
        self._set_observation_space(obs_size)
        self.total_cost, self.total_gmv = None, None

    def _states_to_obs(self, states: np.ndarray, day_total_order_num: int=0, day_roi: float=0.0):
        """Reduce the two-dimensional sequence of states of all users to a state of a user community
            A naive approach is adopted: mean, standard deviation, maximum and minimum values are calculated separately for each dimension.
            Additionly, we add day_total_order_num and day_roi.
        Args:
            states(np.ndarray): A two-dimensional array containing individual states for each user
            day_total_order_num(int): The total order number of the users in one day
            day_roi(float): The day ROI of the users
        Return:
            The states of a user community (np.array)
        """
        assert len(states.shape) == 2
        mean_obs = np.mean(states, axis=0)
        std_obs = np.std(states, axis=0)
        max_obs = np.max(states, axis=0)
        min_obs = np.min(states, axis=0)
        day_total_order_num, day_roi = np.array([day_total_order_num]), np.array([day_roi])
        return np.concatenate([mean_obs, std_obs, max_obs, min_obs, day_total_order_num, day_roi], 0)

    def _get_next_state_by_user_action(self, day_order_num: np.ndarray, day_avg_fee: np.ndarray):
        next_states = np.empty(self.states.shape)
        size_array = np.array([[x[0] / x[1] if x[1] > 0 else 0] for i, x in enumerate(self.states)])
        next_states[:, [0]] = self.states[:, [0]] + day_order_num
        next_states[:, [1]] = self.states[:, [1]] + 1 / (size_array + 1) * (day_order_num - self.states[:, [1]]) * (day_order_num > 0.0).astype(np.float32)
        next_states[:, [2]] = self.states[:, [2]] + 1 / (size_array + 1) * (day_avg_fee - self.states[:, [2]]) * (day_avg_fee > 0.0).astype(np.float32)
        return next_states

    def seed(self, seed_number):
        return np_random(seed_number)[0]

    def _set_action_space(self, num_list=[6, 8]): # discrete platform action
        self.action_space = MultiDiscrete(num_list)

    def _set_observation_space(self, obs_size, low=0, high=100):
        self.observation_space = Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

    def step(self, action):
        coupon_num, coupon_discount = action[0], VirtualMarketEnv.DISCOUNT_COUPON_LIST[action[1]]
        p_action = np.array([[coupon_num, coupon_discount] for _ in range(self.states.shape[0])])
        info_dict = self.venv_model.infer_one_step({'state': self.states, 'action_1': p_action})
        day_user_actions = info_dict['action_2']
        day_order_num, day_avg_fee = day_user_actions[:, [0]], day_user_actions[:, [1]]
        day_order_num = np.clip(day_order_num, 0, 6)
        day_avg_fee = np.clip(day_avg_fee, 0, 100)
        self.states = self._get_next_state_by_user_action(day_order_num, day_avg_fee)

        day_coupon_used_num = np.min(np.concatenate([day_order_num, p_action[:, [0]]], -1), -1, keepdims=True)
        cost_array = (1 - coupon_discount) * day_coupon_used_num * day_avg_fee
        gmv_array = day_avg_fee * day_order_num - cost_array
        day_total_gmv = np.sum(gmv_array)
        day_total_cost = np.sum(cost_array)
        self.total_gmv += day_total_gmv
        self.total_cost += day_total_cost
        if (self.current_env_step+1) < VirtualMarketEnv.MAX_ENV_STEP:
            reward = 0
        else:
            avg_roi = self.total_gmv / self.total_cost
            if avg_roi >= VirtualMarketEnv.ROI_THRESHOLD:
                reward = self.total_gmv / VirtualMarketEnv.ZERO_GMV
            else:
                reward = avg_roi - VirtualMarketEnv.ROI_THRESHOLD

        self.done = ((self.current_env_step + 1) == VirtualMarketEnv.MAX_ENV_STEP)
        self.current_env_step += 1
        day_total_order_num = int(np.sum(day_order_num))
        day_roi = day_total_gmv / max(day_total_cost, 1)
        return self._states_to_obs(self.states, day_total_order_num, day_roi), reward, self.done, {}


    def reset(self):
        """Reset the initial states of all users
        Return:
            The group state
        """
        self.states = self.initial_user_states[self.rng.randint(0, self.initial_user_states.shape[0])]
        self.done = False
        self.current_env_step = 0
        self.total_cost, self.total_gmv = 0.0, 0.0
        return self._states_to_obs(self.states)


def get_env_instance(states_path, venv_model_path, device = torch.device('cpu')):

    initial_states = np.load(states_path)
    venv_model = pk.load(open(venv_model_path, 'rb'), encoding='utf-8')
    venv_model.to(device)

    return VirtualMarketEnv(initial_states, venv_model, device=device)
