import pandas as pd
import torch
import numpy as np
import pickle as pk
from typing import List
from gym import Env
from gym.utils.seeding import np_random
from gym.spaces import Box
from math import ceil, floor
from Cluster import ClusterTry


def states_to_obs(states: np.ndarray, day_total_order_num: int = 0, day_roi: float = 0.0):
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
    # 用户状态要不要加total_gmv，total_cost这些？
    # 聚类算法将1000个用户聚类，直接降维为均值，标准差，最大值，最小值合理吗
    states_clustered = ClusterTry(states)
    day_total_order_num, day_roi = np.array([day_total_order_num]), np.array([day_roi])
    return np.concatenate([states_clustered, day_total_order_num, day_roi], 0)
    # assert len(states.shape) == 2
    # mean_obs = np.mean(states, axis=0)
    # std_obs = np.std(states, axis=0)
    # max_obs = np.max(states, axis=0)
    # min_obs = np.min(states, axis=0)
    # day_total_order_num, day_roi = np.array([day_total_order_num]), np.array([day_roi])
    # return np.concatenate([mean_obs, std_obs, max_obs, min_obs, day_total_order_num, day_roi], 0)


def get_next_state_by_user_action(states: np.ndarray, day_order_num: np.ndarray, day_avg_fee: np.ndarray, action_1: np.ndarray):
    # *****加入新的使用率state,为此next_state除了user action(action2)还需要传入action1促销动作*****
    # *****状态经过了归一化处理*****
    next_states = np.empty(states.shape)

    # *****状态归一化所需的最大值*****
    MAX_STATES = np.array([120, 6, 100, 1, 1])

    size_array = np.array([[(x[0] * MAX_STATES[0]) / (x[1] * MAX_STATES[1]) if x[1] > 0 else 0] for i, x in
                           enumerate(states)])
    next_states[:, [0]] = states[:, [0]] + day_order_num / MAX_STATES[0]
    next_states[:, [1]] = states[:, [1]] + 1 / ((size_array + 1) * MAX_STATES[1]) * (
                day_order_num - states[:, [1]] * MAX_STATES[1]) * (day_order_num > 0.0).astype(np.float32)
    next_states[:, [2]] = states[:, [2]] + 1 / ((size_array + 1) * MAX_STATES[2]) * (
                day_avg_fee - states[:, [2]] * MAX_STATES[2]) * (day_avg_fee > 0.0).astype(np.float32)
    next_states[:, [3]] = np.minimum((next_states[:, [0]] * MAX_STATES[0]) / (
                (states[:, [0]] * MAX_STATES[0]) / np.maximum(states[:, [3]], 0.01) + action_1[:, [0]]),
                                     1) if ((states[:, [0]] * MAX_STATES[0]) / np.maximum(states[:, [3]], 0.01) + action_1[:, [0]]).all() != 0 else 1
    next_states[:, [4]] = states[:, [4]] + (action_1[:, [1]] - states[:, [4]]) * (
                action_1[:, [0]] > 0.0).astype(np.float32) * (day_order_num > 0.0).astype(np.float32) / (size_array + 1)
    return next_states


class VirtualMarketEnv(Env):
    """A very simple example of virtual marketing environment
    """

    MAX_ENV_STEP = 14  # Number of test days in the current phase
    # 直接用离散还是先用连续算出来再舍入效果好？
    # *****尝试使用连续动作空间*****
    # DISCOUNT_COUPON_LIST = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    # 这个阈值能不能用？
    ROI_THRESHOLD = 7
    # In real validation environment, if we do not send any coupons in 14 days, we can get this gmv value
    # 这两个阈值有啥好用的？不用会不会更好？
    ZERO_GMV = 81840.0763705537

    def __init__(self,
                 initial_user_states: np.ndarray,
                 venv_model: object,
                 # 维度[6,8]6是最高六单优惠券，8是优惠券折扣有8档
                 # act_num_size: List[int] = [6],
                 # *****5+5+5+5+5+1+1, 降到了5维度*****
                 obs_size: int = 22,
                 # *****计算设备改用gpu*****
                 device: torch.device = torch.device('cuda'),
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
        self._set_action_space()
        self._set_observation_space(obs_size)
        self.total_cost, self.total_gmv = None, None

    def seed(self, seed_number):
        # 此np_random非np的random，from gym.utils.seeding import np_random看清好伐
        return np_random(seed_number)[0]

    def _set_action_space(self):  # discrete platform action
        # *****动作空间改为连续*****
        self.action_space = Box(low=np.array([0, 0.6]), high=np.array([6, 0.9]), dtype=np.float32)

    def _set_observation_space(self, obs_size, low=0, high=100):
        # *****观测空间最大值全设置100学的时候小数容易学大，归一化high=1*****
        self.observation_space = Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

    def step(self, action):
        # *****动作空间改了最后需要格式化取整,不要，删去格式化处理，只在最后输出策略的时候格式化就行，现在格式化环境状态学习仍是离散的*****
        coupon_num, coupon_discount = action[0], action[1]
        p_action = np.array([[coupon_num, coupon_discount] for _ in range(self.states.shape[0])])
        info_dict = self.venv_model.infer_one_step({'state': self.states, 'action_1': p_action})

        day_user_actions = info_dict['action_2']
        # *****用户订单数检查，order为0时检查设置fee为0*****
        # action_df = pd.DataFrame(day_user_actions, columns=['day_order_num', 'day_avg_fee'])
        # action_df['day_avg_fee'] = action_df.apply(lambda x: 0 if x['day_order_num'] == 0 else x['day_avg_fee'], axis=1)
        # day_user_actions = action_df.to_numpy()
        # *****目前ordernum还是格式化处理了的
        day_order_num, day_avg_fee = np.round(day_user_actions[:, [0]]), day_user_actions[:, [1]]

        # 这两句改不改
        day_order_num = np.clip(day_order_num, 0, 6)
        day_avg_fee = np.clip(day_avg_fee, 30, 100)
        # *****调用_get_next_state_by_user_action新传入了p_action*****
        self.states = get_next_state_by_user_action(self.states, day_order_num, day_avg_fee, p_action)

        # *****加了info*****
        info = {
            "CouponNum": coupon_num,
            "CouponDiscount": coupon_discount,
            "UserAvgOrders": day_order_num.mean(),
            "UserAvgFee": day_avg_fee.mean()
        }

        day_coupon_used_num = np.min(np.concatenate([day_order_num, p_action[:, [0]]], -1), -1, keepdims=True)
        cost_array = (1 - coupon_discount) * day_coupon_used_num * day_avg_fee
        gmv_array = day_avg_fee * day_order_num - cost_array

        day_total_gmv = np.sum(gmv_array)  # 每日gmv与cost
        day_total_cost = np.sum(cost_array)
        self.total_gmv += day_total_gmv
        self.total_cost += day_total_cost
        # if (self.current_env_step+1) < VirtualMarketEnv.MAX_ENV_STEP:  # 稀疏奖励有用吗？
        #     reward = 0
        # else:
        #     avg_roi = self.total_gmv / max(self.total_cost, 1)
        #     if avg_roi >= VirtualMarketEnv.ROI_THRESHOLD:
        #         reward = self.total_gmv / VirtualMarketEnv.ZERO_GMV
        #     else:
        #         reward = avg_roi - VirtualMarketEnv.ROI_THRESHOLD
        #     info["TotalGMV"] = self.total_gmv
        #     info["TotalROI"] = avg_roi

        # *****改reward*****
        reward = 0
        if self.current_env_step + 1 < VirtualMarketEnv.MAX_ENV_STEP:
            reward = day_total_gmv / (VirtualMarketEnv.ZERO_GMV / VirtualMarketEnv.MAX_ENV_STEP)
            # + min((day_total_gmv / max(day_total_cost, 1) - VirtualMarketEnv.ROI_THRESHOLD), 3)
        elif self.current_env_step + 1 == VirtualMarketEnv.MAX_ENV_STEP:
            avg_roi = self.total_gmv / max(self.total_cost, 1)
            if avg_roi <= self.ROI_THRESHOLD:
                reward = self.total_gmv / VirtualMarketEnv.ZERO_GMV - VirtualMarketEnv.MAX_ENV_STEP
            else:
                reward = self.total_gmv / VirtualMarketEnv.ZERO_GMV
            # *****加入一个GMV和reward的输出看一下每次迭代变化*****
            info["TotalGMV"] = self.total_gmv
            info["TotalROI"] = avg_roi

        self.done = ((self.current_env_step + 1) == VirtualMarketEnv.MAX_ENV_STEP)
        self.current_env_step += 1
        day_total_order_num = int(np.sum(day_order_num))
        day_roi = day_total_gmv / max(day_total_cost, 1)
        return states_to_obs(self.states, day_total_order_num, day_roi), reward, self.done, info

    def reset(self):
        """Reset the initial states of all users
        Return:
            The group state
        """
        # 这个reset让人看不懂了，归一化之后要不要改,(不用，但是还是怪了点)
        self.states = self.initial_user_states[self.rng.randint(0, self.initial_user_states.shape[0])]
        self.done = False
        self.current_env_step = 0
        self.total_cost, self.total_gmv = 0.0, 0.0
        return states_to_obs(self.states)


def get_env_instance(states_path, venv_model_path, device=torch.device('cuda')):
    # *****device改用cuda*****
    initial_states = np.load(states_path)
    with open(venv_model_path, 'rb') as f:
        venv_model = pk.load(f, encoding='utf-8')
    venv_model.to(device)

    return VirtualMarketEnv(initial_states, venv_model, device=device)
