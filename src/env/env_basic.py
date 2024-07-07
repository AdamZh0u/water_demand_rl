import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnvBasic(gym.Env):
    def __init__(self, data, train=True, window_size=384, obs_len=10):
        """Summary
        Args:
            data (pd.DataFrame): a pandas DataFrame with columns 'WaterDemandWithLeaks' and 'LeakageLabel' and length > window_size
            window_size (int, optional): the size of the window to be used for the observation space
            obs_len (int, optional): the length of the observation space

        Method:
            reset: reset the environment to the initial state
            step: take an action in the environment

        """
        super(EnvBasic, self).__init__()

        self.data = data
        self.window_size = window_size
        self.obs_len = obs_len
        self.current_step = self.window_size
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.obs_len,), dtype=np.float32)
        self.reward_range = (-10, 10)
        self.train = train

    def reset(self):
        if self.train:
            self.current_step = self.window_size
        else:
            self.current_step = 17520
        return self._next_observation(), self._get_info()

    def step(self, action):
        if self.train:
            terminated = self.current_step >= 17520
        else:
            terminated = self.current_step >= len(self.data) - 1

        # instant reward
        reward = self._calculate_reward(action)
        info = self._get_info()
        self.current_step += 1
        obs = self._next_observation()
        return obs, reward, terminated, info, {}

    def _next_observation(self):
        obs = self.data['WaterDemandWithLeaks'].values[
            self.current_step - self.obs_len:self.current_step]
        return obs

    def _get_info(self):
        return {'step': self.current_step}

    def _calculate_reward(self, action):
        leak_present = self.data['LeakageLabel'].iloc[self.current_step]
        if action == 1:
            if leak_present:
                return 10  # 漏洞被发现
            else:
                return -5  # 错误的警报
        else:
            if leak_present:
                return -10  # 漏洞未被发现
            else:
                return 1  # 正常操作

class EnvRelativeDemand(EnvBasic):
    """_summary_
        low reward for normal operation

    Args:
        WaterLeakEnv (_type_): _description_
    """

    def __init__(self, data, train=True, window_size=384, obs_len=10):
        super(EnvRelativeDemand, self).__init__(data, train, window_size, obs_len)
        self.observation_space = spaces.Box(
            low=-1, high=60, shape=(self.obs_len,), dtype=np.float32)


    def _next_observation(self):
        obs = self.data['WaterDemandWithLeaks'].values[
            self.current_step - self.obs_len:self.current_step]
        obs[:5] = np.clip(np.abs(obs[-1]/obs[-6:-1]),0,40)
        return obs

    def _calculate_reward(self, action):
        leak_present = self.data['LeakageLabel'].iloc[self.current_step]
        if action == 1:
            if leak_present:
                return 10  # 漏洞被发现
            else:
                return -5  # 错误的警报
        else:
            if leak_present:
                return -10  # not detected
            else:
                return 0.1  # 正常操作

class EnvLowR(EnvBasic):
    """_summary_
        low reward for normal operation

    Args:
        WaterLeakEnv (_type_): _description_
    """

    def __init__(self, data, train=True, window_size=384, obs_len=10):
        super(EnvLowR, self).__init__(data, train, window_size, obs_len)

    def _calculate_reward(self, action):
        leak_present = self.data['LeakageLabel'].iloc[self.current_step]
        if action == 1:
            if leak_present:
                return 10  # 漏洞被发现
            else:
                return -5  # 错误的警报
        else:
            if leak_present:
                return -10  # not detected
            else:
                return 0.1  # 正常操作


class EnvLowP(EnvBasic):
    """_summary_
        low reward for normal operation

    Args:
        WaterLeakEnv (_type_): _description_
    """

    def __init__(self, data, train=True, window_size=384, obs_len=10):
        super(EnvCost, self).__init__(data, train, window_size, obs_len)

    def _calculate_reward(self, action):
        leak_present = self.data['LeakageLabel'].iloc[self.current_step]
        if action == 1:
            if leak_present:
                return 10  # 漏洞被发现
            else:
                return -5  # 错误的警报
        else:
            if leak_present:
                return -5 # 漏洞未被发现
            else:
                return -0.1  # 正常操作
