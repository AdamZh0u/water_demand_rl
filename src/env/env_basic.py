import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

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

class EnvComplexR(EnvBasic):
    """_summary_
        low reward for normal operation

    Args:
        WaterLeakEnv (_type_): _description_
    """

    def __init__(self, data, train=True, window_size=384, obs_len=10):
        super(EnvComplexR, self).__init__(data, train, window_size, obs_len)
        self.processed_data = self._gen_cost()


    def _gen_cost(self):
        dft2 = self.data.copy()
        dft2['Timestamp'] = pd.to_datetime(dft2['Timestamp'])
        dft2['hour'] = dft2['Timestamp'].dt.hour

        traffic_flow = np.array([20,15,10,10,10,20,
                                30,50,70,100, 90,80, 
                                60,55,50,50,60,90,
                                95,70,60,40,30,20
                                ])
        dft2['TrafficFlow'] = traffic_flow[dft2['hour'].values]

        # downtime_cost : rolling following 1 hour
        dft2['a_downtime_award'] = -dft2['WaterDemandWithLeaks'].rolling(window=4).sum().shift(-3)
        dft2['a_congestion_award'] = -dft2['TrafficFlow']*0.15
        dft2['a_environmental_award'] = -np.log1p(dft2['LeakageLabel'].rolling(window=12).sum())
        dft2['a_fix_award'] =  np.log1p(dft2['LeakageLabel'].rolling(window=12).sum().shift(-11).sum())
        dft2['a_total_award'] = 7 + dft2['a_downtime_award'] + dft2['a_congestion_award'] + dft2['a_environmental_award'] + dft2['a_fix_award']

        dft2['n_leakage_award'] = -dft2['WaterDemandWithLeaks'].rolling(window=12).sum().shift(-11)
        dft2['n_environmental_award'] = -np.log1p(dft2['LeakageLabel'].rolling(window=30*24*4).sum())
        dft2['n_total_award'] = dft2['n_leakage_award'] + dft2['n_environmental_award'] - 7
        
        # mean fill na
        dft2 = dft2.fillna(dft2.mean())
        return dft2[['LeakageLabel','a_total_award','n_total_award']]

    def _calculate_reward(self, action):
        leak_present = self.processed_data['LeakageLabel'].iloc[self.current_step]
        n_total_award = self.processed_data['n_total_award'].iloc[self.current_step].round(2)
        a_total_award = self.processed_data['a_total_award'].iloc[self.current_step].round(2)

        if action == 1:
            if leak_present:
                return a_total_award  # 漏洞被发现
            else:
                return -5  # error alarm
        else:
            if leak_present:
                return n_total_award  # not detected
            else:
                return 0.1  # 正常操作

if __name__ == '__main__':
    import src.simulation.water_demands as wd

    dft = wd.load(243, 12)

    env = EnvComplexR(dft)
    env.reset()