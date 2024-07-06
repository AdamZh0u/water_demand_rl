
class AgentNeverRepair:
    def __init__(self, env):
        self.env = env

    def run(self):
        obs = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = 0  # 从不派遣维修队
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
        return total_reward

class AgentAlwaysRepair:
    def __init__(self, env):
        self.env = env

    def run(self):
        obs = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = 1  # 总是派遣维修队
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
        return total_reward
