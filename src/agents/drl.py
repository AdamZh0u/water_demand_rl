import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO

class LeakDetector(nn.Module):
    def __init__(self, input_dim):
        super(LeakDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)