import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import src.const as const
import src.env.env_basic as env_basic
import src.simulation.water_demands as wd


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * \
            self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(
            self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(
            states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def dump(self, path):
        torch.save(self.actor.state_dict(), path+'_actor.pth')
        torch.save(self.critic.state_dict(), path+'_critic.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path+'_actor.pth'))
        self.critic.load_state_dict(torch.load(path+'_critic.pth'))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def train_on_policy_agent(env, agent, num_episodes):
    for epiosed in tqdm(range(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [],
                            'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()[0]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info, _ = env.step(action)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
            run.log(
                {'index': info['step'], 'action': action, 'reward': reward}, commit=False)

        agent.update(transition_dict)

        run.log({'epiosed': epiosed, 'sum_reward': episode_return,}, commit=True)
    return agent

def test_agent(env, agent):
    state, info = env.reset()
    done = False
    sum_rewards = 0
    while not done:
        action = agent.take_action(state)
        state, reward, done, info, _ = env.step(action)

        # log to wandb
        sum_rewards += reward
        run.log({'tindex': info['step']-17520, 'taction': action,
                "treward": reward, 'tsum_rewards': sum_rewards})
                
if __name__ == '__main__':

    # ======================== hyperparameters ========================
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    obs_len = 10

    seed = 142
    num_leaks = 12
    Env = env_basic.EnvBasic

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = {'actor_lr': actor_lr, 'critic_lr': critic_lr, 'num_episodes': num_episodes,
            'hidden_dim': hidden_dim, 'gamma': gamma, 'lmbda': lmbda, 'epochs': epochs, 'eps': eps,
            'env': 'EnvBasic', 'obs_len': obs_len, 'seed': seed, 'num_leaks': num_leaks}

    # ======================== init ========================
    run = wandb.init(name=f'agent_ppo', project='cege_test',
                     config=args, monitor_gym=True)

    torch.manual_seed(seed)
    df = wd.load(seed, num_leaks)

    # ======================== train ========================
    env = Env(df, train=True, obs_len=obs_len)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    agent = train_on_policy_agent(env, agent, num_episodes)

    # dump
    agent.dump(str(const.PATH_DATA/'train/ppo'))

    # ======================== test ========================
    env = Env(df, train=False, obs_len=obs_len)
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    agent.load(str(const.PATH_DATA/'train/ppo'))

    test_agent(env, agent)
    run.finish()