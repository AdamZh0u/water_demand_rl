
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random
import collections
import src.const as const
import src.env.env_basic as env_basic
import src.simulation.water_demands as wd

class ReplayBuffer:
    ''' Experience Replay Pool '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' MLP Q Network'''

    def __init__(self, state_dim, hidden_dim=128, action_dim=2):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,64)
        self.fc3 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class AgentDQN:
    ''' Double DQN Agent'''

    def __init__(self, state_dim, hidden_dim=128, action_dim=2, learning_rate=2e-3, gamma=0.98,
                 epsilon=0.01, target_update=10, device='cpu'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state[np.newaxis,:], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def dump(self, path):  # 保存模型
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):  # 加载模型
        self.q_net.load_state_dict(torch.load(path))


def train_on_policy_agent(env, agent, num_episodes):

    for epiosed in tqdm(range(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [],
                            'next_states': [], 'rewards': [], 'dones': []}

        state = env.reset()[0]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info,_ = env.step(action)

            # run.log(
            #     {'index': info['step'], 'action': action, 'reward': reward})
    
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            state = next_state
            episode_return += reward
        
        agent.update(transition_dict)

        run.log({'epiosed': epiosed, 'epiosed_sum_reward': episode_return})
    return agent


def train_off_policy_agent(env, agent, num_episodes):
    replay_buffer = ReplayBuffer(buffer_size)
    for epiosed in tqdm(range(num_episodes)):
        episode_return = 0

        state = env.reset()[0]
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info,_ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            run.log(
                {'index': info['step'], 'action': action, 'reward': reward})
            
            state = next_state
            episode_return += reward
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)

        run.log({'epiosed': epiosed, 'sum_reward': episode_return})
    return agent


def test_agent(env, agent, track=True):
    state, info = env.reset()
    done = False
    sum_rewards = 0
    while not done:
        action = agent.take_action(state)
        state, reward, done, info, _ = env.step(action)

        # log to wandb
        sum_rewards += reward
        if track:
            run.log({'test_action': action,
                    "test_step_reward": reward, 'test_sum_reward': sum_rewards})
    return sum_rewards

if __name__ == '__main__':

    # ======================== hyperparameters ========================
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.6
    epsilon = 0.01
    n_layers = 2
    target_update = 10
    
    obs_len = 10
    seed = 242
    num_leaks = 12
    Env = env_basic.EnvComplexR

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = {'lr': lr, 
            'num_episodes': num_episodes, 
            'hidden_dim': hidden_dim,
            'gamma': gamma, 
            'epsilon': epsilon,
            'n_layers': n_layers,
            # 'target_update': target_update,
            'Env': 'EnvComplexR',
            'obs_len': obs_len,
            'seed': seed, 
            'num_leaks': num_leaks,
            'on_policy': True,}

    # ======================== init ========================
    run = wandb.init(project='water_demand_rl',
                     config=args, monitor_gym=True)
    run.name = f'agent_dqn_{run.id}'

    torch.manual_seed(seed)
    df = wd.load(seed, num_leaks)

    # ======================== train ========================

    env = Env(df, train=True, obs_len=obs_len)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = AgentDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    agent = train_on_policy_agent(env, agent, num_episodes)
    # agent = train_off_policy_agent(env, agent, num_episodes)

    # dump
    agent_file_path = const.PATH_DATA/f'train/dqn_{run.id}'
    agent.dump(str(agent_file_path))

    # ======================== test ========================
    env.train = False
    agent.load(str(agent_file_path))

    test_agent(env, agent)
    run.finish()
