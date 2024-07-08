# Water Demand RL Project

This project is focused on developing a deep reinforcement learning (DRL) agent for water leakage detection and response. Below is the description of the project structure:

## Project Structure

```

├── README.md # Project description and structure
├── data # Directory to store data files
├── docker
│ ├── Dockerfile # Docker configuration for environment setup
│ ├── requirements.txt # Python dependencies for the project
│ └── vscode-extensions # VSCode extensions configuration for the project
├── notebooks
│ ├── 00_simulation.ipynb # Notebook for simulating water demand data
│ ├── 01_env.ipynb # Notebook for defining the environment
│ └── 02_agents.ipynb # Notebook for developing and training the agents
├── src # Source code directory
│ ├── init.py # Init file for the src package
│ ├── const.py # Constants used across the project
│ ├── agents # Directory for agent implementations
│ │ ├── agent_dqn.py # DQN agent implementation
│ │ ├── agent_greedy.py # Greedy agent implementation
│ │ ├── agent_nerdy.py # Nerdy agent implementation
│ │ ├── agent_ppo.py # PPO agent implementation
│ │ └── agent_rnndqn.py # RNN-DQN agent implementation
│ ├── env # Directory for environment implementation
│ │ └── env_basic.py # Basic environment implementation
│ ├── simulation
│ │ └── water_demands.py # Simulation of water demand data
│ └── utils # Utility functions
│ │ └── plot_utils.py # Utility functions for plotting data
└── water_demand_rl.code-workspace # VSCode workspace configuration file
```

## Additional Information

- All training processes were traced using wandb and can be found at [wandb.ai](https://wandb.ai/adamzh0u/water_demand_rl/table).
- The code used to reproduce the results can be accessed from GitHub at [github.com](https://github.com/AdamZh0u/water_demand_rl).

Please refer to the individual notebooks and source files for detailed information on the implementation and usage of the various components of the project.
