import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
from collections import deque
from datetime import datetime           #  ← NEW
from utils import plot_learning_curves, plot_solved_episodes
from tqdm import trange

# ───────────────────────────────────────────────
# helpers ───────────────────────────────────────
# ───────────────────────────────────────────────

def _val_to_str(val):
    """
    Convert *any* hyper-parameter value into a clean, filename–safe string.

    • Floats:  0.001  →  0p001
    • Lists:   [128,128]  →  128x128
    • Others:  str(val)
    """
    if isinstance(val, (list, tuple)):
        val_str = "x".join(map(str, val))
    else:
        val_str = str(val)

    # replace characters that do not play well with filenames
    val_str = (
        val_str.replace(" ", "")
               .replace("[", "")
               .replace("]", "")
               .replace(",", "x")
               .replace(".", "p")
    )
    return val_str


def _make_results_name(param_name, val):
    """
    Build a UNIQUE results filename.

    Adds a UTC timestamp so running the same sweep twice won’t overwrite
    the previous JSON.
    """
    safe_val = _val_to_str(val)
    ts      = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"results_{param_name}_{safe_val}_{ts}.json"


# ───────────────────────────────────────────────
# neural net / replay buffer ────────────────────
# ───────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, s, a, r, s2, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s2, d))
        else:
            self.buffer[self.pos] = (s, a, r, s2, d)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ───────────────────────────────────────────────
# DQN agent ─────────────────────────────────────
# ───────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        memory_size        = 50_000,
        batch_size         = 64,
        gamma              = 0.99,
        lr                 = 1e-3,
        epsilon_start      = 1.0,
        epsilon_min        = 0.01,
        epsilon_decay      = 0.995,
        target_update_freq = 10,
        hidden_dims        = [128, 128],
    ):
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size
        self.epsilon    = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayMemory(memory_size)

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_update_freq = target_update_freq
        self.solved_score       = 200.0

    # ε-greedy
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    # one SGD step
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch       = self.memory.sample(self.batch_size)
        states_np   = np.array([b[0] for b in batch], dtype=np.float32)
        next_np     = np.array([b[3] for b in batch], dtype=np.float32)

        states      = torch.from_numpy(states_np).to(device)
        next_states = torch.from_numpy(next_np).to(device)

        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64,
                               device=device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32,
                               device=device).unsqueeze(1)
        dones   = torch.tensor([b[4] for b in batch], dtype=torch.float32,
                               device=device).unsqueeze(1)

        q_curr  = self.policy_net(states).gather(1, actions)
        q_next  = self.target_net(next_states).max(1)[0].unsqueeze(1)
        q_targ  = rewards + (1 - dones) * self.gamma * q_next

        loss = nn.MSELoss()(q_curr, q_targ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ───────────────────────────────────────────────
# experiment runner ─────────────────────────────
# ───────────────────────────────────────────────

def run_sweep(param_name, param_values, agent_kwargs,
              num_episodes=5000):
    result_paths = []

    for val in param_values:
        # ---------------------------------------------------
        # prepare hyper-parameters for this agent
        # ---------------------------------------------------
        local_kwargs          = agent_kwargs.copy()
        local_kwargs[param_name] = val
        agent = DQNAgent(state_dim, action_dim, **local_kwargs)

        rewards, avgs = [], []
        window        = deque(maxlen=100)
        solved_ep     = None

        for ep in trange(1, num_episodes + 1,
                         desc=f"{param_name}={val}"):
            state, _ = env.reset()
            total_r  = 0
            done     = False

            while not done:
                a = agent.get_action(state)
                s2, r, term, trunc, _ = env.step(a)
                done = term or trunc
                agent.memory.push(state, a, r, s2, done)
                agent.train_step()
                state   = s2
                total_r += r

            agent.decay_epsilon()
            if ep % agent.target_update_freq == 0:
                agent.update_target()

            rewards.append(total_r)
            window.append(total_r)
            avgs.append(sum(window) / len(window))

            if solved_ep is None and avgs[-1] >= agent.solved_score:
                solved_ep = ep

        # ---------------------------------------------------
        # record results
        # ---------------------------------------------------
        solved_ep_val = solved_ep
        fname         = _make_results_name(param_name, val)
        with open(fname, "w") as f:
            json.dump(
                {
                    "episode_rewards": rewards,
                    "average_scores":  avgs,
                    "hyperparameters": local_kwargs,
                    "solved_episode":  solved_ep_val,
                },
                f,
            )
        result_paths.append(fname)

    # -------------------------------------------------------
    # plots for this sweep
    # -------------------------------------------------------
    plot_learning_curves(result_paths,
                         output_file=f"{param_name}_learning_curves.png")
    plot_solved_episodes(result_paths,
                         output_file=f"{param_name}_solved_episodes.png")


# ───────────────────────────────────────────────
# main experiment script ────────────────────────
# ───────────────────────────────────────────────

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # hyper-parameter grids
    lr_list          = [1e-4, 1e-3, 5e-3]
    gamma_list       = [0.98, 0.99, 0.999]
    eps_decay_list   = [0.98, 0.99, 0.995]
    target_freq_list = [1, 10, 50]
    archs            = [[128], [64, 64], [128, 128],
                        [128, 128, 128], [256, 256]]

    # 1) learning-rate sweep
    run_sweep(
        "lr",
        lr_list,
        {
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "target_update_freq": 10,
            "hidden_dims": [128, 128],
        },
    )

    # 2) discount-factor sweep
    run_sweep(
        "gamma",
        gamma_list,
        {
            "lr": 1e-3,
            "epsilon_decay": 0.995,
            "target_update_freq": 10,
            "hidden_dims": [128, 128],
        },
    )

    # 3) ε-decay sweep
    run_sweep(
        "epsilon_decay",
        eps_decay_list,
        {
            "lr": 1e-3,
            "gamma": 0.99,
            "target_update_freq": 10,
            "hidden_dims": [128, 128],
        },
    )

    # 4) target-network-frequency sweep
    run_sweep(
        "target_update_freq",
        target_freq_list,
        {
            "lr": 1e-3,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "hidden_dims": [128, 128],
        },
    )

    # 5) network-architecture sweep
    run_sweep(
        "hidden_dims",
        archs,
        {
            "lr": 1e-3,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "target_update_freq": 10,
        },
    )

    env.close()
