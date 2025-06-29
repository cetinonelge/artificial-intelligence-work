import numpy as np
import matplotlib.pyplot as plt
from utils import plot_value_snapshots, plot_policy_snapshots
from tqdm import trange

class MazeEnvironment:
    def __init__(self):
        # 0=free, 1=obstacle, 2=trap, 3=goal (11×11 layout)
        self.layout = np.array([
            [0,0,0,0,0,0,0,1,1,1,1],
            [1,0,1,1,0,1,0,0,0,0,1],
            [1,0,1,1,1,1,0,1,1,0,1],
            [1,0,0,0,0,1,0,0,1,0,1],
            [1,1,0,1,0,1,0,1,1,0,0],
            [1,1,0,1,0,0,0,0,1,1,0],
            [1,0,0,1,1,0,1,0,0,0,0],
            [0,0,1,1,1,1,2,0,1,1,0],
            [0,0,0,0,0,0,0,1,0,0,0],
            [1,0,1,0,1,1,0,1,1,1,0],
            [1,1,1,0,2,0,1,0,0,3,0]
        ])
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.state_penalty = -1
        self.trap_penalty = -100
        self.goal_reward = 100
        # Actions: up, down, left, right
        self.actions = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
        # Stochastic outcome maps
        self.opposite = {0:1,1:0,2:3,3:2}
        self.perpendicular = {0:[2,3],1:[2,3],2:[0,1],3:[0,1]}

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        probs = [0.75,0.05,0.10,0.10]
        choices = [action,
                   self.opposite[action],
                   self.perpendicular[action][0],
                   self.perpendicular[action][1]]
        move = np.random.choice(choices, p=probs)
        dr,dc = self.actions[move]
        r,c = self.current_pos
        nr,nc = r+dr, c+dc
        if not (0<=nr<self.layout.shape[0] and 0<=nc<self.layout.shape[1]) or self.layout[nr,nc]==1:
            nr,nc = r,c
        self.current_pos = (nr,nc)
        cell = self.layout[nr,nc]
        if cell==2: return (nr,nc), self.trap_penalty, True
        if cell==3: return (nr,nc), self.goal_reward, True
        return (nr,nc), self.state_penalty, False

class MazeTD0(MazeEnvironment):
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.utility = np.zeros_like(self.layout, dtype=float)
        self.convergence = []
        self.record_eps = [1,50,100,1000,5000,episodes]
        self.snapshots = {}

    def get_valid_actions(self, state):
        r,c = state
        valid = []
        for a,(dr,dc) in self.actions.items():
            nr,nc = r+dr, c+dc
            if 0<=nr<self.layout.shape[0] and 0<=nc<self.layout.shape[1] and self.layout[nr,nc]!=1:
                valid.append(a)
        return valid or list(self.actions.keys())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.get_valid_actions(state))
        best_val = -np.inf
        best_actions = []
        for a in self.get_valid_actions(state):
            r,c = state
            dr,dc = self.actions[a]
            nr,nc = r+dr,c+dc
            if not (0<=nr<self.layout.shape[0] and 0<=nc<self.layout.shape[1]) or self.layout[nr,nc]==1:
                nr,nc = r,c
            val = self.utility[nr,nc]
            if val>best_val:
                best_val = val
                best_actions = [a]
            elif val==best_val:
                best_actions.append(a)
        return np.random.choice(best_actions)

    def update_value(self, s, reward, s2):
        old = self.utility[s]
        nxt = self.utility[s2]
        self.utility[s] = old + self.alpha*(reward + self.gamma*nxt - old)

    def run_episodes(self, max_steps=None):
        for ep in trange(1, self.episodes+1,
                         desc=f"α={self.alpha} γ={self.gamma} ε={self.epsilon}"):
            s = self.reset()
            done = False
            delta = 0.0
            steps = 0
            while not done and (max_steps is None or steps < max_steps):
                a = self.choose_action(s)
                s2,r,done = self.step(a)
                old = self.utility[s]
                self.update_value(s, r, s2)
                delta += abs(self.utility[s]-old)
                s = s2
                steps += 1
            self.convergence.append(delta)
            if ep in self.record_eps:
                self.snapshots[ep] = self.utility.copy()
        return self.snapshots, self.convergence


def save_convergence(conv, filename, window=100):
    import numpy as _np
    arr = _np.array(conv)
    if len(arr)>=window:
        ma = _np.convolve(arr, _np.ones(window)/window, mode='valid')
    else:
        ma = arr.copy()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1,len(arr)+1), arr, alpha=0.3, label='raw ΔU')
    ax.plot(range(window, window+len(ma)), ma, color='red', label=f'{window}-episode MA')
    ax.set(xlabel="Episode", ylabel="Sum of |ΔU|", title="Convergence (smoothed)")
    ax.legend()
    fig.savefig(filename)
    plt.close(fig)

if __name__ == "__main__":
    # default parameters
    default_alpha = 0.1
    default_gamma = 0.95
    default_epsilon = 0.2
    episodes = 10000

    experiments = []
    # vary alpha
    for a in [0.001, 0.01, 0.1, 0.5, 1.0]:
        experiments.append(("alpha", a, default_gamma, default_epsilon, None))
    # vary gamma (cap steps to 500)
    for g in [0.10, 0.25, 0.50, 0.75, 0.95]:
        experiments.append(("gamma", default_alpha, g, default_epsilon, 500))
    # vary epsilon
    for e in [0.0, 0.2, 0.5, 0.8, 1.0]:
        experiments.append(("epsilon", default_alpha, default_gamma, e, None))

    for name, alpha, gamma, epsilon, cap in experiments:
        prefix = f"{name}_{alpha if name=='alpha' else (gamma if name=='gamma' else epsilon)}"
        print(f"Running {prefix}, cap={cap}...")
        agent = MazeTD0(alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
        snapshots, conv = agent.run_episodes(max_steps=cap)
        save_convergence(conv, f"{prefix}_convergence.png")
        plot_value_snapshots(snapshots, agent.layout, filename=f"{prefix}_values.png")
        plot_policy_snapshots(snapshots, agent.layout, filename=f"{prefix}_policies.png")
