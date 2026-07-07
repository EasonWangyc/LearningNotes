# Reinforcement Learning (强化学习)

强化学习研究的是：智能体（Agent）如何通过和环境（Environment）交互，逐步学习一种策略（Policy），让长期累积奖励尽可能大。

和监督学习不同，强化学习没有直接给出“每个状态下正确动作是什么”。智能体只能通过试错获得奖励，再从奖励中反推哪些行为更好。

```text
Agent
  -> action
Environment
  -> next state, reward, done
Agent
  -> next action
...
```

## 交互过程

一次强化学习交互通常包含：

| 符号 | 含义 |
|---|---|
| $s_t$ | 时刻 $t$ 的状态 |
| $a_t$ | 智能体在状态 $s_t$ 下采取的动作 |
| $r_t$ | 执行动作后得到的奖励 |
| $s_{t+1}$ | 环境转移到的新状态 |
| $done$ | episode 是否结束 |

```python
class CounterEnv:
    def __init__(self, target=3):
        self.target = target
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 1:
            self.state += 1
        else:
            self.state -= 1

        done = self.state >= self.target
        reward = 1.0 if done else -0.1
        return self.state, reward, done

env = CounterEnv()
state = env.reset()

for _ in range(5):
    action = 1
    next_state, reward, done = env.step(action)
    print(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
```

## Return

强化学习关注的不是单步奖励，而是从当前时刻开始的长期累计奖励。这个量叫 Return：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...
$$

其中 $\gamma$ 是折扣因子，用来控制未来奖励的重要程度。

- $\gamma=0$：只关心当前奖励。
- $\gamma$ 越接近 1：越重视长期回报。

```python
def discounted_return(rewards, gamma=0.9):
    total = 0.0
    power = 1.0
    for reward in rewards:
        total += power * reward
        power *= gamma
    return total

rewards = [-0.1, -0.1, -0.1, 1.0]

print("gamma=0.0:", discounted_return(rewards, gamma=0.0))
print("gamma=0.5:", discounted_return(rewards, gamma=0.5))
print("gamma=0.9:", discounted_return(rewards, gamma=0.9))
```

### 从后往前计算 Return

实际代码中，常常从 episode 末尾往前递推：

$$
G_t = r_t + \gamma G_{t+1}
$$

```python
def compute_returns(rewards, gamma=0.9):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return list(reversed(returns))

print(compute_returns([-0.1, -0.1, -0.1, 1.0], gamma=0.9))
```

# 马尔可夫决策过程

强化学习通常用马尔可夫决策过程（MDP）描述环境。

一个 MDP 包含：

$$
(S, A, P, R, \gamma)
$$

| 组成 | 含义 |
|---|---|
| $S$ | 状态集合 |
| $A$ | 动作集合 |
| $P(s'|s,a)$ | 状态转移概率 |
| $R(s,a)$ | 奖励函数 |
| $\gamma$ | 折扣因子 |

## 马尔可夫性

马尔可夫性指的是：未来只依赖当前状态和动作，不依赖更早之前的历史。

$$
P(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},...) = P(s_{t+1}|s_t,a_t)
$$

这并不是说真实世界一定完全满足这个假设，而是说建模时希望状态已经包含了做决策所需的充分信息。

## 策略

策略 $\pi$ 描述智能体如何根据状态选择动作。

确定性策略：

$$
a = \pi(s)
$$

随机策略：

$$
\pi(a|s) = P(a_t=a|s_t=s)
$$

```python
import numpy as np

action_probs = np.array([0.1, 0.7, 0.2])
actions = np.array(["left", "right", "stay"])

for _ in range(5):
    action = np.random.choice(actions, p=action_probs)
    print(action)
```

# 价值函数

价值函数用来估计“某个状态或动作长期来看有多好”。

## 状态价值函数

状态价值函数 $V^\pi(s)$ 表示：在状态 $s$ 出发，并按照策略 $\pi$ 行动时，期望 Return 是多少。

$$
V^\pi(s) = \mathbb{E}_\pi[G_t|s_t=s]
$$

## 动作价值函数

动作价值函数 $Q^\pi(s,a)$ 表示：在状态 $s$ 下先执行动作 $a$，之后按照策略 $\pi$ 行动时，期望 Return 是多少。

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t|s_t=s,a_t=a]
$$

## Advantage

Advantage 表示某个动作比该状态下的平均水平好多少：

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

```python
import numpy as np

q_values = np.array([1.0, 1.5, 0.2])
v_value = q_values.mean()
advantage = q_values - v_value

print("Q:", q_values)
print("V:", round(v_value, 3))
print("A:", advantage.round(3))
```

# Bellman 方程

Bellman 方程把“当前价值”和“下一步价值”联系起来。

对于某个策略 $\pi$：

$$
V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)[R(s,a,s')+\gamma V^\pi(s')]
$$

最优动作价值函数满足：

$$
Q^*(s,a)=\sum_{s'}P(s'|s,a)[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')]
$$

## 价值迭代

如果环境转移概率已知，可以通过价值迭代求解最优价值函数。

```python
import numpy as np

# 三个状态：0 起点，1 中间，2 终点
# 两个动作：0 留在附近，1 向右走
n_states = 3
n_actions = 2
gamma = 0.9

def transition(state, action):
    if state == 2:
        return 2, 0.0, True
    if action == 1:
        next_state = min(state + 1, 2)
    else:
        next_state = state
    reward = 1.0 if next_state == 2 else -0.1
    done = next_state == 2
    return next_state, reward, done

V = np.zeros(n_states)

for _ in range(20):
    new_V = V.copy()
    for s in range(n_states):
        values = []
        for a in range(n_actions):
            ns, r, done = transition(s, a)
            values.append(r + gamma * V[ns] * (not done))
        new_V[s] = max(values)
    V = new_V

print(V.round(3))
```

# 表格型强化学习

当状态和动作数量都很小时，可以直接用表格存储每个状态-动作对的价值。

## Q-Table

Q-Table 是一个二维表：

```text
rows    -> states
columns -> actions
value   -> Q(s, a)
```

```python
import numpy as np

n_states = 4
n_actions = 2

Q = np.zeros((n_states, n_actions))
Q[0, 1] = 0.5
Q[1, 0] = -0.2

print(Q)
print("best action at state 0:", Q[0].argmax())
```

## epsilon-greedy

如果永远选择当前 Q 值最大的动作，智能体可能过早陷入局部最优。epsilon-greedy 会以小概率随机探索。

```python
import numpy as np

def epsilon_greedy(Q, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    return Q[state].argmax()

Q = np.array([
    [0.1, 1.0, 0.2],
    [0.5, 0.3, 0.4],
])

actions = [epsilon_greedy(Q, state=0, epsilon=0.3) for _ in range(10)]
print(actions)
```

## Q-Learning

Q-Learning 直接学习最优动作价值函数，它的更新公式是：

$$
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
$$

其中：

- $\alpha$ 是学习率。
- $r + \gamma \max_{a'}Q(s',a')$ 是 TD target。
- target 和当前估计的差叫 TD error。

```python
import numpy as np

Q = np.zeros((3, 2))

s = 0
a = 1
r = -0.1
next_s = 1
alpha = 0.5
gamma = 0.9

td_target = r + gamma * np.max(Q[next_s])
td_error = td_target - Q[s, a]
Q[s, a] += alpha * td_error

print("td_target:", td_target)
print("td_error:", td_error)
print(Q)
```

### GridWorld 示例

下面是一个小型格子世界。智能体从左上角出发，希望走到右下角，中间有陷阱。

```python
import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = 0
        self.goal = size * size - 1
        self.holes = {5, 7, 11, 12}
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.size)

        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and col < self.size - 1:
            col += 1
        elif action == 2 and row < self.size - 1:
            row += 1
        elif action == 3 and col > 0:
            col -= 1

        self.state = row * self.size + col

        if self.state in self.holes:
            return self.state, -1.0, True
        if self.state == self.goal:
            return self.state, 1.0, True
        return self.state, -0.01, False

def train_q_learning(env, episodes=1000, alpha=0.2, gamma=0.95, epsilon=0.2):
    Q = np.zeros((env.size * env.size, 4))

    for _ in range(episodes):
        state = env.reset()

        for _ in range(50):
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = Q[state].argmax()

            next_state, reward, done = env.step(action)
            target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state

            if done:
                break

    return Q

env = GridWorld()
Q = train_q_learning(env)
symbols = ["↑", "→", "↓", "←"]

for row in range(env.size):
    cells = []
    for col in range(env.size):
        state = row * env.size + col
        if state == env.goal:
            cells.append("G")
        elif state in env.holes:
            cells.append("X")
        else:
            cells.append(symbols[Q[state].argmax()])
    print(cells)
```

# 深度 Q 网络

当状态空间很大时，无法用表格存储所有 $Q(s,a)$。DQN 使用神经网络近似 Q 函数：

$$
Q_\theta(s,a)
$$

网络输入状态，输出每个动作的 Q 值。

## DQN 网络

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)

model = DQN(state_dim=4, action_dim=2)
state = torch.randn(3, 4)
q_values = model(state)

print(q_values.shape)
```

## 经验回放

DQN 使用经验回放缓冲区保存交互数据：

$$
(s, a, r, s', done)
$$

训练时从缓冲区随机采样，可以减少相邻样本之间的相关性。

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer()
for i in range(10):
    buffer.push((i, 0, 1.0, i + 1, False))

print(buffer.sample(3))
```

## DQN 更新

DQN 的 target 是：

$$
y = r + \gamma \max_{a'}Q_{\theta^-}(s',a')
$$

其中 $\theta^-$ 是目标网络参数。目标网络会周期性从在线网络复制参数，用于稳定训练。

```python
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)

buffer = deque(maxlen=1000)
for _ in range(128):
    s = np.random.randn(4).astype("float32")
    a = np.random.randint(2)
    r = np.random.randn()
    ns = np.random.randn(4).astype("float32")
    done = np.random.rand() < 0.1
    buffer.append((s, a, r, ns, done))

online = DQN(4, 2)
target = DQN(4, 2)
target.load_state_dict(online.state_dict())
optimizer = optim.AdamW(online.parameters(), lr=1e-3)

batch = random.sample(buffer, 32)
states, actions, rewards, next_states, dones = zip(*batch)

states = torch.tensor(np.array(states))
actions = torch.tensor(actions)
rewards = torch.tensor(rewards, dtype=torch.float32)
next_states = torch.tensor(np.array(next_states))
dones = torch.tensor(dones, dtype=torch.float32)

q = online(states).gather(1, actions[:, None]).squeeze(1)

with torch.no_grad():
    next_q = target(next_states).max(dim=1).values
    y = rewards + 0.99 * next_q * (1 - dones)

loss = nn.functional.smooth_l1_loss(q, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("loss:", round(loss.item(), 4))
```

# 策略梯度

值函数方法先学习“每个动作值多少钱”，再选最大值对应的动作。策略梯度方法直接学习策略：

$$
\pi_\theta(a|s)
$$

目标是最大化期望 Return：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G]
$$

策略梯度定理给出：

$$
\nabla_\theta J(\theta)=\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)G]
$$

## REINFORCE

REINFORCE 是最基础的策略梯度算法。它用整条轨迹的 Return 作为动作好坏的估计。

```python
import torch
import torch.nn as nn
import torch.optim as optim

policy = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)

states = torch.randn(5, 4)
actions = torch.tensor([0, 1, 0, 1, 1])
returns = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])

logits = policy(states)
dist = torch.distributions.Categorical(logits=logits)
log_probs = dist.log_prob(actions)

loss = -(log_probs * returns).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("policy gradient loss:", round(loss.item(), 4))
```

### Baseline

直接用 Return 做权重，方差可能很大。可以减去一个 baseline，例如状态价值 $V(s)$：

$$
G_t - V(s_t)
$$

这就是 Advantage 的来源。

```python
import torch

returns = torch.tensor([3.0, 2.0, 1.0, 0.5])
baseline = returns.mean()
advantages = returns - baseline

print("returns:", returns)
print("baseline:", baseline.item())
print("advantages:", advantages)
```

# Actor-Critic

Actor-Critic 同时训练两个网络：

- Actor：输出策略 $\pi_\theta(a|s)$，负责选择动作。
- Critic：估计价值 $V_\phi(s)$，负责评价状态。

Actor 用 Critic 计算出的 Advantage 来更新策略，Critic 用 Return 或 TD target 来更新价值估计。

```python
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
        )
        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        h = self.backbone(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

model = ActorCritic()
state = torch.randn(6, 4)
logits, value = model(state)

print("logits:", logits.shape)
print("value:", value.shape)
```

## Actor-Critic 损失

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.ModuleDict({
    "body": nn.Sequential(nn.Linear(4, 32), nn.ReLU()),
    "actor": nn.Linear(32, 2),
    "critic": nn.Linear(32, 1),
})
optimizer = optim.Adam(model.parameters(), lr=1e-3)

states = torch.randn(8, 4)
actions = torch.randint(0, 2, (8,))
returns = torch.randn(8)

h = model["body"](states)
logits = model["actor"](h)
values = model["critic"](h).squeeze(-1)

dist = torch.distributions.Categorical(logits=logits)
log_probs = dist.log_prob(actions)
advantages = returns - values.detach()

actor_loss = -(log_probs * advantages).mean()
critic_loss = nn.functional.mse_loss(values, returns)
loss = actor_loss + 0.5 * critic_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("actor loss:", round(actor_loss.item(), 4))
print("critic loss:", round(critic_loss.item(), 4))
```

# PPO

PPO 的核心问题是：策略不能一次更新太大。否则新策略和旧策略差距过大，采样数据就不再可靠。

PPO 使用重要性比率：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

然后用 clip 限制更新幅度：

$$
L^{CLIP}=\mathbb{E}[\min(r_tA_t, clip(r_t,1-\epsilon,1+\epsilon)A_t)]
$$

```python
import torch

old_log_probs = torch.tensor([-0.7, -1.2, -0.3, -2.0])
new_log_probs = torch.tensor([-0.6, -1.5, -0.1, -2.1])
advantages = torch.tensor([1.0, 0.5, -0.8, 0.2])
clip_eps = 0.2

ratio = torch.exp(new_log_probs - old_log_probs)
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

loss = -torch.min(unclipped, clipped).mean()

print("ratio:", ratio.round(decimals=3))
print("ppo loss:", round(loss.item(), 4))
```

## 熵奖励

策略如果过早变得确定，就会减少探索。熵越大，说明动作分布越分散。很多策略梯度算法会加入熵奖励鼓励探索。

```python
import torch

logits = torch.tensor([[2.0, 0.1], [0.0, 0.0]])
dist = torch.distributions.Categorical(logits=logits)

print("prob:", dist.probs)
print("entropy:", dist.entropy())
```

# SAC

SAC（Soft Actor-Critic）属于最大熵强化学习。它不仅希望奖励高，也希望策略保持足够随机：

$$
J(\pi)=\mathbb{E}\left[\sum_t r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

其中 $\alpha$ 控制熵奖励的重要程度。

SAC 常用于连续动作控制任务，例如机械臂控制、机器人运动控制等。

```python
import torch

reward = torch.tensor([1.0, 0.5, -0.2])
entropy = torch.tensor([0.7, 1.0, 1.2])
alpha = 0.2

soft_objective = reward + alpha * entropy

print(soft_objective)
```

# 模仿学习

模仿学习不是从奖励中学习，而是从专家示范中学习。最简单的模仿学习是行为克隆（Behavior Cloning），本质上就是监督学习：

$$
(s, a_{expert}) \rightarrow a
$$

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

states = torch.randn(128, 4)
expert_actions = (states[:, 0] + states[:, 1] > 0).long()

policy = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for _ in range(60):
    logits = policy(states)
    loss = loss_fn(logits, expert_actions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    acc = (policy(states).argmax(dim=-1) == expert_actions).float().mean()

print("behavior cloning accuracy:", round(acc.item(), 3))
```

# 离线强化学习

在线强化学习可以继续和环境交互；离线强化学习只能使用已经收集好的静态数据集。

离线强化学习的核心难点是分布外动作：如果模型选择了数据集中很少出现的动作，价值函数可能会给出不可靠的高估。

```text
在线 RL:
  policy -> environment -> new data -> update policy

离线 RL:
  fixed dataset -> update policy
  不再向环境采样
```

离线 RL 和模仿学习很接近，但目标不同：

| 方法 | 学习目标 |
|---|---|
| 行为克隆 | 模仿数据中的动作 |
| 离线 RL | 在静态数据约束下最大化奖励 |

# Gymnasium 最小交互

如果安装了 `gymnasium`，可以用标准环境测试强化学习算法。下面只演示随机策略交互，不下载额外模型。

```python
try:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    state, info = env.reset(seed=42)
    total_reward = 0.0

    for _ in range(200):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    print("total reward:", total_reward)
except ImportError:
    print("gymnasium is not installed")
```

# 学习顺序

如果从零开始学习强化学习，可以按下面顺序推进：

1. 先理解 Agent、Environment、State、Action、Reward 和 Episode。
2. 学 Return 和折扣因子，明确强化学习为什么关心长期奖励。
3. 学 MDP、策略、状态价值和动作价值。
4. 学 Bellman 方程，理解当前价值和未来价值之间的递推关系。
5. 用小型 GridWorld 写 Q-Learning，建立表格型强化学习直觉。
6. 学 DQN，理解为什么需要神经网络、经验回放和目标网络。
7. 学策略梯度和 REINFORCE，理解直接优化策略的思想。
8. 学 Actor-Critic、PPO 和 SAC，把价值估计和策略优化联系起来。
9. 学行为克隆和离线 RL，为机器人/VLA 中的数据驱动策略学习做准备。
