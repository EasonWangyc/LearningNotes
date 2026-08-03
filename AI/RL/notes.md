# Reinforcement Learning (强化学习)

强化学习研究的是：智能体（Agent）如何通过和环境（Environment）交互，逐步学习一种策略（Policy），让长期累积奖励尽可能大。

和监督学习不同，强化学习没有直接给出“每个状态下正确动作是什么”。智能体只能通过试错获得奖励，再从奖励中反推哪些行为更好。

## 强化学习中的基本名词

- **智能体（agent）**：根据当前信息选择动作的学习者。
- **环境（environment）**：接收动作、改变状态并返回反馈的系统。
- **状态（state）**：描述当前环境情况、供智能体决策的信息。
- **动作（action）**：智能体可以执行的操作。
- **奖励（reward）**：环境对刚刚执行动作的即时反馈，可能为正、为负或为零。
- **转移（transition）**：从 $(s_t, a_t)$ 到 $(s_{t+1}, r_t)$ 的环境变化。
- **轨迹（trajectory）**：一段连续的状态、动作和奖励序列。
- **回合（episode）**：从环境重置开始，到终止条件满足为止的一段交互。
- **探索（exploration）**：尝试不熟悉的动作以获取新信息；**利用（exploitation）**：选择当前估计最好的动作。

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
# ===== 构造一个可以逐步向目标移动的最小环境 =====
class CounterEnv:
    def __init__(self, target=3):
        self.target = target
        self.state = 0

    def reset(self):
        # reset 开始一个新 episode，并返回初始状态。
        self.state = 0
        return self.state

    def step(self, action):
        # action=1 向目标前进，其他动作让状态向相反方向移动。
        if action == 1:
            self.state += 1
        else:
            self.state -= 1

        # done 表示当前 episode 是否结束；reward 是当前动作的即时反馈。
        done = self.state >= self.target
        reward = 1.0 if done else -0.1
        return self.state, reward, done

env = CounterEnv()
state = env.reset()

for _ in range(5):
    # 这里固定选择前进动作，只用于观察环境接口，不是学习算法。
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

**Return（回报）**是从某个时间点开始累计的折扣奖励；**折扣因子（discount factor）** $γ$ 让较远的奖励影响更小。Return 是一条轨迹中的目标值，价值函数则是对 Return 的期望或估计。

- $\gamma=0$：只关心当前奖励。
- $\gamma$ 越接近 1：越重视长期回报。

```python
# ===== 正向累加折扣奖励 =====
def discounted_return(rewards, gamma=0.9):
    total = 0.0
    power = 1.0
    for reward in rewards:
        # 第一个奖励乘 gamma^0，之后的奖励依次乘更高次幂。
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
# ===== 用递推关系从 episode 末尾计算每个时刻的 Return =====
def compute_returns(rewards, gamma=0.9):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        # 当前回报 = 当前奖励 + 折扣后的下一时刻回报。
        running = reward + gamma * running
        returns.append(running)
    return list(reversed(returns))

print(compute_returns([-0.1, -0.1, -0.1, 1.0], gamma=0.9))
```

# 马尔可夫决策过程

强化学习通常用马尔可夫决策过程（MDP）描述环境。

**马尔可夫决策过程（Markov Decision Process, MDP）**是一种描述“状态、动作、转移和奖励”的数学模型。**马尔可夫性**要求当前状态已经包含预测未来所需的信息，因此给定当前状态和动作后，未来不再需要完整历史。

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
    # np.random.choice 按 action_probs 给出的策略分布采样动作。
    action = np.random.choice(actions, p=action_probs)
    print(action)
```

# 价值函数

价值函数用来估计“某个状态或动作长期来看有多好”。

**状态价值函数 $V(s)$**评价“处在这个状态有多好”，**动作价值函数 $Q(s,a)$**评价“在这个状态执行这个动作有多好”。二者都不是当前奖励本身，而是对未来 Return 的估计。

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

**优势函数（advantage）**用 $A(s,a)=Q(s,a)-V(s)$ 表示某个动作相对于该状态平均价值的额外收益。它可以作为策略更新的方向和权重。

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

```python
# ===== 用 Q 值减去状态平均价值得到 Advantage =====
import numpy as np

q_values = np.array([1.0, 1.5, 0.2])
# 这里用动作价值的平均值作为一个简单的 V 近似。
v_value = q_values.mean()
advantage = q_values - v_value

print("Q:", q_values)
print("V:", round(v_value, 3))
print("A:", advantage.round(3))
```

# Bellman 方程

Bellman 方程把“当前价值”和“下一步价值”联系起来。

**Bellman 方程**是动态规划和许多强化学习算法的递推基础：当前价值由当前奖励与下一状态价值组成。**TD（temporal-difference）误差**是一次更新中“目标值”和当前估计的差异。

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
# ===== 已知环境转移时进行价值迭代 =====
import numpy as np

# 三个状态：0 起点，1 中间，2 终点
# 两个动作：0 留在附近，1 向右走
n_states = 3
n_actions = 2
gamma = 0.9

def transition(state, action):
    # 返回下一状态、即时奖励和终止标志，模拟一个确定性环境。
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
    # 使用旧的 V 计算所有状态-动作候选值，再同步更新整张 V 表。
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

当状态和动作都是离散且数量有限时，Q-Table 的每个单元格就是一个状态-动作价值估计。表格方法的优点是直观，限制是状态空间增大后存储和探索都会变得困难。

```python
# ===== 创建并读取状态-动作价值表 =====
import numpy as np

n_states = 4
n_actions = 2

Q = np.zeros((n_states, n_actions))
# Q[state, action] 表示在该状态执行该动作的当前价值估计。
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
    # 以 epsilon 概率探索随机动作，否则利用当前价值最大的动作。
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

**Q-Learning**是 off-policy 方法：更新目标使用下一状态的最大 Q 值，不要求下一步真的按照同一个行为策略选动作。`alpha` 控制新信息覆盖旧估计的速度。

```python
# ===== 执行一次 Q-Learning 更新 =====
import numpy as np

Q = np.zeros((3, 2))

s = 0
a = 1
r = -0.1
next_s = 1
alpha = 0.5
gamma = 0.9

td_target = r + gamma * np.max(Q[next_s])
# TD error 衡量目标值和当前 Q[s, a] 的差距。
td_error = td_target - Q[s, a]
# 只向目标移动 alpha 的比例，避免一次更新完全覆盖旧值。
Q[s, a] += alpha * td_error

print("td_target:", td_target)
print("td_error:", td_error)
print(Q)
```

### GridWorld 示例

下面是一个小型格子世界。智能体从左上角出发，希望走到右下角，中间有陷阱。

```python
# ===== 在 GridWorld 中重复执行表格型 Q-Learning =====
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
        # 把一维 state 映射成二维坐标，再根据动作移动。
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
        # 每个 episode 都从起点开始。
        state = env.reset()

        for _ in range(50):
            # 行为策略负责在探索和利用之间做选择。
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = Q[state].argmax()

            next_state, reward, done = env.step(action)
            # 终止状态没有未来价值，因此用 (not done) 屏蔽 bootstrap 项。
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

**DQN（Deep Q-Network）**用神经网络近似原本存放在表格中的 $Q(s,a)$。**在线网络**用于当前预测和梯度更新，**目标网络**提供相对稳定的 TD target；二者参数不会在每一步都同时变化。

## DQN 网络

```python
# ===== 定义一个输出所有动作 Q 值的网络 =====
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
        # 输出形状为 [batch_size, action_dim]，每列对应一个动作。
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

**经验回放（replay buffer）**是保存历史 transition 的队列。随机采样让训练 batch 不再严格按照时间顺序排列，也能重复利用一次交互得到的经验。

```python
# ===== 保存和随机采样 transition =====
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        # deque(maxlen=capacity) 会自动丢弃最旧经验。
        self.buffer.append(transition)

    def sample(self, batch_size):
        # 无放回随机采样一个训练 batch。
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
# ===== 计算一次 DQN 的 TD 损失并更新在线网络 =====
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
# zip(*batch) 把一批 transition 拆成五个字段，再转换成张量。
states, actions, rewards, next_states, dones = zip(*batch)

states = torch.tensor(np.array(states))
actions = torch.tensor(actions)
rewards = torch.tensor(rewards, dtype=torch.float32)
next_states = torch.tensor(np.array(next_states))
dones = torch.tensor(dones, dtype=torch.float32)

q = online(states).gather(1, actions[:, None]).squeeze(1)
# 只取每个样本实际执行动作对应的 Q 值。

with torch.no_grad():
    # 目标网络只用于生成 target，不让目标值参与反向传播。
    next_q = target(next_states).max(dim=1).values
    y = rewards + 0.99 * next_q * (1 - dones)

loss = nn.functional.smooth_l1_loss(q, y)
# Smooth L1 对少量大误差比纯平方损失更稳健。
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

**策略梯度（policy gradient）**直接调整产生动作概率的策略参数，而不是先建立完整的 Q 表。`log_prob(action)` 越大表示策略越倾向于该动作，Return 或 Advantage 会决定这次倾向应被增强还是减弱。

## REINFORCE

REINFORCE 是最基础的策略梯度算法。它用整条轨迹的 Return 作为动作好坏的估计。

```python
# ===== 用 Return 加权动作对数概率 =====
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
# Categorical 把 logits 转成离散动作分布。
dist = torch.distributions.Categorical(logits=logits)
log_probs = dist.log_prob(actions)

loss = -(log_probs * returns).mean()
# 梯度下降最小化负的目标，等价于最大化高回报动作的概率。

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

**Baseline** 是不依赖当前动作的价值基准。减去它不会改变理想梯度的期望，但可以降低 Return 带来的方差，使更新更稳定。

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

**Actor** 是策略网络，输出动作分布；**Critic** 是价值网络，输出状态价值。两者共享特征提取部分时，表示学习可以同时服务于动作选择和价值估计。

```python
# ===== 一个共享骨干的 Actor-Critic 模型 =====
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
        # backbone 提取状态特征，两个 head 分别输出策略和价值。
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
# ===== 使用 Advantage 同时更新 Actor 和 Critic =====
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
# detach 防止 Actor 的损失反向影响 Critic 的 value 估计。

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

**重要性比率（importance ratio）**比较新旧策略对同一动作的概率。PPO 的 `clip` 把比率限制在一个区间内，避免单批采样数据让策略发生过大的更新。

```python
# ===== 计算 PPO 的裁剪目标 =====
import torch

old_log_probs = torch.tensor([-0.7, -1.2, -0.3, -2.0])
new_log_probs = torch.tensor([-0.6, -1.5, -0.1, -2.1])
advantages = torch.tensor([1.0, 0.5, -0.8, 0.2])
clip_eps = 0.2

# 概率比值可用 log 概率之差的 exp 稳定地计算。
ratio = torch.exp(new_log_probs - old_log_probs)
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

loss = -torch.min(unclipped, clipped).mean()

print("ratio:", ratio.round(decimals=3))
print("ppo loss:", round(loss.item(), 4))
```

## 熵奖励

策略如果过早变得确定，就会减少探索。熵越大，说明动作分布越分散。很多策略梯度算法会加入熵奖励鼓励探索。

**熵（entropy）**衡量动作分布的不确定性。均匀分布的熵较大，接近确定性选择的分布熵较小。

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

**最大熵强化学习**把环境奖励和策略熵一起作为目标，因此会在获得较高奖励的同时保留一定探索性。SAC 是连续动作场景中常见的最大熵算法。

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

**专家示范**是已经记录好的状态-动作样本；**行为克隆（behavior cloning）**把状态作为输入、专家动作作为标签，直接训练一个动作分类或回归模型。

$$
(s, a_{expert}) \rightarrow a
$$

```python
# ===== 用专家动作作为监督标签训练策略 =====
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
    # 每一步都让策略拟合同一批专家状态对应的动作。
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

**离线强化学习（offline RL）**只从固定数据集学习，不再请求环境产生新样本。它仍然要利用奖励和价值估计改进策略，因此不只是复现数据中的动作。

| 方法 | 学习目标 |
|---|---|
| 行为克隆 | 模仿数据中的动作 |
| 离线 RL | 在静态数据约束下最大化奖励 |

# Gymnasium 最小交互

如果安装了 `gymnasium`，可以用标准环境测试强化学习算法。下面只演示随机策略交互，不下载额外模型。

```python
# ===== Gymnasium 的标准环境交互接口 =====
try:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    state, info = env.reset(seed=42)
    total_reward = 0.0

    for _ in range(200):
        # 这里只采样随机动作，目的是观察环境 API 的状态和终止信号。
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # terminated 表示任务自然结束，truncated 表示时间上限等外部截断。
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
