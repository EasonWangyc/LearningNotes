## Scikit-Learn

Scikit-Learn(sklearn)是一个开源机器学习库，支持监督和无监督学习。它还提供多种工具，用于模型拟合、数据预处理、模型选择、模型评估及许多其他实用工具。

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 1.Bayesian Optimization Based on Guassian Process Regression：基于高斯过程回归的贝叶斯优化

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
```

```python
# 定义目标函数
def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)
plt.figure()
plt.plot(x, y)
```

```python
# 训练和测试数据
X_train = np.array([-1, 2, 4, 5, 7]).reshape(-1, 1)
y_train = target(X_train)
X_test = np.linspace(-2, 10, 10000).reshape(-1, 1)

# 高斯过程拟合
kernel = ConstantKernel(constant_value=0.2,constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
gpr.fit(X_train, y_train)
mu, cov = gpr.predict(X_test, return_cov=True)
y_test = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov)) #对应95%置信区间

# 绘图
plt.figure()
plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
plt.fill_between(X_test.ravel(), y_test + uncertainty, y_test - uncertainty, alpha=0.4)
plt.plot(X_test, y_test, label="predict")
plt.scatter(X_train, y_train, label="train", c="red", marker="o")
plt.legend()
```

```python
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)
# 创建贝叶斯优化器
optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=30)
# 随机初始化
optimizer.maximize(init_points=0, n_iter=6)
# 定义后验计算函数 
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma
# 自定义UCB函数
def ucb(mean, std, kappa=5):
    return mean + kappa * std
```

```python
def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle("Guassian Process and Bayesian Optimization after {} steps".format(steps),fontdict={'size': 30})
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    ax1.plot(x, y, label="True function")
    ax1.plot(x_obs.flatten(), y_obs, 'o', markersize=8, label=u'Observations', color='r')
    ax1.plot(x, mu, '--', color='k', label='Prediction')
    ax1.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=.4, fc='c', ec='None', label='95% confidence interval')
    ax1.set_xlim((-2, 10))
    ax1.set_ylim((None, None))
    ax1.set_ylabel('f(x)', fontdict={'size': 15})
    ax1.set_xlabel('x', fontdict={'size': 15})

    # 使用UCB计算采集的效用值
    mean, std = optimizer._gp.predict(x, return_std=True)
    utility = ucb(mean, std, kappa=5)
    ax2.plot(x, utility, label='Utility Function', color='purple')
    ax2.plot(x[np.argmax(utility)], np.max(utility),'*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k',
             markeredgewidth=1)
    ax2.set_xlim((-2, 10))
    ax2.set_ylim((0, np.max(utility) + 0.5))
    ax2.set_ylabel('Utility', fontdict={'size': 15})
    ax2.set_xlabel('x', fontdict={'size': 15})

    ax1.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    ax2.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.show()
```

```python
plot_gp(optimizer, x, y)
```

```python
np.random.seed(42)
xs = np.linspace(-2, 10, 10000)

def f(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)

plt.plot(xs, f(xs))
plt.show()
# 定义绘图函数
def plot_bo(f, bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x, f(x))
    plt.plot(x, mean, '--', color='k')
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.5)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()
```

Expolartion(探索参数空间) VS. Exploitation(利用接近当前已知最优点),自由参数kappa允许用户使算法更保守/更不保守。

使用Upper Confidence Bound采集函数(UCB)

```python
# prefer exploitation(kappa=0.1)
acquisition_function = acquisition.UpperConfidenceBound(kappa=0.1)
bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10)

plot_bo(f, bo)
```

```python
# prefer exploration(kappa=10)
acquisition_function = acquisition.UpperConfidenceBound(kappa=10.)
bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10)

plot_bo(f, bo)
```

使用Experted Improvement采集函数（EI）

```python
# prefer exploition(xi=0.0)
acquisition_function = acquisition.ExpectedImprovement(xi=0.0)

bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10)

plot_bo(f, bo)
```

```python
# prefer exploration(xi=0.1)
acquisition_function = acquisition.ExpectedImprovement(xi=100)

bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10)

plot_bo(f, bo)
```

使用Probability of Improvement采集函数（POI）

```python
# prefer exploitation(xi=1e-4)
acquisition_function = acquisition.ProbabilityOfImprovement(xi=1e-4)

bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10)

plot_bo(f, bo)
```

```python
# prefer exploitation(xi=0.1)
acquisition_function = acquisition.ProbabilityOfImprovement(xi=0.1)

bo = BayesianOptimization(
    f=f,
    acquisition_function=acquisition_function,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10)

plot_bo(f, bo)
```

Matern Kernel Function 母核函数的设置

详见：https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

协方差函数是高斯过程预测器的核心要素，在监督学习中，数据点之间的相似性概念显然至关重要：一个基本的相似性假设是，输入值x相近的点很可能具有相似的目标值y，因此靠近测试点的训练点应当对该点的预测提供有效信息。

核函数用来描述两个输入点之间的相似程度，其中的变量决定函数是否平滑、两点之间的影响程度。

对于任意两个输入$x_i$和$x_j$，GP假设两者函数值$f(x_i)$和$f(x_j)$之间的协方差为：
$$k(x_i, x_j) =  \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(
         \frac{\sqrt{2\nu}}{l} d(x_i , x_j )
         \Bigg)^\nu K_\nu\Bigg(
         \frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg)$$

其中，$d(x_i , x_j)=|x_i - x_j|$为两点间的欧式距离，$K_{\nu}()$为修正贝塞尔函数。

若协方差仅为$(x_i , x_j)$的函数，则称其为各向同性，这意味着它对所有刚体运动都具有不变性，也被称为Radial Basis Functions（RBF）。

length_scale($l$)：核的长度尺度，将输入变量变化的范围对应到函数值变化范围的尺度。对于同样的两个输入点，length_scale小，核函数变化大，因此可将length_scale理解为函数的灵敏度。

L-BFGS(limited-memory BFGS)：一种高效求最值的优化算法，相比于BFGS利用过去的梯度变化来估计，只保存最近几次的梯度信息。
