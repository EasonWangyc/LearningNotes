# Machine Learning (机器学习)

机器学习关注的问题是：给定一批样本数据，让模型从数据中学习输入和输出之间的关系，并在新的样本上做出预测、分类、排序、聚类或表示压缩。

和深度学习相比，传统机器学习通常更依赖人工设计好的特征，模型规模更小，可解释性更强，也更适合用来理解“训练、损失、泛化、评估”这些基本概念。

### 先理解几个基本名词

- **样本（sample）**：数据集中的一条记录，例如一个人的一行属性或一张图片。
- **特征（feature）**：描述样本的输入变量，通常记作 $X$。一条样本可以有多个特征。
- **标签（label）**：希望模型预测的目标，通常记作 $y$。监督学习中的标签来自已有标注。
- **模型参数（parameter）**：训练过程中由模型根据数据自动学习的数值，例如线性回归中的权重和偏置。
- **超参数（hyperparameter）**：训练前由使用者设定的配置，例如学习率、树的深度和邻居数量。
- **推理（inference）**：训练完成后，将新输入交给模型并得到预测结果的过程。
- **泛化（generalization）**：模型在没有参与训练的新样本上仍然能够保持较好表现的能力。

### 代码示例的阅读方式

下面的代码不只是调用一个分类器，而是把“构造数据 → 划分数据 → 拟合预处理和模型 → 预测 → 计算指标”完整串起来。代码中的 `X` 表示输入特征矩阵，形状通常是 `[样本数, 特征数]`；`y` 表示每个样本对应的标签。

## 机器学习的基本流程

一个典型的机器学习任务通常包含以下几个阶段：

```text
原始数据
  -> 数据清洗
  -> 特征构造
  -> 训练集 / 验证集 / 测试集划分
  -> 模型训练
  -> 模型评估
  -> 调整特征或超参数
  -> 在新数据上推理
```

在代码里，最小流程可以写成：

```python
# ===== 1. 构造一个可控的二分类数据集 =====
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

# ===== 2. 只用训练集学习模型参数 =====
# stratify=y 让训练集和测试集保持相近的类别比例。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# ===== 3. 把预处理和分类器串成一个流水线 =====
# StandardScaler 只应在训练数据上拟合，Pipeline 会在训练/预测时自动按正确顺序调用。
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# fit 会学习标准化参数和逻辑回归参数；predict 只执行推理。
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# accuracy_score 比较预测标签和真实标签，得到测试集上的准确率。
print("accuracy:", accuracy_score(y_test, y_pred))
```

### 数据集划分

训练集用于让模型学习参数，验证集用于选择模型结构和超参数，测试集用于最后评估模型在未知数据上的表现。

这里的**数据泄漏（data leakage）**指的是本不应该被模型看到的信息进入了训练过程。例如用全量数据计算标准化的均值和方差，或者用测试集分数反复选择超参数，都会让评估结果偏乐观。

如果直接用测试集反复调参，测试集就会被间接“学习”到，最后得到的分数会过于乐观。因此测试集应该尽量只在最终评估时使用。

```python
# ===== 1. 生成一个小型分类数据集 =====
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=12, n_features=4, random_state=0)

# ===== 2. 第一次划分：训练集和临时集 =====
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=0,
)

# ===== 3. 第二次划分：验证集和测试集 =====
# 临时集被均分，因此最终约为 60% / 20% / 20%。
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=0,
)

# 打印形状可以确认每个集合包含的样本数量和标签数量一致。
print("train:", X_train.shape, y_train.shape)
print("val  :", X_val.shape, y_val.shape)
print("test :", X_test.shape, y_test.shape)
```

### 特征缩放

很多模型依赖距离、内积或梯度更新，例如 KNN、SVM、逻辑回归、PCA。如果不同特征量纲差异很大，数值大的特征会在计算中占据主导地位。

标准化通常将每一列特征变成均值为 0、标准差为 1：

$$
x' = \frac{x - \mu}{\sigma}
$$

**特征缩放（feature scaling）**是把不同量纲的特征变换到可比较的数值范围。`fit` 会计算 $μ$ 和 $σ$，`transform` 使用已经计算好的统计量转换新数据；在真实流程中，统计量必须只来自训练集。

```python
# ===== 1. 第二列特征的数值范围明显更大 =====
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([
    [1.0, 1000.0],
    [2.0, 2000.0],
    [3.0, 3000.0],
])

scaler = StandardScaler()
# fit_transform 先计算均值/标准差，再对这批数据完成标准化。
X_scaled = scaler.fit_transform(X)

# axis=0 表示按列统计，每一列对应一个特征。
print("原始均值:", X.mean(axis=0))
print("标准化后均值:", X_scaled.mean(axis=0).round(6))
print("标准化后标准差:", X_scaled.std(axis=0).round(6))
```

### Pipeline

`Pipeline` 将数据预处理和模型封装在一起，可以避免把测试集信息泄漏到训练过程。

**Pipeline（流水线）**是按固定顺序连接多个变换器和最终模型的容器。它的好处是交叉验证的每一折都会只用该折的训练部分拟合 `StandardScaler`，然后再处理验证部分。

错误做法是先对全量数据 `fit_transform`，再切分训练集和测试集。这样测试集的均值、方差已经参与了标准化参数的计算。

```python
# ===== 1. 构造数据 =====
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=300, n_features=20, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

# cross_val_score 会重复进行“训练折 fit + 验证折 predict”，这里 cv=5 表示五折交叉验证。
scores = cross_val_score(pipe, X, y, cv=5)
print("每折分数:", scores.round(3))
print("平均分数:", scores.mean().round(3))
```

# 监督学习

监督学习的数据由输入 $X$ 和标签 $y$ 组成。模型的目标是学习一个映射函数：

$$
f: X \rightarrow y
$$

当 $y$ 是连续值时，通常是回归任务；当 $y$ 是离散类别时，通常是分类任务。

**损失函数（loss function）**把预测结果和真实标签之间的差异变成一个数。训练算法试图让损失下降；**梯度（gradient）**表示损失对各个参数的偏导数，梯度下降会沿着梯度的反方向更新参数。损失用于学习参数，指标用于从任务角度评价结果，两者不一定是同一个量。

## 线性回归

线性回归假设输出可以由输入特征的线性组合表示：

$$
\hat{y} = w_1x_1 + w_2x_2 + ... + w_dx_d + b = \mathbf{w}^{T}\mathbf{x} + b
$$

其中 $\mathbf{w}$ 是权重，$b$ 是偏置，$\hat{y}$ 是预测值。

### 均方误差

回归任务常用均方误差作为损失函数：

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

损失越小，说明模型预测值和真实值越接近。

```python
# ===== 用三个预测值验证均方误差 =====
import numpy as np

y_true = np.array([3.0, 5.0, 7.0])
y_pred = np.array([2.5, 5.5, 8.0])

# 逐元素计算误差，平方后取平均，得到一个标量损失。
mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)
```

### 最小二乘闭式解

对于线性回归，如果矩阵可逆，可以直接通过公式求解最优参数：

$$
\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}
$$

为了同时学习偏置 $b$，通常会给输入矩阵额外拼上一列全 1。

```python
import numpy as np

# 目标数据来自 y = 2 * x + 1。
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])

# 增加一列 1 后，最后一个参数就可以表示偏置 b。
X_with_bias = np.c_[X, np.ones(len(X))]
# 闭式解一次性求出 [w, b]，不需要迭代更新。
w, b = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

print("w:", w)
print("b:", b)
print("predict x=5:", w * 5 + b)
```

### 梯度下降

除了闭式解，也可以通过梯度下降逐步更新参数。梯度下降的基本思想是：计算损失函数对参数的梯度，然后沿着损失下降最快的方向更新。

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

其中 $\eta$ 是学习率。

```python
import numpy as np

X = np.array([1.0, 2.0, 3.0, 4.0])
y = 2 * X + 1

w = 0.0
b = 0.0
lr = 0.05

# 每一步都重新计算当前预测、误差和损失对参数的梯度。
for step in range(100):
    y_pred = w * X + b
    error = y_pred - y

    # 对所有样本的梯度取平均，得到参数的整体更新方向。
    grad_w = np.mean(2 * error * X)
    grad_b = np.mean(2 * error)

    # 沿梯度反方向移动；lr 是学习率，决定每次移动的步长。
    w -= lr * grad_w
    b -= lr * grad_b

print("w:", round(w, 3))
print("b:", round(b, 3))
```

### Ridge 与 Lasso

普通线性回归只关心预测误差，正则化会额外约束参数大小，避免模型过度依赖某些特征。

Ridge 使用 L2 正则：

$$
L = L_{MSE} + \lambda \sum_j w_j^2
$$

Lasso 使用 L1 正则：

$$
L = L_{MSE} + \lambda \sum_j |w_j|
$$

L2 会让权重更平滑，L1 更容易让部分权重变成 0。

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

X, y = make_regression(
    n_samples=100,
    n_features=5,
    noise=20,
    random_state=42,
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=10.0),
    "Lasso": Lasso(alpha=1.0),
}

# 对比不加正则、L2 正则和 L1 正则时学到的权重。
for name, model in models.items():
    model.fit(X, y)
    print(name, model.coef_.round(2))
```

## 逻辑回归

逻辑回归虽然名字里有“回归”，但通常用于二分类。它先计算线性得分：

$$
z = \mathbf{w}^{T}\mathbf{x} + b
$$

然后通过 Sigmoid 函数将得分映射到 0 到 1 之间：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

这个输出可以理解为属于正类的概率。

```python
import numpy as np

# Sigmoid 把任意实数映射到 (0, 1)，可作为二分类正类概率。
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
print(sigmoid(z).round(3))
```

### 决策边界

当概率大于等于 0.5 时，通常预测为正类；小于 0.5 时预测为负类。因为 $\sigma(0)=0.5$，所以二分类的决策边界就是：

$$
\mathbf{w}^{T}\mathbf{x} + b = 0
$$

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([0, 0, 0, 1])

clf = LogisticRegression()
# fit 学习线性决策边界；predict_proba 输出每个类别的概率。
clf.fit(X, y)

print("coef:", clf.coef_.round(3))
print("intercept:", clf.intercept_.round(3))
print("prob:", clf.predict_proba([[1.0, 1.0]]).round(3))
```

### 交叉熵损失

二分类逻辑回归常用交叉熵损失：

$$
L = -[y\log(p) + (1-y)\log(1-p)]
$$

当真实标签为 1 时，如果模型给出的 $p$ 很小，损失会很大；当真实标签为 0 时，如果模型给出的 $p$ 很大，损失也会很大。

```python
import numpy as np

def binary_cross_entropy(y, p):
    # 避免 log(0) 产生无穷大，实际库实现也会做类似的数值保护。
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

y = np.array([1, 1, 0, 0])
p = np.array([0.9, 0.2, 0.1, 0.8])

print(binary_cross_entropy(y, p).round(3))
print("mean loss:", binary_cross_entropy(y, p).mean().round(3))
```

## K 近邻算法

KNN 的思想很直观：一个样本属于什么类别，可以参考离它最近的 $k$ 个样本。

**距离度量（distance metric）**是衡量两个样本相似程度的规则。下面默认使用欧氏距离；距离越小表示样本在特征空间中越接近。KNN 的 `k` 是超参数，不是训练出来的模型参数。

KNN 几乎没有训练过程，主要计算发生在预测阶段。预测一个新样本时，需要计算它到训练集中所有样本的距离，再找最近的 $k$ 个样本投票。

```python
import numpy as np
from collections import Counter

# 训练数据中的每行是一个样本，标签是样本所属类别。
X_train = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [3.0, 3.0],
    [3.0, 4.0],
])
y_train = np.array(["A", "A", "B", "B"])

x = np.array([0.2, 0.1])
k = 3

distances = np.linalg.norm(X_train - x, axis=1)
# 找到距离从小到大的前 k 个样本，再用多数投票决定类别。
nearest_index = np.argsort(distances)[:k]
nearest_labels = y_train[nearest_index]

print("nearest labels:", nearest_labels)
print("prediction:", Counter(nearest_labels).most_common(1)[0][0])
```

### 距离度量

KNN 对特征尺度非常敏感。如果一个特征范围是 0 到 1，另一个特征范围是 0 到 10000，那么距离几乎会被第二个特征决定。

因此 KNN 通常需要标准化。

```python
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=300, n_features=6, random_state=42)

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5)),
])

# 先在每一折训练数据上标准化，再计算邻居距离。
scores = cross_val_score(knn, X, y, cv=5)
print(scores.round(3))
```

## 支持向量机

支持向量机希望找到一个分类超平面，不仅要能把两类样本分开，还希望分类边界离最近样本尽可能远。

距离分类边界最近的样本叫支持向量。它们决定了分类边界的位置。

**超平面（hyperplane）**是由 $\mathbf{w}^{T}\mathbf{x}+b=0$ 定义的分类边界；**间隔（margin）**是边界到最近样本的距离。`C` 是软间隔模型的惩罚系数，控制“扩大间隔”和“减少训练误分类”之间的权衡。

### 硬间隔

当数据完全线性可分时，可以要求所有样本都被正确分类：

$$
y_i(\mathbf{w}^{T}\mathbf{x}_i + b) \geq 1
$$

同时最大化分类间隔，等价于最小化：

$$
\frac{1}{2}\|\mathbf{w}\|^2
$$

```python
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(
    n_samples=60,
    centers=2,
    cluster_std=0.7,
    random_state=42,
)

clf = SVC(kernel="linear", C=1e6)
# C 很大时接近硬间隔：模型强烈惩罚训练样本被错分。
clf.fit(X, y)

print("support vectors:", clf.support_vectors_.shape)
print("coef:", clf.coef_.round(3))
print("intercept:", clf.intercept_.round(3))
```

### 软间隔

真实数据往往有噪声，完全分开可能导致边界过度扭曲。软间隔通过参数 `C` 控制对误分类的惩罚。

`C` 越大，模型越不愿意错分训练样本；`C` 越小，模型越愿意接受一些错分，从而得到更平滑的边界。

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    class_sep=0.8,
    random_state=42,
)

for c in [0.1, 1.0, 100.0]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=c)),
    ])
    # 每个 C 都重新训练一个标准化后的线性 SVM。
    model.fit(X, y)
    print("C =", c, "train accuracy =", round(model.score(X, y), 3))
```

### 核函数

当数据在原始空间中不是线性可分时，可以通过核函数隐式计算高维空间中的内积。

常用核函数：

| 核函数 | 含义 |
|---|---|
| `linear` | 原始空间中的线性分类 |
| `poly` | 多项式特征空间 |
| `rbf` | 高斯径向基核，常用于非线性边界 |

```python
from sklearn.datasets import make_circles
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_circles(n_samples=400, noise=0.08, factor=0.4, random_state=42)

models = {
    "linear": SVC(kernel="linear"),
    "rbf": SVC(kernel="rbf", gamma="scale"),
}

# 圆形数据不是原始二维空间中的线性可分问题，RBF 核可以表达弯曲边界。
for name, svm in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm),
    ])
    scores = cross_val_score(pipe, X, y, cv=5)
    print(name, scores.mean().round(3))
```

## 朴素贝叶斯

朴素贝叶斯基于贝叶斯公式：

$$
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

分类时只需要比较不同类别下的后验概率大小。由于 $P(x)$ 对所有类别相同，可以忽略：

$$
\hat{y} = \arg\max_y P(x|y)P(y)
$$

“朴素”指的是假设各个特征在类别给定时条件独立：

$$
P(x_1, x_2, ..., x_d|y) = \prod_{j=1}^{d}P(x_j|y)
$$

**先验概率（prior）**是观察输入前某个类别的概率，**似然（likelihood）**是给定类别后观察到输入的概率，**后验概率（posterior）**是结合输入后对类别的更新判断。文本示例中的词频就是把文本转换成可计算特征的一种方式。

这个假设在真实世界中通常不完全成立，但能让模型非常简单、快速。

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "gpu cuda tensor",
    "cuda kernel memory",
    "cat dog pet",
    "dog food pet",
]
labels = ["tech", "tech", "life", "life"]

vectorizer = CountVectorizer()
# fit_transform 建立词表并把每句话转换成“词出现次数”向量。
X = vectorizer.fit_transform(texts)

clf = MultinomialNB()
# 朴素贝叶斯在这些词频特征和标签上估计概率分布。
clf.fit(X, labels)

# 测试文本必须复用训练阶段建立的词表，只调用 transform。
test = vectorizer.transform(["cuda memory"])
print(clf.predict(test))
print(clf.predict_proba(test).round(3))
```

## 决策树

决策树通过一系列条件判断对样本进行划分。每个内部节点是一个判断条件，每个叶子节点对应最终预测。

```text
是否有纹理?
  ├── 是 -> 是否颜色偏深?
  │       ├── 是 -> 类别 A
  │       └── 否 -> 类别 B
  └── 否 -> 类别 C
```

### 基尼系数

分类树常用基尼系数衡量一个节点的混杂程度：

$$
Gini = 1 - \sum_{k=1}^{K}p_k^2
$$

如果一个节点里全是同一类样本，基尼系数为 0；如果类别混杂，基尼系数更大。

```python
import numpy as np

def gini(labels):
    # 统计每个类别的比例，比例越混杂，Gini 越大。
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

print("pure node:", gini(["A", "A", "A"]))
print("mixed node:", gini(["A", "A", "B", "B"]))
```

### 信息熵

信息熵也可以衡量不确定性：

$$
Entropy = -\sum_{k=1}^{K}p_k\log_2p_k
$$

信息增益表示一次划分让不确定性减少了多少。

```python
import numpy as np

def entropy(labels):
    # 只对出现过的类别计算概率，避免对 0 取对数。
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

parent = ["A", "A", "B", "B"]
left = ["A", "A"]
right = ["B", "B"]

gain = entropy(parent) - (
    # 信息增益 = 划分前不确定性 - 划分后按样本数加权的不确定性。
    len(left) / len(parent) * entropy(left)
    + len(right) / len(parent) * entropy(right)
)

print("parent entropy:", entropy(parent))
print("information gain:", gain)
```

### 树深度

决策树如果不限制深度，可能会把训练集记得很细，导致泛化能力下降。常见限制包括：

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

for depth in [1, 3, None]:
    # depth=None 允许树继续生长，用来观察过拟合现象。
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    print(
        "max_depth =", depth,
        "train =", round(tree.score(X_train, y_train), 3),
        "test =", round(tree.score(X_test, y_test), 3),
    )
```

## 集成学习

集成学习的思想是把多个模型组合起来，让整体效果比单个模型更稳定。

常见方向有两类：

- Bagging：多个模型并行训练，再投票或平均。
- Boosting：多个模型串行训练，后一个模型关注前面没学好的部分。

## Bagging 与随机森林

Bagging 会从训练集中有放回采样得到多个子数据集，在每个子数据集上训练一个模型。

随机森林在 Bagging 的基础上增加了特征随机性：每棵树的每次分裂只看一部分特征。这样不同树之间差异更大，组合后更稳定。

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=800, n_features=20, random_state=42)

models = {
    "single tree": DecisionTreeClassifier(random_state=42),
    "random forest": RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
    ),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(name, scores.mean().round(3), scores.std().round(3))
```

### 特征重要性

随机森林可以估计特征重要性。它的含义是：某个特征在树分裂中对降低不纯度贡献了多少。

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = make_classification(
    n_samples=500,
    n_features=8,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

order = np.argsort(rf.feature_importances_)[::-1]
print("feature order:", order)
print("importance:", rf.feature_importances_[order].round(3))
```

## Boosting

Boosting 是串行集成。每一轮都会训练一个新模型，用来修正前面模型的不足。

### AdaBoost

AdaBoost 会提高前一轮被错分样本的权重，让后一轮模型更关注这些样本。

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42,
)

scores = cross_val_score(clf, X, y, cv=5)
print(scores.round(3))
```

### Gradient Boosting

Gradient Boosting 可以理解为：后续模型拟合当前模型的残差或负梯度，从而逐步降低损失。

对于回归任务，如果当前模型预测偏低，下一棵树就会学习一个正的修正值；如果预测偏高，下一棵树就会学习一个负的修正值。

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=500, n_features=8, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
```

# 无监督学习

无监督学习只有输入数据 $X$，没有人工标注的标签 $y$。模型需要从数据本身发现结构，例如簇、低维表示、异常点或潜在主题。

## K-Means

K-Means 的目标是把样本划分成 $K$ 个簇，让每个样本尽量靠近自己所属簇的中心。

**质心（centroid）**是一个簇中所有样本的均值向量；**簇内平方和（inertia）**是样本到所属质心的平方距离之和，是衡量聚类紧密程度的数值。

算法流程：

1. 随机初始化 $K$ 个质心。
2. 将每个样本分配给最近的质心。
3. 根据分配结果重新计算每个簇的质心。
4. 重复 2 和 3，直到质心变化很小。

```python
# ===== K-Means 的“分配-更新”循环 =====
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.8,
    random_state=42,
)

rng = np.random.default_rng(42)
# 随机挑选 K 个样本作为初始质心。
centers = X[rng.choice(len(X), size=4, replace=False)]

for _ in range(20):
    # 计算每个样本到每个质心的欧氏距离。
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    # 每个样本归入距离最小的质心。
    labels = distances.argmin(axis=1)
    # 对每个簇内的样本求均值，更新质心位置。
    centers = np.array([X[labels == k].mean(axis=0) for k in range(4)])

print("centers shape:", centers.shape)
print("labels:", sorted(set(labels)))
```

### Inertia

K-Means 的优化目标是最小化每个样本到所属质心的平方距离之和，这个值在 sklearn 中叫 `inertia_`。

随着 $K$ 增大，`inertia_` 一定会下降。但 $K$ 太大时，每个簇会变得过细，不一定有解释意义。

```python
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

def simple_kmeans_inertia(X, k, steps=20):
    # 固定随机种子，使不同 k 的比较可以复现。
    rng = np.random.default_rng(42)
    centers = X[rng.choice(len(X), size=k, replace=False)]

    for _ in range(steps):
        # 交替执行“分配样本”和“更新质心”。
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    final_distances = np.linalg.norm(X - centers[labels], axis=1) ** 2
    return final_distances.sum()

for k in range(1, 8):
    # 比较不同簇数量对应的簇内平方和，可观察肘部趋势。
    inertia = simple_kmeans_inertia(X, k)
    print("k =", k, "inertia =", round(inertia, 1))
```

## DBSCAN

DBSCAN 基于密度聚类，不需要提前指定簇数量。它通过两个参数定义局部密度：

**密度可达（density-reachable）**表示样本可以通过一系列相邻的高密度区域连接起来。DBSCAN 先找核心点，再扩展核心点邻域；无法归入任何密度区域的样本标记为噪声。

- `eps`：邻域半径。
- `min_samples`：邻域内至少有多少点才算核心点。

样本可以分成三类：

- 核心点：邻域内点数足够多。
- 边界点：自己不是核心点，但落在核心点邻域内。
- 噪声点：不属于任何密度区域。

```python
# ===== DBSCAN：按局部密度识别簇和噪声 =====
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

X, _ = make_moons(n_samples=400, noise=0.08, random_state=42)
# DBSCAN 依赖距离，因此先让两个特征处于相近的数值范围。
X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.25, min_samples=5)
labels = dbscan.fit_predict(X)

# sklearn 使用 -1 表示噪声点，其余非负整数表示簇编号。
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print("clusters:", n_clusters)
print("noise:", n_noise)
```

## PCA

PCA 的目标是找到一组新的正交坐标轴，让数据投影到前几个方向后尽可能保留原始方差。

PCA 常用于：

- 数据可视化：高维数据降到 2 维或 3 维。
- 降低特征维度：减少冗余特征。
- 去噪：保留主要变化方向，丢弃小方差方向。

### 协方差矩阵

PCA 会先对数据中心化，然后计算协方差矩阵：

$$
C = \frac{1}{N}X^TX
$$

协方差矩阵描述不同特征之间如何一起变化。对协方差矩阵做特征分解后，特征值越大的方向，表示该方向上数据方差越大。

**中心化（centering）**是每列减去该列均值；**特征值（eigenvalue）**表示对应主成分方向上的方差大小，**特征向量（eigenvector）**表示这个方向本身。PCA 通常保留最大特征值对应的前几个方向。

```python
# ===== 手动计算 PCA 的协方差矩阵和主方向 =====
import numpy as np

X = np.array([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0],
    [5.0, 4.0],
])

# 每列减去均值，消除特征绝对位置对方向计算的影响。
X_centered = X - X.mean(axis=0)
cov = X_centered.T @ X_centered / len(X)

# 特征分解同时得到主方向和每个方向上的方差大小。
eigenvalues, eigenvectors = np.linalg.eig(cov)

print("covariance matrix:")
print(cov.round(3))
print("eigenvalues:", eigenvalues.round(3))
print("eigenvectors:")
print(eigenvectors.round(3))
```

### sklearn 中的 PCA

`explained_variance_ratio_` 表示每个主成分保留的方差比例。

```python
# ===== 使用 sklearn 将高维特征投影到二维 =====
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=8,
    random_state=42,
)

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
# fit_transform 学习两个主成分，并输出每个样本在主成分坐标系中的位置。
X_2d = pca.fit_transform(X_scaled)

print("original shape:", X.shape)
print("pca shape:", X_2d.shape)
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("total:", pca.explained_variance_ratio_.sum().round(3))
```

也可以让 PCA 自动选择保留 95% 方差所需的维度数：

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, _ = make_classification(n_samples=500, n_features=30, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)
# 浮点 n_components 表示至少保留 95% 的累计解释方差。
X_reduced = pca.fit_transform(X_scaled)

print("reduced shape:", X_reduced.shape)
print("components:", pca.n_components_)
```

## t-SNE

t-SNE 是一种非线性降维方法，主要用于可视化。它更关注保留局部邻域结构：原来相似的点，在低维空间里也尽量靠近。

t-SNE 不太适合作为通用特征预处理步骤，因为它的结果不稳定、计算较慢，并且不容易自然地对新样本做同样的变换。

```python
# ===== t-SNE：把局部邻域关系映射到二维 =====
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digits = load_digits()
# 只取少量样本，方便快速观察输出形状。
X = digits.data[:300]
y = digits.target[:300]

try:
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    # t-SNE 的 fit_transform 输出每个样本的二维坐标。
    X_2d = tsne.fit_transform(X)
    print("t-SNE shape:", X_2d.shape)
    print("first point:", X_2d[0].round(3), "label:", y[0])
except AttributeError as exc:
    print("当前 sklearn/threadpoolctl 环境无法运行 t-SNE:", exc)
```

# 模型评估

模型评估的核心问题是：模型在未见过的数据上表现如何。训练集分数高只能说明模型记住或拟合了训练数据，不能直接说明泛化能力好。

## 分类任务指标

二分类可以用混淆矩阵描述预测结果：

| 名称 | 含义 |
|---|---|
| TP | 真实为正，预测为正 |
| FP | 真实为负，预测为正 |
| TN | 真实为负，预测为负 |
| FN | 真实为正，预测为负 |

由此可以定义：

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [1, 1, 1, 0, 0, 0]
y_pred = [1, 1, 0, 1, 0, 0]

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

### Accuracy 的局限

当类别极度不均衡时，Accuracy 可能很有迷惑性。

例如 100 个样本里 95 个是负类，模型全部预测负类，也能得到 95% Accuracy，但它完全没有识别正类的能力。

```python
from sklearn.metrics import accuracy_score, recall_score

y_true = [0] * 95 + [1] * 5
y_pred = [0] * 100

print("accuracy:", accuracy_score(y_true, y_pred))
print("recall for positive class:", recall_score(y_true, y_pred, zero_division=0))
```

## ROC 与 AUC

很多分类模型可以输出概率或置信度。通过改变分类阈值，可以得到不同的 TPR 和 FPR。

$$
TPR = \frac{TP}{TP + FN}
$$

$$
FPR = \frac{FP}{FP + TN}
$$

ROC 曲线描述不同阈值下 TPR 和 FPR 的变化，AUC 是 ROC 曲线下面积。

**混淆矩阵（confusion matrix）**按“真实类别 × 预测类别”统计样本数量。Precision 关注预测为正的样本有多少是真的，Recall 关注真实为正的样本有多少被找出来，F1 是二者的调和平均。ROC/AUC 使用连续分数或概率，可以观察阈值变化带来的整体取舍。

```python
# ===== 用连续正类概率计算 AUC =====
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

clf = LogisticRegression(max_iter=1000)
# 先拟合分类器，再取得测试样本属于正类的概率。
clf.fit(X_train, y_train)

score = clf.predict_proba(X_test)[:, 1]
# roc_auc_score 会在不同阈值下统计 TPR/FPR 并计算曲线面积。
print("AUC:", roc_auc_score(y_test, score).round(3))
```

## 回归任务指标

回归任务常用指标包括 MAE、MSE、RMSE 和 $R^2$。

MAE:

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

MSE:

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

RMSE:

$$
RMSE = \sqrt{MSE}
$$

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([3.0, 5.0, 7.0, 9.0])
y_pred = np.array([2.5, 5.5, 6.5, 10.0])

mse = mean_squared_error(y_true, y_pred)

print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2:", r2_score(y_true, y_pred))
```

## 交叉验证

单次训练集/测试集划分可能受到随机性影响。交叉验证会把数据分成多个折，每次用其中一折验证，其余折训练，最后取平均结果。

```python
# ===== 交叉验证：比较普通划分和分层划分 =====
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=10, random_state=42)
model = LogisticRegression(max_iter=1000)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# StratifiedKFold 在每一折中尽量维持类别比例。
print("KFold:", cross_val_score(model, X, y, cv=kfold).round(3))
print("Stratified:", cross_val_score(model, X, y, cv=stratified).round(3))
```

对于分类任务，`StratifiedKFold` 会尽量保持每一折中的类别比例一致，通常比普通 `KFold` 更稳。

## 超参数搜索

模型训练得到的是参数，例如线性模型的权重；训练前人为设定的是超参数，例如正则强度、树深度、SVM 的 `C` 和 `gamma`。

### 网格搜索

网格搜索会遍历所有候选组合。

```python
# ===== 网格搜索：穷举给定的离散候选值 =====
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=400, n_features=12, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])

param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.1, 1.0],
}

search = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy")
# 搜索过程内部会对每组超参数执行交叉验证。
search.fit(X, y)

print("best params:", search.best_params_)
print("best score:", round(search.best_score_, 3))
```

### 随机搜索

当搜索空间很大时，随机搜索不遍历所有组合，而是随机采样一部分组合。

```python
# ===== 随机搜索：从分布中抽取有限组候选值 =====
from scipy.stats import loguniform
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=400, n_features=12, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf")),
])

param_dist = {
    "svm__C": loguniform(1e-2, 1e2),
    "svm__gamma": loguniform(1e-3, 1e1),
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
)
# n_iter=10 表示只评估 10 组随机采样的超参数。
search.fit(X, y)

print("best params:", search.best_params_)
print("best score:", round(search.best_score_, 3))
```

# 一个完整的小实验

下面用一个小的二分类任务串起数据生成、预处理、模型训练、交叉验证、测试集评估和概率输出。

```python
# ===== 一个完整的端到端实验 =====
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    weights=[0.7, 0.3],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

models = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "svm_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True)),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
    ),
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    print("=" * 50)
    print(name)
    print("cv auc:", cv_scores.mean().round(3))
    print("test auc:", roc_auc_score(y_test, prob).round(3))
    print(classification_report(y_test, pred))
```

# 学习顺序

如果从零开始学习传统机器学习，可以按下面顺序推进：

1. 先理解训练集、验证集、测试集，以及为什么要防止数据泄漏。
2. 学线性回归和逻辑回归，理解损失函数、梯度下降和正则化。
3. 学 KNN、SVM、决策树，体会不同模型的归纳偏置。
4. 学随机森林和 Boosting，理解多个弱模型如何组合成更强模型。
5. 学 K-Means、DBSCAN、PCA、t-SNE，建立无监督学习的直觉。
6. 学分类和回归指标，明确模型分数到底在衡量什么。
7. 用 `Pipeline`、交叉验证和超参数搜索把模型训练流程串起来。
