# GATED DELTA NETWORKS : IMPROVING MAMBA 2 WITH DELTA RULE

论文链接：https://arxiv.org/pdf/2412.06464

代码链接：https://github.com/NVlabs/GatedDeltaNet

## 摘要

Linear Transformers 作为标准 Transformer 的高效替代方案而备受关注，但在检索和长上下文任务中，其性能表现却受到了一定限制。为克服这些局限，**近期研究探索了两种不同的机制**：用于自适应记忆控制的“**门控**”（gating）机制，以及用于精确记忆修改的“**增量更新规则**”（delta update rule）。我们发现，这两种机制具有互补性——门控机制能够实现快速的记忆擦除，而增量更新规则则支持有针对性的记忆更新。基于这一洞察，我们提出了“**门控增量规则**”（gated delta rule），并开发了一种针对现代硬件进行优化的并行训练算法。我们所提出的架构——**Gated DeltaNet**——在多项基准测试（涵盖语言建模、常识推理、上下文内检索、长度外推以及长上下文理解等任务）中，均持续优于 Mamba2 和 DeltaNet 等现有模型。此外，我们通过构建混合架构进一步提升了性能，即将 Gated DeltaNet 层与滑动窗口注意力机制（sliding window attention）或 Mamba2 层相结合，从而在实现更高训练效率的同时，获得了更卓越的任务表现。代码：https://github.com/NVlabs/GatedDeltaNet。

## 1.介绍

Transformer 架构显著提升了大语言模型（LLM）的能力；得益于其高效的注意力机制，该架构在各类任务中均展现出卓越的性能。这种机制在精确的序列建模方面表现出色，且在训练过程中能够充分利用现代 GPU 的并行处理能力。然而，自注意力组件的计算复杂度随序列长度呈二次方增长，导致了巨大的计算需求，从而给模型的训练和推理过程带来了挑战。

为缓解这些问题，研究人员探索了多种替代方案，例如 Linear Transformers。该方案用基于核点积的线性注意力机制取代了传统的基于 Softmax 的注意力机制，并通过将其重构为具有矩阵值状态的线性循环神经网络（RNN），大幅降低了推理过程中的内存需求。尽管早期版本的 Linear Transformer 在语言建模任务中的表现不及标准 Transformer，但近期的一些改进——例如引入类似于 LSTM 中数据依赖型门控机制（以 GLA 和 Mamba2 等模型为代表）——已展现出显著的性能提升。然而，在处理长序列信息方面，挑战依然存在；特别是在上下文检索任务中，传统的 Transformer 仍保持着其固有优势。

这种现象并不令人意外：Linear Transformer 可被视为实现了一种基于外积的键值关联记忆，这让人联想到张量积表示。然而，它们所能存储的正交键值对数量受限于模型的维度。当序列长度超过这一维度时，“记忆冲突”便不可避免，从而阻碍了精确检索。

Mamba2 通过引入一种简单的门控更新规则 $\textbf S_t = \alpha_t \textbf S_{t-1} + \textbf v_t \textbf k_t^\top$ 来解决这一局限性；在该规则中，所有键值关联在每个时间步都会根据一个动态比例 $\alpha_t \in (0, 1)$ 进行统一衰减。

然而，这种方法并未考虑不同键值关联在重要性上的差异，这可能导致记忆利用效率低下。如果模型需要遗忘某个特定的键值关联，它会“一视同仁”地遗忘所有的键值关联，从而使得这一过程缺乏针对性，效率也不高。

相比之下，采用“增量规则”（delta rule）的 Linear Transformer（被称为 DeltaNet）通过按顺序以“软”替换的方式，用新输入的键值对（key-value pair）来更新记忆。该方法在针对上下文检索（in-context retrieval）的合成基准测试中表现出色。然而，由于此过程每次仅修改单个键值对，模型缺乏快速清除过时或无关信息的能力——在上下文发生切换、需要抹除先前数据时，这一局限尤为明显。因此，DeltaNet 在实际任务中的表现仅属一般，这很可能是因为它缺乏一种稳健的记忆清除机制。

**鉴于“门控更新规则”（gated update rule）与“增量更新规则”（delta update rule）在记忆管理方面具有互补优势，我们提出了“门控增量规则”（gated delta rule）**——一种结合了上述两种方法、既简单又直观的机制。这一统一规则实现了灵活的内存控制：通过将 $α_t$ 设为 0，可以迅速清除记忆；而通过将 $α_t$ 设为 1，则能够在不影响其他信息的情况下，有选择地更新特定内容（这实际上等同于切换到了纯粹的增量规则）。

目前尚存的挑战在于，**如何以硬件高效的方式实现门控增量规则（gated delta rule）**。我们基于 Yang 等人（2024b）提出的高效算法—​​—该算法利用 WY 表示法实现了增量规则计算的并行化——对其方法进行了精心扩展，将门控项纳入其中。这一扩展保留了分块并行（chunkwise parallelism）的优势，从而实现了硬件高效的训练。

我们提出的架构——**Gated DeltaNet**——在一系列全面的基准测试中，始终优于 Mamba2 和 DeltaNet。这些测试涵盖了语言建模、常识推理、上下文检索、长度外推以及长上下文理解等多个方面。在此基础上，我们还开发了混合架构，通过将 Gated DeltaNet 层与滑动窗口注意力机制（sliding window attention）或 Mamba2 层进行策略性结合，进一步提升了训练效率与模型性能。

## 2.PRELIMINARY

### 2.1 MAMBA2: LINEAR ATTENTION WITH DECAY

已知在不考虑归一化以及 query 和 key 激活的情况下，Linear Transformer 可表述为如下线性递推形式：

```math
\textbf S_t=\textbf S_{t-1}+\textbf v_t\textbf k^T_t\in\mathbb R^{d_v\times d_k},\quad\textbf o_t=\textbf S_t\textbf q_t\in\mathbb R^{d_v}
```

其中 $d_k$ 和 $d_v$ 分别表示 query/key 和 value 的（头）维度。通过展开递归关系，我们可以将其表示为向量形式（左侧）和矩阵形式（右侧），如下所示：

```math
\textbf o_t=\sum^t_{i=1}(\textbf v_i\textbf k^T_i)\textbf q_t=\sum^t_{i=1}\textbf v_i(\textbf k^T_i\textbf q_t)\in\mathbb R^{d_v},\quad\textbf O=(\textbf Q\textbf K^T\odot\textbf M)\textbf V\in\mathbb R^{L\times d_v}
```

其中 $L$ 为序列长度，$\textbf M ∈ \mathbb R^{L×L}$ 为因果掩码（causal mask），其定义为：当 $i < j$ 时 $\textbf M_{ij} = 0$，否则 $\textbf M_{ij} = 1$。

然而，这种基础的线性注意力机制在语言建模任务上的表现远逊于 Transformer。为解决这一问题，通常会引入一个衰减项，以实现对历史信息的“遗忘”。此处以 Mamba2 为例，它可以用下述线性递归形式来表示（在特定参数化方式下）：

```math
\textbf S_t=\textcolor{blue}{\alpha_t}\textbf S_{t-1}+\textbf v_t\textbf k^T_t,\quad\textbf o_t=\textbf S_t\textbf q_t
```

其中 $\textcolor{blue}{\alpha_t \in (0, 1)}$ 是一个依赖于数据且随 $t$ 变化的标量衰减项。定义累积衰减乘积 $\textcolor{blue}{\gamma_j = \prod_{i=1}^j \alpha_i}$；通过展开该递推关系，我们可以将结果表示为向量形式（左侧）和矩阵并行形式（右侧）：

```math
\textbf o_t=\sum^t_{i=1}\left(\textcolor{blue}{\frac{\gamma_t}{\gamma_i}}\textbf v_i\textbf k^T_i\right)\textbf q_t=\sum^t_{i=1}\textbf v_i\left(\textcolor{blue}{\frac{\gamma_t}{\gamma_i}}\textbf k^T_i\textbf q_t\right),\quad\textbf O=((\textbf Q\textbf K^T)\odot\textcolor{blue}{\Gamma})\textbf V
```

其中，$\textcolor{blue}{\Gamma \in \mathbb{R}^{L \times L}}$ 是一个考虑了衰减机制的因果掩码（causal mask），满足：当 $i \ge j$ 时，$\textcolor{blue}{\Gamma_{ij} = \frac{\gamma_i}{\gamma_j}}$；否则，$\textcolor{blue}{\Gamma_{ij} = 0}$。这种并行形式与循环形式之间的等价性，也被称为 Dao 和 Gu (2024a) 所提出的“**状态空间对偶性**”（State Space Duality, SSD）。这种循环结构也出现在其他多种架构中，包括 Gated RFA、xLSTM 和 Gated RetNet。当 $\gamma_t$ 与数据无关（data-independent）时，该公式便退化为 RetNet 和 Lightning-Attention。此外，若将 $\gamma_t$ 从标量扩展为矩阵，且采用外积结构进行参数化，依然可以实现高效的训练算法；这一点已由 Yang 等人 (2024a) 证明，并被 Yang 等人 (2024a)、Peng 等人 (2024)、Qin 等人 (2024b)、Zhang 等人 (2024)、Chou 等人 (2024)、He 等人 (2025) 以及 Lu 等人 (2025) 等研究工作所采用。

**Chunkwise training**。然而，无论是循环形式还是并行形式，都不利于高效训练。因此，为了实现硬件高效且线性时间复杂度的训练，研究者们提出了“分块并行”（chunkwise parallel）形式，具体介绍如下。简而言之，分块并行形式将输入和输出划分为若干大小为 $C$ 的数据块（chunk）；对于每个数据块，其输出的计算依赖于前一个数据块的最终状态，以及当前数据块中的查询（query）、键（key）和值（value）块。参考 Sun 等人（2023b）以及 Yang 等人（2024a;b）所采用的符号表示，我们以查询块 $\textbf q$ 为例进行说明。我们将第 $t$ 个数据块的查询块记作 $\textbf Q_{[t]} := \textbf q_{tC+1:(t+1)C+1}$，并将该数据块内的第 $r$ 个查询记作 $\textbf q^r_{[t]} := \textbf q_{tC+r}$。第 $t$ 个数据块的初始状态定义为 $\textbf S_{[t]} := \textbf S^0_{[t]} = \textbf S^C_{[t-1]}$。通过对循环过程进行部分展开，我们得到：

```math
\textbf S^r_{[t]}=\textbf S_{[t]}+\sum^r_{i=1}\textbf v^i_{[t]}\textbf k^{iT}_{[t]}\in\mathbb R^{d_v\times d_k},\quad\textbf o^r_{[t]}=\textbf S^r_{[t]}\textbf q^r_{[t]}=\textbf S_{[t]}\textbf q^r_{[t]}+\sum^r_{i=1}\textbf v^i_{[i]}(\textbf k^{iT}_[t]\textbf q^r_{[t]})\in\mathbb R^{d_v}
```

等价地，用矩阵形式表示：

```math
\textbf S_{[t+1]}=\textbf S_{[t]}+\textbf V_{[t]}\textbf K^T_{[t]}\in\mathbb R^{d_v\times d_k},\quad\textbf O_{[t]}=\textbf Q_{[t]}\textbf S^T_{[t]}+\left(\textbf Q_{[t]}\textbf K^T_{[t]}\odot\textbf M\right)\textbf V_{[t]}\in\mathbb R^{C\times d_v}
```

其中 $\textbf M \in \mathbb{R}^{C \times C}$ 为因果掩码（causal mask）。上述公式包含大量的矩阵乘法运算，这使得基于张量核心（Tensor Core）的硬件优化成为可能。这种分块算法可以很容易地扩展至带有衰减机制的线性注意力机制：

```math
\textbf S_{[t+1]}=\textcolor{blue}{\overrightarrow{\textbf S_{[t]}}}+\textbf V^T_{[t]}\textcolor{blue}{\overrightarrow{\textbf K_{[t]}}}\in\mathbb R^{d_v\times d_k},\quad\textbf O_{[t]}=\textcolor{blue}{\overleftarrow{\textbf Q_{[t]}}}\textbf S^T_{[t]}+\left(\textbf Q_{[t]}\textbf K^T_{[t]}\odot\textcolor{blue}{\Gamma_{[t]}}\right)\textbf V_{[t]}\in\mathbb R^{C\times d_v}\tag{1}
```

其中 $(\textcolor{blue}{\Gamma_{[t]})_{ij} =\frac{γ^i_{[t]}}{γ^j_{[t]}}, γ^j_{[t]} =\prod^{tC+j}_{j=tC+1} α_j}$。此处，我们分别使用左箭头（$\overleftarrow{·}$）和右箭头（$\overrightarrow{·}$）来表示某个变量在各自数据块（chunk）内分别向首位置和末位置衰减的过程。

```math
\begin{aligned}
\textcolor{blue}{\overleftarrow{\boldsymbol{q}}^{r}_{[t]}} &= \textcolor{blue}{\gamma^{r}_{[t]}} \boldsymbol{q}^{r}_{[t]} & & \text{decaying each vector to the first position of chunk } t \\
\textcolor{blue}{\overrightarrow{\boldsymbol{k}}^{r}_{[t]}} &= \textcolor{blue}{\frac{\gamma^{C}_{[t]}}{\gamma^{r}_{[t]}}} \boldsymbol{k}^{r}_{[t]} & & \text{decaying each vector to the last position of chunk } t \\
\textcolor{blue}{\overrightarrow{\mathbf{S}}_{[t]}} &= \textcolor{blue}{\gamma^{C}_{[t]}} \mathbf{S}_{[t]} & & \text{decaying the state matrix over the entire chunk } t
\end{aligned}
\tag{2}
```

其他变量（例如 $\textcolor{blue}{\vec{v}}$）的情况亦是如此。Mamba2 中引入的 SSD 分解算法在很大程度上等同于这种分块算法。为了实现更通用的方案，Yang 等人（2024a）提出了一种针对线性注意力机制的扩展分块算法，其中融入了细粒度的衰减机制。

### 2.2 DELTA NETWORKS: LINEAR ATTENTION WITH DELTA RULE

增量更新规则会动态地擦除与当前输入键（$\textbf k_t$）相关联的旧值（$\textbf v^{old}_t$），并写入一个新值（$\textbf v^{new}_t$）；该新值是当前输入值与旧值的一种线性组合，其组合方式取决于“写入强度” $\beta_t \in (0, 1)$。

```math
\textbf S_t=\textbf S_{t-1}-\underbrace{(\textbf S_{t-1}\textbf k_t)}_{\textbf v^{old}_t}\textbf k^{\top}_t+\underbrace{(\beta_t\textbf v_t+(1-\beta_t)\textbf S_{t-1}\textbf k_t)}_{\textbf v^{new}_t}\textbf k^{\top}_t=\textbf S_{t-1}(\textbf I-\beta_t\textbf k_t\textbf k^{\top}_t)+\beta_t\textbf v_t\textbf k^{\top}_t
```

如上所示，DeltaNet 采用了一阶线性递归，并使用了广义 Householder 转移矩阵（即 $\textbf I - \beta_t\textbf k_t\textbf k_t^\top$）。尽管 DeltaNet 在联想记忆和语言建模方面表现优异，但由于计算效率低下，它此前并未受到太多关注；直到 Yang et al. (2024b) 提出了一种对硬件友好的分块训练算法（详见下文），这一状况才有所改观。

**Chunkwise parallel form**。通过部分扩展循环，我们有

```math
\textbf S^r_{[t]}=\textbf S_{[t]}\underbrace{\left(\prod^r_{i=1}\textbf I-\beta^i_{[t]}\textbf k^i_{[t]}\textbf k^{i\top}_{[t]}\right)}_{:=\textbf P^r_{[t]}}+\underbrace{\sum^r_{i=1}\left(\beta^i_{[t]}\textbf v^i_{[t]}\textbf k^{i\top}_{[t]}\prod^r_{j=i+1}(\textbf I-\beta^j_{[t]}\textbf k^j_{[t]}\textbf k^{j\top}_{[t]})\right)}_{:=\textbf H^r_{[t]}}\tag{3}
```

其中 $\textbf P^j_{[t]}$ 涉及广义 Householder 矩阵的累积乘积，这可以通过经典的 WY 表示进行优化：

```math
\mathbf{P}^{r}_{[t]} = \mathbf{I} - \sum_{i=1}^{r} \mathbf{w}^{i}_{[t]} \boldsymbol{k}^{i\mathsf{T}}_{[t]} \in \mathbb{R}^{d_k \times d_k}
\qquad
\mathbf{w}^{r}_{[t]} = \beta^{r}_{[t]} \left( \boldsymbol{k}^{r}_{[t]} - \sum_{i=1}^{r-1} \left( \mathbf{w}^{i}_{[t]} \left( \boldsymbol{k}^{i\mathsf{T}}_{[t]} \boldsymbol{k}^{r}_{[t]} \right) \right) \right) \in \mathbb{R}^{d_k}
\tag{4}
```

同样地，$\mathbf{H}^{r}_{[t]}$ 可以表示为:

```math
\mathbf{H}^{r}_{[t]} = \sum_{i=1}^{r} \mathbf{u}^{i}_{[t]} \boldsymbol{k}^{i\mathsf{T}}_{[t]} \in \mathbb{R}^{d_v \times d_k}
\qquad
\mathbf{u}^{r}_{[t]} = \beta^{r}_{[t]} \left( \boldsymbol{v}^{r}_{[t]} - \sum_{i=1}^{r-1} \left( \mathbf{u}^{i}_{[t]} \left( \boldsymbol{k}^{i\mathsf{T}}_{[t]} \boldsymbol{k}^{r}_{[t]} \right) \right) \right) \in \mathbb{R}^{d_v}
\tag{5}
```

写成矩阵形式为：$\mathbf{P}_{[t]} = \mathbf{I} - \mathbf{W}^{\top}_{[t]} \mathbf{K}_{[t]} \in \mathbb{R}^{d_k \times d_k}$，$\mathbf{H}_{[t]} = \mathbf{U}^{\top}_{[t]} \mathbf{K}_{[t]} \in \mathbb{R}^{d_v \times d_k}$。通过使用 UT 变换，我们可以进一步将 $\mathbf{W}$ 和 $\mathbf{U}$ 写成矩阵形式:

```math
\mathbf{T}_{[t]} = \left[ \mathbf{I} + \operatorname{strictLower}\left( \operatorname{diag}(\beta_{[t]}) \mathbf{K}_{[t]} \mathbf{K}^{\mathsf{T}}_{[t]} \right) \right]^{-1} \operatorname{diag}\left( \beta_{[t]} \right) \in \mathbb{R}^{C \times C}
\tag{6}
```

```math
\mathbf{W}_{[t]} = \mathbf{T}_{[t]} \mathbf{K}_{[t]} \in \mathbb{R}^{C \times d_k},
\qquad
\mathbf{U}_{[t]} = \mathbf{T}_{[t]} \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}
\tag{7}
```

将这些结果代回式 3，可以得到一个面向 DeltaNet 的硬件高效分块 (chunkwise) 算法,该算法利用矩阵乘法，从而支持基于张量核心 (tensor core) 的 GPU 优化:

```math
\mathbf{S}_{[t+1]} = \mathbf{S}_{[t]} \mathbf{P}_{[t]} + \mathbf{H}_{[t]} = \mathbf{S}_{[t]} + \left( \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}^{\mathsf{T}}_{[t]} \right)^{\mathsf{T}} \mathbf{K}_{[t]}
\qquad \in \mathbb{R}^{d_v \times d_k}
\tag{8}
```

```math
\mathbf{O}_{[t]} = \mathbf{Q}_{[t]} \mathbf{S}^{\mathsf{T}}_{[t]} + \left( \mathbf{Q}_{[t]} \mathbf{K}^{\mathsf{T}}_{[t]} \odot \mathbf{M} \right) \left( \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}^{\mathsf{T}}_{[t]} \right)
\qquad \in \mathbb{R}^{C \times d_v}
\tag{9}
```

## 3.GATED DELTA NETWORKS

### 3.1  FORMULATION: GATED DELTA RULE

<img
  src="https://i-blog.csdnimg.cn/direct/a44abd5e842b4ad4857eeda494d2f513.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

所提出的门控增量规则既简单又有效：

```math
\textbf S_t=\textbf S_{t-1}(\textcolor{blue}{\alpha_t}(\textbf I-\beta_t\textbf k_t\textbf k^{\top}_t))+\beta_t\textbf v_t\textbf k^{\top}_t\tag{10}
```

其中，依赖于数据的门控项 $\textcolor{blue}{\alpha_t \in (0, 1)}$ 控制着状态的衰减。这种表述方式融合了门控机制与“增量规则”（delta rule）各自的优势：门控项实现了自适应的记忆管理，而基于增量的更新结构则有助于高效地学习键值（key-value）关联。

我们基于 Liu et al. (2024) 提出的在线学习框架，对“门控增量规则”（gated delta rule）进行了形式化分析。如表 1 所示，在该框架下，循环状态更新可被视为在线学习问题的闭式解。近期的线性循环神经网络（RNN）架构通常会在其在线学习目标函数中引入正则化项，以防止状态偏离先前值，从而实现记忆保持。然而，当状态中充斥着大量信息而趋于饱和时，这种保持机制便会出现问题——此时，每个状态都编码了多条信息的叠加，导致难以从中精确地提取特定信息。为克服这一局限，Mamba2 和 Gated DeltaNet 引入了一个自适应缩放因子 $\alpha_t$，用以放宽正则化项的约束，从而允许 $\textbf S_t$ 与 $\textbf S_{t-1}$ 之间产生受控的偏差。这种改进实现了通过“选择性遗忘”进行的动态记忆管理，有助于过滤掉无关信息（详见第 3.2 节）。

另一方面，线性注意力 (Linear Attention, LA) 和 Mamba2 使用简单的负内积损失 $-\langle \mathbf{S}_t \boldsymbol{k}_t, \boldsymbol{v}_t \rangle$,而 Longhorn 则使用表达能力更强的在线回归目标 $\|\mathbf{S}_t \boldsymbol{k}_t - \boldsymbol{v}_t\|^2$，以更好地建模键值关联。由此得到的 Longhorn 更新规则与 delta 更新规则非常相似，这表明带门控的 delta 规则在上下文关联召回 (in-context associative recall) 方面优于 Mamba2。

从“快速权重编程”（fast weight programming）以及“测试时训练与回归”（test-time training and regression）的角度来看，隐状态 $\textbf S$ 可被视为一个（快速）权重矩阵；其中，通过测试时的随机梯度下降（SGD），“增量规则”（delta rule）能够优化在线回归目标函数 $\mathcal L(\textbf S_t) = \frac{1}{2}\|\textbf S_t\textbf k_t -\textbf v_t\|^2$：

```math
\textbf S_{t+1}=\textbf S_{t}-\beta\nabla\mathcal L(\textbf S_t)=\textbf S_t-\beta_t(\textbf S_t\textbf k_t-\textbf v_t)\textbf k^{\top}_t=\textbf S_t(\textbf I-\beta_t\textbf k_t\textbf k^{\top}_t)+\beta_t\textbf v_t\textbf k^{\top}_t
```

其中 $β_t$ 代表（自适应）学习率。从这一角度来看，门控增量规则（gated delta rule）可被视为在随机梯度下降（SGD）更新中引入了一个自适应权重衰减项 $α_t$——这是一种在深度学习中广泛应用的技术。与此同时，Titans 模型也证明了在循环神经网络（RNN）的测试阶段进行 SGD 更新时，引入权重衰减机制是行之有效的。

### 3.2  CASE STUDY: SINGLE NEEDLE IN A HAYSTACK (S-NIAH)

为了更好地理解“增量规则”（delta rule）与“门控规则”（gated rule）之间的互补优势，我们针对 RULER 基准测试集中的 Single Needle-In-A-Haystack, S-NIAH 任务进行了一项案例研究。在该任务中，一个“键值对”充当了“大海”（即上下文）中的那根“针”；当给定该“键”时，模型必须能够回忆起对应的“值”。表 2 展示了相关结果，我们从中总结出三点主要观察结论：

**Decay hurts memory retention**。在包含重复合成上下文的最简 S-NIAH-1 设置中，模型仅需记忆极少量信息，这旨在考察其长期留存能力。DeltaNet 在各种序列长度下均实现了近乎完美的表现。Mamba2 在序列长度超过 2K 时性能显著下降，这是因为它对历史信息的遗忘速度过快；而得益于 Delta 规则（delta rule）的应用，Gated DeltaNet 的性能降幅则相对较小。

**Gating facilitates filtering**。在涉及“现实世界文章”（real-world-essay）语境的 S-NIAH-2/3 任务中，模型需要存储所有潜在相关的信息，这对记忆管理效率提出了考验。在状态大小固定的情况下，若缺乏清理机制，便会引发记忆冲突——导致信息相互叠加，无法区分。由于记忆清理能力不足，DeltaNet 在处理较长序列时性能会显著下降。而 Mamba2 和 Gated DeltaNet 则通过门控机制过滤掉无关信息，从而保持了更优的性能。

**Delta rule helps memorization**。在 S-NIAH-3 任务中，数值由数字变为 UUID，旨在测试对复杂模式的记忆能力。Mamba2 的性能迅速下降，而 Gated DeltaNet 的表现则更胜一筹，这证实了 Delta 规则确实具备更强的记忆能力。

### 3.3  ALGORITHM: HARDWARE-EFFICIENT CHUNKWISE TRAINING

在本小节中，我们推导用于训练 Gated DeltaNet 的硬件高效分块(chunkwise)算法。通过对式 10 中的递推关系进行部分展开,我们得到:

```math
\mathbf{S}^{r}_{[t]} = \mathbf{S}_{[t]} \underbrace{\left( \prod_{i=1}^{r} \alpha^{i}_{[t]} \left( \mathbf{I} - \beta^{i}_{[t]} \boldsymbol{k}^{i}_{[t]} \boldsymbol{k}^{i\mathsf{T}}_{[t]} \right) \right)}_{:=\mathbf{F}^{r}_{[t]}} + \underbrace{\sum_{i=1}^{r} \left( \beta^{i}_{[t]} \boldsymbol{v}^{i}_{[t]} \boldsymbol{k}^{i\mathsf{T}}_{[t]} \prod_{j=i+1}^{r} \alpha^{j}_{[t]} \left( \mathbf{I} - \beta^{j}_{[t]} \boldsymbol{k}^{j}_{[t]} \boldsymbol{k}^{j\mathsf{T}}_{[t]} \right) \right)}_{:=\mathbf{G}^{r}_{[t]}}
```

容易看出 $\mathbf{F}^{r}_{[t]} = \gamma^{r}_{[t]} \mathbf{P}^{r}_{[t]} = \overleftarrow{\mathbf{P}^{r}_{[t]}}$。对于 $\mathbf{G}^{r}_{[t]}$,我们按如下方式改写式 5：

```math
\mathbf{G}^{r}_{[t]} = \sum_{i=1}^{r} \frac{\gamma^{r}_{[t]}}{\gamma^{i}_{[t]}} \tilde{\mathbf{u}}^{i}_{[t]} \boldsymbol{k}^{i\mathsf{T}}_{[t]} \in \mathbb{R}^{d_v \times d_k}
\qquad
\tilde{\mathbf{u}}^{r}_{[t]} = \beta^{r}_{[t]} \left( \boldsymbol{v}^{r}_{[t]} - \sum_{i=1}^{r-1} \left( \tilde{\mathbf{u}}^{i}_{[t]} \left( \frac{\gamma^{r}_{[t]}}{\gamma^{i}_{[t]}} \boldsymbol{k}^{i\mathsf{T}}_{[t]} \boldsymbol{k}^{r}_{[t]} \right) \right) \right) \in \mathbb{R}^{d_v}
```

(证明见 §A)。通过 UT 变换,我们得到其矩阵形式：

```math
\widetilde{\mathbf{U}_{[t]}} = \left[ \mathbf{I} + \operatorname{strictLower}\left( \operatorname{diag}\left( \beta_{[t]} \right) \left( \boldsymbol{\Gamma}_{[t]} \odot \mathbf{K}_{[t]} \mathbf{K}^{\mathsf{T}}_{[t]} \right) \right) \right]^{-1} \operatorname{diag}\left( \beta_{[t]} \right) \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}
```

类似于 Mamba2 对线性注意力(式 1)的扩展，我们可以将 DeltaNet 的分块算法(式 8-9)改造用于 Gated DeltaNet,以实现硬件高效的训练,具体如下：

```math
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \left( \widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}^{\mathsf{T}}_{[t]} \right)^{\mathsf{T}} \overrightarrow{\mathbf{K}_{[t]}}
\qquad \in \mathbb{R}^{d_v \times d_k}
```

```math
\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}^{\mathsf{T}}_{[t]} + \left( \mathbf{Q}_{[t]} \mathbf{K}^{\mathsf{T}}_{[t]} \odot \mathbf{M} \right) \left( \widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}^{\mathsf{T}}_{[t]} \right)
\qquad \in \mathbb{R}^{C \times d_v}
```

其中 $\overleftarrow{\boldsymbol{q}^{r}_{[t]}} = \gamma^{r}_{[t]} \boldsymbol{q}^{r}_{[t]}$,$\overleftarrow{\mathbf{w}^{r}_{[t]}} = \gamma^{r}_{[t]} \mathbf{w}^{r}_{[t]}$,$\overrightarrow{\boldsymbol{k}^{r}_{[t]}} = \dfrac{\gamma^{C}_{[t]}}{\gamma^{r}_{[t]}} \boldsymbol{k}^{r}_{[t]}$,且 $\overrightarrow{\mathbf{S}_{[t]}} = \gamma^{C}_{[t]} \mathbf{S}_{[t]}$，与式 2 中的定义一致。

### 3.4 GATED DELTA NETWORKS AND HYBRID MODELS

**Token mixer block**。基础版 Gated DeltaNet 沿用了 Llama 的宏观架构，即交替堆叠“Token 混合层”（token mixer layer）与 SwiGLU 多层感知机（MLP）层，但将自注意力机制替换为“门控增量规则”（gated delta rule）进行 Token 混合。图 1（右）展示了其模块设计。对于门控增量规则（公式 10），查询（query）、键（key）和值（value）——即 $\{\textbf q, \textbf k, \textbf v\}$——通过线性投影、短卷积和 SiLU 激活函数生成；其中，$\textbf q$ 和 $\textbf k$ 经过 L2 归一化处理，以确保训练的稳定性。$\alpha$ 和 $\beta$ 则仅通过线性投影生成。参考 Sun 等人（2023a）的做法，输出在经过归一化和门控处理后，再进行最终的输出投影。

**Hybrid models**。线性 Transformer 在建模局部偏移和比较方面存在局限性，且其固定的状态大小也给检索任务带来了困难。借鉴 Griffin 和 Samba 等近期的混合架构，我们将线性循环层与滑动窗口注意力机制（SWA）相结合，构建了 GatedDeltaNet-H1；此外，我们还将 Mamba2、GatedDeltaNet 和 SWA 进行堆叠，构建了 GatedDeltaNet-H2。

## 4.EXPERIMENTS
