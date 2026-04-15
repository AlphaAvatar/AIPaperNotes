论文链接：https://fengyao.notion.site/off-policy-rl#246721e3f6c480ff9fabc9015cc3f970

代码链接：https://github.com/yaof20/verl/tree/flash-rl/recipe/flash_rl

# 摘要

在现代强化学习训练框架（例如 VeRL）中，不同的实现方式分别用于生成 rollout（例如 vLLM）和训练模型（例如 FSDP）。本文指出，这种实现方式的差异会隐式地将 on-policy 强化学习转化为 off-policy 强化学习，并讨论了一种简单而有效的权重采样技术来处理这种差异。

# The Mismatch Problem

<img
  src="https://i-blog.csdnimg.cn/direct/45a18dd8cdfb4c69aa6a793615f54d67.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

为简单起见，我们以 REINFORCE 算法为例，该算法通过以下方式更新策略 $\pi$（一个由 $\theta$ 参数化的 LLM）：

```math
\theta \gets \theta + \mu \cdot  \mathbb{E}_{\underbrace{a \sim{\pi}(\theta)}_{rollout}} [R(a)\cdot \underbrace{\nabla_\theta \log {\pi}(a, \theta)}_{\tiny{training}}].
```

实际上，生成 rollout 的成本很高，现代强化学习框架（例如 [VeRL](https://github.com/volcengine/verl)）通常采用高度优化的推理引擎（例如 vLLM、SGLang）来提高吞吐量，同时使用单独的后端（例如 FSDP、Megatron）进行模型训练。这种混合设计使得更新过程更加高效：

```math
\theta \gets \theta + \mu \cdot  \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)].
```

> 这里，我们使用 $\pi_{\rm sampler}$ 表示加载到推理引擎（例如 vLLM、SGLang）中的模型，使用 $\pi_{\rm learner}$ 表示使用训练后端（例如 FSDP、Megatron）实例化的同一模型。除非另有说明，我们的实验均使用 vLLM 和 FSDP 作为采样器和学习器后端。

观察到意外的部署训练不匹配。如图 1 所示，尽管 $\textcolor{blue}{\pi_{\text{fsdp}}}$ 和 $\textcolor{red}{\pi_{\text{vllm}}}$ 共享相同的模型参数 $\theta$，但它们产生的 token 概率却可能显著不同。对于某些 token $a$，它们甚至会产生相互矛盾的预测，即 $\textcolor{red}{\pi_{\text{vllm}}}(a, \theta)\!=\!1$ 和 $\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)\!=\!0$。这种出乎意料的行为隐含地打破了 **on-policy** 假设，使得强化学习训练实际上变成了 **off-policy** 训练。

# How to Fix It?

## Mitigate the system-level mismatch

更高精度的 vLLM 是否有帮助？我们最初假设 vLLM 是根本原因，因此我们对 vLLM 进行了修补，以解决两个常见的导致匹配错误问题的因素。
- **Inaccessible true sampling probabilities**：vLLM v1 引擎[不支持](https://docs.vllm.ai/en/v0.10.0/usage/v1_guide.html?h=immediately#semantic-changes-to-logprobs)直接返回用于采样的调整概率，从而引入了额外的差距。
→ 我们的补丁强制 vLLM 返回用于采样的实际概率 [[已向上游合并](https://github.com/vllm-project/vllm/pull/22387)]。
- **Backend numerical differences**：vLLM lm_head 的精度与 HuggingFace transformers 的精度不匹配，这一点在 MiniMax-M1 技术报告（https://arxiv.org/pdf/2506.13585#page=8）中也有说明。
→ 我们的补丁提供了强制 vLLM 将 lm_head 转换为 fp32 的选项。

然而，如图 1 所示，应用这两个补丁后，不匹配问题仍然存在。

## Embrace the mismatch — Apply algorithm-level fix

我们不打算在系统层面缓解分布不匹配的问题，而是提出调整模型更新方式，使其能够感知这种不匹配。一种简单的方法是通过重要性采样校正。具体来说，我们通过在模型更新中加入重要性比率来处理 $\textcolor{blue}{\pi_{\text{learner}}}$ 和 $\textcolor{red}{\pi_{\text{sampler}}}$ 之间的不匹配，即改变当前的梯度计算方式。

```math
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)],
```

为

```math
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)} \cdot R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr].
```

虽然已经对如何设计稳定有效的重要性抽样进行了广泛的研究，但在实践中，我们发现通常使用经典技术[截断重要性抽样](https://ionides.github.io/pubs/ionides08-jcgs.pdf)就足够了：

```math
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\underbrace{\min\Bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}, C\Bigr)}_{\text{truncated importance ratio}} \cdot R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr],
```

其中 $C$ 为超参数。

## Extension to Other Algorithms

上述分析很容易推广到其他算法，因为可以将梯度计算的具体形式从 REINFORCE $R(a) \cdot\nabla \log {\pi}(a, \theta)$ 替换为任何形式。这里，我们以常用的 PPO 算法为例，提供类似的分析。

PPO 的策略梯度 $\nabla_\theta L^{\mathrm{CLIP}}(\theta)$ 定义为：

```math
\small{ \mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}}
\Bigl[
\nabla_\theta \min\Bigl(
\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]}.
```

为了提高吞吐量，混合强化学习系统采用 vLLM 引擎生成 rollout——从 $\pi_{\theta_{\text{old}}}$ 中采样 token $a$，同时使用 FSDP 后端从 $\pi_\theta$ 中采样，并重新计算 $\pi_{\theta_{\mathrm{old}}}$ 的 token 概率以进行梯度计算：

```math
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{sampler}}}(\theta_{\mathrm{old}})}
\Bigl[
\nabla_\theta \min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}.
```

与上述分析类似，$\textcolor{blue}{\pi_{\text{learner}}}$ 和 $\textcolor{red}{\pi_{\text{sampler}}}$ 之间的差距再次出现，我们通过截断重要性采样来解决这个问题：

```math
\small{\mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})},  C\Bigr)}_{\text{truncated importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}
```

其中 $C$ 为超参数。

### Additional Discussion on PG, Sequence, and Token

我们上面的讨论并没有涉及状态和动作的具体形式。我们将探讨 token 级和序列级的策略梯度，它们之间的联系，以及学习器与采样器不匹配的影响：

[策略梯度、序列和 token ——第一部分：基本概念](https://www.notion.so/Policy-Gradient-Sequence-and-Token-Part-I-Basic-Concepts-28b721e3f6c480b88b5be1d89512ac3a?pvs=21)

[策略梯度、序列和 token——第二部分：学习器与采样器不匹配](https://www.notion.so/Policy-Gradient-Sequence-and-Token-Part-II-Learner-Sampler-Mismatch-28b721e3f6c480f8a4b0e1f8301d90ac?pvs=21)

## Connection to Classical Wisdom

### Importance Sampling 

**当直接使用蒙特卡罗方法估计目标分布下的期望值较为困难时，重要性抽样允许我们从另一个分布中进行抽样**。在本例中，目标分布为 $\textcolor{blue}{\pi_{\text{learner}}}$，但从中抽样速度极慢。使用单独的后端（例如 vLLM）进行部署生成意味着我们改用 $\textcolor{red}{\pi_{\text{sampler}}}$ 进行抽样。然后，通过为每个样本赋予一个重要性比率来校正这种差异：

```math
\mathbb{E}_{a \sim \textcolor{blue}{\pi_{\text{learner}}}(\theta)} [R(a)] 
= \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \left[ 
\underbrace{\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}}_{\tiny\text{importance ratio}} \cdot R(a) 
\right].
```

### Decoupled PPO

[Decoupled PPO](https://arxiv.org/pdf/2110.00641) 是利用重要性采样弥合 rollout 生成和梯度计算之间差距的一个特例，已被[AReaL](https://arxiv.org/pdf/2505.24298#page=6)等异步强化学习框架所采用。值得一提的是，AReaL并没有像我们这里讨论的那样实现截断重要性比率。相反，如果重要性比率超过预定义的阈值，AReaL[将完全丢弃训练样本](https://github.com/inclusionAI/AReaL/blob/main/realhf/impl/model/utils/ppo_functional.py#L127)。

# Experiments

我们进一步进行了实证分析，以详细说明分布差距的影响以及所提出的截断重要性抽样 (TIS) 修复的有效性。

## Does the gap matter a lot?

我们使用 Qwen2.5-32B 稠密模型，并采用流行的 [DAPO](https://github.com/volcengine/verl/tree/main/recipe/dapo) 配方进行实验。数据处理遵循 [社区指南](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/discussions/3)。结果可视化结果如图 1 所示。

由于资源限制，我们只完成了训练的前 250 步，但考虑分布差距的修正TIS已经显著提升了性能。由于这两次运行的唯一区别在于引入了 TIS，即$\min(\frac{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)}{\textcolor{red}{\pi_{\text{vllm}}}(a, \theta)}, C)$，因此性能的提升凸显了分布差距的潜在影响。

## How well can TIS fix it?

<img
  src="https://i-blog.csdnimg.cn/direct/4659e98a2aec4860a48f54f33db6632d.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们设计了一个对照实验来衡量 TIS 解决该问题的效果。我们按照 [verl 教程中的 GSM8K 示例](https://verl.readthedocs.io/en/latest/start/quickstart.html) 进行 RL 训练，并使用了两种不同的设置：
1. **常规强化学习训练**：最大 token 概率差异显著小于之前的设置（在 Qwen-2.5-32B 密集数据集上的 DAPO 为 1.0）（~0.4）。
2. [**使用 INT8 量化 rollouts 而非 bf16 rollouts 进行强化学习训练**](https://fengyao.notion.site/flash-rl)：最大 token 概率差异显著大于常规强化学习训练（1.0）。

我们在设置 1 中进行常规 PPO 训练，该设置“几乎”是 on-policy 进行的；在设置 2 中，我们同时进行常规 PPO 训练和采用截断重要性采样的 PPO 训练，设置 2 的 rollout 和梯度计算存在较大差异。

如图 2 所示，在设置 2 中执行 PPO 会导致性能显著下降，与设置 1 中的 PPO 相比。同时，应用截断重要性采样可以极大地缩小这种差距，使设置 2 的性能与设置 1 的性能相近。

更多分析请参见下文的“分析部分”。

## Does TIS always help? 

<img
  src="https://i-blog.csdnimg.cn/direct/e3fe712964d84bd89f194b1debb59c88.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**我们还观察到，在概率差异相对较小的情况下，引入额外的截断重要性采样项并不能带来性能提升**。同时值得一提的是，在严格的基于策略的强化学习设置中，重要性采样比率项的值将为 1.0。

# TIS Analysis

## Analysis about different TIS-Variants

我们还总结了两种缓解分布差距的替代方案。

1. **PPO重要性采样（PPO-IS）**

```math
\small{ \mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}
```

> *注：Colossal 框架使用了这种实现方式。

2. **Vanilla 重要性抽样（vanilla-IS）**

```math
\small{\mathbb{E}_{\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})} }_{\text{importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}
```

> *Note: Nemo-RL 使用了这种实现方式。

为了评估 TIS 的有效性并了解其设计选择的影响，我们进行了实验，将 TIS 与上述两种变体进行了比较。TIS的性能始终优于这两种变体，尤其是在差距较大的情况下（例如，FP8/INT8）。

<img
  src="https://i-blog.csdnimg.cn/direct/9f74672855a9435eaf4dbd6175d2a754.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

### 为什么这里的两个变体（PPO-IS 和 vanilla-IS）会导致训练不稳定？

1. **Vanilla-IS v.s. TIS**

对于原始重要性采样（vanilla-IS），其不稳定性主要源于以下情况：当 rollout $a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})$ 的采样概率较低时，重要性比率较大，导致梯度方差放大 $(\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})})^2$。因此，我们在截断重要性采样（TIS）中使用 clamp 操作来实现训练稳定性。例如，当一个 token 的比率 $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}$ 达到 16 时，该 token 的梯度噪声将通过 Vanilla-IS 放大 256 倍，通过 TIS-2 放大 4 倍，或通过 TIS-8 放大 64 倍。

2. **PPO-IS v.s. TIS**

自博客发布以来，很多人询问我们为什么不直接将重要性采样融入 PPO 算法（即上文提到的 PPO-IS 变体）。坦白说，我们最初尝试像 PPO-IS 那样直接修改PPO的裁剪参数，但在我们的实验环境中，这种方法效果并不理想。

至于其根本原理，通过执行 PPO-IS，梯度实际上仍然与 on-policy 的 PPO 版本存在偏差。换句话说，尽管它可能仍然能够优化到无偏目标，但与 PPO 相比，其效率可能较低。

此外，我们注意到，**PPO 信赖域技术旨在约束 rollout $\theta_{\rm old}$与当前模型 $\theta$ 之间的概率比接近 1**，以近似 on-policy REINFORCE 梯度。然而，在 PPO-IS 中，即使 $\theta=\theta_{\rm old}$，由于不匹配，概率比$\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\rm old})}$也已不等于1——这导致裁剪的概率很高，训练信息量也大大减少。此外，在我们的 TIS 方法中，我们分别裁剪 $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}$ 和 $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}$，因此更加温和；注意当 $\theta=\theta_{\rm old}$ 时，$\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}$ 等于 1，这适用于信赖域约束。

## From Ill-conditioned to Benign

除了加速 rollout 之外，rollout 量化还可以作为有效的测试平台，用于检验 rollout 生成和梯度计算之间分布差距的影响。我们证明，使用量化 rollout 的强化学习训练会表现出一些特征性的不稳定性，这些不稳定性在其他未解决此差距的场景中也经常观察到。此外，引入 TIS 项可以使强化学习训练更加稳定和良性。

## Entropy Collapse and Abnormal Response Length

先前的许多研究表明，LLM 中的 RL 训练会导致熵崩溃—— Token 级别的类别分布接近 one-hot 分布，有效地限制了 RL 训练的探索。

我们的 INT8 rollout 实验揭示了严重的熵崩溃现象。图 5 显示熵值降至 0.2 以下，并在整个训练过程中持续下降。我们还观察到异常长的响应生成时间——这是强化学习训练中的另一种故障模式。引入 TIS 项可以逆转这一趋势，使模型能够以稳定且良性的方式进行训练。

<img
  src="https://i-blog.csdnimg.cn/direct/49f3ee28ed0f435a90c3e4cbd7b21fc8.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

相比之下，BF16 的 rollout 实验并未出现严重的熵崩溃。尽管如此，TIS 项仍然增加了熵值。与 INT8 的 rollout 相比，由于分布间隙较小，响应长度仍保持在合理范围内。

<img
  src="https://i-blog.csdnimg.cn/direct/0f576a0df0574f2980d46d7f7a3cf5d2.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## On the Impact of Distribution Gap: A Case Study on KL Estimation

对于 $\rm{KL}(\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}} \Vert \textcolor{blue}{\pi^{\rm fsdp}})$，一个无偏的 kl 估计量是 $k_1$ [估计量](http://joschu.net/blog/kl-approx.html): $\log \textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}(a) - \log \textcolor{blue}{\pi^{\rm fsdp}} (a)$，其中 $a\sim \textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}(a)$。然而，现代 RL 训练框架从 $\textcolor{red}{\pi_{\rm old}^{\rm vllm}}$ 而不是 $\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}$ 生成 rollout，从而给 kl 估计引入偏差，类似于前面讨论的梯度估计偏差。

因此，我们可以以 KL 估计为例，探讨$\textcolor{red}{\pi_{\rm old}^{\rm vllm}}$和$\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}$之间不匹配的影响。**在没有任何偏差的情况下，根据定义，KL散度是非负的**。然而，INT8 部署中显著的分布不匹配导致有偏的 $k_1$ 估计器频繁产生负值，如图 5 所示。这些负的 KL 估计值表明训练动态存在病态条件。

同时，当将 TIS 融入强化学习训练中时，尽管 $k_1$ 估计器仍然受到底层分布不匹配的影响，但在大部分训练过程中仍保持正值。这种对预期符号的保留表明TIS成功地恢复了良好的训练行为。

## Biased Reward in Training Log

集成 TIS的一个有趣现象是，它可能导致奖赏日志记录质量下降，但同时带来更好的下游性能。这是因为 $\textcolor{red}{\pi_{\text{sampler}}}$ 和 $\textcolor{blue}{\pi_{\text{learner}}}$ 之间的差距不仅会引入梯度估计偏差，还会影响日志记录中的奖赏估计。特别地，日志记录的奖赏来自 rollout 策略，即 $E_{\textcolor{red}{\pi_{\text{sampler}}}}[{\rm R}]$，而不是 $E_{\textcolor{blue}{\pi_{\text{learner}}}}[{\rm R}]$。具体而言，如图 6 (右侧两个子图) 所示，对数奖赏指标显示 BF16-Rollout 优于 BF16-Rollout w. TIS。然而，如果我们观察 AIME 准确率的下游性能，则 BF16-Rollout w. TIS 的性能显著优于原始的 BF16-Rollout。

## Intuitions of TIS’s Working Mechanism

虽然 TIS 的确切机制仍是一个开放性问题，但我们提供了关于 TIS 如何缓解分配差距的高层次直觉。

尤其需要注意的是，忽略 $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\rm old})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\rm old})} < 1$ 的 rollout 偏差，可能导致熵崩溃，其机制如下：对于具有负优势的 rollout，策略梯度倾向于减小 $\textcolor{blue}{\pi_{\mathrm{learner}}}$。当参数更新后存在较大的分布间隙时，$\textcolor{blue}{\pi_{\mathrm{learner}}}$ 的减小可能无法反映在 $\textcolor{red}{\pi_{\mathrm{sampler}}}$ 中。因此，策略梯度会继续指向 $\textcolor{blue}{\pi_{\mathrm{learner}}}$ 的进一步减小。直观地说，这种惩罚可能会迫使模型过度倾向于熵较小的输出分布。

同时，TIS 坚持使用未截断的重要性比率 $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\rm old})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\rm old})} < 1$，从而消除了该子集 rollout 的偏差，并破坏了这种机制。

# Rollout-Training Mismatch Analysis

我们进行了一系列受控实验，以识别导致或加剧 rollout 生成与梯度计算之间差异的因素。**具体而言，我们发现并行策略的差异和较长的响应长度是造成这种不匹配的原因**，而采样器后端的选择本身影响甚微。

## Analysis Setup

**Model & Data**。我们使用两个具有代表性的模型——[DAPO-32B](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B) 和 [Polaris-7B](https://huggingface.co/POLARIS-Project/Polaris-7B-Preview) 进行实验，这两个模型分别使用 [DAPO](https://arxiv.org/pdf/2503.14476) 和 [POLARIS](/1dfa954ff7c38094923ec7772bf447a1?pvs=25) 强化学习配方进行训练。为了进行评估，我们使用 [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) 数据集中的前 512 个提示来评估采样器和学习器输出之间的差异。

**Metric**。我们使用两个指标来衡量响应层面的不匹配：
- **Max Mismatch per response**：    $\max_{a\,\in\, \text{response}} |p_{\tiny\text{sampler}}(a) - p_{\tiny\text{learner}}(a)|$
- **Mean Mismatch per response**：$\frac{1}{|\text{response}|}\sum _{a\,\in\, \text{response}} |p_{\tiny\text{sampler}}(a) - p_{\tiny\text{learner}}(a)|$

这些指标使我们能够捕捉到最坏情况下的 token 差异以及响应中的平均差异水平。我们针对不同设置下相同提示的响应计算这些指标，以分离出特定因素的影响。

**Visualization**。我们使用右侧所示的可视化格式呈现这两个指标。这是一个便于理解数据的示例。

<img
  src="https://i-blog.csdnimg.cn/direct/7c10549771d643d683778a31bd6c2542.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## Larger Parallelism Difference, Larger Max Gap

<img
  src="https://i-blog.csdnimg.cn/direct/b1ab0e96aec94cc38c93eeffb8fa3b8e.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们观察到采样器和学习器之间的并行性差异对最大不匹配度量有显著影响。

### Simplest Setting

使用 DAPO-32B 模型，我们从最简单的配置开始：采样器运行在 TP1 的 vLLM 上，学习器使用 SP1 的 FSDP。由于采样器和学习器具有相同的并行度设置，我们称之为“same parallelism”，其分布差距归因于并行度差异以外的其他因素。

### Adding Tensor Parallelism

为了研究 TP 差异的影响，我们将采样器从 TP1 更改为 TP2，同时保持学习器为 SP1（Different TP）。如图 7 左图所示，随着并行度差异的增大，最大不匹配度较高（> 0.5）的响应数量也随之增加。相同并行度情况下仅产生一个此类响应，而不同TP情况下则增加到两个。

### Adding Sequence Parallelism

为了研究 [Ulysses序列并行](https://arxiv.org/abs/2309.14509) 差异的影响，我们将学习器从 SP1 更改为 SP8（TP和 SP 不同）。如图 7 中间所示，SP差异的增加使最大不匹配次数从两位数增加到两位数。

### Disentangling Parallelism and Sharding

### Mean Mismatch and KL

## Longer Response, Larger Max Gap

## Altering Sampler Alone, Gap Still There

## What’s More

# Discussion

## The gap can be amplified in MoE RL

虽然我们目前的实验和分析主要集中在密集模型上，但我们认为这种分布差异在多专家强化学习（MoE RL）中也存在，甚至可能更为严重。主要原因有二：

- **动态路由**：与密集模型不同，MoE 使用路由器动态激活特定专家。这种路由机制本质上对精度非常敏感；即使是微小的数值差异也可能导致专家激活结果的显著不同。

- **专门优化的内核**：MoE 模型通常规模庞大，而且现代推理引擎（例如 vLLM）针对 MoE 模型进行了独特的优化，这使得后端数值不一致性更加明显。

这些特性共同作用，会显著放大分布不匹配，使得像 TIS 这样的解决方案在 MoE RL 中尤为重要。

## TIS is orthogonal and compatible with existing GxPOs
