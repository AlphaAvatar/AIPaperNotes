论文链接：https://arxiv.org/pdf/2503.14476

代码链接：https://dapo-sia.github.io/

# 摘要

推理缩放赋予大语言模型（LLM）前所未有的推理能力，强化学习是实现复杂推理的核心技术。然而，目前最先进的推理 LLM 的关键技术细节往往被隐藏（例如在OpenAI o1博客和DeepSeek R1技术报告中），因此，社区仍然难以复现其强化学习训练结果。我们提出了一种**D**ecoupled **C**lip and **D**ynamic s**A**mpling **P**olicy **O**ptimization (DAPO) 算法，并完全开源了一个最先进的大规模强化学习系统，该系统使用 Qwen2.5-32B 基础模型在AIME 2024数据集上取得了50分的成绩。与以往隐藏训练细节的工作不同，我们介绍了算法的四项关键技术，这些技术使得大规模LLM强化学习得以成功。此外，我们还开源了基于verl框架的训练代码，以及一个精心整理和处理的数据集。我们开源系统的这些组成部分增强了可复现性，并为未来大规模LLM强化学习的研究提供了支持。

# 1.介绍

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a7e35b2fa3e48fa93bac0b6de0ecf92.png)

诸如 OpenAI 的 o1 和 DeepSeek 的 R1 等测试时扩展技术，为大语言模型 (LLM) 带来了深刻的范式转变。测试时扩展技术能够支持更长的思维链，并诱导出更复杂的推理行为，这使得这些模型在 AIME 和 Codeforces 等数学和编程竞赛任务中表现优异。

推动这场变革的核心技术是大规模强化学习（RL），它能够激发诸如自我验证和迭代改进等复杂的推理行为。然而，可扩展 RL 训练的实际算法和关键方法仍然是个谜，在现有推理模型的技术报告中鲜有提及。本文揭示了大规模 RL 训练中存在的主要障碍，并开源了一个可扩展的 RL 系统，该系统包含完全开源的算法、训练代码和数据集，旨在提供具有行业级RL成果的民主化解决方案。

我们使用 Qwen2.5-32B 作为强化学习 (RL) 的预训练模型进行实验。在最初的 GRPO 测试中，我们在 AIME 数据集上仅获得 30 分，远低于 DeepSeek 的 RL 模型（47 分）。深入分析表明，简单的 GRPO 基线模型存在熵崩溃、奖赏噪声和训练不稳定等几个关键问题。更广泛的研究群体在复现 DeepSeek 的结果时也遇到了类似的挑战，这表明 R1 论文中可能遗漏了开发工业级、大规模且可复现的 RL 系统所需的关键训练细节。

为了弥补这一差距，我们发布了一个开源的、最先进的大规模 LLM 强化学习系统。该系统基于 Qwen2.5-32B 模型，在AIME 2024数据集上取得了50分，超越了之前由DeepSeek-R1-Zero-Qwen-32B 取得的最先进结果（47分），且仅使用了50%的训练步数（图1）。我们提出了解耦裁剪和动态采样策略优化（**DAPO**）算法，并引入了4项关键技术，使强化学习在长CoT场景下表现出色。详细信息请参见第3节。

1. **Clip-Higher**。这促进了系统的多样性，避免了熵崩溃；
2. **Dynamic Sampling**。这有助于提高训练效率和稳定性；
3. **Token-Level Policy Gradient Loss**。这在长 CoT 强化学习场景中至关重要；
4. **Overlong Reward Shaping**。这样可以减少奖励噪音，稳定训练。

我们的实现基于 verl。通过全面发布我们最先进的强化学习系统，包括训练代码和数据，我们旨在揭示对大规模 LLM 强化学习有价值的见解，从而造福更广泛的社区。

# 2.Preliminary

## 2.1 Proximal Policy Optimization (PPO)

PPO 引入了一种用于策略优化的裁剪代理目标。通过**使用裁剪将策略更新限制在先前策略的邻近区域内**，PPO 可以稳定训练并提高样本效率。具体来说，PPO 通过最大化以下目标来更新策略：

$$\mathcal J_{PPO}(\theta)=\mathbb E_{(q,a)\sim\mathcal D,o_{\le t}\sim\pi_{\theta_{old}}(\cdot|q)}[min(\frac{\pi_{\theta}(o_t|q,o_{\le t})}{\pi_{old}(o_t|q,o_{\le t})}\hat A_t,clip(\frac{\pi_{\theta}(o_t|q,o_{\le t})}{\pi_{old}(o_t|q,o_{\le t})}, 1-ε,1+ε)\hat A_t)],\tag{1}$$

其中 $(q, a)$ 是来自数据分布 $\mathcal D$ 的问答对，$ε$ 是重要性抽样率的裁剪范围，$\hat A_t$ 是时刻 $t$ 的优势估计值。给定价值函数 $V$ 和奖赏函数 $R$，$\hat A_t$ 使用广义优势估计 (GAE) 计算：

$$\hat A^{GAE(γ,λ)}_t=\sum^{∞}_{l=0}(γλ)^lδ_{t+l},\tag{2}$$

其中，

$$δ_l = R_l + γV(s_l+1) − V(s_l),\quad 0 ≤ γ, λ ≤ 1.\tag{3}$$

## 2.2 Group Relative Policy Optimization (GRPO)

与 PPO 相比，**GRPO 消除了价值函数**，并以租相对的方式估计优势函数。对于特定的问答对 $(q, a)$，行为策略 $π_{θ_{old}}$ 对 $G$ 个个体响应 $\{o_i\}^G_{i=1}$ 进行采样。然后，通过对组层面的奖励 $\{R_i\}^G_{i=1}$ 进行归一化来计算第 $i$ 个响应的优势：

$$\hat A_{i,t}=\frac{r_i-mean(\{R_i\}^G_{i=1})}{std(\{R_i\}^G_{i=1})}.\tag{4}$$

与 PPO 类似，GRPO 采用截断目标函数，并直接施加 KL 惩罚项：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{(q,a)\sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \big( r_{i,t}(\theta)\hat{A}_{i,t}, \mathrm{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t} \big) - \beta D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}) \right) \right],\tag{5}$$

其中，

$$r_{i,t}(\theta) =
\frac{
\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})
}{
\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})
}.\tag{6}$$

值得注意的是，**GRPO是在样本级别计算目标函数的**。更准确地说，GRPO首先计算每个生成序列内的平均损失，然后再对不同样本的损失进行平均。正如我们将在3.3节讨论的那样，这种差异可能会影响算法的性能。

## 2.3 Removing KL Divergence

KL 惩罚项用于调节在线策略与冻结参考策略之间的偏差。在RLHF场景中，强化学习的目标是在不偏离初始模型过远的情况下调整模型行为。然而，在训练长CoT推理模型的过程中，模型分布可能与初始模型存在显著偏差，因此这种限制并非必要。故此，我们将从提出的算法中排除KL项。

## 2.4 Rule-based Reward Modeling

使用奖赏模型通常会遇到奖赏作弊问题。因此，我们直接使用可验证任务的最终准确率作为结果奖赏，计算规则如下：

$$
R(\hat{y}, y) =
\begin{cases}
1, & \text{if } \mathrm{is\_equivalent}(\hat{y}, y) \\
-1, & \text{otherwise}
\end{cases}\tag{7}
$$

其中 $y$ 是真实答案，$\hat y$ 是预测答案。这已被证明是激活基础模型推理能力的有效方法，如自动定理证明、计算机编程和数学竞赛等多个领域所示。

# 3.DAPO

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1a18f3ba9b1411a9791b802456f8542.png)

我们提出了一种**解耦裁剪**和**动态采样**策略优化（DAPO）算法。DAPO 为每个问题 $q$ 及其答案 $a$ 生成一组输出 $\{o_i\}^G_{i=1}$，并通过以下目标函数优化策略：

$$
\mathcal{J}_{\text{DAPO}}(\theta) =
\mathbb{E}_{(q,a)\sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)}
\left[
\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\left(
r_{i,t}(\theta)\hat{A}_{i,t},
\mathrm{clip}\!\big(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\big)\hat{A}_{i,t}
\right)
\right],\tag{8}
$$

$$
\text{s.t.} \quad
0 < \big|\{ o_i \mid \mathrm{is\_equivalent}(a, o_i) \}\big| < G.
$$

其中，

$$
r_{i,t}(\theta) =
\frac{
\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})
}{
\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})
},
\quad
\hat{A}_{i,t} =
\frac{
R_i - \mathrm{mean}(\{R_i\}_{i=1}^{G})
}{
\mathrm{std}(\{R_i\}_{i=1}^{G})
}.\tag{9}
$$

完整的算法见算法 1。本节将介绍与 DAPO 相关的关键技术。

## 3.1 Raise the Ceiling: Clip-Higher

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7733c88070a54f27bc720c4584747bc6.png)

在我们最初使用 PPO 或GRPO​​ 的实验中，我们观察到了熵崩溃现象：**随着训练的进行，策略的熵迅速下降（图2b）。某些组的采样响应趋于完全相同**。这表明探索有限，并且早期策略确定性较强，这可能会阻碍扩展过程。

我们提出了一种名为 **Clip-Higher** 的策略来解决这个问题。在 Clipped Proximal Policy Optimization (PPO-Clip) 中，我们引入了对重要性采样率的裁剪，以限制信赖域并增强强化学习的稳定性。我们发现，上限裁剪会限制策略的探索范围，因为提高“利用” token 的概率相对容易，而不太可能出现的“探索” token的概率却被限制得太紧，难以提升。

具体而言，当 $ε = 0.2$（大多数算法的默认值）且 $\hat A_{i,t} > 0$（系统尝试提高概率）时，考虑概率分别为 $π_{θ_{old}} (o_i | q) = 0.01$ 和 $0.9$ 的两个动作。概率 $π_θ(o_i | q)$ 提高的上限分别为 0.012 和 1.08 (即，$π_{θ_{old}} · (1 + ε)$)。这意味着概率较高的“利用” token（例如 0.9）的概率并非必然达到极高的水平，例如 0.999。相反，对于概率较低的“探索” token，要实现显著的概率提升则更具挑战性。经验上，我们也观察到向上裁剪token 的平均概率较低：$π_θ(o_i | q) < 0.2$（图 3a）。这一发现支持了我们的直觉，即上限裁剪阈值确实限制了低概率“探索” token的概率增加，从而可能限制了对系统的探索。

遵循 Clip-Higher 策略，我们将较低和较高的裁剪范围解耦为 $ε_{low}$ 和 $ε_{high}$，如公式 10 所示：

$$
\mathcal{J}_{\text{DAPO}}(\theta) =
\mathbb{E}_{(q,a)\sim \mathcal{D}, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot|q)}
\left[
\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\left(
r_{i,t}(\theta)\hat{A}_{i,t},
\mathrm{clip}\!\big(
r_{i,t}(\theta),
1-\textcolor{red}{\varepsilon_{\text{low}}},
1+\textcolor{red}{\varepsilon_{\text{high}}}
\big)\hat{A}_{i,t}
\right)
\right],
$$

$$
\text{s.t.} \quad
0 < \big|\{ o_i \mid \mathrm{is\_equivalent}(a, o_i) \}\big| < G.
\tag{10}
$$

我们增大 $ε_{high}$ 的值，以便为低概率 token 的增加留出更多空间。如图 2 所示，这一调整有效地提高了策略的熵，并有助于生成更多样化的样本。我们保持 $ε_{low}$ 的值不变，因为增大 $ε_{low}$ 会将这些 token 的概率抑制到 0，导致采样空间坍缩。

## 3.2 The More the Merrier: Dynamic Sampling

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ec624f5cc7d4cefb79872e68b5e11da.png)

当某些提示的准确率等于 1 时，现有的强化学习算法会面临梯度下降问题。例如，对于 GRPO 算法，如果某个提示的所有输出 $\{o_i\}^G_{i=1}$ 都正确且获得相同的奖赏，则该组的优势为零。零优势会导致策略梯度为零，从而减小当前 batch 梯度的幅度并增加其对噪声的敏感性，进而降低样本效率。如图 3b 所示，经验表明准确率为 1 的样本数量会持续增加。这意味着每个 batch 中有效提示的数量会不断减少，这会导致梯度方差增大，并抑制模型训练的梯度信号。

为此，我们提出对提示进行过采样，并过滤掉准确率等于 1 和 0 的提示（如公式 11 所示），从而保留 batch 中所有具有有效梯度的提示，并保持提示数量一致。每个 batch 的采样成本是动态的。在训练之前，我们会持续采样，直到 batch 完全被准确率既非 0 也非 1 的样本填充。

$$
\mathcal{J}_{\text{DAPO}}(\theta) =
\mathbb{E}_{(q,a)\sim \mathcal{D}, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot|q)}
\left[
\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\left(
r_{i,t}(\theta)\hat{A}_{i,t},
\mathrm{clip}\!\big(
r_{i,t}(\theta),
1-\varepsilon_{\text{low}},
1+\varepsilon_{\text{high}}
\big)\hat{A}_{i,t}
\right)
\right],
$$

$$
\text{s.t.} \quad
\textcolor{red}{
0 < \big|\{ o_i \mid \mathrm{is\_equivalent}(a, o_i) \}\big| < G
}.
\tag{11}
$$

需要注意的是，这种策略并不一定会降低训练效率，因为如果强化学习系统是同步的且生成阶段不是流水线式的，那么生成时间通常主要取决于长尾样本的生成。此外，如图 6 所示，我们发现采用动态采样后，实验能够更快地达到相同的性能。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1d70baba3d948bba8a8fac1da0f098c.png)

## 3.3 Rebalancing Act: Token-Level Policy Gradient Loss

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5f4c295b680d484c8eb2f367e8a2ab18.png)

原始的 GRPO 算法采用样本级损失计算方法，首先对每个样本内的每个 token 的损失进行平均，然后将所有样本的损失进行聚合。在这种方法中，每个样本在最终损失计算中被赋予相同的权重。然而，我们发现，在长 CoT 强化学习场景下，这种损失降低方法会带来一些挑战。

由于所有样本在损失计算中都被赋予相同的权重，因此较长响应（包含更多 token ）中的 token 对整体损失的贡献可能不成比例地偏低，这会导致两个不利影响。首先，对于高质量的长样本，这种影响会阻碍模型学习其中与推理相关的模式。其次，我们观察到过长的样本通常包含低质量的模式，例如无意义的词语和重复的词语。因此，由于样本级损失计算无法有效惩罚长样本中的这些不良模式，会导致熵和响应长度的异常增加，如图 4a 和图 4b 所示。

为了解决上述局限性，我们在长CoT RL场景中引入了Token级策略梯度损失：

$$
\mathcal{J}_{\text{DAPO}}(\theta) =
\mathbb{E}_{(q,a)\sim \mathcal{D}, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot|q)}
\left[
\textcolor{red}{
\frac{1}{
\sum_{i=1}^{G} |o_i|
}
\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
}
\min\!\left(
r_{i,t}(\theta)\hat{A}_{i,t},
\mathrm{clip}\!\big(
r_{i,t}(\theta),
1-\varepsilon_{\text{low}},
1+\varepsilon_{\text{high}}
\big)\hat{A}_{i,t}
\right)
\right],
$$

$$
\text{s.t.} \quad
0 < \big|\{ o_i \mid \mathrm{is\_equivalent}(a, o_i) \}\big| < G.
\tag{12}
$$

在这种情况下，较长的序列对整体梯度更新的影响可能比短序列更大。此外，从单个 token 的角度来看，如果某种特定的生成模式能够导致奖励的增加或减少，那么无论它出现在响应中的长度如何，它都会被同等地触发或抑制。

## 3.4 Hide and Seek: Overlong Reward Shaping

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bbfd71324d5b41509b35041713961143.png)

在强化学习训练中，我们通常会设置样本的最大长度，过长的样本会被相应地截断。**我们发现，对截断后的样本进行不恰当的奖赏设计会引入奖赏噪声，并显著干扰训练过程**。

默认情况下，我们会对截断的样本赋予惩罚性奖赏。这种方法可能会给训练过程引入噪声，因为一个合理的推理过程可能仅仅因为过长而受到惩罚。这种惩罚可能会使模型对其推理过程的有效性产生混淆。

为了探究这种奖赏噪声的影响，我们首先应用了一种**过长过滤策略**，该策略可以掩盖截断样本造成的损失。如图 5 所示，我们发现这种方法显著提高了训练的稳定性并增强了性能。

此外，我们提出了一种名为“**Soft Overlong Punishment**”（公式 13）的机制，这是一种长度感知惩罚机制，旨在调整截断样本的奖励。具体而言，当响应长度超过预定义的最大值时，我们定义一个惩罚区间。在该区间内，响应越长，受到的惩罚越大。该惩罚会添加到基于规则的原始正确性奖励中，从而提醒模型避免过长的响应。

$$
R_{\text{length}}(y) =
\begin{cases}
0, &
|y| \le L_{\max} - L_{\text{cache}} \\[6pt]
\dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, &
L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\[10pt]
-1, &
L_{\max} < |y|
\end{cases}
\tag{13}
$$

## 3.5 Dataset Transformation

我们的数据集来源于网络和官方竞赛网站，通过网络爬虫和人工标注相结合的方式获取。数学数据集的答案通常以多种格式出现，例如表达式、公式和数字，这使得设计全面的解析规则极具挑战性。为了利用规则提供准确的奖励信号并最大限度地减少公式解析器引入的错误，我们借鉴AIME的思路，选择答案并将其转换为易于解析的整数。例如，如果原始答案的形式为 $\frac{a+\sqrt b}{c}$，我们会指示LLM修改问题，使预期答案变为 $a + b + c$。经过选择和转换，我们得到了DAPO-Math-17K数据集，其中包含17000个题目，每个题目都对应一个整数答案。

# 4.Experiments

## 4.1 Training Details

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/359903f3996645ceaa364b6e0d75f210.png)

本文重点关注数学任务，以评估我们的算法，该算法可轻松迁移到其他任务。我们采用 verl 框架进行训练。我们使用朴素 GRPO 作为基线算法，并使用组奖赏归一化来评估算法优势。

对于超参数，我们使用 AdamW 优化器，学习率为 $1 × 10^{-6}$，并包含 20 个 rollout 步骤的线性预热。rollout 阶段，提示 batch size 为 512，每个提示采样 16 个响应。训练阶段，mini-batch 大小设置为 512，即每个 rollout 步骤进行 16 次梯度更新。对于 **Overlong Reward Shaping**，我们将预期最大长度设置为 16,384 个 token，并额外分配 4,096 个 token 作为软惩罚缓存。因此，生成的最大 token 数设置为 20,480 个 token。对于 **Clip-Higher** 机制，我们将裁剪参数 $ε_{low}$ 设置为 0.2，$ε_{high}$ 设置为 0.28，这有效地平衡了探索和利用之间的权衡。为了在 AIME 数据集上进行评估，我们重复测试 32 次，并报告 avg@32 以验证结果的稳定性。评估的推理超参数设置为 temperature 1.0 和 topp 0.7。
