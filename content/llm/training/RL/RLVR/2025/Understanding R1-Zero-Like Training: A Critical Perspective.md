论文链接：https://arxiv.org/pdf/2503.20783

代码链接：https://github.com/sail-sg/understand-r1-zero

# 摘要

DeepSeek-R1-Zero 已证明，大规模强化学习 (RL) 无需有监督微调即可直接提升 LLM 的推理能力。本研究通过分析其两个核心组件：基础模型和强化学习 (RL)，对类似 R1-Zero 的训练进行了批判性研究。我们研究了包括 DeepSeek-V3-Base 在内的多种基础模型，以**了解预训练特性如何影响 RL 性能**。分析表明，DeepSeek-V3-Base 已展现出“顿悟时刻”，而 Qwen2.5 基础模型即使在没有提示模板的情况下也展现出强大的推理能力，这表明预训练可能存在偏差。此外，我们还发现组相对策略优化 (GRPO) 中存在优化偏差，它会在训练过程中人为地增加响应长度（尤其是对于错误输出）。为了解决这个问题，我们引入了 Dr.GRPO，这是一种无偏差的优化方法，可以在保持推理性能的同时提高 token 效率。利用这些见解，我们提出了一种极简的 R1-Zero 方案，使用 7B 基础模型在 AIME 2024 上实现了 43.3% 的准确率，建立了新的最佳水平。

# 1.Introduction
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f0b784d988146c4a748275078356d94.png)

DeepSeek-R1-Zero 引入了类似 R1-Zero 的训练范式，彻底革新了大语言模型 (LLM) 的后训练流程：直接将强化学习 (RL) 应用于基础 LLM，而无需依赖有监督微调 (SFT) 作为初始步骤。这种新范式因其简单易用性和已证实的强化学习扩展现象而备受青睐：随着模型响应长度的不断增加，模型推理能力也随之提升。这种现象还伴随着“顿悟时刻”，即模型学习到诸如自我反思等新兴技能。

在本文中，我们旨在通过研究两个基本组成部分：基础模型和强化学习 (RL) 来理解类似 R1-Zero 的训练。在第一部分中，我们研究了基础模型的各种属性，重点关注 Qwen2.5 模型系列（该系列在近期复现 R1-Zero 的尝试中被使用）以及 DeepSeek-V3-Base（真实的 R1-Zero 模型正是基于该模型进行强化学习调优）。在第二部分中，我们识别了 GRPO 优化中的偏差，这种偏差可能会导致错误响应逐渐增加。为此，我们提出了一种简单的修改方法来消除偏差，即“GRPO Done Right”（Dr. GRPO），从而提高 token 效率（如图 1 所示）。

我们对基础模型和强化学习 (RL) 的分析表明，我们可以使用 R1-Zero 式的极简训练方法：我们在Qwen2.5-Math-7B模型上利用（无偏差的）Dr. GRPO 算法，结合 Qwen-Math 模板，对数学 3-5 级题目进行 RL 训练，并在 8 块 A100 GPU 上仅耗时 27 小时便实现了最佳性能（图 2）。我们希望本文中提出的研究成果、发布的模型以及开源的代码库能够助力该领域的未来研究。

作为概述，我们总结了本文的要点如下：
- （第 2.1 节）模板对于使基础模型能够回答问题而不是完成句子至关重要。此外，所有基础模型在强化学习之前都已经具备了数学求解能力。
- （第 2.2 节）有趣的是，Qwen-2.5 基础模型通过不使用模板立即获得了 ∼ 60% 的改进，这让我们假设它们在烹饪模型时可能会对连接的问答文本进行预训练。
- （第 2.3 节）几乎所有基础模型都已经展现出“顿悟时刻”，包括 DeepSeek-V3-Base。
- （3.1节、3.2节）Dr. GRPO 有效地修复了GRPO在优化中的偏差，实现了更好的 token 效率。
- （第 3.3 节）模型模板不匹配会在 RL 重建之前破坏推理能力。
- （第 3.4 节）对 Llama-3.2-3B 进行数学预训练可提高其 RL 上限。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c63cfb2285548d48d168f46f694c69e.png)

# 2.Analysis on Base Models

在本节中，我们仔细研究了各种基础模型，包括 Qwen-2.5 系列、Llama-3.1 和 DeepSeek 系列，向它们询问从 MATH 训练集中抽取的 500 个问题并分析它们的回答。

## 2.1 R1-Zero Trainability: Templates Construct Exploratory Base Policies

由于从基础模型进行训练是 R1-Zero 类范式的基本设定，我们首先研究广泛使用的开源基础模型（通常用于句子补全，即 $p_θ(x)$）是否可以通过适当的模板有效地激发其问答能力，从而充当问答基础策略 $π_θ(·|q)$。除了 Guo et al. (2025) 中的 R1 模板（模板 1）之外，我们还考虑了 Zeng et al. (2025) 使用的 Qwen-Math 模板（模板 2）以及 No template（模板 3）：

> **Template 1 (R1 template)**. A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within $\text{<think> </think>}$ and answer is enclosed within $\text{<answer> </answer>}$ tags, respectively, i.e., $\text{<think> reasoning process here </think> <answer> answer here </answer>}$.\nUser: {question}\nAssistant: $\text{<think>}$

> **Template 2 (Qwen-Math template)**. <|im start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im end|>\n<|im start |>user\n{question} <|im end|>\n<|im start|>assistant\n

> **Template 3 (No template)**. {question}

**Experimental settings**。我们选取 Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-7B、Llama-3.1-8B、DeepSeek-Math-7B 和 DeepSeek-V3-Base-685B 进行实验。对于每个模型，我们首先应用 No template 获取模型响应，然后使用 GPT-4o-mini 判断模型响应是采用问答格式（无论质量如何）还是采用句子补全模式。我们记录倾向于回答问题的响应百分比作为指标（**Answering Rate**）。然后，我们同时应用 R1 模板和 Qwen-Math 模板获取模型响应，并根据指标确定每个模型最合适的模板。最后，我们使用相应模板评估每个模型的 pass@8 准确率，以评估基础策略是否能够探索出有利于强化学习改进的奖赏轨迹。

**Results**。图 3 左图展示了基础模型（是否使用模板）对所提问题的回答效果。我们观察到，Llama 和 DeepSeek 模型均通过使用合适的模板（R1 模板）提升了回答能力。然而，Qwen2.5 模型在不使用模板时效果最佳（回答率高达 100%）。这一有趣的特性激发了进一步的研究，正如第 2.2 节所述。同时，**不使用模板时最低的回答率表明 DeepSeek-V3-Base 是一个近乎纯粹的基础模型**。这一观察结果促使我们探索像 DeepSeek-V3-Base 这样的纯粹基础模型是否能够展现出“顿悟时刻”（Aha moment）（第 2.3 节）。图 3 中图展示了不同基础模型（使用模板）在不同采样温度下的 pass@8 准确率。该指标可以作为基础策略探索能力的指标。例如，如果一个基础策略甚至无法采样一条能够得出正确最终答案的轨迹，那么强化学习就无法改进该策略，因为没有奖赏信号。我们的结果表明，所有测试模型都是探索性的（因此可以用于强化学习），其中 Qwen2.5 模型表现最佳（甚至超过了 DeekSeek-V3-Base）。这或许可以部分解释为什么大多数 R1-Zero 项目都基于 Qwen2.5 模型。

## 2.2  Qwen-2.5 Models Unlock the Best Performance When Discarding Template

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/849de231f87449deabb68cfacc323674.png)

接下来，我们深入探讨一个有趣的观察结果（参见图 3（左））：所有 Qwen2.5 基础模型即使无需任何模板，也能轻松用作聊天模型。我们进一步评估了 Qwen2.5-Math 模型在五个标准基准测试（AIME 2024、AMC、MATH500、Minerva Math 和 OlympiadBench）上的推理能力。按照惯例，我们使用贪婪解码，并将采样预算限制为 3000 个 token。

如表 1 所示，不使用任何模板可以显著提升平均性能，与传统的 4-shot 提示相比，性能提升约 60%。由于 Qwen2.5-Math 在预训练阶段使用了聊天模型的数据（问答对），我们假设它们可能在拼接文本上进行预训练，以直接最大化 $log~p_θ (\textbf q;\textbf o)$。如果我们的假设成立，我们将更加谨慎地使用 Qwen2.5 模型来复现 DeepSeek-R1-Zero，因为基础模型在没有模板的情况下就已经类似于 SFT。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ef0557a755394ffaa90d7cffd2785328.png)

## 2.3 Aha Moment Already Appears in Base Models Including DeepSeek-V3-Base

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b20b30b87c2e458385ae8c10551afbed.png)

DeepSeek-R1-Zero 最鼓舞人心的成果之一是，通过纯强化学习训练，**它能够激发自我反思行为**，也就是“顿悟时刻”。此前一些研究表明，开源 R1 复现模型中可能不存在“顿悟时刻”，因为它们使用的基础模型已经展现了自我反思关键词。然而，他们尚未测试 DeepSeek-V3-Base，而真正的 R1-Zero 模型正是在其上进行强化学习调优的。我们自行托管了 DeepSeek-V3-Base-685B，并使用 R1 模板测试了 500 道数学题，从而弥补了这一缺失。从图 3 右侧的图中，我们可以看到 DeepSeek-V3-Base 也产生了相当数量的自我反思，这进一步验证了 Liu et al. (2025b) 的论断。我们还在图 4 中展示了 DeepSeek-V3-Base 生成“顿悟”、“等待”和“验证问题”等关键词的示例。

另一个重要问题是，**自我反思行为是否与强化学习训练后模型性能的提升相关**。为了探究这一点，我们托管了 DeepSeek-R1-Zero，并分析了其对 MATH 数据集中相同问题的响应。虽然自我反思行为在 R1-Zero 中出现的频率更高，但我们观察到这些行为并不一定意味着更高的准确率。详细分析请参见附录 D。

# 3.Analysis on Reinforcement Learning

语言模型生成可以表述为一个 token 级的马尔可夫决策过程 (MDP) $\mathcal M = (\mathcal S, \mathcal A,r, \mathcal p_{\mathcal Q})$。在每个生成步骤 $t$，状态 $s_t ∈ \mathcal S$ 是输入问题与迄今为止生成的输出响应的拼接：$s_t = \textbf q; \textbf o_{<t} = [q_1, ..., q_M, o_1, ..., o_{t−1}]$。策略 $π_θ (·|s_t)$ 将从词表 $\mathcal A$ 中选择下一个token $o_t$，从而确定性地过渡到下一个状态 $s_{t+1} = s_t;[o_t]$。生成过程从从一组问题中采样初始状态 $s_1 = \textbf q ∼ p_{\mathcal Q}$ 开始，并在自回归策略生成 [eos] 个 token 或耗尽预算时停止。

通常，我们最大化熵正则化目标：

$$\mathcal J(\pi_{\theta})=\mathbb E_{\textbf q\sim p_{\mathcal Q}}[\mathbb E_{\textbf o\sim\pi_{\theta}(\cdot|\textbf q)}[R(\textbf q,\textbf o)]-\beta\mathbb D_{KL}[\pi_{\theta}(\cdot|\textbf q)||\pi_{ref}(\cdot|\textbf q)]],\tag{1}$$

其中 $R(\textbf q, \textbf o) = \sum^{|o|}_{t=1}r(s_t, o_t)$ 是轨迹 $\textbf q;\textbf o$ 的回报，$π_{ref}$ 是参考策略。对于从人类反馈中进行强化学习，通常采用 KL 正则化项（$β > 0$），其中 $r$ 是从 $π_{ref}$ 收集的数据中学习到的**奖赏模型**。在这种情况下，正则化有助于防止 $π_θ$ 偏离奖赏模型准确的分布太远。然而，强化学习调优推理模型通常使用基于规则的**验证器**作为 $r$，从而消除了分布偏移的担忧。这使我们能够删除 KL 项，这不仅节省了 $π_{ref}$ 在训练期间所需的内存和计算量，而且还可能为类似 R1-Zero 的训练带来更好的性能。我们将在本文中通篇假设 $β = 0$。

**Policy optimization algorithms**。为了根据上述目标（等式 (1)，$β = 0$）优化 $π_θ$，近端策略优化 (PPO) 会最大化以下替代目标：

$$\mathcal J_{PPO}(\pi_{\theta})=\mathbb E_{\textbf q\sim p_{\mathcal Q},\textbf o\sim\pi_{\theta}(\cdot|\textbf q)}\\
\sum^{|\textbf o|}_{t=1}\{min[\frac{\pi_{\theta}(o_t|\textbf q,\textbf o_{<t})}{\pi_{\theta_{old}}(o_t|\textbf q,\textbf o_{<t})}\hat A_t,clip(\frac{\pi_{\theta}(o_t|\textbf q,\textbf o_{<t})}{\pi_{\theta_{old}}(o_t|\textbf q,\textbf o_{<t})},1-ϵ, 1+ϵ)\hat A_t]\},\tag{2}$$

其中 $π_{θ_old}$ 是更新前的策略，$ϵ$ 是裁剪超参数，$\hat A_t$ 是第 t 个 token 的优势函数估计器。估计 $\hat A_t$ 的标准方法是使用学习到的价值模型 $V_ϕ$ 来计算广义优势估计 (GAE)。然而，在 LLM RL 调优的背景下，学习价值模型的计算成本很高，因此不使用 $V_ϕ$ 来估计 $\hat A_t$ 的方法实际上更受欢迎。例如，Shao et al. (2024) 提出了 GRPO，它首先对每个问题采样一组响应 $\{\textbf o_1, ...,\textbf o_G\}$ 并计算它们的回报 $\textbf R = \{R_1, ..., R_G\}$，然后将 $o_i$ 中所有 token 的优势设为 $\hat A_t =\frac{R_i−mean(\textbf R)}{std(\textbf R)}$。

## 3.1 GRPO Leads to Biased Optimization

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ed44701332f4c46806cdd803769747b.png)

在 Deepseek-R1-Zero 中，一个显著的趋势是响应长度在整个训练过程中持续增加。这通常被解读为高级推理能力（例如自我反思）发展的标志。近期研究使用各种算法和实现方式复制了这一现象。然而，我们认为，观察到的响应长度增加也可能归因于 GRPO 目标函数中固
有的偏差：
$$\mathcal J_{GRPO}(\pi_{\theta})=\mathbb E_{\textbf q\sim p_{\mathcal Q},\textbf o\sim\pi_{\theta}(\cdot|\textbf q)}\\
\frac{1}{G}\sum^{|\textbf G|}_{i=1}\frac{1}{|o_i|}\sum^{|\textbf o_i|}_{t=1}\{min[\frac{\pi_{\theta}(o_{i,t}|\textbf q,\textbf o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|\textbf q,\textbf o_{i,<t})}\hat A_{i,t},clip(\frac{\pi_{\theta}(o_{i,t}|\textbf q,\textbf o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|\textbf q,\textbf o_{i,<t})},1-ϵ, 1+ϵ)\hat A_{i,t}]\},\tag{3}$$

其中，

$$\hat A_{i,t}=\frac{R(\textbf q,\textbf o_i)-mean(\{R(\textbf q,\textbf o_1), ..., R(\textbf q,\textbf o_G)\})}{std(\{R(\textbf q,\textbf o_1), ..., R(\textbf q,\textbf o_G)\})},$$

其中回报 $R(\textbf q, \textbf o_i)$ 通常仅包括 LLM 推理中可验证的结果奖赏（该分析也适用于过程奖赏案例）。

与等式（2）中的目标函数相比，GRPO 引入了两个偏差（另见图5）：
- **Response-level length bias**：这是由除以 $|o_i|$ 得出的。对于正例优势（$\hat A_{i,t} > 0$，表示正确答案），这种偏差会导致较短答案的梯度更新更大，从而使策略更倾向于选择简洁的正确答案。相反，对于负例优势（$\hat A_{i,t < 0}$，表示错误答案），较长答案由于其 $|o_i|$ 较大而受到的惩罚较少，从而使策略在错误答案中更倾向于选择较长的答案。
- **Question-level difficulty bias**：这是由于将居中结果奖赏除以 $std(\{R(\textbf q,\textbf o_1), ..., R(\textbf q,\textbf o_G)\})$ 造成的。标准差较低的问题（例如，结果奖励几乎全部为 1 或 0 的过难或过易问题）在策略更新期间会被赋予更高的权重。虽然优势归一化是强化学习中的常用技巧，但它通常针对整个批次进行计算。相比之下，问题级归一化会导致不同问题的目标函数权重不同，从而导致优化过程中出现难度偏差。

**Length Bias Also Exists in Open-Source PPO Implementations**。我们还研究了几种流行的用于 LLM 后训练的 vanilla PPO 算法的开源实现。令人惊讶的是，所有这些实现都根据响应长度对损失进行归一化（参见清单 1 和表 2），这与公式 (2) 中定义的 PPO 目标不一致。这种公式实现上的不一致甚至在 GRPO 发布之前就已存在。我们推测这种不一致可能源于预训练阶段，在此阶段，所有 token 都被打包到固定长度的上下文中，并且根据上下文长度对损失进行归一化（即计算 loss.mean(-1)）可以提高数值稳定性。然而，在 RL 调优阶段，典型的实现根据响应长度对损失进行归一化，而响应长度并非常数，从而引入了意想不到的长度偏差。

## 3.2  Dr. GRPO: Group Relative Policy Optimization Done Right

为了避免 GRPO 中上述优化偏差，我们建议直接删除 $\frac{1}{|o_i|}$ 和 $std(\{R(\textbf q,\textbf o_1), ..., R(\textbf q,\textbf o_G)\})$ 归一化项。同时，为了准确实现无偏优化目标，我们可以将 List1 中 MASK 均值函数中的 mask.sum(axis=dim) 替换为一个常量值（例如，生成预算），如绿色线条所示。值得注意的是，这些简单的修改恢复了公式 (2) 中的 PPO 目标，其优势函数通过蒙特卡洛回报估计得出，且基线为无偏。我们在附录 A 中给出了详细的推导。我们将新的优化算法称为 Dr. GRPO。接下来，我们将通过实验验证其有效性。

## 3.3 A Duet of Template and Question Set Coverage in RL dynamics
