# Self-Distilled Agentic Reinforcement Learning

论文链接：https://arxiv.org/pdf/2605.15155v1

代码链接：https://github.com/ZJU-REAL/SDAR

## 摘要

强化学习（RL）已成为大语言模型（LLM）智能体后训练阶段的核心范式；然而，其基于完整轨迹的奖赏信号在长程交互任务中仅能提供较为粗糙的监督信息。On-Policy Self-Distillation（OPSD）通过引入“教师分支”（该分支可获取特权上下文信息）所提供的细粒度（Token 级别）指导，对强化学习形成了有效补充。但在应用于多轮交互智能体时，OPSD 面临诸多挑战：多轮交互中累积的不稳定性会干扰监督信号；此外，在基于特定技能的特权指导中，必须采取非对称的处理策略——因为“教师模型”给出的负向反馈（即否定信号），可能源于智能体对技能的检索或利用不当，而非技能本身的问题。为此，我们提出了 **SDAR** (**S**elf-**D**istilled **A**gentic **R**einforcement Learning)。SDAR 将 OPSD 视为一种“门控辅助目标”，同时保留强化学习作为主要的优化基干。SDAR 通过一个 Sigmoid 门控机制来处理 Token 级别的信号：对于教师模型认可的、具有正向增益的 Token，强化其蒸馏效果；而对于教师模型给出的负向反馈，则通过软衰减（softly attenuate）机制进行抑制。在基于 Qwen2.5 和 Qwen3 系列模型的 ALFWorld、WebShop 及 Search-QA 任务中，SDAR 较 GRPO 取得了显著提升（在 ALFWorld 上提升 9.4%，在 Search-QA 上提升 7.0%，在 WebShop 准确率上提升 10.2%）。该方法不仅规避了直接结合 GRPO 与 OPSD 所带来的不稳定性，而且在不同模型规模下，均持续优于各类混合型 RL-OPSD 基线方法。代码已开源：https://github.com/ZJU-REAL/SDAR。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/9cf65a7a1a8b448380bd9449c9af995d.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

“智能体化”后训练已成为大语言模型（LLM）面临的核心挑战。与静态的单轮推理不同，多轮智能体会在较长的时间跨度内与环境进行交互；在此过程中，每一次行动都会改变后续的观测结果，而生成的每一条响应也会成为后续决策的上下文信息。

两种范式自然地呈现为互补的力量：强化学习（RL）基于环境或验证器的反馈，实现任务层面的优化；而在线策略蒸馏（OPD）及在线策略自蒸馏（OPSD）则通过“教师”分支，提供细致的、token层面的指导。然而，OPSD 无法直接且有效地应用于多轮智能体训练。我们认为这源于以下两点发现：[1] 多轮 OPSD 的不稳定性；[2] 对特权指导（privileged guidance）存在非对称的信任。

<img
  src="https://i-blog.csdnimg.cn/direct/b7b45a08829241ac982c21f36aeebeb2.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**[Observation-1] Multi-turn OPSD Instability**。一旦智能体偏离了由教师策略所引导的轨迹（这是不可避免的），原本有效的 token-level 监督信号便会变得愈发不可靠。这种误差的累积会导致每轮 KL 散度（per-turn KL divergence）急剧上升，并引发任务性能的灾难性下降，如图2（左）所示。TCOD 试图通过课程学习（curriculum learning）来解决这一问题，但其方法依赖于固定的时间进度安排或基于轨迹深度的阈值设定。

**[Observation-2] Asymmetric Trust in Privileged Guidance**。在 OPSD 中，教师分支并非一个独立且更强大的模型，而是基于同一策略，并辅以仅在训练阶段可用的特权上下文（例如检索到的技能）进行增强。这使得其在 Token 级别的引导呈现出本质上的非对称性。对于学生模型采样出的 Token $y_t$，如果拥有特权信息的教师模型给出的概率高于学生模型，那么检索到的技能便提供了一种“背书”信号：它支持了学生模型本已能够生成、但尚未完全内化的“在线策略”（on-policy）行为。这种正向引导对于知识蒸馏尤为适用。

相比之下，如果“特权教师”（privileged teacher）模型为所采样的 token 赋予了较低的概率，那么对于这一信号，应当持更加谨慎的态度。负向差距（negative gap）可能意味着该 token 确实应当被抑制；但在基于技能条件的 OPSD（特权监督偏好蒸馏）框架下，这**种现象也可能源于“特权上下文”（privileged context）的不稳定性，具体包括**：(1) 技能质量：所检索到的技能可能是不相关的、不完整的或冗余的。(2) 技能利用：教师模型可能未能将相关的技能有效地转化为可靠的、token 级别的偏好。(3) 多轮交互中的偏差累积：随着交互轨迹的展开，教师模型与学生模型之间的差距往往会随轮次增加而扩大（见图 3 中图），从而使早期的不匹配在后续决策中被进一步放大。我们针对 Qwen2.5-3B-Instruct 模型进行的初步研究显示，出现负向差距的 token 占比超过了 50%（见图 3），这表明该问题相当普遍。基于此，我们提出对“特权引导”采取非对称的处理策略：**即更充分地信任教师模型给出的正面偏好，而对教师模型给出的负面排斥则采取更为审慎的态度**。

一个鲜明的认识浮现出来：对于多轮交互智能体而言，强化学习（RL）有望成为核心优化支柱，而 OPSD 则退居为一种受到严格管控的辅助手段。

那么，该如何管控这一辅助作用呢？RLSD 直接利用“自散度”（self-divergence）来对 token 层面的强化学习优势值（RL advantages）进行重加权；然而，这种做法可能会显著放大更新幅度——尤其是在训练初期，当教师模型与学生模型之间存在较大偏差时（见图 2 右侧）。

我们采取了不同的路径：将 OPSD 损失视为一种直接的辅助优化目标，同时保持由验证器驱动的强化学习（RL）策略损失不变，从而严格保留了 RL 优势函数（advantage）的语义及其无偏性。为了克服多轮 OPSD 及特权引导（privileged guidance）所带来的不稳定性，**我们并未对所有 Token 统一执行蒸馏；而是通过一种自适应且平滑的门控机制，对 Token 进行选择性蒸馏**，而非采用人工预设的刚性调度策略（如 Skill-SD 和 HDPO 中所用的方式）。受 TIP 的启发，我们利用 Token 级别的信号（例如学生模型的熵，或教师模型与学生模型之间的散度）来控制门控机制的激活。其核心理念很简单：让每个 Token 自行决定所受监督的强度。这种方法实现了一种动态的、自定步调（self-paced）的课程学习机制，且其作用粒度达到了最细的层面——即单个 Token 级别。

我们在 Qwen2.5 和 Qwen3 模型系列上，针对三种不同的大语言模型（LLM）智能体基准测试——ALFWorld、WebShop 和 Search-QA——验证了我们的方法。与 GRPO 相比，SDAR 取得了显著的性能提升（在 7B 规模下，ALFWorld 提升了 9.4%，Search-QA 提升了 7.0%，WebShop-Acc 提升了 10.2%）；它彻底规避了原始 GRPO+OPSD 方法中出现的灾难性不稳定问题；此外，在所有三种模型规模（包括 Qwen3-1.7B）下，SDAR 的表现均持续优于 Skill-SD 和 RLSD 等“强化学习（RL）与 OPSD 混合”方法。进一步的鲁棒性分析表明，SDAR 在检索质量下降时，性能仅会出现平缓的衰减：即使是随机检索，其效果也优于 GRPO 基线；这是因为我们的门控（gating）设计能够过滤掉低质量技能中的噪声，仅提取出有益的信号。

<img
  src="https://i-blog.csdnimg.cn/direct/f8aa9ce3af774e3d96151eca0aa926a2.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## 2.Method

### 2.1 Problem Setup

我们考虑一个在有限时域内与环境进行交互的多轮智能体。给定初始提示或任务描述 $x$，在第 $k$ 轮，智能体接收观测 $o_k$ 并生成响应 $a_k$，随后环境返回下一个观测 $o_{k+1}$。每个响应 $a_k$ 既可包含中间推理 tokens，也可包含可执行的动作 token。为简化符号表示，我们将单次交互轨迹中所有有效的响应 token 展平为一个统一的 token 序列。

```math
y=(y_1,...,y_T)\sim\pi_{\theta}(\cdot|x),
```

其中 $\pi_\theta$ 表示学生策略，$T$ 为有效响应 token 的总数。在 token 位置 $t$ 处，我们将“自学生上下文”（self-student context）表示为

```math
s_t=(x,y_{\lt t}),
```

“自教师上下文”（self-teacher context）表示为

```math
s^+_t=(x,c^+,y_{\lt t}),
```

其中 $c^+$ 表示仅在训练阶段可供“教师”分支使用的特权上下文——例如参考答案、技能（本文提出的技能）或其他在测试阶段无法获取的辅助信息。

**Skills Retrieval**。我们检索与任务相关的 skills ——即紧凑且结构化的演示，其中编码了特定领域的知识（如子目标分解或动作模板）。我们采用了四种质量各异的检索策略，旨在评估我们的框架对于所检索上下文保真度的稳健性：(1) UCB 检索、(2) 关键词匹配 (KM)、(3) 全量检索，以及 (4) 随机检索。

Skill 检索被建模为针对技能库 $\mathcal E = \{e_1, \dots, e_M\}$ 的多臂老虎机问题。对于每个到来的任务，UCB 检索（UCB Retrieval）会根据“置信上限”（Upper Confidence Bound, UCB）准则，从中选出得分最高的那一个技能文件。

```math
score(e)=\bar{r}(e)+c\sqrt{\frac{ln~N_{ucb}}{n(e)}},\tag{1}
```

其中，$\bar r(e)$ 表示技能 $e$ 先前作为上下文提供时获得的运行平均奖励，$N_{ucb}$ 表示针对同一任务类型发出的检索查询总数，$n(e)$ 表示 $e$ 被选择的次数，$c$ 控制探索与利用之间的权衡。关键词匹配绕过了多臂老虎机算法，而是通过将任务描述中的关键词与预定义的类别标签进行匹配来识别任务场景，并直接检索与匹配类别关联的技能文件。

### 2.2 Optimization Goals

<img
  src="https://i-blog.csdnimg.cn/direct/ee81f294a6db4d4399810ac688bbfafe.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们的方法被设计为标准策略优化 GRPO 损失之上的辅助目标。总体训练目标是：

```math
\mathcal L(\theta)=\mathcal L_{GRPO}(\theta)+\lambda_{SDAR}\cdot\mathcal L_{SDAR}(\theta),
```

其中 $\mathcal L_{GRPO}$ 是原始策略损失，$\mathcal L_{SDAR}$ 是我们的 on-policy 自蒸馏目标函数。令 $m_t ∈ \{0,1\}$ 为响应掩码，指示 token $t$ 是否有效。我们将掩码 token 平均定义为

```math
Agg(z_{1:T})=\frac{\sum^T_{t=1}m_tz_t}{\sum^T_{t=1}m_t}.
```

**RL Optimization**。对于每个输入 x，GRPO 会采样出一组回答。

```math
\{y^{(i)}\}^G_{i=1}\sim\pi_{\theta}(\cdot|x),
```

并根据环境奖赏计算序列级优势 $A^{(i)}$。使用参考策略 $π_{ref}$，GRPO 目标可以写成如下形式：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{GRPO}}(\theta)
=& -\frac{1}{G}\sum_{i=1}^{G}
\operatorname{Agg}\left(
\min\left(
r_t^{(i)}A^{(i)},
\operatorname{clip}(r_t^{(i)},1-\epsilon,1+\epsilon)A^{(i)}
\right)
\right) \\
&+\beta\cdot\frac{1}{G}\sum_{i=1}^{G}
\operatorname{Agg}\left(
D_{\mathrm{KL}}\left(
\pi_\theta(\cdot\mid s_t^{(i)})
\parallel
\pi_{\mathrm{ref}}(\cdot\mid s_t^{(i)})
\right)
\right),
\end{aligned}
\tag{2}
```

其中 $r^{(i)}_t = \frac{\pi_\theta(y^{(i)}_t | s^{(i)}_t)}{\pi_{\theta_{old}}(y^{(i)}_t |s^{(i)}_t)}$ 是重要性采样比。

**OPSD Optimization**。在固定的 token 位置 $t$ 处，教师模型和学生模型分别导出条件 token 分布 $\pi_T(\cdot \mid s^+_{t})$ 和 $\pi_\theta(\cdot \mid s_t)$。每个 token 对应的反向 KL 散度定义如下：

```math
D_{\mathrm{RKL}}^{(t)}
= D_{\mathrm{KL}}\left(
\pi_\theta(\cdot \mid s_t)
\parallel
\pi_T(\cdot \mid s_t^+)
\right)
= \sum_{v \in \mathcal{V}}
\pi_\theta(v \mid s_t)
\log \frac{\pi_\theta(v \mid s_t)}
{\pi_T(v \mid s_t^+)}.
```

为了在无需计算开销巨大的全词表求和的情况下高效获取重要性信号，我们基于学生模型采样的 token $y_t \sim \pi_\theta(\cdot | s_t)$ 进行单样本估计。该估计值的相反数即直接给出了教师-学生对数概率差 $\Delta_t$：

```math
\Delta_t = -\widehat{D}_{\mathrm{RKL}}^{(t)}
= \log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t).
```

### 2.3 Token-Level Gating

核心思想是将特权教师的指导转化为 token 级信任权重，同时保持验证器驱动的强化学习（RL）目标不变。我们引入一个 token 级门控 $g_t \in [0,1]$，用于调节每个学生采样 token 上的 OPSD 信号，并将其应用于采样 token 的代理目标，使不同的门控策略共享相同的优化方式。

设

```math
\Delta_t = \operatorname{sg}\left(\log \pi_{\theta}^{+}(y_t \mid s_t^{+}) - \log \pi_{\theta}(y_t \mid s_t)\right)
```

表示学生采样 token 上经过梯度分离（detached）的教师-学生对数概率差距，并设

```math
h_t = -\sum_{v \in \mathcal{V}} \pi_{\theta}(v \mid s_t) \log \pi_{\theta}(v \mid s_t)
```

表示学生模型在位置 $t$ 处的熵。我们将每个原始分数通过 Logistic Sigmoid 函数 $\sigma$ 进行变换，使每个门控都具有平滑、可微的特性，并自然地限制在 $(0,1)$ 区间内。陡峭度参数 $\beta > 0$ 控制门控从保守抑制到强激活的过渡。

我们设计了三种互补的门控策略：

1. *熵门控（Entropy gating）*：$g_t = \sigma(\beta h_t)$，针对学生模型最不确定的高熵位置。

2. *差距门控（Gap gating）*：$g_t = \sigma(\beta \Delta_t)$，对获得特权教师认可的正差距 token 分配更大的权重，同时降低负差距 token 的权重。

3. *软或门控（Soft-OR gating）*：$g_t = \sigma(\beta[1-(1-h_t)(1-\Delta_t)])$，结合学生模型的不确定性与教师-学生差距，作为另一种门控策略。

在所有情况下，门控均通过 $\operatorname{sg}(\cdot)$ 进行梯度分离，因此梯度仅通过学生模型的对数概率传播。token 级损失定义为

```math
\ell_t^{\mathrm{SDAR}} = g_t \cdot \left(\log \pi_{\theta}^{+}(y_t \mid s_t^{+}) - \log \pi_{\theta}(y_t \mid s_t)\right),
\qquad
\mathcal{L}_{\mathrm{SDAR}} = \operatorname{Agg}(\ell_t^{\mathrm{SDAR}}).
```

采用差距门控时，Sigmoid 门控实现了非对称的 token 级调节：正差距 token 获得更强的辅助蒸馏信号，而负差距 token 的信号则被柔性抑制。我们还在附录 A 中提供了该设计的理论分析。

## 3.Experiment

<img
  src="https://i-blog.csdnimg.cn/direct/1b60125b8dd1429d9a9090b1a0efaae7.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>
