# RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning翻译

论文链接：https://arxiv.org/pdf/2609.20784

代码链接：https://github.com/ZJU-REAL/SDAR

## 摘要

在强化学习（RL）训练的多轮交互智能体中，每个完整轨迹仅获得一个标量奖赏。为了解决这一问题，研究者提出了“自 on-policy 蒸馏”（Self-On-Policy Distillation, OPD）方法：利用一个具备特权任务技能的“自教师”（self-teacher）提供密集的 token 级（token-level）监督，从而让不具备特定技能的“学生”模型内化这些能力。然而，在智能体任务中，这种方法面临两个挑战：仅凭特权信息并不总能保证教师的可靠性，且教师监督的有效性往往取决于训练阶段。为此，我们提出了 **RetireOPD** (Self-**Retiring** **On-Policy** **Distillation**)。该方法首先利用环境奖赏优化一个解耦的、基于技能条件的教师模型，随后通过强化学习（RL）与 OPD 联合训练学生模型。RetireOPD 摒弃了预设的蒸馏计划，转而采用“自适应退出”（Adaptive Retirement）机制：当学生与教师之间的差异不再缩小，且学生在任务上的成功率达到教师成功率的特定比例时，学生便会自动“脱离”教师，后续训练仅依靠强化学习进行。在从 1.5B 到 7B 参数规模的 Qwen2.5 模型上，RetireOPD 在 ALFWorld 任务中的成功率较纯强化学习基线提升了 14.1% 至 18.8%，在 WebShop 任务中的准确率提升了 11.8% 至 19.0%；在所有实验设置下，其表现均优于作为指导的、基于技能条件的教师模型。代码已开源至：https://github.com/ZJU-REAL/SDAR。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/8c1ae8f6906c47ba9cf0beb93924242f.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

在针对多轮任务训练大语言模型（LLM）智能体时，基于可验证奖赏的强化学习（RLVR）是一种标准方法；然而，由于每个交互序列仅对应一个最终结果奖赏，长交互过程中的中间决策往往缺乏监督。On-Policy Distillation（OPD）通过引入“教师”模型，在“学生”模型采样出的交互序列上提供密集的 token 级监督，从而弥补了稀疏奖赏的不足。在智能体训练中，教师模型通常与学生模型相同，但它能够获取仅在训练阶段可见的特权上下文信息（例如检索到的任务技能）。学生模型在无法获取这些上下文信息的前提下，通过学习复现教师模型的行为，从而将相关技能内化到自身的参数中，确保在推理阶段无需额外的上下文输入。

这一范式基于两项假设：其一，教师模型被认为始终优于学生模型，因为它能够获取特权上下文信息；其二，在整个训练过程中，向教师模型看齐始终是有益的。这两项假设实际上引出了两个问题——**即学生模型应向哪位教师学习，以及学习应持续多久**——而现有方法通常通过预设或固定的时间表来回答这些问题。我们在 ALFWorld 环境中对这些假设进行了检验，结果发现这两项假设均不成立（图 2）。

<img
  src="https://i-blog.csdnimg.cn/direct/95e4e14ebf62495d9da491704c80d04e.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Privileged context alone does not make a teacher reliable**。在“组相对策略优化”（GRPO）与“在线策略自蒸馏”（OPSD）相结合的框架下——其中教师模型与学生模型同属一个共享策略下的两个分支——受特定技能引导的分支，其表现并不总是优于它所监督的学生模型（图2，左侧）。这表明教师模型并未有效利用所提供的技能信息：尽管它能获取特有的技能上下文，但并未经过充分训练，无法将这些信息整合到其决策过程中。单纯增加模型规模也无法解决这一问题——在ALFWorld任务中，一个未经专门优化、且被赋予相同技能提示的 7B 模型，其成功率仅为23.4%。因此，教师模型必须经过优化，学会利用其所掌握的特有上下文信息，才能提供有效的监督。

**The benefit of teacher supervision is stage-dependent**。在 GRPO 中引入 OPD，既克服了纯 GRPO 启动缓慢的问题，也解决了纯 OPD 性能上限较低的局限（图 2，中）。“学生”与“教师”之间的差异会先缩小，后扩大（图 2，右）。一旦“学生”内化了由这些技能所引发的行为，奖赏优化便会倾向于采取超越“教师”水平的行动，此时两者对应的梯度开始产生冲突（图 1，左）。若在此阶段之后仍坚持与“教师”保持一​​致，则会使“学生”的性能被限制在“教师”的性能上限附近（第 4.3 节）。一阶分析（附录 A.1）表明，若两者间的差异停滞不前，则意味着 OPD 带来的收益正在递减，或者梯度冲突正在加剧。因此，“教师”的监督仅是一种临时的“脚手架”；问题的关键在于，何时将其移除。

现有研究通过预设的调度方案来处理这一过渡过程，例如采用“退火蒸馏”（annealing distillation）策略，或在固定的训练步数点从蒸馏切换至强化学习（RL）。虽然也存在基于更细粒度（如针对每个提示词或每个 token）的在线决策机制，但这些方法均未完全移除“教师”模型。**在我们的实验中，不同模型和任务下，差异（discrepancy）停止下降的时刻各不相同，分布在第 50 步到第 90 步之间（见附录 A.9）；这意味着，单一的预设调度方案在某些场景下可能过早撤去引导，而在另一些场景下则可能撤去得太晚**。这些观察结果引出了一个简单的原则：教师模型是否仍有价值，可以从训练信号本身判断出来。当两个优化目标发生冲突时，差异便会停止下降；而学生模型相对于教师模型的成功率，则反映了学生是否已经内化了相关技能。因此，这种过渡应当根据训练过程中的实时情况（在线）来确定，而非预先固定。

受上述发现的启发，我们提出了 **RetireOPD**（Self-Retiring On-Policy Distillation）。该方法将“特权技能”内化至一个不依赖预设技能的学生模型中；当教师模型提供的监督信号不再能为学生模型带来增益时，便将其“退役”。RetireOPD 首先将教师模型与学生模型解耦，并在技能语境下利用环境奖赏对教师模型进行训练，从而确保蒸馏过程始于一个已经学会如何利用这些技能的策略。随后，学生模型通过 GRPO（组相对策略优化）和 OPD（同策略蒸馏）联合训练，同时系统会定期监测前述的两个信号。一旦差异不再减小，且学生模型的成功率达到教师模型成功率的预设比例，教师模型即被“退役”，后续训练仅由 GRPO 继续进行。由此，学生模型不再受限于教师模型，且无需再进行教师模型的前向计算。

在 ALFWorld 和 WebShop 任务中，RetireOPD 在 Qwen2.5-1.5B、3B 和 7B 模型规模上均优于强化学习（RL）、知识蒸馏及混合基线方法；与 GRPO 相比，RetireOPD 将 ALFWorld 的成功率提升了 14.1% 至 18.8%，并将 WebShop 的准确率提升了 11.8% 至 19.0%。此外，在各项设置中，RetireOPD 的表现均超越了其所依赖的“技能条件化教师模型”（skill-conditioned teacher）。我们的主要贡献如下：
- 我们确定了智能体在“特权信息蒸馏”过程中存在的两种失效模式：一是未经优化的、基于技能条件的教师模型不可靠；二是当学生模型已经内化了教师模型的知识后，教师匹配（teacher matching）会与奖赏优化产生冲突。
- 我们提出了 RetireOPD。该方法利用环境奖赏来训练一个与学生模型容量相当、且以技能为条件的“教师”模型；当“教师”与“学生”之间的差异不再减小，且“学生”的成功率达到“教师”成功率的预设比例时，该“教师”模型便会在在线训练过程中被“退役”（即停止使用）。
- 在 ALFWorld 和 WebShop 环境下，针对三种模型规模开展的实验表明，RetireOPD 的表现始终优于强化学习（RL）、知识蒸馏及混合型基线方法，甚至超越了其自身的教师模型；这验证了该方法在智能体任务中的有效性。

## 2.Related Work

### 2.1  On-Policy Distillation with Privileged Teachers

On-policy distillation（OPD）通过教师模型提供的 token-level 反馈，让学生模型在自身采样的数据上进行训练；目前，这已成为模型后训练阶段的标准流程。在没有更强教师模型的情况下，On-policy self-distillation（OPSD）通过引入“特权信息”（如参考解或文本反馈）作为条件，引导模型生成特定输出，进而将这一“受条件约束”的分支蒸馏至“无条件约束”的分支中。对于智能体而言，这类特权信息通常表现为检索到的 skill 或事后总结出的经验；在此过程中，教师模型的监督作用通过 token-level 的门控（gating）或引导（shaping）机制进行调控。尽管指导质量取决于教师模型与学生模型之间的兼容性及其自身超越学生模型的能力，且并非所有 token 都包含有效信号，但“教师模型的可靠性”这一问题此前却鲜受关注。与之不同，RetireOPD 方法在蒸馏之前，会先利用环境奖赏（environment rewards）对教师模型进行独立训练。

### 2.2  Combining On-Policy Distillation with Reinforcement Learning

RLVR 是多轮交互智能体（multi-turn agents）训练中的主流方法，但由于其轨迹级奖赏较为稀疏，一些混合方法会在强化学习（RL）目标函数中引入“蒸馏项”或“由教师模型引导的优势函数”。随着学生模型能力的提升，这两种信号之间会产生干扰；为解决这一问题，现有研究通常会采取措施削弱教师模型在训练过程中的影响，例如：逐渐降低蒸馏权重、缩短学生模型可见的轨迹长度，或者将蒸馏过程与强化学习过程按顺序先后执行。在这些方法中，教师模型在强化学习中的过渡策略通常在训练开始前就已经固定好了。虽然也存在一些基于具体信号的动态调整机制（例如仅在所有尝试均失败的提示词上进行蒸馏，或针对每个 Token 决定是否引入教师模型），但这些方法本质上都未能彻底移除教师模型的作用。相比之下，RetireOPD 能够根据“教师模型与学生模型之间的差异”以及“学生模型的相对能力水平”，在训练过程中在线决定何时结束教师模型的引导（即实现全局过渡）。

## 3.Method

<img
  src="https://i-blog.csdnimg.cn/direct/41b89fa63d764544bde9aa1a6bce6169.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

如图 3 所示，我们的方法包含三个阶段：(1) **教师构建**：利用环境奖赏优化一个基于技能条件的教师模型（第 3.1 节）；(2) **联合 GRPO-OPD 训练**：由具备技能的教师基于学生生成的轨迹，对不具备特定技能的学生进行监督（第 3.2 节）；(3) **自适应教师退出机制**：当行为迁移陷入停滞且学生已具备足够的任务能力时，移除 OPD 模块（第 3.3 节）。

**Task Definition**。我们考虑一个与环境进行多轮交互的智能体，该交互过程包含一系列决策步骤。给定任务输入 $x \sim\mathcal D$，智能体根据交互历史迭代生成响应，并接收来自环境的反馈，从而形成轨迹 $\tau = (x, y)$；其中 $y = (y_1, \dots, y_T)$ 表示在各交互轮次中产生的所有有效响应 token 所构成的序列，$R(\tau)$ 则代表环境给予的标量奖赏。

### 3.1 Privileged Teacher Construction

基础模型无法有效利用特权技能上下文，因此需要进行“教师”训练。为此，我们训练了一个“教师”策略（teacher policy），使其学会如何利用技能进行决策。具体而言，在训练过程中，教师策略 $\pi_\phi$ 可获取特权技能上下文 $c^+$。我们采用 GRPO 算法对该教师策略进行优化，从而获得一个具备熟练技能的教师模型；其优化目标如下：

```math
\phi^{*}=arg\mathop{max}\limits_{\phi}\mathbb E_{x\sim\mathcal D,\tau\sim\pi_{\phi}(\cdot|x,c^+)}[R(\tau)].\tag{1}
```

环境奖赏优化旨在促使教师将特权信息转化为有效的任务行为。训练完成后，我们将 $\phi^*$ 固定，并将由此得到的策略记为“熟练教师” $\pi_T$。此后，该固定后的教师仅用于为学生提供密集监督。关于教师构建的更多细节，请参阅附录 A.5。

### 3.2 Joint GRPO-OPD Optimization

获得熟练教师后，我们接下来训练学生策略 $π_θ$ 以内化教师的行为。遵循 SDAR 原则，学生在无法访问特权技能上下文 $c^+$ 的情况下进行训练。对于每个输入 4x$，学生 $π_θ$ 采样一组 $G$ 条轨迹。基于环境奖赏，我们首先使用 GRPO 优化学生，GRPO 的定义如下：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{GRPO}}(\theta)
=& -\mathbb{E}_{x\sim\mathcal{D}}
\left[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|y^{(i)}|}
\sum_{t=1}^{|y^{(i)}|}
\min\left(
r_t^{(i)}(\theta)\hat{A}^{(i)},
\operatorname{clip}\left(r_t^{(i)}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}^{(i)}
\right)
\right] \\
&+\alpha_{KL}\mathbb{E}_{x\sim\mathcal{D}}
\left[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|y^{(i)}|}
\sum_{t=1}^{|y^{(i)}|}
D_{\mathrm{KL}}\left(
\pi_{\theta}(\cdot\mid x,y_{<t}^{(i)})
\parallel
\pi_{\mathrm{ref}}(\cdot\mid x,y_{<t}^{(i)})
\right)
\right].
\end{aligned}
\tag{2}
```

其中 $r_t^{(i)}(\theta) = \frac{\pi_\theta(y_t^{(i)} \mid x, y_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(y_t^{(i)} \mid x, y_{<t}^{(i)})}$ 是 token 层面的重要性比率。

对于学生生成的每一条轨迹，我们在给定相同轨迹前缀 $y_{<t}$ 的条件下，比较学生与教师的分布。教师直接针对学生所经历的状态提供监督。我们利用反向 KL 散度（reverse KL divergence）来定义 OPD 目标函数：

```math
\mathcal{L}_{\mathrm{OPD}}(\theta)=
\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\theta}(\cdot\mid x)}
\left[
\frac{1}{|y|}
\sum_{t=1}^{|y|}
D_{\mathrm{KL}}\left(
\pi_{\theta}(\cdot\mid x,y_{<t})
\parallel
\pi_T(\cdot\mid x,c^+,y_{<t})
\right)
\right].
\tag{3}
```

为了降低精确 KL 散度计算的开销，我们对学生模型生成的轨迹采用基于采样 token 的近似方法，即在学生模型和教师模型的分布下分别对每个 $y_t$ 进行评估：

```math
\Delta_t = \log \pi_T(y_t \mid x, c^+, y_{<t}) - \log \pi_{\theta}(y_t \mid x, y_{<t}).
```

最后，完整的优化目标表述为：

```math
\mathcal{L}_{\mathrm{student}}(\theta)=
\mathcal{L}_{\mathrm{GRPO}}(\theta)
+
\lambda\mathcal{L}_{\mathrm{OPD}}(\theta).
```

其中 $λ$ 控制教师监督的贡献。

### 3.3 Adaptive Teacher Retirement

在训练初期，教师的指导是有益的；但从长远来看，它可能会因给学生设定了能力上限，从而限制了学生性能的进一步提升。因此，我们引入了一种**自适应退出机制**，通过监测训练动态来决定何时将教师从优化过程中移除。

我们将训练过程划分为 $W$ 个步长的监控窗口，窗口索引为 $m$。在步骤 $n$，令 $y^{(i)}_n$ 表示第 i 个采样轨迹。给定师生对数概率差距 $∆^{(i)}_t$，我们计算步长级和窗口级对齐进度 $K_n$ 和 $\bar K_m$：

```math
K_n = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_n^{(i)}|}\sum_{t=1}^{|y_n^{(i)}|}\Delta_t^{(i)},
\qquad
\bar{K}_m = \frac{1}{W}\sum_{n=mW}^{mW+W-1}K_n.
\tag{6}
```

为了捕捉师生匹配度何时开始停滞或逆转，我们将反弹指标 $ρ^{(K)}_m$ 和相对能力 $η_m$ 定义为

```math
\rho_m^{(K)} = \frac{\bar{K}_m - \bar{K}_{m-1}}{\bar{K}_m},
\qquad
\eta_m = \frac{SR_m + SR_{m-1}}{2SR_T}.
\tag{7}
```

其中 $SR_m$ 和 $SR_T$ 分别表示学生和教师的成功率。当监测窗口满足以下条件时，教师监督即告结束：

```math
m^* = \inf\left\{m \geq 2 : \rho_m^{(K)} \geq \delta \;\land\; \eta_m \geq \gamma\right\}.
\tag{8}
```

其中，$δ$ 和 $γ$ 分别代表“反弹指标”和“相对能力阈值”。同时满足这两项标准，可避免因短暂波动而导致过早“退出”（即停止指导），从而确保学生在转为独立学习之前，已从教师的指导中获得了充分的助益。

## 4.Experiments
