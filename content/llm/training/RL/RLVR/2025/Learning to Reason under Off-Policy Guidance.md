论文链接：https://arxiv.org/pdf/2504.14945

代码链接：https://github.com/ElliottYan/LUFFY

# 摘要

大型推理模型 (LRM) 的最新进展表明，诸如多步推理和自我反思等复杂行为可以通过基于简单规则的奖赏强化学习 (RL) 实现。然而，现有的零强化学习 (Zero-RL) 方法本质上是 on-policy 的，将学习限制在模型自身的输出上，无法获得超越其初始能力的推理能力。我们推出了 **LUFFY** (**L**earning to reason **U**nder o**FF**-polic**Y** guidance)，这是一个通过离线策略推理轨迹增强零强化学习的框架。**LUFFY 通过在训练过程中结合离线策略演示和在线策略部署来动态平衡模仿和探索**。值得一提的是，我们提出通过正则化重要性抽样进行策略设计，以避免在混合策略训练期间进行肤浅而僵化的模仿。值得注意的是，LUFFY 在六个数学基准测试中实现了超过 +7.0 的平均增益，并在分布外任务中实现了超过 +6.2 的优势。它还显著超越了基于模仿的有监督微调 (SFT)，尤其是在泛化方面。分析表明，LUFFY 不仅能够有效地模仿，还能进行超越演示的探索，从而为训练具有离线策略指导的泛化推理模型提供了一条可扩展的途径。

# 1.介绍

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45852bae39c04359afe6866f3d9886bb.png)

大型推理模型（包括 OpenAI-o1、DeepSeek-R1 和 Kimi-1.5）近期取得的突破，在复杂推理任务中展现出卓越的能力。这些模型在生成广泛的思维链 (CoT) 响应以及展现诸如自我反思和自我纠正等复杂行为方面展现出前所未有的能力。尤其值得注意的是，这些成就是如何通过基于规则的奖赏机制的强化学习实现的，正如 DeepSeek-R1 所展示的那样。通过这种简单的奖赏机制，能够实现长 CoT 推理和自我反思能力，这被称为“顿悟时刻”，代表着该领域的重大进步。

成功背后的一个关键因素是零强化学习 (zero-RL)，它将强化学习直接应用于基础语言模型，利用模型自身的 rollout 来挖掘推理潜力。然而，它有一个值得强调的根本局限性：它本质上是 on-policy 的，从而将学习完全限制在模型通过迭代试验和反馈循环产生的自我输出上。尽管零强化学习展现出了良好的效果，但它仍然受限于基础语言模型本身。**本质上，在这种环境下，强化学习只是放大了现有的行为，而不是引入真正新的认知能力**。最近的研究证实了这一限制，表明像 Llama 3.2 这样的模型在零强化学习训练下很快就会达到性能平台期，正是因为它们缺乏进一步发展所必需的某些基础认知行为。

这种固有的局限性引发了关于零强化学习范式中学习有效性和范围的关键问题：我们如何才能使 LLM 获得超越其初始认知边界的推理行为？**从更强大的策略中引入外部指导的一种自然方法是模仿学习，其中使用由 DeepSeek-R1 等强大的 LRM 产生的推理轨迹对模型进行微调**。然而，最近的研究对通过纯粹模仿学习获得的泛化限制提出了担忧，这种限制将模型锁定在肤浅而僵化的推理模型中，阻碍了进一步的学习。同时，离线策略学习（off-policy learning）已被证明在各种强化学习任务中能够有效扩展 Agent 初始能力之外的学习能力，但在零强化学习中仍然基本上未被探索。这留下了一些关键问题：如何在简单的模仿学习之外，有效地将离线策略知识与在线策略学习的探索结合起来。

在本研究中，我们旨在将离线策略指导与统一的零强化学习范式相结合，并提出 **LUFFY**: **L**earning to reason **U**nder o**FF**-polic**Y** guidance。LUFFY 基于 GRPO 等传统零强化学习方法，引入离线策略推理轨迹（例如来自 DeepSeek-R1 的轨迹），并将其与模型在优势计算之前的在线策略roll-outs相结合，如图 1 所示。直观地讲，由于离线策略轨迹始终能够获得正向奖赏，因此 LUFFY 使模型能够在自身roll-outs无法达到正确性时选择性地模仿这些高质量的推理轨迹，同时在其生成的推理步骤成功时保留自主探索的能力。通过这种方式，LUFFY 在模仿和探索之间实现了动态且自适应的平衡。然而，单纯地组合离线策略轨迹可能会导致过快的收敛和熵崩溃，使模型陷入肤浅的模式，而不是获得真正的推理能力。为了解决这些问题，我们通过**正则化重要性采样**引入了策略设计，在离线策略指导下，放大低概率但关键动作的学习信号。这种机制鼓励模型在整个训练过程中保持探索，最终使其能够内化更深层、更泛化的推理行为。

如图 2 所示，LUFFY 在 AIME24/25、AMC、OlympiadBench、Minerva 和 MATH-500 等基准测试中，与之前的 RL-zero 方法相比，平均提升了 +7.0 分，证明了离线策略学习在类零范式中的有效性。此外，LUFFY 还展现出卓越的泛化能力，在 SFT 所欠缺的分布外任务中，平均提升超过 +6.2 分。实证结果表明，LUFFY 能够激励模型模仿高质量的推理轨迹，同时保持对自身采样空间的探索。这与我们更深入的分析结果一致，表明 LUFFY 能够灵活有效地吸收离策略推理行为，而 SFT 则将模型限制于对外部推理模式的刻板记忆，从而阻碍了泛化和探索。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7e72d86c056e428b9ac5397fd2f29bed.png)

# 2.Learning to Reason under Off-Policy Guidance

为了促进模型自身能力之外的探索，我们将离线策略指导（即由更强大的推理模型（例如 Deepseek R1）生成的现成推理轨迹）融入到零强化学习中。我们期望模型能够从离线策略中学习到可泛化的知识，而不仅仅是进行表面模仿，并保持与零强化学习训练一样有效且高效的探索。

在以下章节中，我们首先介绍强化学习的骨干算法 GRPO，然后演示混合策略 GRPO，它简单地集成了离线策略轨迹。最后，我们介绍 LUFFY，它利用策略设计来缓解熵崩溃并鼓励持续探索。

## 2.1 GRPO and Importance Sampling

由于 Deepseek-R1 的成功，GRPO 已成为基本的零强化学习训练方法。与广泛使用的 PPO 相比，GRPO 使用从 query 中采样 N 个解决方案的奖赏分数来估计优势函数，从而无需额外的价值模型。正式地，我们将更新前后的策略模型分别表示为 $π_{θ_{old}}$ 和 $π_θ$。给定一个问题 $q$、由 $π_{θ_{old}}$ 生成的一组解决方案 $τ_i$ 以及奖励函数 $R(·)$，GRPO 目标定义如下：

$$\mathcal J_{GRPO}(\theta)=\frac{1}{\sum^N_{i=1}|τ_i|}\sum^N_{i=1}\sum^{|τ_i|}_{t=1}min[r_{i,t}(\theta)A_i,clip(r_{i,t}(\theta);1-ϵ,1+ϵ)A_i]-\beta\cdot\mathbb D_{KL}[\pi_{\theta}||\pi_{ref}]$$

$$where~r_{i,t}(\theta)=\frac{\pi_{\theta}(τ_{i,t}|q,τ_{i,<t})}{\pi_{old}(τ_{i,t}|q,τ_{i,<t})},A_i=\frac{R(τ_i)-mean(\{R(τ_i)|τ_i\sim\pi_{\theta_{old}}(τ),i=1,2,...,N\})}{std(\{R(τ_i)|τ_i\sim\pi_{\theta_{old}}(τ),i=1,2,...,N\})}\tag{1}$$

$\mathbb D_{KL}$ 是 KL 散度。在 RL 目标函数中，GRPO 遵循 PPO，使用重要性采样（等式 1 中的 $r_{i,t}$）来校准梯度，因为 rollouts 是由 $π_{θ_{old}}$ 生成的。

裁剪项的裁剪率为 $ϵ$，经验上确保当前策略 $π_θ$ 位于旧策略 $π_{θ_{old}}$ 的信赖域内。我们将此方法大致归类为“同策略强化学习”，表明该模型使用与当前策略紧密相关的分布样本进行优化。然而，最近的实践越来越多地忽略了 KL 散度项，使得这些方法在某种程度上“不再符合策略”。

## 2.2 Mixed-Policy GRPO

我们将离线策略 rollout 直接添加到模型自身生成的在线策略 rollout 组中，从而将离线策略 rollout 融入 GRPO。假设存在一个离线策略分布 $π_ϕ$，这将以如下方式影响优势函数计算：
$$\hat A_i=\frac{R(τ_i)-mean(\mathcal G_{on}\cup\mathcal G_{off})}{std(\mathcal G_{on}\cup\mathcal G_{off})},\tag{2}$$
其中 $\mathcal G_{on} = \{R(τ_i) | τ_i ∼ π_{θ_{old}}(τ), i=1, 2, . . . , N_{on}\}$ 和 $\mathcal G_{off} = \{R(τ_j) | τ_j ∼ π_{ϕ(τ)}, j = 1, 2, . . . , N_{off}\}$。当模型难以独立生成正确解决方案时，这种群体计算自然会赋予离线策略 rollouts 更高的优势，而一旦模型开始产生成功的推理轨迹，在线策略 rollouts 就会优先考虑，从而鼓励自我驱动的探索。

然而，这种混合优势计算会给策略梯度算法的估计**引入偏差**，因为该算法假设策略分布会生成rollout。在我们的初步实验中，这会导致训练过程中的性能大幅下降。因此，我们使用重要性抽样来校准梯度估计，并将此方法称为 **Mixed-Policy GRPO**：

$$\nabla_{\theta} \mathcal{J}(\theta) 
= \mathbb{E}_{\tau_j \sim \pi_{\theta}(\tau)}\left[ \nabla_{\theta}\log \pi_{\theta}(\tau_j)\hat{A}_j \right]
= \mathbb{E}_{\tau_j \sim \pi_{\phi}(\tau)}\left[ \frac{\pi_{\theta}(\tau_j)}{\pi_{\phi}(\tau_j)}\nabla_{\theta}\log \pi_{\theta}(\tau_j)\hat{A}_j \right].
\tag{3}$$

重要性采样项有效地将梯度从期望值 $π_θ$ 校正为 $π_ϕ$。这与在线策略强化学习（等式1）中使用的重要性采样项$r_{i,t}(θ)$形成对比，其中分母对应于更新前的roll-out模型策略$π_{θ_{old}}$。由于 $π_θ$ 和 $π_{θ_{old}}$ 之间的差异通常远小于πθ与离线策略策略 $π_ϕ$ 之间的差异，因此等式3中的重要性采样率往往较小，用于校准来自不同分布的梯度估计。

Mixed-Policy GRPO 的 RL 目标是从原始 GRPO 目标（公式 1）扩展而来的。

$$\begin{aligned}
\mathcal{J}_{\text{Mixed}}(\theta) = & \frac{1}{Z}\Bigg(\underbrace{\sum_{j=1}^{N_{\text{off}}}\sum_{t=1}^{|\tau_j|}\min\left[\hat{r}_{j,t}(\theta,\phi)\hat{A}_j,\,\text{clip}\big(\hat{r}_{j,t}(\theta,\phi);\,1-\epsilon,1+\epsilon\big)\hat{A}_j\right]}_{\text{off-policy objective}}\\
&+\underbrace{\sum_{i=1}^{N_{\text{on}}}\sum_{t=1}^{|\tau_i|}\min\left[r_{i,t}(\theta)\hat{A}_i,\,\text{clip}\big(r_{i,t}(\theta);\,1-\epsilon,1+\epsilon\big)\hat{A}_i\right]}_{\text{on-policy objective}}\Bigg),
\end{aligned}
\quad\text{where}\quad \hat{r}_{j,t}(\theta,\phi)=\frac{\pi_{\theta}(\tau_{j,t}|q,\tau_{j,<t})}{\pi_{\phi}(\tau_{j,t}|q,\tau_{j,<t})}\quad\text{and}\quad r_{i,t}(\theta)=\frac{\pi_{\theta}(\tau_{i,t}|q,\tau_{i,<t})}{\pi_{\theta_{\text{old}}}(\tau_{i,t}|q,\tau_{i,<t})}.\tag{4}$$

其中，$Z =\sum^{N_{off}}_{j=1} |τj |+\sum^{N_{on}}_{i=1} |τ_i|$ 是归一化因子。基于非凸优化中随机梯度下降的理论分析，我们在定理1中给出了收敛性分析，表明式(3)中的重要性加权策略梯度估计器能够稳定并收敛到一个驻点，收敛速度为$O(1/\sqrt K)$，其中K是迭代次数。证明见附录A。

**Theorem 1**。假设策略梯度算法的目标函数 $J ∈ \mathcal J_n$，其中 $\mathcal J_n$ 是有限和 Lipschitz 光滑函数类，具有 $σ-bounded$ 梯度，重要性权重 $w=π_θ/π_ϕ$ 通过 $[\underline w, \overline w]$被裁剪为有界。令 $α_k = α = c/\sqrt K$，其中 $c =\sqrt{\frac{2(J(θ^∗)−J(θ^0))}{Lσ^2\underline w\overline w}}$，$θ^∗$ 为最优解。则公式 (3) 中算法的迭代满足：

$$\min_{0 \leq k \leq K-1}\mathbb{E}[||\nabla J(\boldsymbol{\theta}^k)||^2] 
\leq \sqrt{\frac{2(J(\boldsymbol{\theta}^*) - J(\boldsymbol{\theta}^0))L\overline{w}}{K\underline{w}}}\sigma.$$

离线策略学习中的重要性采样率通常涉及 $π_{\phi}$，它表示行为策略在离线策略轨迹中的概率。理论上，我们的推导和保证适用于任何定义明确的 $π_{\phi}$ 分布。在实践中，为了便于直接集成来自大型强大模型（例如DeepSeek-R1）的高质量演示，我们采用 $π_{\phi} = 1$ 以提高计算效率。这种实用选择避免了在线策略和离线策略模型之间不同的 tokenizer 方法所带来的复杂性。它有助于轻松合并现成的数据集，而无需重新计算 $π_{\phi}$，同时保留了理论保证。此外，我们省略了离线策略部署的裁剪操作，因为当 $π_{\phi} = 1$ 时，裁剪操作会不平衡。以下小节将演示**LUFFY**，它将策略设计集成到 Mixed-Policy GRPO 中。

## 2.3 Policy Shaping via Regularized Importance Sampling

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/efd806c8d46048e8a85b4cecf37a0877.png)

虽然 Mixed-Policy GRPO 通过重要性采样成功地整合了离策略 rollout，但一个新的实际挑战出现了：重要性采样加速了收敛，但显著减少了探索（图 3 左）。具体来说，熵的崩溃速度比在线策略 RL 中快得多，这表明 rollout 的确定性越来越强，而探索多样化推理轨迹的能力则有所下降。

这源于对混合策略目标的“黑客攻击”。当同时学习离线策略和在线策略信号时，模型倾向于快速收敛，**强化那些可能出现在在线策略 $π_θ$ 分布中的离线策略 token**，而忽略那些偏离线模型原始策略的离线策略 token，即那些可能代表模型尚未获得的基本推理能力的低概率 token。我们将在4.2节中对此问题进行详细的实证分析。

为了解决这个问题，我们引入了通过**正则化重要性采样**进行策略设计的技术，该技术通过重新加权离线策略分布的梯度来增强对低概率 token 的学习。具体而言，我们的方法将重要性采样率 $π_θ(τ_{j,t}|q, τ_{j,<t})/π_ϕ(τ_{j,t}|q, τ_{j,<t})$ 替换为 $f(π_θ(τ_{j,t}|q, τ_{j,<t})/π_ϕ(τ_{j,t}|q, τ_{j,<t}))$，其中 $f(·)$ 表示一个变换函数，它改变离线策略分布和在线策略分布之间的动态关系，从而增强对模型标准分布中低概率 token 的梯度强调。回想一下，我们省略了离线策略部署的裁剪操作。策略设计后的损失函数可以写成如下形式：

$$\begin{aligned}
\mathcal{J}_{\text{SHAPING}}(\theta) 
= & \frac{1}{Z}\left(\sum_{j=1}^{N_{\text{off}}}\sum_{t=1}^{|\tau_j|} f(\hat{r}_{j,t}(\theta,\phi)) \cdot \hat{A}_j\right)\\
&+ \sum_{i=1}^{N_{\text{on}}}\sum_{t=1}^{|\tau_i|}\min\left[r_{i,t}(\theta)\cdot\hat{A}_i,\text{clip}(r_{i,t}(\theta);1-\epsilon,1+\epsilon)\cdot\hat{A}_i\right],\\
&\text{where }\hat{r}_{j,t}(\theta,\phi)=\frac{\pi_{\theta}(\tau_{j,t}|q,\tau_{j,<t})}{\pi_{\phi}(\tau_{j,t}|q,\tau_{j,<t})}\text{ and } r_{i,t}(\theta)=\frac{\pi_{\theta}(\tau_{i,t}|q,\tau_{i,<t})}{\pi_{\theta_{\text{old}}}(\tau_{i,t}|q,\tau_{i,<t})}.
\end{aligned}\tag{5}$$

为了进一步说明塑造函数 $f$ 的含义，我们推导出离线策略目标的梯度如下，

$$\begin{aligned}
\nabla_{\theta}\mathcal{J}_{\text{SHAPING-OFF}}(\theta) 
&= \mathbb{E}_{\tau\sim\pi_{\phi}}\left[\nabla_{\theta}f\left(\frac{\pi_{\theta}}{\pi_{\phi}}\right)\cdot\hat{A}_j\right] \\
&= \mathbb{E}_{\tau\sim\pi_{\phi}}\left[f'\left(\frac{\pi_{\theta}}{\pi_{\phi}}\right)\frac{1}{\pi_{\phi}}\nabla_{\theta}\pi_{\theta}\cdot\hat{A}_j\right] \\
&= \mathbb{E}_{\tau\sim\pi_{\phi}}\left[f'(\pi_{\theta})\frac{\pi_{\theta}}{\pi_{\phi}}\underbrace{\nabla_{\theta}\log\pi_{\theta}\cdot\hat{A}_j}_{\text{importance sampling}}\right].
\end{aligned}\tag{6}$$

为简便起见，我们将 $π(τ_{j,t}|q, τ_{j,<t})$ 记为 $π$。从推导中可以看出，$f′(π_θ)$ 是梯度的加权函数。原始的混合策略GRPO可以视为使用线性整形函数，即 $f(π) = π$，其中应用了原始重要性采样率 $π_θ/π_ϕ$。

为了进一步了解 $f′(·)$ 如何改变模型行为，我们可以分解对数概率并推导出每个输出 logit 的梯度：

$$\begin{aligned}
\frac{\partial\mathcal{J}_{\text{SHAPING-OFF}}(\theta)}{\partial M_{\theta}(\tau'_{j,t})}
&= \mathbb{E}_{\tau\sim\pi_{\phi}}\left[f'(\pi_{\theta})\,\pi_{\theta}\left[\mathbb{I}(\tau'_{j,t}=\tau_{j,t}) - \pi_{\theta}\right]\cdot\hat{A}_j\right] \\
\Rightarrow\left|\frac{\partial\mathcal{J}_{\text{SHAPING-OFF}}(\theta)}{\partial M_{\theta}(\tau'_{j,t})}\right|
&\leq \mathbb{E}_{\tau\sim\pi_{\phi}}\left[\left|f'(\pi_{\theta})\right|\pi_{\theta}(1-\pi_{\theta})\cdot|\hat{A}_j|\right].
\end{aligned}\tag{7}$$

其中 $τ′_{j,t}$ 是在第 $j$ 条轨迹和第 $t$ 个位置上动作空间中任何可能的动作/token，$M_θ(τ)$ 表示该动作的对数。$\mathbb{I}(\cdot)$表示动作为离线策略动作 $τ =τ′$ 时的梯度，这提高了预测离策略动作的概率。从等式 7 中我们可以看出，梯度的尺度上限为 $π_θ(1 − π_θ)$，当 πθ → 0 和 πθ → 1 时，梯度的值较小。为了鼓励低概率但至关重要的动作，我们使用 $f(x) = x/(x + γ)$ 作​​为我们的变化函数（图 3 中间部分），其中所有实验的 γ 均设置为 0.1。

考虑相等情况（$τ = τ′$）和 $π_ϕ = 1$，具有策略塑造的梯度可以写成：

$$\Rightarrow \mathbb{E}_{\tau\sim\pi_{\phi}}\left[
\frac{\gamma}{(\pi_{\theta}+\gamma)^2}\pi_{\theta}(1-\pi_{\theta})\cdot\hat{A}_j
\right].\tag{8}$$

如图 3（右）所示，塑造函数重新加权梯度，赋予低概率动作更多的重要性，从而改善从离线策略轨迹中不熟悉但有效的决策中学习的能力。

## 2.4  Removing On-policy Clip

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70097542d7b64480b46542931f88f4a1.png)

在 PPO 中，引入了裁剪机制，将策略更新限制在信赖区域内，从而确保训练的稳定性。然而，当引入离线策略指导时，目标行为可能会与模型的当前策略产生显著偏差，尤其是在训练初期。

如图 4 所示，LUFFY 与 On-Policy 强化学习相比，会经历更频繁的裁剪，这会抑制对高质量离线策略轨迹的学习。为了解决这个问题，我们移除了 on-policy 裁剪，以便更灵活地更新不熟悉但有效的动作，从而释放模型更好地整合离策略推理行为的能力。

# 3.Experimental Setup
