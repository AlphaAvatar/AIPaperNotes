论文链接：https://arxiv.org/pdf/2404.04656v1

代码链接：

# 摘要

通过偏好优化将大语言模型 (LLM) 与人类偏好对齐至关重要，但却十分耗时费力，需要评估者对每个提示的接受文本补全和拒绝文本补全进行比较。最近，Kahneman-Tversky 优化 (KTO) 证明，LLM 仅需在每个提示-补全对上使用二元“点赞”或“踩”信号即可进行对齐。本文提出了解释通过这些二元信号实现成功对齐的理论基础。我们的分析揭示了一个新的视角：优化一个以 logit 为奖赏的二元分类器，隐式地实现了直接偏好优化 (DPO) 损失的最小化。在这一发现过程中，我们确定了两种有效的对齐技术：奖赏偏移和底层分布对齐。因此，我们提出了一种新的算法—— Binary Classifier Optimization，该算法整合了这两种技术。我们在两种设置下验证了我们的方法：首先，在一个配对偏好数据集上，我们的方法与 DPO 和 KTO 的性能相当；其次，在模拟真实世界场景的二元信号数据集上，我们测试了点赞和点踩数据之间存在显著差异的潜在分布。我们的模型在两个基础 LLM 和三个不同的二元信号数据集上均表现出有效且稳健的对齐能力，充分展现了我们方法在从二元反馈中学习方面的强大优势。

# 1.Introduction

大语言模型 (LLM) 的对齐一直是 LLM 在生产环境中部署的关键环节，因为预训练的 LLM 容易产生不理想的输出。Ouyang et al. (2022) 提出了基于人工反馈的强化学习 (RLHF) 方法，该方法首先基于单个提示的各种补全及其比较结果训练一个奖赏模型，然后优化 LLM 以最大化这些奖赏。随后，Rafailov et al., (2023) 提出了直接偏好优化 (DPO) 方法，该方法无需训练奖赏模型，而是直接基于已选择和已拒绝的补全之间的偏好来优化模型。RLHF 和 DPO 都已成为 LLM 对齐的标准方法，但它们仍然需要一个包含已选择和已拒绝文本补全的比较数据集，而收集这样的数据集非常耗时耗力。

由 Ethayarajh et al. (2024) 提出的 Kahneman-Tversky 优化算法 (KTO) 源于经济学中的前景理论，它为对齐提供了一种很有前景的方法，该方法仅需每个提示进行一次补全，并附带一个二元偏好信号，例如“点赞”或“点踩”。这一进展使得无需再费力地比较完成情况以创建偏好数据集，从而使对齐过程更加灵活便捷。

本文提出了基于二元信号进行对齐的有效性的理论基础，是对 Ethayarajh et al. (2024) 研究成果的补充。我们的分析表明，训练一个以 logit 作为奖赏的二元分类器，有效地将 $\texttt{\{prompt, thumbs-up completion\}}$ 对映射到 1，将 $\texttt{\{prompt, thumbs-down completion\}}$ 对映射到 0，可以隐式地最小化直接偏好优化 (DPO) 损失。具体而言，分类器训练中使用的二元交叉熵 (BCE) 损失可以作为最小化 DPO 损失的上界。

基于研究结果，**我们提出了两种技术：1）奖赏转移和 2）底层分布对齐**。奖赏转移技术旨在最小化 BCE 损失和 DPO 损失之间的差异。底层分布对齐技术能够实现模型对齐，假设点赞和点踩数据集来自相同的底层 $\texttt{\{prompt, chosen completion, rejected completion\}}$ 分布，这在现实世界中是不合理的。我们采用重要性抽样和密度比技巧来实现分布对齐。

我们在两类数据集上验证了我们的方法：成对偏好数据集和真实世界二元信号数据集。在成对偏好数据集上，我们证明了我们的方法与 DPO 和 KTO 的性能相当。在具有不同潜在分布的真实世界二元信号数据集（该数据集中“点赞”和“点踩”子集的分布不同）上，实证结果证实了 BCO 在两种不同的基础 LLM（StableLM-2-1.6B 和 Mistral-7B-v0.1）以及三个不同的数据集上均具有优越性。

# 2.Related Work

基于人类反馈的强化学习（RLHF）作为一种将 LLM 与人类偏好对齐的有前景的方法，已引起广泛关注。尽管RLHF有效，但由于需要经历三个阶段：有监督微调（SFT）、奖赏建模和强化学习（RL），因此较为繁琐。其中，强化学习阶段尤其耗费内存，因为它需要将策略、参考模型、奖赏模型和价值函数加载到内存中。DPO 的引入通过消除奖赏建模阶段，仅需在优化过程中将策略和参考模型加载到内存中，从而提高了 LLM 对齐的便捷性。DPO使用源自Bradley-Terry（BT）模型的损失函数直接优化策略以满足人类偏好。

DPO 的一个潜在缺点是容易过拟合偏好数据集。为了解决这个问题，身份偏好优化（Identity Preference Optimization）引入了一个正则化项来缓解过拟合。拒绝采样优化（Rejection Sampling Optimization）则采用拒绝采样从估计的最优策略中生成偏好对。尽管这些方法与我们的工作有共同之处，例如它们都为 BT 模型提供了理论见解并提出了改进的对齐方法，但它们仍然依赖于偏好数据集，这使它们与我们的工作有所区别。

为了减少收集偏好数据集所需的工作量，一些方法被提出，这些方法要么让语言学习模型（LLM）自身进行补全比较，要么将 LLM 的补全结果视为被拒绝的补全结果。相比之下，受前景理论启发的 KTO 旨在仅使用点赞和点踩数据集来对齐LLM，而无需构建偏好数据集。在基于二元信号对齐LLM方面，KTO 与我们的工作最为相似。与 KTO 不同的是，我们从理论上论证了基于二元信号的对齐与偏好优化之间的联系，并在此基础上，提出了两种在实际场景中实现稳健对齐的技术。

# 3.Preliminaries

将 LLM 与人类偏好对齐遵循 Ouyang et al. (2022) 提出的广泛应用的惯例，该惯例包含三个主要阶段：SFT、奖赏建模和强化学习 (RL)。在 SFT 阶段，给定一个来自数据集 $\mathcal D$ 中的输入提示 $x$ 和对应的补全 $y$，最大化给定 $x$ 的 $y$ 的生成概率，即 $−\mathbb E_{(x,y)∼\mathcal D} [log~p(y|x)]$。在奖赏建模阶段，训练一个独立的奖赏模型，为给定的 $\{prompt, completion\}$ 对分配反映人类偏好的适当标量奖赏。最后，应用强化学习，利用人类偏好标注的数据集进一步对齐从 SFT 获得的模型，这通常涉及使用获得的奖赏模型优化策略。

**Reward Model**。奖赏建模从一个三元组数据集 $\mathcal D = \{x^{(i)}, y^{(i)}_w , y^{(i)}_l\}^N_{i=1}$ 开始，其中 $(x, y_w, y_l) ∼ \mathcal D$ 分别表示一个采样的提示、选择的补全结果和拒绝的补全结果的集合。根据人类偏好，选择和拒绝的对表示 $y_w ≻ y_l|x$。使用数据集 $\mathcal D$，训练底层奖赏模型 $r: \mathcal X × \mathcal Y → \mathbb R$ 的一种常用方法是最大化 BT 模型。

$$p(y_w ≻ y_l|x)=\frac{exp(r_{\phi}(x,y_w))}{exp(x,y_w)+exp(x,y_l)}=\sigma(r_{\phi}(x,y_w),r_{\phi}(x,y_l)),$$

其中$\phi$是一个参数。

**RL**。本阶段的主要目标是优化策略 $π$，使其收益最大化，收益由奖赏模型衡量。更正式地说，目标是：

$$\mathcal J(\theta)=\mathbb E_{(x,y)\sim\mathcal D}[r(x,y)]-\beta KL(\pi_{\theta}(\cdot|x),\pi_{ref},\pi_{ref}(\cdot|x))\tag{1}$$

其中第二项是对参考策略 $π_{ref}$ 的 Kullback-Leibler (KL) 约束，其系数为 $β$，$θ$ 为策略模型参数。参考策略$π_{ref}$ 通常设置为 SFT 模型，并且通常使用训练好的奖赏函数 $r_ϕ$ 代替 r。然后使用诸如近端策略优化 (PPO) 之类的强化学习算法最大化目标函数 $\mathcal J$。

**DPO**。虽然使用训练好的奖赏模型的RLHF已被证明是成功的，但它也存在一些挑战，例如计算量大且需要额外的训练阶段。DPO 通过证明可以使用偏好数据集 $\mathcal D$，**并利用公式1中的奖赏-策略关系直接优化策略 $π_θ$，从而巧妙地规避了这些挑战**。然后，奖赏函数可以定义为策略的函数，即 $r_θ(x, y) = βlog\frac{π_θ(y|x)}{π_{ref}(y|x)}$，而不会损失DPO理论基础的一般性。将BT模型与奖赏模型相结合，DPO的损失函数为：

$$\mathcal L_{DPO}(\theta)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}[log\sigma(r_{\theta}(x,y_w)-r_{\theta}(x,y_l))].\tag{2}$$

**KTO**。KTO 提出了一种对齐框架，该框架基于每个提示补全情况的二元信号（点赞或点踩）进行训练。然后，给定一个包含 $\texttt{\{prompt, completion\}}$ 对及其相应二元信号的数据集，KTO基于 Tversky & Kahneman (1992) 提出的**人类价值函数**形式定义了一个价值函数。

$$v_{KTO}(x,y;\theta)=\begin{cases}
\sigma(r_{\theta}(x,y)-z_{ref}) & if~r_{\theta}(x,y)\ge z_{ref}\\
\sigma(z_{ref}-r_{\theta}(x,y)) & if~r_{\theta}(x,y)\lt z_{ref},
\end{cases}\tag{3}$$

其中 $z_{ref}$ 为参考点。有关 $z_{ref}$ 选择的详细信息，请参见附录 C。最后，KTO 的损失函数定义为：

$$\mathcal L_{KTO}(\theta)=\mathbb E_{(x,y)\sim \mathcal D}[w(y)(1-v_{KTO}(x,y;\theta))]\tag{4}$$

其中，如果 $y$ 是来自点赞数据集的补全结果，则权重因子 $w(y)$ 为 $λ_D$；如果 y 是来自点踩数据集的补全结果，则权重因子 $w(y)$ 为 $λ_U$。

# 4.Binary Classifier Optimization

本文探讨了利用二元反馈信号进行 LLM 对齐的理论基础，二元反馈信号比成对偏好数据集更容易收集。我们提出了一种名为  **Binary Classifier Optimization (BCO)** 的新方法，该方法融合了奖赏偏移和底层分布匹配技术，从而在理论基础上，利用二元信号实现稳健的对齐。

在本节中，我们将通过优化奖赏来阐述对齐过程。需要注意的是，由于奖赏-策略关系 $r_θ (x, y) = βlog\frac{π_θ(y|x)}{π_{ref}(y | x)}$（如第 3 节所述），奖赏优化足以实现对齐。

## 4.1 Theoretical Analysis

为简化起见，我们暂时假设公式 3 中的 $z_{ref}$ 为 0。如第 3 节所述，DPO 损失函数是最小化 $−log~σ(r_θ (x, y_w) − rθ (x, y_l))$，而 KTO 损失函数最小化 $−σ(r_θ (x, y_w)) − σ(−r_θ (x, y_l))$。通过建立这两个项之间的联系，我们可以桥接 DPO 与基于二元信号的对齐方法之间的差距。

**Theorem 1**。对于分配奖赏 logit 的二元分类器，其中 $\texttt{\{prompt, choen completion\}}$ 对映射到1，$\texttt{\{prompt, rejected completion\}}$ 对映射到0，最小化真实标签和预测标签之间的二元交叉熵损失是直接偏好优化损失的上界。即：

$$\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}
\left[ -\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right) \right]
<\ 
\mathbb{E}_{(x, y_w)\sim \mathcal{D}}
\left[ -\log \sigma\left(r_\theta(x, y_w)\right) \right]
+
\mathbb{E}_{(x, y_l)\sim \mathcal{D}}
\left[ -\log \left(1 - \sigma\left(r_\theta(x, y_l)\right)\right) \right]$$


为证明上述定理，我们先证明下面的引理。

**Lemma 2**。 Sigmoid 的 log 对和大于 log 对 Sigmoid 的和。即  $log~σ(x + y) > log~σ(x) + log~σ(y)$ 对所有 $x, y ∈ ℝ$ 都成立。

**证明.**

$$
\log \sigma(x+y) = -\log\left(1 + e^{-(x+y)}\right) \tag{5}
$$

$$
\begin{aligned}
\log \sigma(x) + \log \sigma(y)
&= -\log(1+e^{-x}) - \log(1+e^{-y}) \\
&= -\log\big((1+e^{-x})(1+e^{-y})\big) \\
&= -\log(1 + e^{-(x+y)} + e^{-x} + e^{-y}) \tag{6}
\end{aligned}$$

由于 $e^{-x}$ 和 $e^{-y}$ 都大于 0，命题成立。

简单应用引理 2 和期望的线性性即可完成**定理 1**的证明。

$$
\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}
\big[-\log \sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\big]
$$

$$
< \mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}
\big[-\log \sigma(r_\theta(x,y_w)) - \log \sigma(-r_\theta(x,y_l))\big] \tag{7}
$$

$$= \mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[-\log\sigma(r_\theta(x,y_w))] + \mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[-\log\sigma(-r_\theta(x,y_l))] \tag{8}$$

方程 8 是二元交叉熵（BCE）损失，其中二元分类器的 logit 是由 policy 模型和 reference 模型隐式定义的奖励。由于 BCE 损失作为 DPO 损失的上界，仅使用二元信号即可对 LLM 进行 alignment。

根据**方程 6**，BCE 损失作为 DPO 损失上界的紧度依赖于误差项  $e^{-x} + e^{-y}$，其中 $x = r_\theta(x,y_w)$、$y = -r_\theta(x,y_l)$。随着训练进行且 BCE 损失被最小化，$r_\theta(x,y_w)$ 的幅度增大，而 $r_\theta(x,y_l)$ 的幅度减小，从而使误差项减小。于是，BCE 损失作为 DPO 损失的上界变得更紧。第 5 节的实验证明，误差项不会显著影响真实数据上的训练。

## 4.2 Reward Shift

## 4.3 Matching Underlying Distributions

由于期望的线性性，**方程 7** 和 **方程 8** 中的期望可以分布化，从而在优化过程中无需同时考虑 $y_w$ 和 $y_l$。然而，这里有一个重要假设：当分别对 $-\log\sigma(r_\theta(x,y_w))$ 和 $-\log\sigma(-r_\theta(x,y_l))$ 取平均时，平均必须在同一个三元组分布 $\mathcal{D}$ 上进行。

在实际场景中，thumbs-up 和 thumbs-down 数据集很可能具有不同的底层分布，违反了这一假设。例如，考虑一个部署给用户的 LLM：如果模型在写作任务上表现很好，但在编码任务上表现较差，那么 thumbs-up 数据集会被写作相关的 prompts 主导，而 thumbs-down 数据集则会包含主要由编码相关 prompts 组成的内容。

因此，训练模型时需要假设收集到的 thumbs-down 数据集是从 thumbs-up 数据集的底层分布中采样的。令 $p^+(x, y_w, y_l)$ 表示 thumbs-up 数据集 $D^+$ 的底层分布，$p^-(x, y_w, y_l)$ 表示 thumbs-down 数据集 $D^-$ 的底层分布。假设：

$$
p^+(x,y_w,y_l)/p^-(x,y_w,y_l) = p^+(x)/p^-(x),
$$

则有：

$$
\mathbb{E}_{(x,y_w)\sim D^+}[-\log\sigma(r_\theta(x,y_w))] +
\mathbb{E}_{(x,y_l)\sim D^+}[-\log\sigma(-r_\theta(x,y_l))]
$$

$$
=\mathbb{E}_{(x,y_w)\sim D^+}[-\log\sigma(r_\theta(x,y_w))] +
\mathbb{E}_{(x,y_l)\sim D^-}\left[-\frac{p^+(x)}{p^-(x)} \log\sigma(-r_\theta(x,y_l))\right] \tag{9}
$$

这一结果使得我们可以从 thumbs-up 和 thumbs-down 数据集中学习，仿佛它们来自相同的分布，尽管它们在底层分布中存在实际差异。

密度比 $p^+(x)/p^-(x)$ 使用 density-ratio trick 来估计，即：

$$
\frac{p_\psi(f=1\mid x)}{p_\psi(f=0\mid x)},
$$

其中 $f$ 是一个二元反馈变量，表示 thumbs-up ($f = 1$) 或 thumbs-down ($f = 0$)，而 $p_\psi(f=1\mid x)$ 表示 prompt $x$ 从 thumbs-up 数据集中采样的概率。该概率由参数 $\psi$ 定义的模型建模。一个小尺寸的 text-embedding 模型加 logistic 回归模型即可实现 $p_\psi$。更多细节见 Appendix E。

将 density-ratio trick 代入后，方程 9 可写为：

$$
-\mathbb{E}_{(x,y_w)\sim D^+}[\log\sigma(r_\theta(x,y_w))] =
-\mathbb{E}_{(x,y_l)\sim D^-}\left[\frac{p_\psi(f=1\mid x)}{p_\psi(f=0\mid x)} \log\sigma(-r_\theta(x,y_l))\right] \tag{10}
$$

结合奖励偏移与重要性采样，最终的损失函数为：

$$
\mathcal{L}_{BCO}(\theta) =
-\mathbb{E}_{(x,y)\sim D^+}[\log\sigma(r_\theta(x,y) - \delta)]
-\mathbb{E}_{(x,y)\sim D^-}\left[\frac{p_\psi(f=1\mid x)}{p_\psi(f=0\mid x)} \log\sigma(-(r_\theta(x,y)-\delta))\right] \tag{11}
$$
# 5.Experiments
