# Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding

论文链接：https://arxiv.org/pdf/2606.21906

代码链接：https://github.com/QwenLM/Confident-Decoding

## 摘要

大语言模型（LLM）中的自回归生成通常基于最后一层进行解码，其前提假设是更深层的表征能提供更可靠的下一个 token 预测。**我们重新审视了这一假设，揭示了一种反复出现的“猜测-精炼-扰动”（Guess–Refine–Perturb）动态过程**：浅层生成粗略的猜测，中间层精炼与推理相关的语义，而最终层则可能将这些精炼后的预测扰动至通用词汇或符合对齐（alignment）偏好的词汇。为此，我们提出了一种无需训练的解码策略—— **Confident Decoding**，该策略通过熵引导的保守反向搜索，动态选择最可靠的“近末层”（near-final layer）。我们进一步将层选择问题形式化为最优停止问题，证明在存在有限投影噪声和主导性后期对齐扰动的情况下，我们的搜索规则既能过滤扰动，又能将相对于“最优精炼层”（oracle refinement layer）的损失控制在一定范围内。针对稠密模型（dense LLMs）和混合专家模型（MoE LLMs）的实验表明，该方法在 GPQA-Diamond、Omni-MATH 和 HLE 等具有挑战性的推理基准测试中均取得了持续的性能提升，且无需额外的内存开销，延迟增加也不足 2%。这些结果表明，**动态规避最终层的扰动能够激发已对齐 LLM 更强的推理能力**。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/46c959d4a0084f19b7015c2eff75ba2a.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

在大语言模型（LLM）的自回归生成过程中，通常的做法是从最后一个解码器层（decoder layer）解码出每一个 token。这种常规做法隐含了一个假设：即随着层深度的增加，表征的可靠性会逐步提升——浅层仅进行不完整的计算，中间层负责精炼上下文信息，而最深层则生成最准确的下一个 token 分布。**基于这一观点，模型最后一层被视为模型内部计算与输出词表之间自然且最优的接口**。

然而，近期证据表明，这种“**深度单调性**”假设可能会失效。探测（probing）与逐层分析显示，中间层往往编码了与任务高度相关的丰富语义，而后续层则可能对已精炼的信息进行压缩、重定向或扰动。通过对残差流（residual-stream）动态、贡献范数及预测轨迹的深入分析，我们发现大语言模型（LLM）在前向传播过程中存在一种反复出现的“猜测-精炼-扰动”（Guess–Refine–Perturb）模式（见 §2.1 及图 2）。具体而言，浅层生成粗略的统计性猜测；中间层沿稳定的语义轨迹对这些猜测进行精炼；而在针对部分 Token 的处理中，深层（最终层）会引入剧烈的表征转变，导致预测结果偏离那个与推理过程最相关的中间状态。

我们推测，这种发生在后期的扰动在一定程度上与当前后训练（post-training）阶段所采用的目标函数有关。当前的各类大语言模型（LLM）通常会经历持续预训练、有监督微调以及偏好对齐（如 RLHF、RLAIF 或 DPO）等过程。这些步骤虽然能提升模型在指令遵循、安全性及回复风格方面的表现，但也可能**导致模型最终层的输出分布倾向于生成高频、安全或平庸的续写内容**。对于常规对话或侧重安全性的提示词而言，这种对齐行为能起到有益的约束作用；然而，在处理复杂的推理任务时，同样的后期偏差可能会与模型中间层所形成的特定任务推理路径发生冲突，从而产生我们所谓的“规划与语用之间的权衡”（planning–pragmatics tradeoff）：即模型内部可能已形成了强有力的、以推理为导向的预测结果，但最终层却可能出于语用层面的考量，将输出分布转向更通用或更符合对齐偏好的 Token。

近期的一些解码策略开始利用中间层信息。例如，“层间对比解码”（Decoding by Contrasting Layers）等对比解码方法通过比较不同层的预测结果来抑制不理想的续写内容，而“Logits 演变”（logits-evolution）类方法则研究 Token 分布随网络深度变化的过程；“提前退出”（early-exit）方法也利用中间层的置信度信号来降低推理成本。然而，**这些方法要么仍将最后一层视为主要的语义锚点，要么主要关注效率而非推理能力的退化**；它们并未直接探讨：用于输出下一个 Token 分布的层是否理应总是最后一层。

为了解决这一问题，我们提出了 **Confident Decoding** ——这是一种无需训练且可直接应用的解码策略，能够在每个生成步骤中动态选择最可靠的“近末层”（near-final layer）。关键在于，置信解码既不会截断 Transformer 架构，也不会修改模型的前向传播过程。其独特之处在于：该方法不再总是将最后一层的 Logits（未归一化对数概率）直接传递给采样器，而是基于近末层的一个小窗口计算候选 Logits，并通过一种保守的向后搜索策略，选取遇到的第一个**局部熵极小值点**。鉴于较低的 Token 熵对应着更尖锐的预测分布，这一规则能够识别出模型在潜在的“深层扰动”出现之前达到高置信度的时刻（典型示例见图 1）。

我们在涵盖通用推理、数学问题求解、长上下文理解、代码生成及安全性等多个具有挑战性的基准测试上评估了“置信解码”（Confident Decoding）的表现，所用数据集包括 GPQA-Diamond、HLE、Omni-MATH、LongBench v2、LiveCodeBench v6 和 Air-Bench-2024。无论是在稠密（dense）架构还是混合专家（MoE）架构中，置信解码的表现均持续优于标准的贪婪解码及强有力的对比基线方法，且仅带来极低的内存开销和不到 2% 的额外延迟。我们进一步发现，更大、更强的模型往往表现出更显著的深层（late-layer）扰动特征；这意味着随着模型能力增强和任务难度提升，动态层选择策略的优势将愈发明显。

总之，我们的贡献主要体现在三个方面：
- 我们发现在大语言模型（LLM）的前向传播过程中存在一种反复出现的“猜测-精炼-扰动”（Guess–Refine–Perturb）动态模式，这表明对于依赖推理能力的下一个 token 预测而言，最终层的表征并不总是最可靠的信息源。
- 我们提出了一种名为“置信解码”（Confident Decoding）的策略；这是一种无需训练且可直接应用的解码方法，它在保留完整前向传播过程的同时，通过基于熵引导的保守反向搜索，动态选择接近最终结果的 Logits。
- 我们提供了理论（第3节）和实证（第5节）证据，表明“置信解码”（Confident Decoding）能够过滤高层（late-layer）扰动，保留精细的推理信号，并在推理、编程、长上下文及安全性等各类基准测试中提升性能。

## 2.Preliminaries

本节将正式刻画大语言模型（LLM）的前向传播动态，以此阐明我们所提出解码策略的动机。我们首先量化各层的表征变化，从而在数学上定义这一三阶段演进过程；随后，我们展示了一项初步研究，验证了基于熵的动态层选择机制优于静态退出范式。

### 2.1 Layer-wise Dynamics

<img
  src="https://i-blog.csdnimg.cn/direct/6fc458898d4a45fb80c2eed748df3785.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

考虑一个基于 Transformer 的 $L$ 层自回归语言模型。令 $\mathcal V$ 表示词表空间。对于给定的输入序列（截至步骤 $t$），第 $l$ 层的残差流，记为 $\textbf h^{(l)}_t ∈ \mathbb R^d$，通过注意力机制和前馈层进行更新。在标准的序列（pre-norm）公式中：

```math
\tilde{\textbf h}^{(l)}_t=\textbf h^{(l-1)}_t+f^{(l)}_{Attn}(\textbf h^{(l-1)}_t),\tag{1}
```

```math
\textbf h^{(l)}_t=\tilde h^{(l)}_t+f^{(l)}_{FFN}(\tilde{\textbf h}^{(l)}_t),\tag{2}
```

其中 $f^{(l)}_{Attn}$ 表示注意力子层（全注意力或线性注意力），$f^{(l)}_{FFN}$ 表示前馈子层（MLP 或 MoE）。我们将层贡献向量定义为由第 $l$ 层施加的总更新量：

```math
\mathbf{m}_t^{(l)}
\triangleq
\mathbf{h}_t^{(l)}-\mathbf{h}_t^{(l-1)}=
f_{\mathrm{Attn}}^{(l)}\left(\mathbf{h}_t^{(l-1)}\right)
+
f_{\mathrm{FFN}}^{(l)}\left(\tilde{\mathbf{h}}_t^{(l)}\right)
```

从而满足 $\textbf h^{(l)}_t = \textbf h^{(l-1)}_t + \textbf m^{(l)}_t$。该定义自然适用于并行注意力（parallel-attention）变体，即两个子层均直接接收 $h^{(l-1)}_t$ 作为输入。为了理解每一层如何改变 Token 预测的语义轨迹，我们分析了随网络深度变化的两个关键指标：相对贡献范数（Relative Contribution Norm）和残差输入输出余弦相似度（Residual I/O Cosine Similarity）。

```math
\mathrm{Norm\ Ratio}^{(l)}=
\frac{\left\|\mathbf{m}_t^{(l)}\right\|_2}
{\left\|\mathbf{h}_t^{(l-1)}\right\|_2},
\qquad
\mathrm{IO\text{-}CosSim}^{(l)}=
\frac{
\mathbf{h}_t^{(l)} \cdot \mathbf{h}_t^{(l-1)}
}{
\left\|\mathbf{h}_t^{(l)}\right\|_2
\left\|\mathbf{h}_t^{(l-1)}\right\|_2
}
```

在残差流分解 $\textbf h^{(l)}_t = \textbf h^{(l-1)}_t + \textbf m^{(l)}_t$ 中，Norm Ratio$^{(l)}$ 刻画了写入强度。当 Norm Ratio$^{(l)} \gg 1$ 时，该贡献主导了现有状态（即 $h^{(l)}_t \approx m^{(l)}_t$），从而有效地覆盖了累积的表征；当 Norm Ratio$^{(l)} \ll 1$ 时，该层在不改变原有内容的基础上进行增量式修正；而当 Norm Ratio$^{(l)} \lesssim 1$ 时，新写入内容的量级与所承载的残差相当。在这种中间状态下，仅凭 Norm Ratio$^{(l)}$ 无法区分是“建设性增强”还是“破坏性重写”，因为最终效果取决于更新的方向。因此，IO-CosSim$^{(l)}$ 用于刻画方向保真度：在高维表征空间中（其中方向编码了语义内容），接近 1 的数值表明该层保留并细化了现有的语义轨迹，而较低的数值则表明表征发生了旋转，转向了一个语义上截然不同的子空间。

如图2所示，针对 Qwen3.5-35B-A3B 模型（该模型在固定位置交替采用了30层线性注意力层 [DeltaNet] 和10层全注意力层，图中以灰色标示）的实证观察揭示了一种一致的三阶段结构，我们将其称为“猜测-精炼-扰动”（Guess-Refine-Perturbation）演进过程：
1. **Phase I: Guess (Shallow Layers, $l ≲ 0.15L$)**。在初始层中，Norm Ratio$^{(l)}$ 的值极高：Norm Ratio$^{(1)} ≈ 1.6$，这意味着第一个解码器层的贡献向量其模长是输入嵌入的 1.6 倍；因此，输出 $\textbf h^{(1)}_t$ 几乎完全由该层的计算结果决定，而非取决于 Token 嵌入本身。相应地，IO-CosSim$^{(1)} ≈ 0.67$——这一数值显著低于第二阶段的平台期水平——表明模型在此阶段经历了显著的方向性转变，在高度不确定的情况下迅速构建了初始的潜在表征。
2. **Phase II: Refine (Intermediate Layers, $0.15L ≲ l ≲ 0.95L$)**。Norm Ratio$^{(l)}$ 急剧下降并稳定在 0.23–0.57（< 1）的范围内：这意味着子层的贡献远小于现有的残差，因此每一层都进行增量式且方向保持一致的更新，在不改变已积累表征的前提下逐步整合上下文信息。相应地，IO-CosSim$^{(l)}$ 始终保持高位（0.91–0.97），证实了语义轨迹在不断精细化的同时并未发生旋转。该模型沿着一条稳定的语义轨迹不断优化其对 Token 的预测。
3. **Phase III: Perturbation (Post Layers, $l ≳ 0.95L$)**。在最后几层中，Norm Ratio$^{(l)}$ 显著回升，超过了第二阶段的平台期，并在 Norm Ratio$^{(40)}$ 处达到峰值——约为第二阶段水平的 2 到 3 倍。该数值仍低于 1，因此未进入第一阶段的“覆盖”（overwriting）区间。这表明该写入操作的幅度与输入残差相当，而非微小的修正性更新。佐证这一点的依据是 IO-CosSim^{(40)} ≈ 0.69（即相对于输入残差的输出状态旋转），该值远低于第二阶段的区间范围（0.91–0.97），标志着除初始嵌入阶段外最剧烈的方向偏转。这两个指标结合表明，这是一次在结构上具有显著影响且方向不一致的更新，它部分重写并改变了第二阶段所优化的语义轨迹。我们在第 3.2 节中对这一扰动背后的机制进行了形式化描述。

这种第三阶段的扰动打破了“单调提升”的假设，表明在第二阶段结束时提取预测结果，可能比基于最终层进行解码获得更高的推理准确率。附录 B.1 中的进一步实验在其他架构上验证了这一三阶段模式。

### 2.2 Motivation: Emerging Entropy Valley in Intermediate Layer

<img
  src="https://i-blog.csdnimg.cn/direct/7ea5b9121d984903af70499330d28dfb.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

第 2.1 节中的既往研究与观察表明，**大语言模型（LLM）往往在进入第三阶段之前便已收敛至稳定的语义表征**。这一发现促使我们寻找一个可靠的中间输出点或代理指标，以便在无需承担重新训练辅助分类器额外开销的情况下，规避潜在的后期扰动。

**Limitations of Static Early Exit**。我们首先考察了一种简单的静态提前退出策略的性能。如图 3(a) 所示，一个独立的伯努利概率 $p$ 代表了在任意 Token 生成步骤中，跳过最终层并强制从固定的较浅层（$L - k$）进行解码的可能性。随着该执行概率 $p$ 的增加，模型的整体准确率急剧下降。这一现象表明，静态截断策略忽略了 Token 复杂度固有的差异性；统一的退出策略过早地中断了处理“困难” Token 所必需的关键计算，从而破坏了模型推理的完整性。因此，稳健的提取机制必须具备 Token 自适应特性，即需要利用动态指标来确定每个隐藏状态的最佳处理深度。

**Entropy as a Confidence Metric**。此类指标的一个理想候选是模型的预测置信度。通过将预训练的语言建模头（$W_U$）应用于中间隐藏状态 $\textbf h_t^{(l)}$，我们可以得到逐层的概率分布 $p_t^{(l)}$。为了确保维度一致性并避免 Softmax 坍塌（collapse），我们在进行投影之前先应用最终层归一化：

```math
p_t^{(l)}=
\mathrm{Softmax}\left(
W_U \cdot \mathrm{RMSNorm}_L\left(\mathbf{h}_t^{(l)}\right)
\right)\tag{3}
```

随后，每一层的不确定性通过香农熵 $H(p^{(l)}_t) = − \sum p^{(l)}_t \log p^{(l)}_t$ 进行量化。
我们注意到，直接使用原始 $W_U$（词嵌入矩阵）作为零样本探测（zero-shot probe）工具会引发基底偏移（basis shift）问题，因为 $W_U$ 是专门针对模型最终层的潜在空间进行优化的。尽管现代大语言模型（LLM）后半部分层级的表征同质性在一定程度上缓解了这一问题，但随着层级位置远离最终层，映射误差仍会增加。这就产生了一种权衡：较深层级的映射偏差较小；而某些中间层级在受到对齐（alignment）过程引入的噪声或过度平滑（over-smoothing）效应影响之前，可能蕴含着更具“置信度”或更“纯粹”的语义信息。

为了平衡映射可靠性与预测置信度，我们提出了“熵谷”（Entropy Valley）这一概念——即从最后一层反向扫描时遇到的局部熵极小值。直观而言，该“熵谷”代表了一种状态：在后续层可能引入不利的振荡之前，模型已就输出分布达成了一致。

**Empirical Validation**。图 3 展示了这种动态选择的优越性。与静态退出策略不同，即使频繁应用，熵谷策略也能保持较高的准确率。此外，图 3(b) 表明，熵谷本身就是一个精确的最优边界：从其相邻节点（Valley$±k$）解码会导致性能持续下降。这些结果证实，熵谷可以作为细化阶段峰值的可靠且与 token 相关的代理，从而证明了将其用作我们提出的回滚机制的核心边界的合理性。

## 3.Theoretical Grounding

为了从理论上确立“置信解码”（Confident Decoding）的优越性，我们从信息论和表征工程的视角分析了其前向传播的动态特性。我们阐明了为何表征最优状态出现在中间层深度，并证明了我们采用的保守式反向搜索机制相当于一种数学意义上的最优自适应滤波器——它能选择性地抵消对齐扰动，同时保留安全防护机制。

### 3.1 Information-Theoretic View of Layer Dynamics

<img
  src="https://i-blog.csdnimg.cn/direct/ab733d0ed33440279f18cb15aabd65f7.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

尽管信息瓶颈（IB）原理通常用于刻画神经网络的训练过程，但所习得的权重矩阵已将这一优化机制内化，从而决定了推理过程中逐层的信息流动。设 $X$ 表示输入上下文，$Y_{logic}$ 表示严格基于特定领域推理得出的最优目标 token；残差流 $\textbf h^{(l)}_t$ 的演变过程即是在 IB 权衡（tradeoff）中进行调节的过程：

```math
\mathcal L_{IB}=I(X;\textbf h^{(l)}_t)-\beta I(\textbf h^{(l)}_t;Y_{logic})
```

在 **Phase I (Guess)**，网络会剧烈压缩表层输入 $X$（即最小化 $I(X; \textbf h)$）。这种大幅度的降维会导致几何波动，使得预测熵 $H(p^{(l)}_t)$ 极不稳定，并容易受到退化的单字（unigram）偏差的影响。至关重要的是，这种压缩噪声的残留影响会延续至第二阶段的初期，导致预测熵维持在一个高熵平台期；如图 4 所示，该状态在经验层面上与第一阶段无法区分。

随后，网络进入 **Phase II (Refine)**；在此阶段，连续的注意力机制严格整合上下文依赖关系，主导信息流向以最大化互信息 $I(\textbf h^{(l)}_t; Y_{\text{logic}})$。在第二阶段中，我们确定了一个内部转折层 $V_{\text{onset}}$，在该层处，第一阶段的遗留效应消退，网络进入单调精炼模式。理想情况下，在 $l \ge V_{\text{onset}}$ 这一特定区间内，预测熵 $H(p^{(l)}_t)$ 构成了真实条件熵 $H(Y_{\text{logic}}|\textbf h^{(l)}_t)$ 的经验上界。随着互信息在这一后期精炼窗口内单调增加，$H(p^{(l)}_t)$ 呈现出严格单调下降的趋势。

$V^∗$ 处的极小值点代表了潜在扰动介入之前的最佳表征状态。如图 4(a) 所示，在 Qwen3.5-35B-A3B 模型于 GPQA Diamond 数据集上进行推理（每个提示生成 4,096 个 token）的过程中，第二阶段（Phase II）在处理 202,935 个 token 时累计产生了约 −10.8 nats 的熵变 $∆H$（其中 $V_{onset}$ ≈ 28，$V^∗$ = 39）。关键在于，第三阶段（Phase III）并非一个固定的架构区域，而是一种取决于最终层是否扰动了 token 预测结果的逐 token 现象：16.2% 的 token 在第 40 层（l = 40）表现出 +0.37 nats 的熵增——这些 token 经历了第三阶段。图 1 展示了此类 token 被来自较早层的低熵替代项所取代的例子。其余 83.8% 的 token 在 $V^∗$ 处继续演化，表现出进一步的熵降（−2.52 nats）——这些 token 只是通过最终层完成了正常的“第一阶段至第二阶段”的细化过程。

### 3.2 Modeling Alignment: Tax vs. Guardrail

现代大语言模型（LLM）会经历严格的后训练过程，以实现与人类偏好的对齐。表征工程（Representation Engineering）领域的最新研究表明，这种对齐行为受**低秩引导向量**（low-rank steering vectors）的调控；这些向量主要在模型末层被激活，从而在不覆盖预训练知识的前提下调节输出。因此，在 **Phase III (Perturbation)** 中，潜在表征会针对一个通用的安全分布 $P_{align}$ 优化某种经正则化处理的风险目标：

```math
\mathcal R^{(l)}=\mathbb E[-log~p^{(l)}_t(Y_{logic}|X)]+\lambda\mathcal D_{KL}(p^{(l)}_{logic}||p_{align})
```

其中，$λ$ 决定了引导强度。令 $V^∗$ 表示第二阶段与第三阶段之间的理想（oracle）转换边界。当 $l → L$ 时，网络引入加性扰动 $δ^{(l)}_{align}$ 以最小化 KL 散度：$\textbf h^{(l)}_t ≈ \textbf h^{(V^∗)}_t + ∑^l_{k=V^∗+1}δ^{(k)}_{align}$。

至关重要的是，这种扰动的破坏性是有**严格条件**的：
- **Alignment as a Guardrail (Safety Tasks)**。对于标准的对话或安全类 query，$P_{logic}$ 与 $P_{align}$ 基本相当，KL 散度可忽略不计。后期的层级主要用于优化格式，而不会改变语义。
- **Alignment as a Tax (Complex Reasoning)**。在严谨数学与科学等领域，专用的逻辑分布 $P_{\text{logic}}$ 与通用的 $P_{\text{align}}$ 存在剧烈冲突。这种加性扰动会强行将潜在状态（latent state）从推理子空间中拉离，在数学上表现为一种非自然的“熵振荡”，从而导致既定的逻辑链条发生偏离。

### 3.3 Minimax Optimality of Conservative Backward Search

## 4.Methodology

标准的自回归解码过程通过 Transformer 最后一层的反嵌入（unembedding）操作来输出下一个 token 的概率分布。我们提出了一种名为“置信解码”（Confident Decoding）的方法，这是一种无需训练、可直接应用于推理阶段的即插即用方案。在该方法中，模型在每个解码步骤不再使用最后一层的输出，而是选择使用“置信度最高”的层（通常位于模型末端附近）所生成的 Logits 进行后续处理；这里的置信度通过反嵌入后分布的香农熵（Shannon entropy）来衡量。整个前向传播过程、模型权重、键值（KV）缓存以及后续的采样步骤均保持不变；唯一的变动在于动态选择将哪一层的 Logits 传递给采样器。

### 4.1 Confident Decoding

<img
  src="https://i-blog.csdnimg.cn/direct/0d702379166541a396f865c6bbfbae4c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>


**Notation.** 考虑一个仅包含解码器（decoder-only）的语言模型，其包含 $L$ 个 Transformer 块，隐藏层维度为 $d$，词表为 $\mathcal{V}$，且反嵌入（unembedding）矩阵为 $W_U \in \mathbb{R}^{|\mathcal{V}|\times d}$。在生成步骤 $t$，令 $x_t^{(\ell)} \in \mathbb{R}^d$ 表示经过第 $\ell$ 个 Transformer 块（$\ell \in \{1,\ldots,L\}$）后的残差流状态。在 Pre-norm 架构中，在进行反嵌入之前会执行一次最终的归一化操作 $\mathrm{Norm}(\cdot)$，从而得到候选表示、Logits 以及 Softmax 分布：

```math
\tilde{\mathbf{h}}_t^{(\ell)}=
\mathrm{Norm}\left(\mathbf{x}_t^{(\ell)}\right),
\qquad
\mathbf{z}_t^{(\ell)}=
W_U\tilde{\mathbf{h}}_t^{(\ell)},
\qquad
\mathbf{p}_t^{(\ell)}=
\mathrm{softmax}\left(\mathbf{z}_t^{(\ell)}\right),
\tag{4}
```

连同香农熵：

```math
H_t^{(\ell)}=
-\sum_{v\in\mathcal{V}}
\mathbf{p}_t^{(\ell)}(v)
\log \mathbf{p}_t^{(\ell)}(v).
\tag{5}
```

较低的 $H_t^{(\ell)}$ 对应于第 $\ell$ 层更尖锐、置信度更高的预测分布。标准的解码过程使用 $\mathbf{z}_t^{(L)}$。尽管将最终层的反嵌入矩阵 $W_U$ 应用于中间状态会引入潜在的“基底偏移”（basis shift），但这对于“置信解码”（Confident Decoding）而言依然是最优选择。由于解码过程在物理上受限于必须使用 $W_U$ 进行词表投影，因此中间层的熵在数学上精确反映了实际解码时的不确定性。此外，针对“相对单调梯度” $(\Delta \hat{H}(\ell))$（而非绝对量值）进行分析时，均匀投影噪声会被自然滤除。

**Candidate set.** “置信解码”（Confident Decoding）将注意力限制在接近最终结果的候选集合上。

```math
\mathcal{C}=
\{L-M+1,\ldots,L\},
\tag{6}
```

大小为 $M \in [1, L]$，受显式最大回溯窗口或模型深度固定比例的控制。回溯程度越高，计算成本也越高。

**Entropy-trough selection.** 默认规则对于每个 token，选择从 $\ell=L$ 向后扫描 $\mathcal{C}$ 时遇到的第一个*局部熵谷*。一旦向下移动一层未能严格降低 $H_t^{(\ell)}$，则标记的选择将被冻结。等价地，对于每个 token 的扫描窗口 $K\in[1,M]$，所选层为

```math
\ell_t^\star=
\min
\left\{
\ell\in[L-K+1,L]:
H_t^{(\ell)}
<
H_t^{(\ell+1)}
<
\cdots
<
H_t^{(L)}
\right\}.
\tag{7}
```

### 4.2 Systems Implementation

## 5.Experiments
