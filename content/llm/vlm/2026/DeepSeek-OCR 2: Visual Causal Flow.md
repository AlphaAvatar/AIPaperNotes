论文链接：https://arxiv.org/pdf/2601.20552

代码链接：http://github.com/deepseek-ai/DeepSeek-OCR-2

# 摘要

我们提出 DeepSeek-OCR 2，旨在研究一种新型编码器的可行性 — **DeepEncoder V2**。**该编码器能够根据图像语义动态地重新排列视觉 token**。传统的视觉语言模型（VLM）在将图像输入 LLM 时，总是以固定的位置编码，按照固定的光栅扫描顺序（从左上到右下）处理视觉 token。然而，这与人类视觉感知相悖。人类视觉感知遵循灵活但语义连贯的扫描模式，这种模式由其固有的逻辑结构驱动。尤其对于布局复杂的图像，人类视觉会表现出因果关系驱动的顺序处理能力。受此认知机制的启发，DeepEncoder V2 的设计旨在赋予编码器因果推理能力，使其能够在基于 LLM 的内容解释之前智能地重新排列视觉 token。这项工作探索了一种新的范式：能否通过两个级联的一维因果推理结构有效地实现二维图像理解，从而提供一种新的架构方法，有可能实现真正的二维推理。代码和模型权重可在 http://github.com/deepseek-ai/DeepSeek-OCR-2 公开获取。

# 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/2297a21fe2ae4d6abbf5bff53d90b885.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

人类视觉系统与基于Transformer的视觉编码器非常相似：中央凹注视点作为视觉 token，局部清晰且具有全局感知能力。然而，与现有编码器从左上到右下刚性扫描 token 不同，人类视觉遵循由语义理解引导的因果驱动流程。**以追踪螺旋为例——我们的眼球运动遵循内在逻辑，其中每一次后续注视都因果依赖于之前的注视**。类似地，模型中的视觉 token 也应根据视觉语义而非空间坐标进行选择性处理，其顺序高度依赖于视觉语义。

这一洞见促使我们从根本上重新思考视觉语言模型（VLM）的架构设计，尤其是编码器组件。LLM 本质上是基于一维序列数据进行训练的，而图像是二维结构。**直接按照预定义的栅格扫描顺序展平图像块会引入不必要的归纳偏差，忽略语义关系**。为了解决这个问题，我们提出了DeepSeek-OCR 2，并采用了一种新的编码器设计——DeepEncoder V2——以期实现更接近人类的视觉编码。与 DeepSeek-OCR 类似，我们选择文档阅读作为主要实验平台。文档带来了丰富的挑战，包括复杂的布局顺序、复杂的公式和表格。这些结构化元素本身就蕴含着因果视觉逻辑，需要强大的推理能力，这使得文档 OCR 成为验证我们方法的理想平台。

我们的主要贡献体现在三个方面：

首先，我们提出了 **DeepEncoder V2**，它具有以下几个关键创新：（1）如图 1 所示，我们用紧凑的 LLM 架构替换了 DeepEncoder 中的 CLIP 组件，以实现视觉因果流；（2）为了实现并行处理，我们引入了可学习 query，称为因果流 token，视觉 token 作为前缀添加到因果流 token 中——通过定制的注意力掩码，视觉 token 保持全局感受野，而因果流 token 可以获得视觉 token 重排序能力；（3）我们保持因果 token 和视觉 token 之间的基数相等（通过填充和边界等冗余），以提供足够的容量进行重新注视；（4）只有因果流 token（编码器输出的后半部分）被送入 LLM 解码器，从而实现级联因果感知视觉理解。

其次，我们利用 DeepEncoder V2 提出了 **DeepSeek-OCR 2**，它在保持 DeepSeek-OCR 图像压缩比和解码效率的同时，实现了显著的性能提升。我们将输入到 LLM 的视觉 token 数量限制在 256 到 1120 之间。下限 (256) 对应于 DeepSeek-OCR 对 1024×1024 图像的 tokenizer，而上限 (1120) 与 Gemini-3 Pro 的最大视觉 token 预算相匹配。这种设计使得 DeepSeek-OCR 2 既可以作为一种用于研究探索的新型 VLM 架构，也可以作为一种用于生成高质量 LLM 预训练数据的实用工具。

最后，我们初步验证了将语言模型架构用作 VLM 编码器的可行性——这是实现**统一全模态编码的**一条很有前景的途径。该框架只需配置特定模态的可学习 qeury，即可实现跨多种模态（图像、音频、文本）的特征提取和 token 压缩。至关重要的是，它能够自然地兼容语言模型（LLM）领域的高级基础设施优化，包括混合专家（MoE）架构、高效注意力机制等。

综上所述，我们提出了用于 DeepSeek-OCR 2 的 DeepEncoder V2，它采用专门的注意力机制来有效地建模文档阅读的因果视觉流程。与 DeepSeek-OCR 基线相比，DeepSeek-OCR 2 在 OmniDocBench v1.5 上的性能提升了 3.73%，并在视觉阅读逻辑方面取得了显著进步。

# 2.Related Works

<img
  src="https://i-blog.csdnimg.cn/direct/f8011744a5264099986f4f5c700da35e.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## 2.1 Parallelized Queries in Decoder

DETR 开创性地将 Transformer 架构集成到目标检测中，从根本上突破了传统的检测范式。为了克服 Transformer 模块中串行解码的效率限制，DETR 引入了预设的**并行可学习 query**——一组包含 100 个目标 query 的集合，这些 query 通过训练编码了目标的先验信息，例如形状和位置。这些 query 通过交叉注意力机制与特征图交互，同时通过自注意力机制进行双向信息交换。DETR 建立了一个基础范式，使 Transformer 能够处理并行化的 token。此后，目标 query 设计已成为后续基于 Transformer 的目标检测方法中事实上的标准架构组件。

## 2.2 Parallelized Queries in Projector

近年来，视觉语言模型发展迅速，其架构逐渐趋向于 **encoder-projector-LLM 范式**。投影器将视觉 token 与LLM的嵌入空间对齐，作为连接两者的关键桥梁，使 LLM 能够理解视觉内容。BLIP-2 中提出的 Q-former 就是一个有效的投影器设计范例，它利用可学习 query 进行视觉 token 压缩。Q-former 采用了类似 BERT 的架构，并借鉴了 DETR 的目标查询，利用32个可学习 query，通过交叉注意力机制与数百个CLIP 视觉 token 进行交互。这些压缩后的 query 表示随后被输入到LLM中，实现了从视觉空间到语言空间的有效映射。Q-former 的成功表明，**并行化的可学习 qeury 不仅对检测任务中的特征解码有效，而且对多模态对齐中的 token 压缩也同样有效**。

## 2.3 LLM-based Multimodal Initialization

基于大规模互联网数据训练的 LLM 已被证明可有效用于多模态模型的初始化。Pang et al. [35]  证明，冻结的 LLM Transformer 层能够增强视觉判别任务的性能。此外，诸如视觉领域的 Fuyu 和 Chameleon 等无编码器或轻量级编码器模型，以及语音领域的 VALL-E 等模型，进一步验证了 LLM 预训练权重在多模态初始化方面的潜力。

# 3.Methodology

## 3.1 Architecture

<img
  src="https://i-blog.csdnimg.cn/direct/0f1afefc5a2842de9f9fac0d59efc5f1.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

如图 3 所示，DeepSeek-OCR 2 继承了 DeepSeek-OCR 的整体架构，该架构由编码器和解码器组成。编码器将图像离散化为视觉 token，而解码器则根据这些视觉 token 和文本提示生成输出。关键区别在于编码器：我们将 DeepEncoder 升级为 DeepEncoder V2，它保留了前代的所有功能，并通过一种全新的架构设计引入了因果推理。我们将在以下章节中详细阐述 DeepSeek-OCR 2 的细节。

## 3.2 DeepEncoder V2

<img
  src="https://i-blog.csdnimg.cn/direct/43377c4a9495402893ddc3396caa1828.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

原始编码器是一个重要的组件，它通过注意力机制提取和压缩图像特征，其中每个 token 都关注所有其他 token，从而实现类似于人类中央凹和周边视觉的全图像感受野。然而，将二维图像块展平为一维序列会通过面向文本的位置编码（例如 RoPE）引入一种僵化的排序偏差。这与自然的视觉阅读模式相悖，尤其是在光学文本、表单和表格等非线性布局中。

### 3.2.1 Vision tokenizer

DeepEncoder V2 的第一个组件是视觉 tokenizer。与 DeepEncoder 类似，我们采用了一种架构，该架构结合了一个 80M 参数的  SAM-base 模型和两个卷积层。为了与后续流程保持一致，我们将最后一个卷积层的输出维度从 DeepEncoder 中的 1024 降低到 896。需要注意的是，这种基于压缩的 tokenizer 并非必需，可以用简单的块嵌入来替代。我们保留它是因为它可以通过**窗口注意力机制**以最少的参数实现 16 倍的 token 压缩，从而显著降低后续全局注意力模块的计算成本和激活内存。此外，它的参数数量（80M）与 LLM 中用于文本输入嵌入的典型 100M 参数相当。

### 3.2.2 Language model as vision encoder

在 DeepEncoder 中，视觉 tokenizer 之后会引入一个 CLIP ViT 组件来压缩视觉知识。DeepEncoder V2 将此组件重新设计为 LLM 风格的架构，并采用双流注意力机制。视觉 token 利用双向注意力来保持 CLIP 的全局建模能力，而新引入的因果流 query 则采用因果注意力。这些可学习的 query 作为后缀附加在视觉 token 之后，每个 query 都会关注所有视觉 token 及其前面的 query。通过保持 query 和视觉 token 的基数相等，这种设计在不改变 token 数量的情况下，**对视觉特征施加了语义排序和提炼**。最后，只有因果 query 的输出会被送入 LLM 解码器。

我们使用 Qwen2-0.5B 实现此架构，其 500M 参数与 CLIP ViT (300M) 相当，且不会引入过多的计算开销。**仅解码器架构结合视觉 token 的前缀连接被证明至关重要**：在类似 mBART 的编码器-解码器结构中使用交叉注意力机制的额外实验未能收敛。我们推测，这种失败源于视觉 token 在单独的编码器中交互不足。相比之下，前缀连接设计使视觉 token 在所有层中保持活跃，从而促进了与因果 query 的有效视觉信息交换。

这种架构实际上建立了一个两阶段级联因果推理：编码器通过可学习的 qeury 对视觉 token 进行语义重排序，而 LLM 解码器则对排序后的序列执行自回归推理。与通过位置编码强制执行严格空间顺序的传统编码器不同，我们基于因果顺序的 query 能够适应平滑的视觉语义，同时自然地与LLM的单向注意力模式保持一致。这种设计有望弥合二维空间结构和一维因果语言建模之间的鸿沟。

### 3.2.3 Causal flow query

如前所述，因果查询 token 的数量等于视觉 token 的数量，计算公式为 $\frac{W\times H}{16^2\times 16}$，其中 $W$ 和 $H$ 分别表示编码器输入图像的宽度和高度。为了避免为不同分辨率维护多个 qeury 集，我们采用多裁剪策略，在预定义的分辨率下使用固定的 query 配置。

具体来说，全局视图采用 1024 × 1024 的分辨率，对应于 256 个 query 嵌入，记为 $query_{global}$。局部裁剪采用 768 × 768 的分辨率，裁剪数量 $k$ 的范围为 0 到 6（当图像的两个尺寸均小于 768 时，不进行裁剪）。所有局部视图共享一组统一的 144 个 query 嵌入，记为 $query_{local}$。因此，输入到 LLM 的重排序视觉 token 总数为 $k × 144 + 256$，范围为 $[256, 1120]$。此最大 token 数量 (1120) 低于 DeepSeek-OCR 的 1156（高达模式），并且与 Gemini-3-Pro 的最大视觉 token 预算相匹配。

### 3.2.4 Attention mask

<img
  src="https://i-blog.csdnimg.cn/direct/8806fa7031d048e89f3654b3576be23e.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

为了更好地阐释 DeepEncoder V2 的注意力机制，我们在图 5 中可视化了注意力 mask。该注意力 mask 由两个不同的区域组成。左侧区域对原始视觉 token 应用双向注意力（类似于 ViT），从而实现 token 间的完全可见性。右侧区域对因果流 token 采用因果注意力（三角形掩码，与纯解码器的 LLM 相同），其中每个 token 仅关注其前面的 token。这两个组件沿序列维度连接起来，构建了 DeepEncoder V2 的注意力掩码 (M)，如下所示：

```math
M =
\begin{bmatrix}
\mathbf{1}_{m \times m} & \mathbf{0}_{m \times n} \\
\mathbf{1}_{n \times m} & \text{LowerTri}(n)
\end{bmatrix},
\quad \text{where } n = m
\tag{1}
```

其中 $n$ 是因果 query token 的数量，$m$ 表示普通视觉 token 的数量，LowerTri 表示下三角矩阵（对角线及其下方为 1，上方为 0）。

## 3.3 DeepSeek-MoE Decoder

由于 DeepSeek-OCR 2 主要侧重于编码器的改进，因此我们没有升级解码器组件。遵循这一设计原则，我们保留了 DeepSeek-OCR 的解码器——一个具有约 500M 有效参数的 3B 参数 MoE 结构。DeepSeek-OCR 2 的核心前向传播过程可以表述如下：

```math
\mathbf{O}=\mathcal D(\pi_{Q}(\mathcal T^L(\mathcal E(\mathbf{I})⊕\mathbf Q_0;\mathbf M)))\tag{2}
```

其中 $\mathbf I ∈ \mathbb R^{H×W×3}$ 是输入图像，$\mathcal E$ 是视觉 tokenizer，将图像映射到 $m$ 个视觉 token $\mathbf V ∈ \mathbb R^{m×d}$，$\mathbf Q_0 ∈ \mathbb R^{n×d}$ 是可学习的因果 query 嵌入，$⊕$ 表示序列拼接，$\mathcal T^L$ 表示带有掩码注意力的 $L$ 层 Transformer，$\mathbf M ∈ \{0, 1\}^{2n×2n}$ 是公式 1 中定义的块因果注意力 mask，$\pi_Q$ 是投影算子，用于提取最后 $n$ 个 token（即 $\mathbb Z = \mathbb X_{m+1:m+n}$），$\mathcal D$ 是语言解码器，$\mathbf O ∈ \mathbb R^{n× |\mathcal V|}$ 是 LLM 词表的输出 logits。

# 4.Experimental Settings

<img
  src="https://i-blog.csdnimg.cn/direct/2c74da4bfb6d422f855cf4ad972d1a82.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>
