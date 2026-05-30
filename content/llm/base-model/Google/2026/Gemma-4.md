# Gemma 4：面向端侧、长上下文与智能体的开放多模态模型家族

## 摘要

Google DeepMind 的 Gemma 4 系列多模态开放权重模型已经在 Hugging Face、Kaggle 等平台发布，并开始被 `transformers`、`llama.cpp`、MLX、WebGPU、Rust 推理框架以及多种微调工具支持。相比前代 Gemma，Gemma 4 的定位更加清晰：它既要覆盖移动端、浏览器和边缘设备，也要覆盖服务器端的长上下文推理、多模态理解和智能体工作流。Google 官方文档将 Gemma 4 分为小尺寸模型、31B Dense 模型和 26B A4B MoE 模型三类，并强调其开放权重、商业可用、多模态、函数调用和系统提示词支持等能力。([Google AI for Developers][1])

Gemma 4 的核心价值不只是“模型更大”或者“分数更高”，而是通过一组相互配合的架构设计，把长上下文、端侧部署、低成本推理和多模态能力整合到同一个模型家族里。它采用交替局部/全局注意力、双 RoPE 配置、Per-Layer Embeddings、共享 KV Cache、可变视觉 token 预算，以及面向小模型的音频编码器。26B A4B 版本还引入了 MoE 稀疏专家结构，在总参数量较大的同时，每次前向只激活部分专家，从而兼顾质量和推理成本。([Hugging Face][2])

在 Hugging Face 的预发布测试中，Gemma 4 展现出较强的开箱即用能力：OCR、语音转文本、目标检测、GUI 元素定位、图像理解、视频理解、函数调用、代码生成和纠错都可以直接完成。对于开发者来说，它不是单一的聊天模型，而是一套可以用于端侧助手、浏览器智能体、多模态工具调用、企业私有部署和低成本微调的模型基础设施。([Hugging Face][2])

## Gemma 4 新在哪里？

与 Gemma-3n 类似，Gemma 4 支持文本、图像、音频等输入，并生成文本响应。更准确地说，Gemma 4 全系列支持文本、图像和视频输入；E2B 与 E4B 两个小尺寸版本原生支持音频输入，大尺寸模型则主要面向文本、图像和无音频视频理解。Google 文档还指出，小模型支持 128K 上下文窗口，31B 和 26B A4B 等更大模型支持 256K 上下文窗口。([Google AI for Developers][1])

文本解码器延续了 Gemma 系列的 Transformer 路线，但在长上下文和推理效率上做了更激进的取舍。Gemma 4 不再让所有层都做全上下文注意力，而是交替使用局部滑动窗口注意力和全局全上下文注意力。局部层负责高频、近距离的信息建模，全局层周期性地把长距离依赖重新注入上下文。这样可以降低长序列推理时的注意力开销，同时保持跨文档、跨段落和多轮对话中的全局信息流动。([Hugging Face][2])

图像编码器相比 Gemma 3 有两项关键改进：第一，它保留原始图像宽高比，而不是强行拉伸到固定尺寸；第二，它允许用户配置图像 token 预算，在速度、显存和视觉细节之间做取舍。官方博客中提到，Gemma 4 可以把图像编码为 70、140、280、560、1120 等不同 token 数量，从而适配从浏览器端快速问答到服务器端高精度视觉理解的不同场景。([Hugging Face][2])

Gemma 4 共提供四种尺寸，并且每种尺寸都有 base 和 instruction-tuned 版本：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7231d6c789054e6faab89b6a4ec48f05.png)

### 不同尺寸模型对比

下表把 Gemma 4 四个尺寸的定位、参数、上下文窗口、模态能力和关键架构差异放在一起。参数与上下文窗口整理自 Hugging Face 和 Google 文档；层数、滑动窗口大小、注意力模式、MoE 专家数等架构信息参考 Google DeepMind 官方代码；显存为 Google 给出的近似推理加载内存，不代表训练显存。([Hugging Face][2])

| 模型                  |                             架构与参数 | 上下文窗口 | 输入模态        | 关键架构配置                                                                           | 近似推理显存 BF16 / 8-bit / 4-bit |
| ------------------- | --------------------------------: | ----: | ----------- | -------------------------------------------------------------------------------- | --------------------------: |
| **Gemma 4 E2B**     | 2.3B effective，含 embedding 约 5.1B |  128K | 文本、图像、视频、音频 | 35 层；4 个局部滑窗层 + 1 个全局层循环；512 token 滑窗；PLE；共享 KV Cache；端侧/浏览器优先                   |    9.6 GB / 4.6 GB / 3.2 GB |
| **Gemma 4 E4B**     |   4.5B effective，含 embedding 约 8B |  128K | 文本、图像、视频、音频 | 42 层；5 个局部滑窗层 + 1 个全局层循环；512 token 滑窗；PLE；共享 KV Cache；质量与端侧成本折中                  |       15 GB / 7.5 GB / 5 GB |
| **Gemma 4 31B**     |                         31B Dense |  256K | 文本、图像、视频    | 60 层；5 个局部滑窗层 + 1 个全局层循环；1024 token 滑窗；Dense 架构；更强通用推理和编码能力                      | 58.3 GB / 30.4 GB / 17.4 GB |
| **Gemma 4 26B A4B** |    26B MoE，总参数 26B，每 token 激活约 4B |  256K | 文本、图像、视频    | 30 层；5 个局部滑窗层 + 1 个全局层循环；1024 token 滑窗；128 experts/layer，top-k experts=8；高吞吐推理优先 |     48 GB / 25 GB / 15.6 GB |

## 架构与能力总览

Gemma 4 的架构目标可以概括为三个关键词：**长上下文、端侧友好、多模态统一**。为了实现这些目标，它并没有简单堆叠参数，而是围绕推理效率做了多处结构调整。

Gemma 4 的主要架构特点包括：

* **交替使用局部滑动窗口和全局全上下文注意力层**：小型密集模型使用 512 token 滑动窗口，大模型使用 1024 token 滑动窗口。
* **双 RoPE 配置**：局部滑动窗口层使用标准 RoPE，全局层使用面向长上下文的裁剪/比例化 RoPE 配置。
* **Per-Layer Embeddings（PLE）**：增加第二个嵌入表，为每个解码器层注入轻量残差信号。
* **共享 KV Cache**：模型后部若干层复用前面同类型注意力层产生的 key-value 状态，减少重复 KV 投影。
* **视觉编码器**：使用学习到的 2D 位置编码和多维 RoPE，保留原始宽高比，并支持多种视觉 token 预算。
* **音频编码器**：采用 USM 风格的 Conformer 编码器，基础架构与 Gemma-3n 中的音频编码器一致。
* **MoE 稀疏专家结构**：26B A4B 版本使用 128 个专家，每层通过路由器选择部分专家参与计算。([Hugging Face][2])

### 交替注意力：局部滑窗 + 全局上下文

长上下文模型最大的瓶颈之一是注意力复杂度。标准全注意力需要让每个 token 看到前面所有 token，在 128K 或 256K 上下文下，计算量和 KV Cache 内存都会快速膨胀。Gemma 4 的做法是把注意力层分成两类：局部滑动窗口层和全局层。

在局部层中，每个 token 只关注附近一段窗口。例如 E2B 和 E4B 使用 512 token 的滑动窗口，31B 和 26B A4B 使用 1024 token 的滑动窗口。这样可以让大部分层以更低成本处理局部语义、短程依赖、语法结构和多轮对话中的近邻上下文。([Hugging Face][2])

全局层则周期性地出现，用来恢复跨长距离的信息传递。官方代码中，E2B 是“4 个局部层 + 1 个全局层”的循环模式；E4B、31B 和 26B A4B 则是“5 个局部层 + 1 个全局层”的循环模式。这种设计使模型不必在每一层都付出全上下文注意力的成本，又不会完全失去长距离信息整合能力。([GitHub][3])

### 双 RoPE：短程精细建模与长程扩展并存

Gemma 4 使用双 RoPE 配置来服务两种注意力层。局部滑动窗口层使用标准 RoPE，适合处理窗口内的相对位置信息；全局层则使用不同的 RoPE 配置，以支持更长的上下文范围。官方代码中可以看到，局部层和全局层分别使用不同的 base frequency：局部为 `10_000`，全局为 `1_000_000`，并且设置了 `local_rope_proportion=1.0` 和 `global_rope_proportion=0.25`。这相当于让局部层保留更完整的短程位置分辨率，而全局层把位置建模能力更多用于长距离泛化。([GitHub][3])

这种双 RoPE 的意义在于：模型不必用同一套位置编码同时兼顾短上下文精度和超长上下文外推。局部层更像“细节处理器”，负责局部 token 的精细排列；全局层更像“全局路由器”，负责把远距离信息压缩进主残差流。对于代码、长文档、多轮工具调用和长链路推理任务，这种分工会比单一全局位置编码更稳定。

### Per-Layer Embeddings（PLE）

Gemma 4 小型模型中最有辨识度的设计之一是 **Per-Layer Embeddings**，即 PLE。传统 Transformer 通常只在输入阶段为 token 查一次 embedding，后续所有层都沿着同一个残差流不断变换。这意味着初始 embedding 需要“预先携带”大量潜在信息，后面的层只能从同一个初始表示中提取不同特征。

PLE 的思路是：除了主 embedding 之外，再为每一层提供一个小型的、逐层专用的输入信号。Hugging Face 文档描述，PLE 会把 token identity 分量和 context-aware 分量结合起来，生成每一层自己的轻量残差信号；Transformers 文档进一步说明，这两个分量会相加并按 `1/√2` 缩放后送入对应解码器层。([Hugging Face][2])

从实现上看，PLE 大致包含两条路径：

1. **Token-identity 路径**：根据 `input_ids` 查找第二个 embedding 表，得到和 token 身份相关的逐层向量。
2. **Context-aware 路径**：把主 embedding 通过一个线性投影映射到逐层 PLE 空间，再经过归一化得到上下文感知向量。

每个解码器层只消费属于自己的那一小段 PLE 信号，因此它可以获得更强的逐层特化能力，而不需要显著增加主隐藏层宽度。对于 E2B 和 E4B 这类小模型来说，PLE 的意义尤其大：它让模型的“有效参数”更高，同时仍然适合端侧和浏览器部署。

多模态输入下，PLE 还需要处理一个特殊问题：图像、音频、视频 token 通常不是普通文本 token ID，而是被编码器产生的 soft token 替换进文本序列。由于 PLE 的 token-identity 路径依赖 `input_ids`，多模态位置会采用更中性的逐层信号，避免对视觉或音频 token 施加错误的文本 token 身份。Hugging Face 博客也指出，多模态输入中的 PLE 会在 soft token 合并进 embedding 序列之前计算。([Hugging Face][2])

### Shared KV Cache：减少长上下文推理中的重复开销

KV Cache 是自回归推理中最占显存的部分之一。序列越长，模型需要缓存的 key/value 张量越多；层数越多，缓存成本越高。Gemma 4 在部分模型中引入 **Shared KV Cache**：模型最后若干层不再重新计算自己的 key/value，而是复用前面某个同类型注意力层已经计算好的 KV 张量。

具体来说，共享策略会区分局部滑动窗口层和全局层。对于后部共享层，如果它是全局注意力层，就复用前面某个全局层的 KV；如果它是局部滑窗层，就复用前面某个局部层的 KV。Google DeepMind 代码中的 `KVCacheSharingConfig` 也明确包含 `share_global` 和 `share_local` 两个配置项，并通过 `create_kv_cache_sharing_patterns` 为不同层生成对应的 KV 复用索引。([GitHub][4])

这个机制的收益主要体现在长上下文和端侧推理中：模型可以减少重复的 KV 投影和缓存占用，从而降低显存压力，提高吞吐。代价是部分层不再拥有完全独立的 KV 表达，但官方博客认为这种质量影响很小，而内存和计算效率收益明显。([Hugging Face][2])

### 视觉编码器：可变宽高比与可配置 token 预算

Gemma 4 的视觉编码器面向真实应用场景做了不少实用优化。首先，它保留图像原始宽高比，而不是统一缩放到固定正方形尺寸。这对截图、网页、表格、票据、代码界面和移动端 UI 尤其重要，因为这些场景里，布局关系往往比单个对象本身更关键。

其次，Gemma 4 支持多档图像 token 预算。官方博客给出的 token 预算包括 70、140、280、560、1120。较低 token 预算适合快速预览、移动端问答和批量推理；较高 token 预算则适合 OCR、细粒度目标检测、复杂图表理解和 GUI 元素定位。([Hugging Face][2])

位置编码方面，Gemma 4 的视觉编码器使用学习到的 2D 位置表，并结合多维 RoPE。Transformers 文档提到，Gemma 4 的视觉位置表每个轴最多支持 10,240 个位置；2D RoPE 会把注意力头的一部分维度用于 x 轴旋转，另一部分用于 y 轴旋转，从而帮助模型理解“上方、下方、左侧、右侧”等空间关系。([GitHub][5])

这解释了为什么 Gemma 4 在 GUI detection、网页复现、图像问答和目标定位任务中表现比较自然：它不是简单把图像压成一个全局向量，而是保留了更细致的空间结构。

### 音频编码器：USM 风格 Conformer

Gemma 4 的 E2B 和 E4B 原生支持音频输入，适合语音转文本、音频问答、视频带音频理解等任务。Hugging Face 博客将其描述为 USM 风格的 Conformer 音频编码器，基础架构与 Gemma-3n 中的音频编码器相同。([Hugging Face][2])

从官方代码看，Gemma 4 的音频模块包含一个 `AudioTokenizer`，注释中明确称其为 Conformer-based audio encoder。音频输入通常会先被转换为声学特征，再经过降采样和 Conformer 层编码，最后投影到语言模型的 embedding 空间，与文本 token、图像 token 一起进入统一的 Transformer 解码器。([GitHub][6])

这类结构的好处是，音频不是作为外部工具结果被拼接进提示词，而是作为一等输入模态进入模型。对于端侧语音助手、会议记录、短视频理解、语音问答等应用，E2B 和 E4B 的原生音频支持会比“先 ASR 再 LLM”的流水线更紧凑。

### 26B A4B：MoE 稀疏专家模型

Gemma 4 26B A4B 是系列中最值得关注的效率型大模型。它的总参数量约为 26B，但每个 token 前向时只激活约 4B 参数，因此名字中的 A4B 可以理解为 “Activated 4B”。Hugging Face 博客提到，26B A4B 在仅激活约 4B 参数的情况下，取得了接近 31B Dense 的竞技场表现。([Hugging Face][2])

官方代码显示，Gemma 4 26B A4B 每层包含 128 个 experts，`top_k_experts=8`，并带有一个 dense shared MLP 分支。也就是说，每个 token 会通过路由器选择少数专家参与计算，同时仍保留共享 MLP 分支来提供稳定的通用表示。([GitHub][3])

MoE 的优势在高吞吐推理场景中尤其明显：总参数量提供容量，稀疏激活控制单次计算成本。对于需要服务大量请求的多模态问答、长上下文检索增强生成、智能体任务规划等场景，26B A4B 往往会比同规模 dense 模型更经济。

## 多模态能力

Gemma 4 的多模态能力并不局限于“看图说话”。Hugging Face 的测试覆盖了 OCR、语音转文本、目标检测、pointing、函数调用、代码补全和纠错等任务，并且模型可以在不少任务中直接输出结构化 JSON，而无需额外语法约束。([Hugging Face][2])

### Object Detection and Pointing

#### GUI Detection

我们使用以下图像和文本提示测试 Gemma 4 在不同尺寸下检测和指向 GUI 元素的能力：

> “What's the bounding box for the `view recipe` element in the image?”

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9d51e431cf5f413d9d8b995c5056ceba.png)

在这个提示下，模型可以直接以 JSON 格式返回检测到的边界框，而不需要额外规定输出语法。Hugging Face 测试中发现，模型返回的坐标对应 1000×1000 的归一化图像坐标系，再映射回原图即可可视化边界框。([Hugging Face][2])

示例输出：

```json
[
  {
    "box_2d": [171, 75, 245, 308],
    "label": "view recipe element"
  }
]
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8258ee2e85a24608a5474755029e350c.png)

这个能力对 GUI agent 很关键。模型不只是回答“按钮在哪里”，而是能把自然语言目标转成可执行的屏幕坐标。结合浏览器自动化、桌面自动化或移动端操作框架，就可以构建“看屏幕—定位控件—执行点击”的闭环。

#### Object Detection

在普通目标检测任务中，Gemma 4 可以根据自然语言指令定位图像里的对象，并输出边界框。例如用户可以要求模型“找到图片中的自行车”“框出所有路牌”“标出页面里的提交按钮”。这种输出方式比传统分类式 VLM 更适合智能体系统，因为边界框和标签可以直接传给下游工具执行裁剪、点击、OCR 或二次识别。

Gemma 4 的优势来自两点：第一，视觉编码器保留宽高比和二维空间关系；第二，语言解码器已经具备结构化输出和工具调用倾向。因此，它可以把视觉理解结果直接组织成 JSON、坐标、列表或函数参数，而不仅是自然语言描述。

### Video Understanding

Gemma 4 支持视频输入。小模型 E2B 和 E4B 可以处理带音频的视频，大模型则主要处理无音频视频帧。虽然 Hugging Face 博客提到这些模型并非显式针对视频做后训练，但测试中它们仍然可以理解视频画面内容、场景变化和音频语义，尤其适合短视频摘要、视频问答和多模态检索。([Hugging Face][2])

从架构角度看，视频理解可以被视为多帧图像 token 加上可选音频 token 的联合建模。对于带音频的视频，E2B/E4B 可以同时利用视觉帧和音频编码器；对于大模型，则可以用更高的文本/视觉推理能力理解关键帧序列。

### Captioning

图像描述是 Gemma 4 最基础但也最实用的能力之一。得益于可变 token 预算，开发者可以根据场景选择不同精度：低 token 预算用于快速生成概览，高 token 预算用于保留更多细节，例如图表中的文字、UI 层级、页面布局或物体之间的空间关系。

在实际应用中，Captioning 不只是“描述图片”。它可以作为 OCR、检索增强、多模态数据清洗、视频切片标注和自动测试报告生成的前置步骤。Gemma 4 的长上下文能力还允许把多张图片、多段说明和历史对话放在一起，让模型生成更连贯的上下文描述。

### Audio Question Answering

E2B 和 E4B 的音频输入能力让 Gemma 4 可以直接处理语音或带音频视频。例如，用户可以上传一段会议录音并询问“谁提出了预算问题？”“这段音频里提到的截止时间是什么？”“请把这段语音总结成行动项”。

相较于“ASR 模型转文本 + LLM 总结”的两段式流程，Gemma 4 的原生音频输入更适合轻量端侧应用。当然，在高精度转写场景中，专用 ASR 仍然可能更稳定；但在问答、摘要、短视频理解和多模态助手里，原生音频能力可以显著简化工程链路。

### Multimodal Function Calling

Gemma 4 支持文本和多模态函数调用。Google 文档也强调 Gemma 4 在编码、智能体能力和内置函数调用方面有所增强，并新增了系统角色支持。([Google AI for Developers][1])

多模态函数调用的典型流程是：模型先理解图像、视频或音频内容，再把结果转成结构化参数调用工具。例如：

* 识别截图中的按钮坐标，然后调用浏览器自动化工具点击；
* 读取票据图片中的金额和日期，然后调用财务系统录入；
* 分析视频中的异常动作，然后调用告警接口；
* 解析语音指令，然后调用日程、邮件或搜索工具。

这种能力让 Gemma 4 更适合作为 agent backbone，而不是仅仅作为聊天机器人。

## Benchmark Results

Gemma 4 在推理、知识、编码、视觉和长上下文任务中都展现了较强的综合能力。Hugging Face 博客提到，31B Dense 模型在仅文本 LMArena 估计分上达到 1452，26B A4B MoE 在每 token 仅激活约 4B 参数的情况下达到 1441。([Hugging Face][2])

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d1d6bad12754fe99f3f42f02fb721f4.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e535a146abda412fbc93fcf9e7a04891.png)

从模型规模和性能关系来看，Gemma 4 的策略不是单点冲榜，而是构造一条覆盖端侧到服务器侧的 Pareto 前沿。E2B 和 E4B 面向低显存、低功耗、浏览器和移动端；31B Dense 面向高质量通用推理；26B A4B MoE 面向高吞吐、低激活参数成本的大模型服务。

## 部署与微调生态

Gemma 4 的另一个重要特点是生态覆盖广。Hugging Face 博客中提到，Gemma 4 已经支持 `transformers`、`llama.cpp`、MLX、WebGPU、Rust 等推理路径，并且提供 TRL、Vertex AI、Unsloth Studio 等微调方案。([Hugging Face][2])

对于本地推理，开发者可以根据硬件选择不同格式：GPU 环境下使用 `transformers` 或 vLLM 类框架；Mac 设备上使用 MLX；CPU 或边缘设备上使用 GGUF/llama.cpp；浏览器中则可以通过 WebGPU 跑小尺寸模型。对于企业场景，E4B 和 26B A4B 是两个很有吸引力的选择：前者便于私有化和端侧部署，后者适合服务器端高吞吐服务。

Gemma 4 还提供 Multi-Token Prediction drafter，用于 speculative decoding。drafter 会一次预测多个未来 token，目标模型再一次性验证，从而在不改变输出质量和推理行为的前提下提升生成速度。Hugging Face 博客提到，相关 drafter 覆盖 E2B、E4B、26B A4B 和 31B 四个尺寸，端到端加速在不同硬件和任务下最高可达约 3 倍。([Hugging Face][2])

## 总结

Gemma 4 的发布说明开放模型正在进入一个新的阶段：模型不再只是“更大的文本聊天模型”，而是变成了跨设备、跨模态、跨推理框架的基础能力层。

从架构上看，Gemma 4 的优势来自一组细粒度工程选择：交替局部/全局注意力降低长上下文成本，双 RoPE 改善长程位置建模，PLE 提升小模型参数效率，共享 KV Cache 降低推理内存，视觉编码器增强空间理解，音频编码器扩展输入模态，26B A4B MoE 则用稀疏激活平衡质量和成本。

对于开发者来说，Gemma 4 最值得尝试的场景包括：端侧多模态助手、浏览器/GUI agent、长文档问答、音频视频摘要、多模态函数调用、企业私有知识库、低成本 LoRA/QLoRA 微调，以及需要高吞吐的推理服务。E2B 和 E4B 适合轻量部署，31B 适合追求最高 dense 质量的场景，26B A4B 则适合在服务器端追求性能与成本平衡的应用。

[1]: https://ai.google.dev/gemma/docs/core "Gemma 4 model overview  |  Google AI for Developers"
[2]: https://huggingface.co/blog/gemma4 "Welcome Gemma 4: Frontier multimodal intelligence on device"
[3]: https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_gemma4.py "gemma/gemma/gm/nn/gemma4/_gemma4.py at main · google-deepmind/gemma · GitHub"
[4]: https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py "gemma/gemma/gm/nn/gemma4/_config.py at main · google-deepmind/gemma · GitHub"
[5]: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gemma4.md "transformers/docs/source/en/model_doc/gemma4.md at main · huggingface/transformers · GitHub"
[6]: https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/audio/_model.py "gemma/gemma/gm/nn/gemma4/audio/_model.py at main · google-deepmind/gemma · GitHub"
