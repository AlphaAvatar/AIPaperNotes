论文链接：https://unipat.ai/blog/SWE-Vision#swe-vision-system-design

代码链接：https://github.com/UniPat-AI/SWE-Vision

# 摘要

<img
  src="https://i-blog.csdnimg.cn/direct/dd963cff49ab41a08acb9f7d4cbb0dec.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

视觉理解和编码是前沿多模态大语言模型的两大核心能力——然而，它们与人类能力相比，表现却截然不同。在编码方面，模型已经远远超越了专家级水平，能够生成、调试和优化复杂的项目。但在视觉方面，差距依然巨大：正如我们之前的项目 BabyVision 所展示的那样，模型仍然难以完成人类能够轻松解决的任务。这种不对称性引出了一个自然的问题：编码能否用于提升视觉能力？

许多后续研究都遵循了 OpenAI 提出的 “thinking with images” 范式，并整合工具链来增强视觉理解能力。然而，在实践中，这些努力收效甚微。这些系统通常依赖于人工设计的工具，并且针对特定基准（例如高分辨率图像理解）进行了狭义的优化，从而导致两个主要局限性：（1）人工设计的工具对于基础模型而言并不熟悉，这阻碍了有效的学习和探索，尤其是在强化学习（RL）中；（2）评估范围有限，难以评估真正的泛化能力。因此，目前社区仍然缺乏一个简洁的开源框架，能够使模型以广泛可泛化的方式提升视觉能力。

<img
  src="https://i-blog.csdnimg.cn/direct/96e3adedbce94d6e8ca984b2bc389332.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

这就是 SWE-Vision 的核心理念——一个智能体循环，**它为视觉语言模型配备了一个简单且有状态的环境，使其能够编写和执行 Python 代码来推理视觉输入**。SWE-Vision 并非仅仅依赖内部视觉表征或临时工具，而是允许模型通过编程来找到答案：使用 PIL 加载图像，使用 NumPy 进行像素级分析，使用 matplotlib 和其他库生成可视化图表，并将这些计算工具与基础模型的原生视觉理解能力和多轮编码能力无缝集成。

基准测试结果也令人瞩目。在涵盖基础感知、图表推理、数学问题解决、空间理解和复杂多步骤视觉挑战的五个不同的视觉基准测试中，SWE-Vision 始终优于 GPT-5.2-xhigh 和 Seed-2.0-Pro 等前沿 LLM 模型，并取得了最先进的成果：BabyVision 得分 64.4，MathVision 得分 94.0，Zero-Bench-Sub 得分 50.1，OmniSpatial 得分 69.0，CharXiv-RQ 得分 82.5。

实验结果表明，引入通用编码工具是提升前沿 LLM 在视觉任务上性能的**有效测试时扩展**方向。然而，我们也发现，相对性能提升与不同的基础模型有关，这与基础模型的函数调用、编码能力和长上下文处理能力密切相关。为了充分发挥 SWE-Vision 在智能视觉理解方面的潜力，我们需要使用深度交错的视觉编码 SFT 和 RL 数据进行更全面的训练——使模型不仅能够使用工具，还能原生地整合感知和程序推理。

# Cases: Visual Test Time Scaling with Coding

【请参考原博客文章】

# SWE-Vision: System Design

> Why Give LLMs Code Execution?

关键在于，**许多视觉推理错误并非源于视觉感知，而是源于信息处理**。例如，当模型查看图表并估计“大约 75%”时，它可能正确感知了柱状图的高度，但却未能计算出精确的比例。又如，当模型统计杂乱场景中的物体数量时，它可能看到了每个物体，但却无法准确统计总数。而这些错误，只需几行 Python 代码就能轻松消除。

SWE-Vision 的设计理念很简单：让模型自行决定何时以及如何使用代码。模型并非必须为每个问题编写代码——它保留在有把握的情况下直接回答问题的选项。但当任务需要精确计算、像素级分析或迭代探索时，模型可以调用其 Jupyter Notebook，系统地解决问题。

## Architecture Overview

SWE-Vision 采用 Agent 循环架构，包含两个核心工具：`execute_code` 和 `finish`。

<img
  src="https://i-blog.csdnimg.cn/direct/89682dc56068456f9da3d5906713180c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

工作流程如下：
1. **User Input**。用户提供查询语句以及一张或多张图片。这些图片会 (a) 作为对话中的视觉内容传递给 LLM，以及 (b) 复制到 Jupyter 内核可访问的共享文件系统中。
2. **Reasoning & Tool Selection**。LLM 接收查询和图像，对任务进行推理，并决定是直接回答还是调用 `execute_code` 工具来运行 Python 代码。
3. **Code Execution**。当模型调用 `execute_code` 函数时，提供的 Python 代码会在运行于 Docker 容器内的持久化 Jupyter 内核中执行。该内核会在多次调用之间保留所有状态——变量、导入、加载的数据——从而支持多步骤分析。
4. **Result Feedback**。执行结果（包括文本输出（stdout）、错误信息以及任何生成的图像（例如 matplotlib 图表））都会作为持续对话的一部分反馈给 LLM。生成的图像会被捕获并作为视觉内容返回，以便模型可以检查自身的图表和中间可视化结果。
5. **Iteration**。该模型检查结果，决定是否需要更多计算，然后要么再次调用 `execute_code`，要么使用最终答案调用 `finish` 工具。

## Key Design Decisions

**Stateful Jupyter Kernel**。与无状态代码执行不同，持久化的 Jupyter 内核允许智能体在多次工具调用中构建上下文。模型可以在一次调用中加载图像，检查其属性，在下一次调用中应用转换，并在第三次调用中计算最终结果——所有这些都无需重新加载数据或重新导入库。这与人类数据科学家在 Jupyter Notebook 中的工作方式非常相似。

**Docker Sandboxing**。所有代码都在一个隔离的 Docker 容器内运行，该容器具有受控的软件包环境（包括 NumPy、Pandas、OpenCV、PIL、scikit-image、matplotlib 等）。共享卷挂载点 (`/mnt/data/`) 实现了主机和内核之间的文件交换。这既保证了安全性（任意代码执行都在沙盒环境中进行），又保证了可复现性（每次运行环境保持一致）。

**Image-In / Image-Out**。SWE-Vision 支持双向图像流。用户图像被传递给 LLM 并存储在内核的文件系统中。内核生成的图像（例如，带注释的图像、图表、热图）被捕获并作为视觉内容返回给 LLM。这形成了一个强大的反馈循环：模型可以生成可视化结果来验证自身的推理，检查边缘检测结果，或在原始图像上叠加注释。

**OpenAI Function Calling Interface**。SWE-Vision 使用标准的 OpenAI 函数调用（工具使用）API，因此与任何 OpenAI 兼容的接口都兼容。这两个工具——`execute_code` 和 `finish`——被定义为结构化函数模式。这意味着 SWE-Vision 不仅可以与 OpenAI 模型直接配合使用，还可以通过 OpenAI 兼容的 API 与其他模型提供商合作。

## System Prompt Design

系统提示指示模型作为专家助手运行，并具备笔记本电脑访问权限。关键要素包括：
- 明确记录文件系统布局（`/mnt/data/`）
- 指导使用 `print()` 进行文本输出，使用 `plt.show()` 进行绘图
- 鼓励逐步操作并检查中间结果
- 允许多次调用 `execute_code` 进行迭代探索
- 指示仅在确信答案正确时才调用 `finish`。

这种提示设计刻意保持简洁——它告诉模型有哪些工具可用以及如何使用它们，但并不规定何时使用代码或应用哪些算法。模型自身的推理能力决定了其解决问题的策略。

# Experiments

## Benchmarks

我们使用五项不同的视觉基准测试来评估 SWE-Vision，这些基准测试旨在涵盖广泛的视觉推理能力：

<img
  src="https://i-blog.csdnimg.cn/direct/b22eb81a656a4b36bfa4c5aa1fb1dd34.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## Setup

- **Model under test**: Gemini-3-Pro (high), GPT-5.2 (xhigh), Seed-2.0-Pro (high).
- **SWE-Vision configuration**: GPT-5.2/Seed-2.0-Pro with max reasoning effort, max 100 agent iterations per task.
- **Judge model**: GPT-5 as LLM-as-judge for answer correctness evaluation.

## Results

<img
  src="https://i-blog.csdnimg.cn/direct/c9cd1b151ed44d5f9ccc310d04c3e610.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

<img
  src="https://i-blog.csdnimg.cn/direct/c94f1c43f3bc49b1ba18e87faafebba1.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## Analysis

**在所有测试的基准测试中，两种模型均取得了持续的提升**。SWE-Vision 在所有测试基准测试中都优于基础模型。使用 GPT-5.2 时，得分提升范围为 +2.3 至 +18.2 分；使用 Seed-2.0-Pro 时，得分提升范围为 +1.6 至 +3.8 分。这种在两个截然不同的模型系列中均保持一致的提升，证实了代码执行是视觉推理中一项广泛适用的能力，而非任何单一模型缺陷的体现。

**感知任务的提升最为显著**。在 BabyVision 测试中，GPT-5.2 的提升最为显著（+18.2 分，从 34.4 提升至 52.6），该测试旨在评估基本的视觉感知能力，例如计数、颜色识别和空间关系。这些任务正是基于代码的分析（例如像素计数、颜色直方图绘制、轮廓检测）能够最直接地弥补模型视觉局限性的领域。 Seed-2.0-Pro 在 BabyVision 上的得分也显著提升了 3.8 分（60.6 → 64.4），尽管增幅较小，但反映了其更强的基线感知能力。

**数学推理能力显著提升**。MathVision 在 GPT-5.2 上的提升幅度位居第二（+7.9，86.1 → 94.0），这反映了该模型能够从视觉图形中提取数据并以编程方式计算答案，而非进行估算。即使是得分已达 88.8 的 Seed-2.0-Pro，也提升至 90.7 分（+1.9）——这表明即使接近性能上限，代码执行仍然能够带来显著提升。

**空间和综合推理能力**。 OmniSpatial（GPT-5.2 提升 4.7 分，Seed-2.0-Pro 提升 1.6 分）和 ZeroBench-Sub（GPT-5.2 提升 3.9 分，Seed-2.0-Pro 提升 2.5 分）均取得了显著提升。空间推理能力得益于模型测量像素距离和计算几何关系的能力。ZeroBench-Sub 需要跨多个视觉技能进行多步骤推理，而笔记本的迭代特性使其受益匪浅——智能体可以将复杂问题分解为一系列顺序分析步骤。

**编码能力更强的模型获益更多**。双模型评估结果显示出一个显著的规律：GPT-5.2 拥有更强大的代码生成能力，在所有五个基准测试中，其从 SWE-Vision 获得的绝对提升始终大于 Seed-2.0-Pro。这一点在 BabyVision（提升 18.2 分 vs 提升 3.8 分）和 MathVision（提升 7.9 分 vs 提升 1.9 分）中尤为明显。结果符合直觉——SWE-Vision 的强大之处在于编写和执行代码，因此，能够生成更正确、更具针对性的程序的模型将从该框架中获得更多价值。这表明，投资于模型的编码能力会带来复利效应：模型不仅会成为更优秀的程序员，而且当与 SWE-Vision 等工具增强型流程结合使用时，它还会成为更优秀的视觉推理者。

# Discussions

以下几点局限性值得探讨：
- **对基础模型视觉推理和编码能力的依赖**。SWE-Vision 的有效性受限于 LLM 编写正确、相关代码并正确分析结果的能力。GPT-5.2 和 Seed-2.0-Pro 之间的性能差距可能部分反映了编码能力的差异，而不仅仅是视觉能力。如果模型编写的代码存在缺陷或选择了不合适的算法，该工具可能无法提供帮助，甚至可能降低性能。
- **故障模式**。我们观察到智能体进入无效循环的情况——反复尝试类似的失败方法，或为简单的任务生成过于复杂的代码。最大迭代次数限制提供了一定的安全保障，但这些情况仍然会消耗资源，却无法产生更好的结果。
- **代码引入的错误**。程序化分析并非总是优于纯粹的 LLM 视觉推理。在某些情况下，编写和执行代码来分析图像可能会引入新的错误——无论是由于错误的假设、不正确的实现，还是对中间结果的误解。因此，智能体不仅要执行代码，还要批判性地评估其输出，在必要时修正错误的逻辑，并动态地思考代码是否对特定问题有益。下面，我们将展示实验中一个有趣的失败案例，以说明这一挑战。

## Future Directions

SWE-Vision 在多个模型后端上的成功表明了几个有前景的研究方向：
- **更丰富的工具库**。除了通用的 Python notebook 之外，专用工具——例如 3D 渲染引擎、物理模拟器、OCR 系统、符号数学求解器，甚至 CLI 和 GUI 工具——可以进一步扩展模型的功能，尤其是在目前提升幅度较小的领域。
- **学习何时使用工具**。目前，模型根据其通用推理能力来决定何时执行代码。通过高质量的人类演示或强化学习进行微调可以优化这一决策，从而减少在基础模型已经很强大的任务上不必要的代码执行。
- **反思代码执行**。智能体不仅要执行代码，还要批判性地评估其输出，在必要时修正错误的逻辑，并动态地思考代码对于特定问题是否有益。
- **更多模态**。当前的图像输入/图像输出设计可以扩展到支持音频、视频和交互式可视化，从而实现更丰富的反馈循环。
- **模型自适应策略**。鉴于不同任务的收益各不相同，未来的研究可以探索自适应执行策略，将更多计算资源分配给基础模型可能难以处理的任务。
- **视觉 Agent 数据工程与训练**。与用于训练多模态 LLM 的传统数据不同，训练视觉 Agent 模型需要多模态交错的 Agent 轨迹。此外，还需要一个交互式环境来支持强化学习、工具使用和评估，使模型不仅能够回答问题，还能感知、行动和反思。
