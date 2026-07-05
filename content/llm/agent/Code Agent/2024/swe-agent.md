# SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

论文链接：https://arxiv.org/pdf/2405.15793

代码链接：https://swe-agent.com/

## 摘要

语言模型（LM）Agent 正日益广泛地应用于数字环境中复杂任务的自动化。正如人类受益于强大的软件应用（例如集成开发环境）来完成软件工程等复杂任务一样，我们认为语言模型 Agent 代表了一类拥有自身需求和能力的新型终端用户，并且能够从专门构建的软件接口中获益。**我们研究了接口设计如何影响语言模型 Agent 的性能**。基于这项研究，我们提出了 **SWE-agent**：一个能够帮助语言模型 Agent 自主使用计算机解决软件工程任务的系统。SWE-agent 的定制 **agent-computer interface**（ACI）显著增强了 Agent 创建和编辑代码文件、浏览整个代码库以及执行测试和其他程序的能力。我们在 SWE-bench 和 HumanEvalFix 上对 SWE-agent 进行了评估，在这两个平台上均取得了最先进的性能，pass@1 率分别为 12.5% 和 87.7%，远超以往非交互式语言模型所达到的最先进水平。最后，我们深入探讨了 ACI 的设计如何影响智能体的行为和性能。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/c78426783d944d8d8f70382cb7252730.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

近期研究表明，LM Agent 在代码生成和执行反馈方面具有显著优势。然而，将 Agent 应用于更复杂的代码任务（例如软件工程）仍有待探索。为了解决编程任务，LM Agent 通常被设计为使用现有应用程序，例如 Linux shell 或 Python 解释器。然而，对于更复杂的编程任务（例如软件工程），人类工程师可以受益于 VSCode 等功能强大的应用程序及其丰富的工具和扩展。受人机交互（HCI）领域关于用户接口对人类有效性的研究的启发，我们探讨 LM Agent 是否也能通过设计更完善的接口来提升软件工程任务的执行效率。

考虑一个简单场景：智能体直接与 Linux shell 交互。在实践中，我们发现 LM 智能体在这种环境下难以可靠地执行操作。例如，它无法提供编辑小文件片段的简单命令，并且在用户进行无效编辑时也不会提供任何反馈。这些缺陷严重影响了性能，因此需要**智能体-计算机接口 (ACI)**，即 LM 智能体和计算机之间的抽象层，以增强 LM 智能体在计算机环境中的能力（图 1）。

基于此项研究，我们提出了 **SWE-agent**，这是一个由 LM 和 ACI 组成的智能体，能够与计算机交互，解决具有挑战性的实际软件工程问题，例如 SWE-bench 中提出的问题。与 Linux Shell 细粒度、高度可配置的动作空间不同，SWE-agent 的 ACI 提供了一组用于查看、搜索和编辑文件的简单动作。ACI 使用安全机制来防止常见错误，并且智能体在每次执行命令时都能收到关于命令效果的具体、简洁的反馈。我们证明，专门为语言模型定制的 ACI 的性能优于现有的面向人类用户的用户接口 (UI)，例如 Linux Shell。

以 GPT-4 Turbo 为基础语言模型，SWE-agent 解决了 2294 个 SWE-bench 测试任务中的 12.47%，显著优于之前非交互式检索系统 3.8% 的最佳解决率。我们对 300 个 SWE-bench 测试实例子集（SWE-bench Lite）进行了消融研究，以分析我们的 ACI 设计选择。结果表明，SWE-agent 比仅使用默认 Linux shell 的基线 Agent 多解决了 10.7 个百分点的实例。尽管我们的 ACI 是为 GPT-4 Turbo 开发的，但我们证明它可以移植到不同的语言模型；使用 Claude 3 Opus 的 SWE-agent 可以解决 10.5% 的基准测试任务。

我们的贡献主要体现在两个方面：
1. 首先，我们引入了 Agent-Computer 接口（ACI）的概念，并展示了如何通过精心设计 ACI 来显著提升语言模型（LM）Agent 的性能，而无需修改底层语言模型的权重。
2. 其次，我们构建、评估并开源了 SWE-agent 系统，该系统为语言模型提供了一个 ACI 接口，用于解决实际的软件工程任务。

与以往分别探讨工具使用、提示技术和代码执行在交互式环境中优劣的工作不同，我们的方法将这些因素统一到 ACI 框架内。我们证明，构建以语言模型为中心的交互式组件能够显著提升下游任务的性能。

## 2.The Agent-Computer Interface

<img
  src="https://i-blog.csdnimg.cn/direct/fce55ec199a44ae7a5a6aa9baefc7ec4.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**当预言模型（LM）通过迭代执行动作并接收反馈与环境交互时，它就扮演着 Agent 的角色**。通常，环境具有硬性约束，例如在机器人领域，Agent 控制着物理世界中的执行器。另一方面，数字环境可以通过抽象形式进行塑造，例如应用程序编程接口（API）和用户界面（UI），分别面向软件和人类用户。当然，现有的接口设计主要针对其中一种用户。我们认为，语言模型 Agent 代表了一种新的终端用户类别，他们拥有自身的需求和能力。我们将语言模型 Agent 用于与计算机交互的接口称为智能体-计算机接口（ACI）。图 2 展示了 ACI 如何为语言模型 Agent 提供与计算机交互的重要功能，类似于代码编辑器如何帮助人们更有效地使用计算机。

人类和 LM 在能力和局限性上的差异，促使人们采用不同的接口设计指南。例如，当前生成式语言模型缺乏直接操作包含丰富视觉组件和信号的图形用户界面（GUI）应用程序所需的视觉理解能力。然而，如果以合适的方式呈现，这些应用程序提供的许多功能（例如语法检查和导航工具）对语言机器来说可能非常有用。此外，人类可以灵活地忽略不必要的信息，而所有内容对语言模型来说都具有固定的内存和计算成本，并且干扰性的上下文会损害其性能。因此，如果为语言模型提供一个能够充分考虑这些差异的接口，它们在与计算机交互时可能会更加有效。

最终，一个设计良好的 ACI 应该能够帮助 LM Agent 理解应用程序在先前变更下的状态，管理历史记录以避免引入不必要的先前观察结果，并提供模型可以高效可靠地使用的操作。ACI 不仅规定了 LM 可用的命令，还规定了环境状态如何反馈给 LM。它还会跟踪所有先前命令和观察结果的历史记录，并在每个步骤中管理如何将这些信息格式化，并将其与高级指令组合成 LM 的单个输入。

本文假 LM 是固定的，重点在于设计智能体计算机交互系统（ACI）以提升其性能。这意味着我们需要调整智能体的动作、动作文档以及环境反馈，以弥补语言模型的不足并增强其功能。我们从人机交互（HCI）领域汲取灵感，该领域的用户研究揭示了不同界面在人类直觉和操作表现方面的兼容性。**我们采用两种方法来提升开发集上的性能**：（1）手动检查智能体的行为，识别问题并提出改进建议；（2）运行网格搜索以选择最佳的智能体交互系统配置。

采取这两项措施后，我们对设计原则有了一些新的认识，这些认识对于构建有效的 ACI 系统尤为重要：
1. **Actions should be simple and easy to understand for agents**。许多 Bash 命令的文档都包含数十个选项。而选项较少、文档简洁的命令更便于 Agent 使用，从而减少了演示或微调的需要。这是我们在第 3 节中描述的所有 SWE Agent 命令的核心原则。
2. **Actions should be compact and efficient**。重要的操作（例如文件导航、编辑）应尽可能简化为最少的动作。高效的动作有助于智能体一步完成目标并取得实质性进展。因此，糟糕的设计会包含许多简单的动作，这些动作必须跨越多个回合才能组合成更高阶的操作。我们在第 5.1 节的编辑和搜索界面分析中展示了这一理念的实际应用。
3. **Environment feedback should be informative but concise**。高质量的反馈应该向智能体提供关于当前环境状态（以及智能体近期操作的影响）的实质性信息，而无需包含不必要的细节。例如，在编辑文件时，告知智能体已修改的内容就很有帮助。图 3a、3b 和表 3 展示了这一点。
4. **Guardrails mitigate error propagation and hasten recovery**。与人类一样，语言模型在编辑或搜索时也会犯错，并且难以从这些错误中恢复。内置一些防护机制，例如能够自动检测错误的代码语法检查器，可以帮助智能体识别并快速纠正错误。表 3 展示了编辑防护机制的效果。

第 5 节中的分析和消融研究表明，不同的 ACI 如何影响 LM 性能。我们的研究表明，这些原则在动作、反馈和工作流程中反复出现。

## 3.SWE-agent: Designing an ACI for Software Engineering

<img
  src="https://i-blog.csdnimg.cn/direct/5fb8269943e54a25abb8f24373d1cd6f.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>
本文描述了 SWE-agent 如何为 LM 提供一个 ACI，使其能够作为软件工程 Agent，从而高效地搜索、导航、编辑和执行代码命令。该 ACI 由几个主要组件构成，包括搜索/导航、文件查看器、文件编辑器和上下文管理。在每个步骤中，SWE-agent 都会生成一个想法和一个命令，然后将命令执行的反馈融入到环境中（ReAc）。SWE-agent 构建于 Linux shell 之上，因此在需要时还可以访问常用的 Linux 命令和实用程序。

**Search and navigation**。浏览代码库需要找到相关的文件和内容。一种常见的策略是查找可能有用的术语，例如问题中提到的文件、函数或类定义。​​我们引入了特殊的命令 `find_file`、`search_file` 和 `search_dir`，它们会在搜索文件或目录中的文件名和字符串时输出搜索结果摘要。图 10 展示了这些搜索结果格式的示例。`find_file` 命令在代码库中搜索文件名，而 `search_file` 和 `search_dir` 命令则在子目录的文件中查找字符串。我们的界面通过抑制冗长的结果来鼓励高效搜索。每个搜索 query 最多返回 50 个结果；如果搜索结果超过此数量，我们将不报告结果，而是建议用户编写更具体的 query。

**File viewer**。找到想要查看的文件后，Agent 会调用相应文件路径上的 open 命令来打开交互式文件查看器。文件查看器一次最多显示文件中的 100 行。Agent 可以使用 scroll_down 和 scroll_up 命令移动窗口，或使用 goto 命令跳转到特定行。为了方便文件内导航和代码定位，我们会显示：打开文件的完整路径、文件总行数、当前窗口前后省略的行数以及行号（显示在每行可见行的前面）。图 3a 展示了该界面的示例。

**File editor**。我们提供了一些命令，允许 LM 创建和编辑文件。编辑命令与文件查看器配合使用，允许智能体替换打开文件中特定范围的行。此命令需要三个参数：起始行、结束行和替换文本。智能体可以一步完成操作，将起始行和结束行之间的所有行替换为替换文本，如图 3b 所示。编辑完成后，文件查看器会自动显示更新后的内容，帮助智能体立即查看编辑效果，而无需调用其他命令。图 3b 显示了一个智能体响应示例，其中包含文件编辑操作。

就像人类可以使用语法高亮等工具在集成开发环境 (IDE) 中编辑文件时发现格式错误一样，我们将代码检查器集成到编辑功能中，以便在智能体编辑文件时提醒其可能引入的错误。代码检查器会将选定的错误以及错误前后的文件内容片段显示给智能体。无效的编辑将被丢弃，并提示智能体重新尝试编辑文件。

**Context management**。SWE-agent 系统利用信息提示、错误信息和历史处理器来保持 Agent 上下文的简洁性和信息量。Agent 会收到关于正确使用 bash 和 ACI 命令的说明、文档和演示。在每个步骤中，系统都会提示 Agent 生成一个想法和一个动作。格式错误的生成会触发错误响应（如图 32 所示），要求 Agent 重试，直到收到有效的生成为止。一旦收到有效的生成，除第一个错误信息外，所有过去的错误信息都将被忽略。

Agent 的环境响应会使用图 30 所示的模板显示计算机输出；但是，如果没有生成任何输出，则会包含一条特定消息（“您的命令已成功运行，但未产生任何输出”），以提高清晰度。为了进一步提高上下文相关性，最后 5 条之前的观察结果会被合并成一行，如图 31 所示。通过移除先前观察结果中的大部分内容，我们保留了有关计划和行动历史的关键信息，同时减少了不必要的上下文，从而允许更多的交互循环，并避免显示过时的文件信息。§A 提供了更多实现细节。

## 4.Experimental Setup

**Datasets**。我们主要在 SWE-bench 数据集上进行评估，该数据集包含来自 12 个不同常用 Python 包仓库的 2294 个任务实例。除非另有说明，否则我们报告的是 Agent 在完整 SWE-bench 测试集上的主要结果，以及在 SWE-bench Lite 测试集上的消融和分析结果。SWE-bench Lite 是 SWE-bench 的一个标准子集，包含 300 个实例，专门用于评估独立的、功能性的错误修复。我们还使用 HumanEvalFix（一个简短的代码调试基准测试）测试了 SWE-agent 的基本代码编辑能力。

**Models**。所有结果、消融实验和分析均基于两个领先的语言模型：GPT-4 Turbo (gpt-4-1106-preview) 和 Claude 3 Opus (claude-3-opus-20240229)。我们也尝试了一些其他的闭源和开源模型，包括 Llama 3 和 DeepSeek Coder，但发现它们在智能体环境下的性能欠佳。许多语言模型的上下文窗口过小，例如 Llama 3 的上下文窗口只有 8k。GPT-4 Turbo 和 Claude 3 Opus 的上下文窗口分别为 128k 和 200k 个 token，这为语言模型在输入系统提示、问题描述以及可选的演示后进行多轮交互提供了足够的空间。

**Baselines**。我们将 SWE-agent 与两个基线模型进行比较。第一个基线模型是 Jimenez et al. 建立的非交互式检索生成 (RAG) 基线模型。在该模型中，BM25 检索系统以问题为查询条件检索最相关的代码库文件；给定这些文件，模型需要直接生成一个能够解决该问题的补丁文件。

第二种设置称为“纯 Shell 模式”，改编自 Yang et al. 提出的交互式编码框架。与 InterCode 环境类似，该基线系统要求语言模型（LM）通过与 Linux 上的 shell 进程交互来解决问题。与 SWE-agent 类似，模型预测是基于交互后代码库的最终状态自动生成的。

**Metrics**。我们以“**Resolved**”或“**pass@1**”作为主要指标，该指标是指在将模型生成的补丁应用到代码库后，所有测试均成功通过的实例比例。我们还报告“平均成本”指标，该指标是指 SWE-agent 在所有成功解决的实例上产生的 API 推理成本的平均值。由于预算限制，我们将每个实例的预算设置为 4 美元；如果运行超出此预算，则会自动提交现有修改。

**Configuration search**。在 SWE-agent 的设计过程中，我们通过对 SWE-bench 开发分支中精心挑选的一小部分示例进行系统行为的定性分析，最终确定了 ACI 设计。对于其余的超参数选择，我们对窗口大小、历史处理和解码温度进行了扫描，如 §B.1 所示。

## 5.Results

<img
  src="https://i-blog.csdnimg.cn/direct/72ce2c954ba2499bbed4e3c849cc8f45.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

在所有系统中，搭载 GPT-4 Turbo 的 SWE-agent 表现最佳，成功解决了完整 SWE-bench 测试集 12.47% (286/2294) 的问题，以及 Lite 测试集 18.00% (54/300) 的问题。如表 1 所示，与 Lite 测试集上的 RAG 相比，SWE-agent 的计算成本高出 8-13 倍，但其问题解决率提高了 6.7 倍。SWE-agent 相较于仅使用 Shell 算法的算法，在搭载 GPT-4 Turbo 的情况下，其相对性能提升了 64%，这进一步证实了其作为 LM 友好型 ACI 的价值。

表 2 显示，SWE-agent 在 HumanEvalFix 测试中表现出色，通过率高达 88.3%。图 4 显示，平均性能波动相对较低，但单个实例的解析度可能存在显著差异。更多结果见附录：§B.2 表明成功率与问题年龄无关（已控制可能的测试污染），B.5 详细介绍了性能波动和 pass@k，B.7 讨论了额外的评估细节。

<img
  src="https://i-blog.csdnimg.cn/direct/f3e881ca5cd14caba079016371b1b95d.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

### 5.1  Analysis of ACI Design

<img
  src="https://i-blog.csdnimg.cn/direct/e7c1eed4812b4c08a46e2b97d740e007.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们对 SWE-Agent 接口进行了多次消融实验，特别是针对 SWE- Agent 与 GPT-4 配置的实验，结果总结在表 3 中。我们的案例研究揭示了 Agent 的有趣行为以及不同 ACI 设计的影响。

**Human user interfaces are not always suitable as agent-computer interfaces**。当前的语言模型在 Linux shell 环境下搜索相关内容时存在诸多缺陷。某些搜索模式（例如，cd、ls、cat 等命令链）效率极低。虽然 grep 或 find 命令的性能有所提升，但偶尔也会产生大量无关结果。我们假设，通过更快的导航速度和更丰富的搜索界面，可以实现更佳的语言定位效果。

**图 <span style="color:red">5</span>** 比较了仅 Shell 设置与两种不同搜索接口。*迭代搜索* 直接受到传统搜索用户界面的启发，例如 `vim` 或 VSCode，它通过文件查看器逐个显示结果。Agent 可以使用 `next` 和 `prev` 操作浏览结果。每个结果都会显示匹配行以及周围的 n 行上下文。一个优点是，agent 在搜索中看到相关代码后，可以直接开始编辑。然而，当给定大量搜索结果时，agent 往往会穷尽式地查看每一个匹配项，不断调用 `next`，直到每个结果都被检查过。这种低效行为可能会耗尽 agent 的成本预算或上下文窗口，导致性能甚至比完全没有额外搜索工具时更差（无搜索为 15.7%<span style="color:red">↓2.3</span>，而迭代搜索为 12.0%<span style="color:red">↓6.0</span>）。

<img
  src="https://i-blog.csdnimg.cn/direct/c178d48a0a9c42398a2f4bad59748355.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Compact, efficient file editing is critical to performance**。SWE-agent 的文件编辑器和查看器旨在将编辑过程整合为一个单一命令，从而实现轻松的多行编辑并提供一致的反馈，并在编辑后自动更新 Agent 对文件的视图。在“无编辑”设置下，编辑选项受限且容易出错；主要方法要么是通过重定向和覆盖来替换整个文件，要么使用 sed 等工具进行单行或查找替换编辑。这两种方法都存在明显的缺陷。重定向需要复制并重写整个文件，即使是微小的更改也需要这样做，这既低效又容易出错。虽然 sed 可以方便地进行特定编辑，但执行多行编辑却很繁琐，并且可能导致难以检测的意外后果。此外，这两种策略都缺乏关于文件更新的即时反馈，使得这些静默操作可能会让模型难以理解，并增加出错的风险。如果没有 SWE-agent 的文件编辑器界面，性能会下降到（10.3% ↓ 7.7）。我们还发现，Agent 对文件查看器显示的行数非常敏​​感。内容太少（30 行，性能下降 14.3%，实际性能下降 3.7）或内容太多（整个文件，性能下降 12.7%，实际性能下降 5.3）都会降低性能。

**Guardrails can improve error recovery**。当模型反复编辑同一段代码时，会出现一种常见的故障模式。造成这种行为的常见原因是 Agent 在错误编辑过程中引入了语法错误（例如，缩进错误、多余的括号）。如第 3 节所述，我们在编辑逻辑中添加了一个干预措施，仅当修改不会产生重大错误时才允许应用。我们在图 6 中将此接口与“不编辑”和“不进行代码检查”两种方案进行了比较。该干预措施显著提高了性能（不进行代码检查时，性能提升 15.0%，下降 3.0）。

<img
  src="https://i-blog.csdnimg.cn/direct/c7a0fa46e7c2493d9869b101dafbb9a7.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

### 5.2 Analysis of Agent Behavior

## 6.Related Work

## Appendix

### A SWE-agent Design

#### A.1 ACI Design

#### A.2 Implementation

#### A.3 Configuration

### B.Extended Results

#### B.1 Hyperparameter Sweep

#### B.2 Model Performance

#### B.3 Trajectory Analysis

##### B.3.1 Turns to Resolution

##### B.3.2 Walkthrough of Trajectory Phases

##### B.3.3 Breakdowns of Action Sequences

#### B.4 Failure Modes

#### B.5  Performance Variance and Pass@k Rate

#### B.6 Patch Generations

#### B.7 HumanEvalFix Evaluation

#### B.8 Dataset Information

#### B.9 Miscellaneous

### C.Prompts

### D.Qualitative Analysis
