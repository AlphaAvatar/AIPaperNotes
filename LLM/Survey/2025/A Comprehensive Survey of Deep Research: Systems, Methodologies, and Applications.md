论文链接：https://arxiv.org/pdf/2506.12594

# 摘要

本综述探讨了快速发展的 Deep Research 系统领域——由人工智能驱动的应用程序，通过集成大语言模型、高级信息检索和自主推理功能，实现复杂研究工作流程的自动化。我们分析了自2023年以来出现的80多个商业和非商业实现，包括OpenAI/DeepResearch、Gemini/DeepResearch、Perplexity/DeepResearch以及众多开源替代方案。通过全面的研究，我们提出了一种新的**层次化分类**法，该分类法根据**四个基本技术维度**对系统进行分类：（1）基础模型和推理引擎、（2）工具利用和环境交互、（3）任务规划和执行控制，（4）以及知识综合和输出生成。我们探索了这些系统在学术、科学、商业和教育应用中的架构模式、实现方法和特定领域的适应性。我们的分析揭示了当前实现的强大功能，以及它们在信息准确性、隐私、知识产权和可访问性方面所带来的技术和道德挑战。本综述最后指出了高级推理架构、多模态集成、领域专业化、人机协作以及生态系统标准化等有前景的研究方向，这些方向可能会塑造这项变革性技术的未来发展。通过提供一个理解深度研究系统的全面框架，本综述不仅有助于加深对人工智能增强知识工作的理论理解，也有助于开发更强大、更可靠、更易于获取的研究技术。论文资源可在 https://github.com/scienceaix/deepresearch 查看。

# 1.Introduction

人工智能的快速发展，促使学术界和产业界在知识发现、验证和应用方式上发生了范式转变。传统的研究方法依赖于人工文献综述、实验设计和数据分析，而如今，能够自动化端到端研究工作流程的智能系统正日益补充（在某些情况下甚至被取代）。这种演变催生了一个我们称之为 **Deep Research** 的全新领域，它融合了大语言模型 (LLM)、高级信息检索系统和自动推理框架，重新定义了学术研究和实际问题解决的界限。

## 1.1 Definition and Scope of Deep Research

Deep Research 是指系统地应用人工智能技术，通过三个核心维度实现研究过程的自动化和增强：
（1） **Intelligent Knowledge Discovery**：跨异构数据源自动进行文献检索、假设生成和模式识别。
（2）**End-to-End Workflow Automation**：将实验设计、数据收集、分析和结果解释集成到统一的 AI 驱动流程中。
（3）**Collaborative Intelligence Enhancement**：通过自然语言界面、可视化和动态知识表示促进人机协作。

为了清晰地界定 Deep Research 的界限，我们将其与类似的人工智能系统进行如下区分：
- **Differentiating from General AI Assistants**：虽然像 ChatGPT 这样的通用 AI 助手可以回答研究问题，但它们缺乏 Deep Research 系统所具备的自主工作流能力、专业研究工具以及端到端的研究协调能力。最近的调查凸显了专业研究系统与通用 AI 能力之间的关键区别，尤其强调了**特定领域工具**与通用助手相比如何从根本上改变研究工作流程。
- **Differentiating from Single-Function Research Tools**：引用管理器、文献搜索引擎或统计分析软件包等专业工具虽然能够处理孤立的研究功能，但缺乏深度 Deep Research 所具备的集成推理和跨功能协同能力。scispace 和 You.com 等工具代表了早期研究辅助的尝试，但缺乏真正的 Deep Research 系统所具备的端到端功能。
- **Differentiating from Pure LLM Applications**：仅仅用面向研究的提示包装 LLM 的应用程序缺乏真正的 Deep Research 系统所具有的环境交互、工具集成和工作流自动化功能。

本综述专门考察至少具备三个核心维度中的两个维度的系统，重点关注那些以大语言模型为基础推理引擎的系统。我们的范围涵盖 **OpenAI/DeepResearch**、谷歌的 **Gemini/DeepResearch** 和 **Perplexity/DeepResearch** 等商业产品，以及 **dzhng/deepresearch**、**HKUDS/Auto-Deep-Research** 等开源实现，以及后续章节中详细介绍的众多其他产品。我们排除了纯粹的文献计量工具或缺乏集成认知能力的单阶段自动化系统，例如 Elicit、ResearchRabbit、Consensus 等研究辅助工具，或 Scite 等引文工具。其他一些专业工具，例如专注于科学文本检索和组织的 STORM，虽然很有价值，但缺乏我们调查范围所核心的端到端深度研究能力。

## 1.2 Historical Context and Technical Evolution

Deep Research 的发展轨迹可以通过三个反映技术进步和实现方法的演变阶段来描绘：

*1.2.1 Origin and Early Exploration (2023 - February 2025)*。值得注意的是，像 **n8n**、**QwenLM/Qwen-Agent** 等工作流自动化框架早在 Deep Research 兴起之前就已存在。它们的早期建立体现了相关技术领域的既有基础，突显出发展格局并非仅仅由 Deep Research 的兴起所塑造，而是具有更加多样化和更早的根源。Deep Research 的概念源于人工智能助手向智能 Agent 的转变。2024 年 12 月，Google Gemini 率先推出了此功能，其初始 Deep Research 实现专注于基本的多步推理和知识集成。这一阶段为后续发展奠定了基础，为更复杂的人工智能驱动研究工具奠定了基础。其中许多进展建立在早期的工作流自动化工具（如 n8n）和 Agent 框架（如 **AutoGPT** 和 **BabyAGI**）的基础上，这些框架已经为自主任务执行奠定了基础。该生态系统的其他早期贡献包括开创了集成研究工作流程的 **cline2024** 和开发了基于网络的研究所必需的基础浏览器自动化功能的 **open_operator**。

*1.2.2 Technological Breakthrough and Competitive Rivalry (February - March 2025)*。DeepSeek 开源模型的兴起，以其高效的推理能力和经济高效的解决方案彻底改变了市场。2025 年 2 月，OpenAI 发布 Deep Research，标志着 Deep Research 的重大飞跃。基于 o3 模型，Deep Research 展示了自主研究规划、跨领域分析和高质量报告生成等先进能力，在复杂任务中实现了超越以往基准的准确率。与此同时，Perplexity 于 2025 年 2 月推出了免费使用的 Deep Research，强调快速响应和易用性，以占领大众市场。**nickscamara/open-deepresearch**、**mshumer/OpenDeepResearcher**、**btahir_open_deep_research** 和 **GPT-researcher** 等开源项目逐渐兴起，成为由社区驱动的商业平台替代方案。生态系统不断扩展，出现了轻量级实现，例如专为有限资源的本地执行而设计的 **Automated-AI-Web-Researcher-Ollama**，以及模块化框架，例如为自定义研究工作流程提供可组合组件的 **Langchain-AI/Open_deep_research**。

*1.2.3 Ecosystem Expansion and Multi-modal Integration (March 2025 - Present)*。第三阶段的特点是多样化生态系统的成熟。像 **Jina-AI/node-DeepResearch** 这样的开源项目实现了本地化部署和定制，而 OpenAI 和谷歌的商业闭源版本则继续通过多模态支持和多 Agent 协作功能突破界限。高级搜索技术和报告生成框架的集成进一步增强了该工具在学术研究、金融分析和其他领域的实用性。与此同时，**Manus** 和 **AutoGLM-Research**、**MGX** 和 **Devin** 等平台正在整合先进的人工智能研究功能，以增强其服务。与此同时，Anthropic 于 2025 年 4 月推出了 **Claude/Research**，引入了 Agent 搜索功能，可以系统地探索问题的多个角度，并提供具有可验证引文的全面答案。**OpenManus**、**Camel-AI/OWL** 和 **TARS** 等智能体框架通过专门的功能和特定领域的优化进一步扩展了生态系统。

## 1.3 Significance and Practical Implications
Deep Research 表明跨多个领域的变革潜力：

（1）**Academic Innovation**：通过自动化文献综合（例如，HotpotQA 性能基准）加速假设验证，并使研究人员能够探索更广泛的跨学科联系，否则这些联系可能未被发现。Deep Research 的变革潜力超越了单个应用，可以从根本上重塑科学发现过程。正如 Sourati 和 Evans 所论证的那样，具有人类意识的人工智能可以通过增强研究人员的能力，同时适应他们的概念框架和方法论，从而显著加速科学发展。这种人机协同代表着从传统自动化向尊重和增强人类科学直觉的协作智能的根本转变。Khalili 和 Bouchachia 的补充研究进一步表明，构建科学发现机器的系统方法如何通过集成的人工智能驱动的研究工作流程来改变假设生成、实验设计和理论完善。

（2）**Enterprise Transformation**：通过 **Agent-RL/ReSearch** 和 **smolagents/open_deep_research** 等系统实现大规模数据驱动的决策，这些系统可以前所未有的深度和效率分析市场趋势、竞争格局和战略机遇。

（3）**Democratization of Knowledge**：通过像 **grapeot/deep_research_agent** 和 **OpenManus** 这样的开源实现降低进入门槛，使个人和组织能够访问复杂的研究能力，而无需考虑技术专长或资源限制。

## 1.4 Research Questions and Contribution of this Survey

本综述探讨了三个基本问题：
- 架构选择（系统架构、实现方法、功能能力）如何影响 Deep Research 的有效性？
- 在 Deep Research 实现的范围内，LLM 微调、检索机制和工作流程编排出现了哪些技术创新？
- 现有系统如何平衡性能、可用性和伦理考虑？通过比较 **n8n** 和 **OpenAI/AgentsSDK** 等方法，可以得出哪些模式？

我们的贡献体现在三个方面：
- **Methodological**：提出一种新的分类法，根据系统的技术架构对系统进行分类，从基础模型到知识综合能力
- **Analytical**：对评估指标中的代表性系统进行比较分析，突出不同方法的优势和局限性
- **Practical**：确定关键挑战并制定未来发展路线图，特别关注新兴架构和集成机会

本文的其余部分遵循结构化探索，从概念框架（第 2 节）、技术创新和比较分析（第 3-4 节）、实现技术（第 5 节）、评估方法（第 6 节）、应用和用例（第 7 节）、道德考虑（第 8 节）和未来方向（第 9 节）开始。

# 2.The Evolution and Technical Framework of Deep Research
本节将围绕定义 Deep Research 系统的四项基本技术能力，对 Deep Research 系统进行全面的技术分类。针对每项能力，我们将探讨其演进轨迹和技术创新，并重点介绍体现每种方法的代表性实现。

## 2.1  Foundation Models and Reasoning Engines: Evolution and Advances

Deep Research 系统的基础在于其底层人工智能模型和推理能力，这些模型和推理能力已经从通用语言模型发展为专门的研究型架构。

### 2.1.1 From General-Purpose LLMs to Specialized Research Models

从通用 LLM 到研究专业模式的进展代表了 Deep Research 能力的根本转变：

**Technical Evolution Trajectory**。早期的实现依赖于通用的 LLM，且针对特定任务的优化极少。当前的系统通过架构修改、专门的训练语料库以及专注于分析和推理能力的微调机制，专门针对研究任务增强了模型。从 GPT-4 等模型到 OpenAI 的 o3 模型的过渡，展现了抽象、多步推理和知识集成能力的显著提升，这些能力对于复杂的研究任务至关重要。

**Representative Systems**。OpenAI/DeepResearch 以其基于 o3 的模型（该模型专门针对网页浏览和数据分析进行了优化）为例，展示了这一演变。该系统利用思维链和思维树推理技术来探索复杂的信息环境。谷歌的 Gemini/DeepResearch 也采用了 Gemini 2.5 Pro，该模型具有增强的推理能力和百万个 token 上下文窗口来处理海量信息。这些方法建立在推理增强技术的基础之上，例如思维链提示 、自洽性和人类偏好校准，这些技术已专门针对研究密集型任务进行了调整。在开源领域，**AutoGLM-Research** 展示了专门的训练方案如何优化 ChatGLM 等现有模型，使其能够胜任研究密集型任务，并通过有针对性地增强推理组件，显著提升性能。

### 2.1.2 Context Understanding and Memory Mechanisms

### 2.1.3 Enhancements in Reasoning Capabilities
