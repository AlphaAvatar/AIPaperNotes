论文链接：https://claude.com/blog/building-agents-with-skills-equipping-agents-for-specialized-work

代码链接：

# 摘要

过去一年发生了许多变化。MCP 迅速被行业领导者和开发者社区采用，成为 Agent 连接的标准。Claude Code 作为一款通用编码 Agent 正式发布。此外，我们还推出了 **Claude Agent SDK**，它现在提供了一个开箱即用、可用于生产环境的 Agent。

但随着我们构建和部署这些智能体，我们不断遇到同样的问题：**智能体拥有智能和能力，但并非总能有效应对实际工作所需的专业知识**。这促使我们创建了 **Agent Skills**。Skills 是组织有序的文件集合，其中打包了领域专业知识——工作流程、最佳实践、脚本——并以智能体可以访问和应用的格式呈现。它们可以将一个能力全面的智能体转变为一个知识渊博的专家。

在这篇文章中，我们将解释为什么我们停止构建专业 Agent，转而构建 Skills，以及这种转变如何改变我们对扩展 Agent 能力的思考方式。

# The new paradigm: code is all you need

我们过去认为不同领域的智能体应该截然不同。编码智能体、研究智能体、金融智能体、市场营销智能体——每个领域似乎都需要专属的工具和框架。业界最初也接受了这种领域特定智能体的模式。但随着模型智能水平的提升和智能体能力的进步，我们逐渐转向了另一种方法。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8e9d357741ad4dea84d2a1364e1f9601.png)

我们逐渐意识到，**代码不再仅仅是一种使用场景，而是智能体执行几乎所有数字工作的接口**。Claude Code 是一个编码智能体，但同时也是一个通用智能体，它恰好通过代码来工作。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8a3245e518584aa5af318d80f7c04070.png)

不妨考虑使用 Claude Code 生成财务报告。它可以调用 API 进行研究，将数据存储在文件系统中，使用 Python 进行分析，并提炼出有价值的见解。所有这些操作都通过代码完成。搭建框架非常简单，只需 bash 脚本和文件系统即可。

但通用能力并不等同于专业技能。当我们开始将 Claude Code 应用于实际工作时，差距就显现出来了。

# The missing piece: domain expertise

你会选择谁来帮你报税：一个数学天才从零开始推算，还是一个经验丰富的税务专家，他已经处理过成千上万份报税单？大多数人会选择税务专家。这并非因为他们更聪明，而是因为他们拥有相关的专业知识。

如今的 Agent 就像数学天才：他们善于推理解决新问题，但往往缺乏经验丰富的专业人士所积累的专业知识。在适当的指导下，他们能做出令人惊叹的事情。然而，他们常常忽略重要的背景信息，难以吸收组织的专业知识，也无法从重复性任务中自动学习。

Skills 通过将领域专业知识打包成 Agent 可以逐步访问和应用的形式来弥合这一差距。

# What are Agent Skills?

Skills 包含 Agent 的领域专业知识和程序知识。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3abee4d1fb90408e9ec40744c21baac2.png)

Skills 的简洁性是刻意设计的。文件是一种通用的基本单位，可以与你现有的资源兼容。你可以使用 Git 进行版本控制，将它们存储在 Google 云端硬盘中，并与团队共享。这种简洁性也意味着技能创建不再局限于工程师。产品经理、分析师和领域专家也已经在构建技能，以规范他们的工作流程。

# Progressive disclosure

Skills 可以包含大量信息。为了保护上下文窗口并使 Skills 可组合，它们采用渐进式披露：在运行时，仅向模型显示元数据（来自 YAML 前置元数据的名称和描述）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c49d58aef0be45d5aebe72c9c547296d.png)

如果 Claude 判断需要某项技能，它会读取完整的 SKILL.md 文件。为了提供更多详细信息，Skills 可以包含一个 references/ 目录，其中的辅助文档仅在需要时加载。

这种**三层方法**意味着您可以为 Agent 配备数百种技能，而不会使其上下文窗口过载——元数据使用约 50 个 token，完整的 SKILL.md 文件使用约 500 个 token，参考文件使用 2,000 多个 token，并且仅在需要时才使用。

# Skills can include scripts as tools

传统工具存在一些问题：有些工具的说明文档编写不完善，模型并非总能对其进行修改或扩展，而且它们常常会占用大量的上下文窗口空间。而代码则不同，它具有自文档性，可修改，并且无需始终处于上下文中。

举个真实的例子：我们发现 Claude 一直在编写同一个脚本，将 Anthropico 样式应用到幻灯片上。所以我们请 Claude 把它保存为一个工具供自己使用：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b302a6c6e964a4d9eb88c5bac6a833f.png)

slide-decks.md 中的相应文档只是简单地引用了以下脚本：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8edd37033f394017aaadad334604f486.png)

# The skills ecosystem

Skills 生态系统迅速发展，目前我们已经看到三种主要类型的 Skills 正在构建：

## Foundational skills

这些 Skills 涵盖了每个人都需要的核心功能：处理文档、电子表格、演示文稿等等。它们总结了文档生成和处理的最佳实践。您可以通过探索我们公共知识库中的基础技能，了解这些技能在实践中的应用。

## Partner skills

随着 Skills 规范化 Agent 与专业功能交互的方式，各公司正在构建 Skills，使其服务更易于 Agent 使用。K-Dense、Browserbase、Notion 等众多公司正在创建可直接集成其服务的技能，在保持技能格式简洁性的同时，扩展 Claude 在特定领域的功能。

## Enterprise skills

组织会构建专有 Skills，以编码其内部流程和领域专业知识。这些 Skills 有助于捕捉特定的工作流程、合规性要求和机构知识，从而使 Agent 能够胜任企业工作。

## The complete architecture

随着 Skills 应用的普及，一些趋势正在显现，预示着这一范式未来的发展方向。这些趋势影响着我们对技能设计的思考方式，以及我们为支持技能开发者而构建的工具。

## Increasing complexity

早期的 Skills 仅限于简单的文档查阅。现在我们看到的是复杂的多步骤工作流程，它能够协调跨多个工具的数据检索、复杂计算和格式化输出。
- **Simple**：Status report writer" (~100 lines) - Templates and formatting
- **Intermediate**："Financial model builder" (~800 lines) - Data retrieval, Excel modeling with Python
- **Complex**："RNA sequencing pipeline" (2,500+ lines) - Coordinates HISAT2, StringTie, DESeq2 analysis

# Skills and MCP

Skills 能和 MCP 服务器可以自然地协同工作。例如，一项竞争分析技能可以协调网络搜索、通过 MCP 获取的内部数据库、Slack 消息历史记录以及 Notion 页面，从而综合生成一份全面的报告。

# Non-developer adoption

Skills 创建正从工程师扩展到产品经理、分析师和各领域的专家。他们可以使用 Skills 创建工具，在 30 分钟内创建并测试自己的第一个技能。该工具会以交互式的方式引导他们完成整个过程。我们正在努力让技能创建更加便捷，通过改进工具和模板，让任何人都能轻松获取和分享专业知识。

# The complete architecture

综合来看，新兴的 Agent 架构看起来像是以下几方面的组合：

1. **Agent loop**：决定下一步行动的核心推理系统
2. **Agent runtime**：执行环境（代码、文件系统）
3. **MCP servers**：与外部工具和数据源的连接
4. **Skills library**：领域专业知识和程序知识

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4174b2ba5b5486eab1e00e360c59dba.png)
每一层都有明确的用途：循环层负责推理，运行时负责执行，MCP 负责连接，技能层负责指导。这种分离使得系统易于理解，并允许每个部分独立演进。

想象一下，如果在这个架构中添加一项技能会发生什么。前端设计技能可以立即提升 Claude 的前端能力。它提供关于排版、色彩理论和动画的专业指导，并且仅在构建 Web 界面时激活。渐进式披露意味着它仅在需要时加载。添加新功能也非常简单。
