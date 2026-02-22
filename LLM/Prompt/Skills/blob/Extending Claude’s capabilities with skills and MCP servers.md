论文链接：https://claude.com/blog/extending-claude-capabilities-with-skills-mcp-servers

代码链接：

# 摘要

自推出 Skills 以来，我们从客户那里听到的两个最大的问题是：“**Skills 和 MCP 如何协同工作？我应该在什么情况下使用其中一个，而不是另一个？**”

模型上下文协议 (MCP) 将 Claude 与第三方工具连接起来，而 Skills 则教会 Claude 如何高效地使用这些工具。将两者结合起来，您可以构建遵循团队工作流程的 Agent，而不是需要不断修正的通用流程。

例如，通过 MCP 连接到 Notion，Claude 就可以搜索您的工作区。添加一项会议准备 skill 后，Claude 就能知道要从中提取哪些页面、如何格式化准备文档，以及您的团队提交会议记录的标准是什么。这样一来，连接就变得实用而不仅仅可用了。

在本文中，我们将分析 Skills 与 MCP 之间的关系，如何将它们结合起来构建能够遵循您的工作流程并产生一致输出的 Agent，并通过一些现实世界的例子来说明它们在实践中是如何协同工作的。

# Understanding Skills and MCP

你走进一家五金店，想修理一个坏掉的橱柜。店里什么都有（木工胶、夹子、替换铰链），但知道该买哪些东西以及如何使用它们却是另一个问题。

MCP 就像拥有进入货架通道的权限。而 Skills 则如同员工的专业技能。如果你不知道需要哪些物品或如何使用它们，即使库存再多也无济于事。Skills 就像一位乐于助人的员工，他会指导你完成维修流程，指出合适的耗材，并教你正确的操作方法。

更具体地说，MCP 服务器使 Claude 能够访问您的外部系统、服务和平台，而 Skills 则为 Claude 提供有效使用这些连接所需的上下文，教会 Claude 在获得此访问权限后应该做什么。

如果没有 Skills 提供的上下文信息，Claude 就只能猜测你的意图。有了 Skills，Claude 就可以按照你的计划行事了。

# Why Skills and MCP work well together

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14e90b3208cf4fbba1cb687f567fd769.png)

**MCP负责连接**：提供对外部系统的安全、标准化访问。无论您连接的是 GitHub、Salesforce、Notion，还是您自己的内部 API，MCP 服务器都能让 Claude 访问您的工具和数据。

**Skills 体现了专业知识**：它包含领域知识和工作流程逻辑，能够将原始工具访问权限转化为可靠的结果。熟练的技术人员知道何时查询 CRM 系统、在结果中查找什么、如何格式化输出以及哪些特殊情况需要特殊处理。

这种分离方式保证了架构的可组合性。单个技能可以协调多个 MCP 服务器，而单个 MCP 服务器也可以支持数十个不同的技能。添加新的连接后，现有技能可以将其集成到现有连接中。改进某个技能后，它就可以在所有已连接的工具中生效。

## When you combine skills and MCP, you get:

**Clear discovery**：Claude 不再猜测该从何入手。一项会议准备 Skills 可能会明确指出：首先查看项目页面，然后查看之前的会议记录，最后查看利益相关者资料。一项研究 Skills 可能会指出：从共享驱动开始，与客户关系管理系统 (CRM) 进行交叉比对，然后通过网络搜索填补空白。这项 Skills 体现了机构知识，即哪些资源对哪些任务至关重要。

**Reliable orchestration**：多步骤工作流程变得可预测。如果没有相关 Skills，Claude 可能会先提取数据并进行格式化，然后再检查数据是否完整。Skills 明确定义了操作顺序，因此 Claude 每次都能以相同的方式执行工作流程。

**Consistent performance**：实际产出符合标准。通用结果需要编辑。Skills 决定了团队对“完成”的定义：合适的结构、恰当的细节程度以及适合受众的语气。

随着时间的推移，团队会积累各种相互关联的 Skills 和链接，使 Claude 在其特定领域拥有专业知识。

## Where skills and MCP may overlap

MCP 服务器可能包含工具使用提示和常用任务指导等形式的说明。这样可以将特定于工具的知识与工具紧密结合。但是，这些说明的设计初衷应该是通用的。

**一般来说**：MCP 指南涵盖如何正确使用服务器及其工具。Skills 指南则涵盖如何将它们用于特定流程或在多服务器工作流中使用。

例如，Salesforce MCP 服务器可能会指定查询语法和 API 格式。Skills 则会指定首先检查哪些记录，如何将它们与 Slack 对话进行交叉引用以获取最新上下文，以及如何组织输出以供团队进行流程审查。

**在组合使用 MCP 服务器和 Skills 时，请注意是否存在冲突指令**。如果 MCP 服务器指示返回 JSON 格式，而 Skills 指示格式化为 Markdown 表格，Claude 就只能猜测哪个才是正确的。让 MCP 处理连接，让 Skills 处理展示、排序和工作流逻辑。

# Real-world examples of using skills and MCP together

现在我们来看看 Skills 和 MCP 如何在实际工作流程中结合使用。我们将通过两个例子来讲解：财务分析师提取实时市场数据进行公司估值，以及项目经理使用 Notion 的会议智能技能进行会议准备。

在这两种情况下，MCP 服务器提供对工具的访问权限，而 Skills 则定义了如何使用这些工具。

## Financial analysis: Automating company valuations skill

Anthropic 发布了一套预置 Skills，用于常见的财务工作流程，包括可比公司分析。可比公司分析是一种标准的估值方法。分析师在进行可比公司分析时，需要花费数小时从多个来源提取财务指标，应用相同的估值方法，并格式化输出以符合合规标准。这项工作重复性高、容易出错，而这正是 Skills 和 MCP 协同工作能够显著提升效率的工作流程。

**Skill**：可比公司分析可自动执行此估值工作流程，从多个来源提取数据，应用一致的方法，并按照特定标准格式化输出。

**MCP servers**：连接至标普全球市场财智 (S&P Capital IQ)、Daloopa 和晨星 (Morningstar) 获取实时市场数据

**Workflow**：
1. Skills 用于确定要查询哪些数据源（发现）
2. MCP 连接获取实时财务数据
3. Skills 应用方法论并格式化输出（编排）
4. Skills 符合合规要求（性能）

## Meeting preparation: Notion's Meeting Intelligence skill

会议准备工作很繁琐。你需要从多个渠道收集信息，例如项目文档、之前的会议记录以及利益相关者的信息，然后将其整合到会前阅读材料和议程中。这是一个多步骤的过程，每次你都得重新解释一遍。

**Skill**：会议智能功能定义了要搜索的页面、输出的结构以及要包含的章节。

**MCP server**：Notion 连接，用于搜索、读取和创建页面

**Workflow**：
1. Skills 可识别要搜索的相关页面，包括项目、以往会议、利益相关者信息（发现）
2. MCP 连接从 Notion 搜索并检索内容
3. Skills 结构化两份文件：内部预读文件和外部议程（统筹安排）。
4. MCP 连接会将两个文档保存到 Notion 中，并进行整理和链接。
5. Skills 确保输出符合格式标准（性能）

# When to use skills vs. MCP

Skills 和 MCP 解决的是不同的问题，但对于特定的工作流程来说，决定使用哪一个并不总是显而易见的。

## What to use skills for

Skills 是指那些原本只存在于你脑海中，或者每次有新成员加入团队都需要重新解释的知识。它们最适合用于：
- **涉及工具的多步骤工作流程**：会议准备工作从多个来源提取信息，然后创建结构化文档
- **需要保持一致性的流程**：季度财务分析必须每次都遵循相同的方法，合规性审查必须设有强制性检查点。
- **您希望获取和分享的领域专业知识**：研究方法、代码审查标准、写作指南
- **即使团队成员离职，以下工作流程仍应保留**：以可重用指令形式编码的机构知识

## What to use MCP servers for

MCP 扩展了 Claude 的访问权限和使用权限。当您需要以下情况时，请使用 MCP：
- **实时数据访问**：搜索 Notion 页面、阅读 Slack 消息、查询数据库
- **外部系统中的操作**：创建 GitHub 问题、更新项目管理工具、发送通知
- **文件操作**：从 Google 云端硬盘读取数据和写入数据，访问本地文件系统
- **API 集成**：连接到不具备原生 Claude 支持的服务

如果你是在解释如何做某件事，那是一种 Skills。如果你需要 Claude 访问某个东西，那就是 MCP。

## Quick reference table: How skills and MCP differ

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c4d71b61443c44819e07c2b321519702.png)

# Common questions

## Skills 可以取代 MCP 吗？

不能。Skills 和 MCP 解决的是不同的问题。MCP 提供与外部工具和数据的连接。Skills 则提供如何有效利用这种连接的操作知识。大多数强大的工作流程都会同时使用两者。

## 一项 Skills 可以使用多个 MCP 服务器吗？

可以。一项 Skills 可以同时协调多个 MCP 服务器。例如，一项技术竞争分析技能可能会在 Google 云端硬盘中搜索内部研究资料，从 GitHub 拉取竞争对手的代码库，并通过网络搜索收集市场数据。

## 我可以为一个 MCP 服务器创建多个 Skills 吗？

可以。Skills 可以提升您从单个 MCP 连接中获得的价值。Notion 通过从会议准备、研究、知识获取和规范到实施等独立 Skills 展示了这种模式——[[点击此处查看](https://claude.com/connectors/notion)]。
