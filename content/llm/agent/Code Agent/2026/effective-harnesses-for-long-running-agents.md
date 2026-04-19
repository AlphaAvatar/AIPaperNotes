# Effective harnesses for long-running agents

论文链接：https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

代码链接：

## 摘要

随着人工智能 Agent 能力的不断提升，开发者们越来越多地要求它们承担复杂的任务，这些任务可能需要数小时甚至数天才能完成。然而，如何让 Agent 在多个上下文窗口中持续取得进展仍然是一个尚未解决的问题。

**长时间运行的智能体面临的核心挑战在于，它们必须在离散的会话中工作，每个新会话开始时都对之前的工作一无所知**。想象一下，一个软件项目由工程师轮班工作，每个新工程师都对上一班的工作内容毫无记忆。由于上下文窗口有限，而且大多数复杂项目无法在单个窗口内完成，智能体需要一种方法来弥合编码会话之间的空白。

我们开发了一套双重解决方案，使 Claude Agent SDK 能够在多个上下文窗口中高效运行：一个**初始化 Agent**，用于在首次运行时设置环境；以及一个**编码 Agent**，负责在每个会话中逐步推进，同时为下一个会话留下清晰的痕迹。您可以在随附的快速入门指南中找到代码示例。

## The long-running agent problem

Claude Agent SDK 是一款功能强大的通用 Agent 框架，它不仅擅长编码，还能胜任其他需要模型使用工具来收集上下文、规划和执行任务的工作。它具备上下文管理功能，例如上下文压缩，使 Agent 能够在不耗尽上下文窗口的情况下完成任务。理论上，在这种配置下，Agent 应该能够持续执行任意长时间的有效工作。

然而，压缩是不够的。即使是像 Opus 4.5 这样前沿的编码模型，如果仅仅给出诸如“构建 claude.ai 的克隆”之类的高级提示，并在多个上下文窗口中循环运行于 Claude Agent SDK 上，也无法构建出生产级的 Web 应用程序。

**Claude 的失败主要体现在两个方面**：
- 首先，Agent 倾向于一次性执行过多操作——本质上是试图一次性完成应用程序。这通常会导致模型在执行过程中失去上下文，使得下一个会话开始时，某个功能只实现了一半，而且没有文档说明。Agent 随后不得不猜测发生了什么，并花费大量时间试图让应用程序的基本功能重新运行。即使使用了压缩机制，这种情况仍然会发生，因为压缩机制并非总是能将清晰明确的指令传递给下一个 Agent。
- 第二种故障模式通常会在项目后期出现。在某些功能已经构建完成后，后续的 Agent 实例会检查项目进展，发现已经取得了一些进展，然后宣布任务完成。

这会将问题分解为两部分。首先，我们需要搭建一个初始环境，为给定提示所需的所有功能奠定基础，使 Agent 能够逐步、逐个功能地完成任务。其次，我们应该引导每个 Agent 逐步实现其目标，同时在会话结束时将环境清理干净。所谓“清理干净”的状态，是指代码适合合并到主分支：没有重大错误，代码结构清晰、文档齐全，通常情况下，开发人员可以轻松地开始开发新功能，而无需先清理无关的混乱代码。

在内部试验中，我们采用两步解决方案来解决这些问题：
1. **Initializer agent**：第一个 Agent 会话使用一个特殊的提示，要求模型设置初始环境：一个 `init.sh` 脚本、一个 `claude-progress.txt` 文件（用于记录 Agent 执行的操作）以及一个初始 git 提交（用于显示已添加的文件）。
2. **Coding agent**：后续每次会话都要求模型逐步取得进展，然后提供结构化的更新信息（我们之所以将它们称为不同的 Agent，仅仅是因为它们的初始用户提示不同。除此之外，系统提示、工具集和整个 Agent 框架都完全相同）。

关键在于找到一种方法，让 Agent 在打开一个全新的上下文窗口时能够快速了解​​工作状态，这可以通过 `claude-progress.txt` 文件以及 Git 历史记录来实现。**这些实践的灵感来源于了解高效软件工程师的日常工作**。

## Environment management

在更新后的 Claude 4 提示指南中，我们分享了一些多上下文窗口工作流程的最佳实践，其中包括一种框架结构，该结构使用“针对第一个上下文窗口的不同提示”。这种“不同提示”要求初始化 Agent 设置环境，使其包含后续编码 Agent 有效工作所需的所有必要上下文。本文将深入探讨此类环境的一些关键组件。

### Feature list

为了解决 Agent 程序一次性完成应用程序或过早认为项目已完成的问题，我们要求初始化 Agent 程序编写一份全面的功能需求文件，以扩展用户的初始提示。在 claude.ai 克隆示例中，这意味着超过 200 项功能，例如“用户可以打开一个新的聊天窗口，输入查询内容，按下回车键，然后查看 AI 回复”。这些功能最初都被标记为“未完成”，以便后续的编码 Agent 程序能够清楚地了解完整功能应有的样子。

我们仅通过更改 `pass` 字段的状态来引导编码 Agent 编辑此文件，并使用措辞强烈的指令，例如“删除或编辑测试是不可接受的，因为这可能导致功能缺失或出现错误”。经过一些实验，我们最终决定使用 JSON 格式，因为与 Markdown 文件相比，模型不太可能错误地更改或覆盖 JSON 文件。

### Incremental progress

在搭建好初始环境框架后，下一轮编码 Agent 被要求每次只处理一个功能。这种增量式方法对于解决 Agent 一次性处理过多任务的倾向至关重要。

即使采用增量式开发，模型在每次代码更改后保持环境的清洁状态仍然至关重要。**在我们的实验中，我们发现实现这一目标的最佳方法是要求模型使用描述性的提交信息将其开发进度提交到 Git，并在进度文件中写入进度摘要**。这样，模型就可以使用 Git 回滚错误的代码更改，并恢复代码库的正常状态。

这些方法也提高了效率，因为它们消除了 Agent 需要猜测发生了什么事，并花费时间试图让基本应用程序重新运行的需要。

### Testing

我们观察到的最后一个主要故障模式是 Claude 倾向于在没有进行充分测试的情况下将某个功能标记为已完成。如果没有人明确提醒，Claude 往往会修改代码，甚至使用单元测试或 `curl` 命令在开发服务器上进行测试，但他却无法意识到该功能并未全面正常运行。

在构建 Web 应用程序的过程中，一旦明确要求 Claude 使用浏览器自动化工具并像人类用户一样进行所有测试，Claude 在端到端验证功能方面大多表现良好。

为 Claude 提供这类测试工具显著提高了性能，因为该 Agent 能够识别并修复仅从代码中无法明显看出的错误。

仍然存在一些问题，例如 Claude 的视野和浏览器自动化工具的局限性，导致难以识别所有类型的错误。例如，Claude 无法通过 Puppeteer MCP 查看浏览器原生弹出框，因此依赖这些弹出框的功能往往更容易出现错误。

## Getting up to speed

完成以上所有步骤后，每个编码 Agent 都会被提示执行一系列步骤来熟悉操作，其中一些步骤虽然很基础，但仍然很有帮助：
1. 运行 `pwd` 命令查看当前工作目录。您只能编辑此目录中的文件。
2. 阅读 git 日志和进度文件，了解最近的工作进展。
3. 阅读功能列表文件，选择尚未完成的最高优先级功能进行开发。

这种方法可以节省 Claude 在每个会话中的一些 token，因为它无需自行计算如何测试代码。此外，还可以让初始化 Agent 编写一个 `init.sh` 脚本来运行开发服务器，然后在实现新功能之前运行一个基本的端到端测试。

以 claude.ai 克隆版为例，这意味着 Agent 程序始终启动本地开发服务器，并使用 **Puppeteer MCP** 发起新的聊天、发送消息并接收回复。这确保了 Claude 能够快速识别应用程序是否处于故障状态，并立即修复任何现有错误。如果 Agent 程序转而开始实现新功能，则很可能会使问题更加严重。

鉴于以上情况，典型的会话开始时会显示以下助手消息：

```
[Assistant] I'll start by getting my bearings and understanding the current state of the project.
[Tool Use] <bash - pwd>
[Tool Use] <read - claude-progress.txt>
[Tool Use] <read - feature_list.json>
[Assistant] Let me check the git log to see recent work.
[Tool Use] <bash - git log --oneline -20>
[Assistant] Now let me check if there's an init.sh script to restart the servers.
<Starts the development server>
[Assistant] Excellent! Now let me navigate to the application and verify that some fundamental features are still working.
<Tests basic functionality>
[Assistant] Based on my verification testing, I can see that the fundamental functionality is working well. The core chat features, theme switching, conversation loading, and error handling are all functioning correctly. Now let me review the tests.json file more comprehensively to understand what needs to be implemented next.
<Starts work on a new feature>
```

## Future work

这项研究展示了一种可能的解决方案，用于构建一个能够让模型在多个上下文窗口中逐步推进的长期运行 Agent 框架。然而，仍有一些问题悬而未决。

最值得注意的是，目前尚不清楚单个通用编码 Agent 是否在所有上下文中都能发挥最佳性能，或者多 Agent 架构是否能带来更佳的性能。诸如测试 Agent、质量保证 Agent 或代码清理 Agent 之类的专用Agent，在软件开发生命周期的各个子任务中可能表现更佳。

此外，此演示针对全栈 Web 应用开发进行了优化。未来的研究方向是将这些发现推广到其他领域。这些经验教训很可能可以应用于科学研究或金融建模等领域所需的长期运行 Agent 任务。
