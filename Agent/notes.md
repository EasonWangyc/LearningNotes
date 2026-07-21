# AI Agent 工具体系

> Agent 生态变化非常快。本文以 2026-07-08 前后常见的 Claude Code、Codex、OpenCode 等代码 Agent 工具为参照，整理 Agent 的核心概念、配置方式和工程组织思路。不同工具的命令和配置文件会变化，但底层问题基本稳定：模型如何获得上下文、如何调用工具、如何被约束、如何拆分任务、如何形成可重复的工作流。

## Agent 是什么

Agent 可以理解为“带工具、带状态、带执行循环的模型应用”。普通 LLM 更像一次问答：输入 prompt，输出文本；Agent 则会在一次任务中反复经历“理解目标、读取环境、调用工具、观察结果、继续决策”的过程。

```text
用户目标
  ↓
模型理解任务
  ↓
选择工具或直接回答
  ↓
执行工具：读文件 / 改文件 / 跑命令 / 查文档 / 调 API
  ↓
观察结果
  ↓
继续推理，直到完成或遇到边界
```

用最简单的伪代码表示：

```python
task = "修复 README 中失效的链接"
context = collect_initial_context(task)

while not is_finished(task, context):
    action = model.decide(task=task, context=context)

    if action.type == "tool_call":
        observation = run_tool(action)
        context.append(observation)
    else:
        final_answer = action.content
        break
```

这里最重要的变化是：模型不只是在“生成文字”，而是在一个受约束的环境里持续做决策。

### Agent 与传统脚本的区别

| 维度 | 传统脚本 | Agent |
| --- | --- | --- |
| 输入 | 固定参数、配置文件 | 自然语言目标 + 项目上下文 |
| 流程 | 预先写死 | 由模型动态选择下一步 |
| 工具 | 脚本内部调用 | 由 Agent 根据权限调用 |
| 状态 | 程序变量、文件、数据库 | 对话上下文、文件观察、工具输出、记忆 |
| 适合任务 | 稳定重复任务 | 需要理解、搜索、判断和多步修改的任务 |

例如，“把所有 `.py` 文件格式化”更适合脚本；“理解这个项目为什么 CI 失败，并只修复真正的问题”更适合 Agent。

### Code Agent 的组成

一个代码 Agent 通常由下面几部分组成：

| 组件 | 作用 |
| --- | --- |
| Model | 负责理解、规划、生成代码和解释结果 |
| Tools | 读文件、写文件、运行命令、访问浏览器、调用 MCP 等 |
| Context | 当前任务需要进入模型窗口的信息 |
| Instructions | 系统指令、用户指令、项目级 `AGENTS.md` / `CLAUDE.md` |
| Permissions | 沙箱、审批、工具白名单、危险操作限制 |
| Memory | 跨任务保留的用户偏好、项目习惯和长期事实 |
| Verifier | 测试、lint、构建、人工检查、运行结果 |

可以把它想成一个“会看项目文档、会调用工具、会受规则约束的循环系统”。

## Agentic Loop

Agentic Loop 指 Agent 执行任务时不断重复的核心循环。很多工具的能力差异最终都会落到这个循环的某个环节上。

```text
┌──────────────┐
│ 用户目标      │
└──────┬───────┘
       ↓
┌──────────────┐
│ 构建上下文    │  读取指令、目录、文件、历史消息、工具描述
└──────┬───────┘
       ↓
┌──────────────┐
│ 模型决策      │  回答、提问、调用工具、拆分任务
└──────┬───────┘
       ↓
┌──────────────┐
│ 工具执行      │  shell、文件编辑、浏览器、MCP、子代理
└──────┬───────┘
       ↓
┌──────────────┐
│ 观察结果      │  stdout、diff、测试结果、错误日志
└──────┬───────┘
       ↓
┌──────────────┐
│ 判断是否完成  │  未完成则继续循环
└──────────────┘
```

### 一次代码修改中的循环

以“修复一个 Python 函数的边界条件”为例：

```text
1. 读用户描述
2. 搜索相关函数和测试
3. 阅读实现
4. 推断失败边界
5. 修改代码
6. 运行最小测试
7. 如果失败，读取错误并继续修改
8. 如果通过，整理修改说明
```

这个过程里，模型的每一步都依赖上下文。如果上下文里没有测试失败信息，它可能只能猜；如果上下文里有真实错误日志，它就能把注意力放在实际边界上。

### Loop 与 `/loop`

需要区分两个概念：

- `agentic loop`：Agent 的基本运行机制，几乎所有 Agent 都有。
- `/loop`：某些工具中的命令或技能，用来重复执行某个 prompt，直到满足停止条件。

例如，可以把一个验证任务写成循环：

```text
/loop
每轮执行以下步骤：
1. 运行当前测试命令。
2. 如果测试通过，停止。
3. 如果测试失败，只根据第一个失败原因做最小修改。
4. 修改后继续下一轮。
最多执行 5 轮。
```

循环任务必须写清楚停止条件，否则 Agent 可能在“继续尝试”里消耗大量 token 或反复改动无关内容。

## 项目级指令：AGENTS.md、CLAUDE.md 与 README

代码 Agent 需要知道“在这个仓库里应该怎么工作”。这类信息通常放在项目级指令文件中，例如 Codex 使用 `AGENTS.md`，Claude Code 常用 `CLAUDE.md`，OpenCode 也会读取项目中的代理说明或配置。

### 三类文档的分工

| 文档 | 主要读者 | 适合写什么 |
| --- | --- | --- |
| `README.md` | 人类使用者和开发者 | 项目介绍、安装、运行、主要入口 |
| `AGENTS.md` / `CLAUDE.md` | AI Agent | 编辑规则、验证命令、目录边界、禁止操作 |
| `docs/` | 人类和 Agent | 架构说明、长流程、设计背景、专题文档 |

`AGENTS.md` 不应该变成完整教材，它更像一张仓库操作地图：告诉 Agent 先看哪里、能改哪里、改完跑什么、哪些行为需要确认。

### 指令的层级与覆盖

不同工具的文件名和层级略有不同，但思想相似：越靠近当前工作目录的指令，越应该描述局部规则。

| 工具 | 常见项目指令 | 层级思想 |
| --- | --- | --- |
| Codex | `AGENTS.md` | 从全局到项目根目录，再到当前目录逐级读取；更近的规则覆盖更远的规则 |
| Claude Code | `CLAUDE.md`、设置文件、技能描述 | 项目、用户、本地、托管等配置共同决定行为 |
| OpenCode | 项目配置、技能、插件、说明文件 | 通过配置和扩展把规则注入 Agent 流程 |

一个简单的 `AGENTS.md` 可以这样写：

```markdown
# Repository Guidelines

## Project Overview

本仓库是个人学习笔记仓库，长期维护格式为 Markdown。
默认修改 `notes.md`，只有明确需要交互实验时才修改 notebook。

## Editing Guidelines

- 中文说明为主，代码标识符使用英文。
- 代码块必须标注语言，例如 `python`、`bash`、`yaml`。
- 不要新建零散脚本文件；示例代码写入 Markdown 或 notebook。

## Verification Guidelines

- 修改 Markdown 后检查标题层级和代码块闭合。
- 新增主题时同步更新根目录 `readme.md`。

## Safety Guidelines

- 不要删除用户已有笔记。
- 不要恢复用户已经删除的 notebook。
```

这类文件越具体越好。“保持高质量”这种句子对 Agent 帮助很小；“新增主题时同步更新根目录 `readme.md`”才是可执行规则。

### 局部指令示例

大型项目可以在子目录放更具体的指令：

```text
repo/
├── AGENTS.md              # 全局规则
├── frontend/
│   └── AGENTS.md          # 前端局部规则
└── backend/
    └── AGENTS.md          # 后端局部规则
```

当 Agent 在 `frontend/` 中修改文件时，会同时考虑根目录规则和 `frontend/AGENTS.md`。这种设计可以避免根目录指令无限膨胀。

## Context：Agent 真正看到什么

Context 是 Agent 当前能用来推理的信息集合。它不等于整个仓库，也不等于电脑上的所有文件。Agent 只能基于已经进入上下文窗口的内容做判断。

### Context 的来源

常见来源如下：

| 来源 | 示例 |
| --- | --- |
| 用户消息 | “帮我重构 Agent 笔记” |
| 系统和开发者指令 | 工具权限、输出格式、仓库规则 |
| 项目指令 | `AGENTS.md`、`CLAUDE.md` |
| 文件读取结果 | `notes.md`、测试文件、配置文件 |
| 命令输出 | `git diff`、测试失败日志、目录列表 |
| 工具描述 | shell、浏览器、MCP、图片工具 |
| Skills 描述 | 可用技能名称、触发条件、文件路径 |
| Memory | 长期保存的用户偏好和项目经验 |

一个常见误解是“Agent 已经知道整个项目”。实际上，Agent 需要先搜索和读取文件，相关内容才会进入上下文。

### Context Window

Context Window 是模型一次推理能看到的最大信息量。窗口越大，不代表越应该塞入更多内容。无关内容太多会导致模型注意力分散，也会挤占真正重要的信息。

可以把上下文看成一个背包：

```text
上下文窗口
├── 必须携带：用户目标、项目指令、当前文件、错误日志
├── 可以携带：相关文档、相邻模块、测试命令
└── 不应携带：无关目录、长篇历史输出、过期推测
```

### Context Pollution

Context Pollution 指上下文被无关或低质量信息污染。比如任务只是修改一个 Markdown 文件，却把整个仓库的所有文件内容都读进来，模型就可能在不相关信息里分散注意力。

示例：

```text
任务：补充 Agent/notes.md

有用上下文：
- Agent/notes.md 当前内容
- LLM/notes.md 风格样本
- readme.md 分类结构
- 官方 Agent 工具文档

低价值上下文：
- 每个历史主题的完整笔记
- 无关 notebook 输出
- 没有来源的零散网文总结
```

### Context Rot

Context Rot 指一次任务持续太久后，上下文里积累了太多历史分支、失败尝试和过期信息，导致模型对当前真实状态的判断变差。

典型表现：

```text
1. 早期假设已经被后续命令推翻，但仍留在上下文里。
2. 多轮失败日志混在一起，模型不知道哪个是最新错误。
3. 文件已经修改，但模型还在引用旧版本内容。
```

解决思路不是“永远保留所有历史”，而是让最新事实占主导：重新读取关键文件、重新运行验证命令、用简洁总结替代过长历史。

### Compaction

Compaction 是把长对话压缩成摘要，让任务继续进行。压缩后通常会保留目标、关键决策和当前状态，但细节可能丢失。

适合保留的信息：

```markdown
## 当前目标

重构 `Agent/notes.md`，风格对齐 `LLM/notes.md`。

## 已确认事实

- 用户希望保持纯学习笔记风格，只围绕概念、机制和示例展开。
- 需要补充 subagent、loop、context 等 Agent 概念。
- `readme.md` 需要新增 Agent 入口。

## 下一步

编辑 `Agent/notes.md`，然后检查 Markdown 结构。
```

这类摘要比保留所有中间搜索输出更适合继续工作。

### Progressive Disclosure

Progressive Disclosure 可以翻译为“渐进式披露”。它的核心思想是：不要一开始把全部资料塞给模型，而是先提供目录和摘要，需要时再加载细节。

Skills 就是典型例子：

```text
启动时只加载：
- skill 名称
- description
- SKILL.md 路径

真正触发时才加载：
- 完整 SKILL.md
- references/
- scripts/
- assets/
```

这种方式可以减少上下文占用，也能让技能库变得更可扩展。

## Tools：Agent 如何影响外部世界

工具是 Agent 与外部环境交互的接口。对代码 Agent 来说，最常见的工具就是文件系统和 shell。

### 常见工具类型

| 工具类型 | 作用 |
| --- | --- |
| 文件读取 | 查看源码、文档、配置、日志 |
| 文件编辑 | 修改代码、Markdown、配置 |
| Shell | 运行测试、格式化、构建、搜索 |
| 浏览器 | 查看网页、调试前端、验证 UI |
| MCP | 连接外部系统和数据源 |
| 图片/文档工具 | 查看图片、处理 PDF、生成图像 |
| 子代理 | 把独立任务交给另一个 Agent 线程 |

### Tool Call 的生命周期

一个工具调用大致有三个阶段：

```text
PreToolUse
  ↓
Tool Execution
  ↓
PostToolUse
```

- `PreToolUse`：工具执行前，可以做权限检查、命令审查、参数补充。
- `Tool Execution`：实际执行命令或读取文件。
- `PostToolUse`：工具执行后，可以记录日志、触发验证、提取结果。

这也是 Hooks 能发挥作用的位置。

### 工具权限

Agent 工具必须受权限约束。特别是 shell 和文件编辑能力，如果不加边界，就可能误删文件、泄露密钥、连接生产环境。

一个简单的权限模型可以写成：

```yaml
tools:
  read_file: allowed
  edit_file: workspace_only
  shell:
    allowed:
      - git status
      - rg
      - pytest
    require_approval:
      - git push
      - rm
      - deploy
```

真实工具不会一定使用这个 YAML 格式，但权限思想类似：读和写分开，普通命令和高风险命令分开，自动执行和需要确认分开。

## Skills

Skill 是一组可复用的工作流说明。它通常不是代码库功能本身，而是教 Agent “遇到某类任务时应该怎么做”。

### Skill 的本质

一个 Skill 通常包含：

| 组成 | 作用 |
| --- | --- |
| `name` | 技能名称 |
| `description` | 触发条件，决定 Agent 何时使用它 |
| `SKILL.md` | 完整工作流程 |
| `references/` | 额外参考资料 |
| `scripts/` | 可复用脚本 |
| `assets/` | 模板、图片、示例文件 |

与普通文档相比，Skill 更强调“可执行工作流”。它不是解释某个概念，而是告诉 Agent 在某个任务里应该遵循哪些步骤。

### Codex Skill 目录示例

Codex 的仓库级技能通常可以放在 `.agents/skills/` 下：

```text
.agents/
└── skills/
    └── markdown-note-writer/
        ├── SKILL.md
        ├── references/
        │   └── style-guide.md
        └── templates/
            └── notes-template.md
```

一个最小 `SKILL.md` 示例：

```markdown
---
name: markdown-note-writer
description: Use when rewriting a learning note into structured Chinese Markdown.
---

# Markdown Note Writer

## Workflow

1. Read the current note and one nearby style sample.
2. Preserve useful existing content.
3. Organize the note by concepts, mechanisms, examples, and references.
4. Use Chinese explanations and language-tagged code blocks.
5. Verify heading hierarchy and fenced code blocks before finishing.
```

其中最关键的是 `description`。Agent 启动时通常只看到技能名称和描述，只有判断相关时才会读取完整技能。

### Claude Code Skill 示例

Claude Code 的技能也以 `SKILL.md` 为核心，可以附带脚本和资源。示意结构如下：

```text
.claude/
└── skills/
    └── code-review/
        ├── SKILL.md
        ├── scripts/
        └── references/
```

技能也可以在 frontmatter 中限制工具或指定执行方式：

```markdown
---
name: docs-maintainer
description: Use when editing repository documentation and validating links.
allowed-tools:
  - Read
  - Grep
  - Edit
---

# Docs Maintainer

Follow the local documentation style before editing.
```

不同工具的字段名会变化，因此学习时更应该关注思想：技能通过“描述 + 详细步骤 + 可选资源”实现可复用工作流。

### Skill 与普通 Prompt 的区别

| 维度 | 普通 Prompt | Skill |
| --- | --- | --- |
| 存放位置 | 当前对话 | 文件系统或插件包 |
| 复用方式 | 每次手写 | 自动或手动触发 |
| 上下文占用 | 直接进入当前对话 | 需要时再加载 |
| 适合内容 | 临时要求 | 稳定流程、团队规范、领域方法 |

如果某段 prompt 被反复复制，就可以考虑沉淀成 Skill。

## Commands 与 Slash Commands

很多 Agent 工具都有命令系统，例如 `/compact`、`/hooks`、`/mcp`、`/loop`。命令通常是“用户主动触发的动作入口”。

### 命令和 Skill 的关系

| 维度 | Command | Skill |
| --- | --- | --- |
| 触发 | 用户显式输入为主 | 显式或隐式触发 |
| 作用 | 执行某个工具动作或工作流入口 | 提供完整任务方法 |
| 生命周期 | 一次命令调用 | 可以跨项目长期复用 |
| 示例 | `/compact`、`/mcp`、`/hooks` | `debugging`、`docs-maintainer` |

有些工具会把自定义命令逐渐合并到技能体系中，因为二者本质上都在表达“如何让 Agent 执行一类任务”。

### 常见命令

| 命令 | 作用 |
| --- | --- |
| `/compact` | 压缩上下文，保留任务摘要 |
| `/mcp` | 查看或管理 MCP 连接 |
| `/hooks` | 查看、信任或禁用 hooks |
| `/agent` | 查看或切换 Agent / 子代理线程 |
| `/loop` | 重复执行某个 prompt，直到满足条件 |

命令适合处理“当前会话状态”，Skill 更适合沉淀“长期工作方法”。

## Plugins 与 Extensions

Plugin 和 Extension 都是把能力打包、分发、启用的方式。不同工具命名不同，但可以先按下面方式理解：

| 概念 | 更像什么 | 主要作用 |
| --- | --- | --- |
| Skill | 工作流说明书 | 教 Agent 如何做一类任务 |
| Plugin | 能力包 | 打包技能、MCP、hooks、工具配置等 |
| Extension | 分发单元 | 在 OpenCode 等工具中组织 plugins + skills |
| Hook | 生命周期拦截器 | 在特定事件前后执行逻辑 |
| MCP Server | 外部工具服务 | 给 Agent 暴露外部系统能力 |

### Codex Plugin 示例

Codex 插件通常包含 `.codex-plugin/plugin.json`，再声明技能路径、hooks、MCP 配置等内容。示意结构：

```text
my-plugin/
├── .codex-plugin/
│   └── plugin.json
├── skills/
│   └── docs-maintainer/
│       └── SKILL.md
└── hooks/
    └── hooks.json
```

一个简化的 `plugin.json`：

```json
{
  "name": "my-docs-plugin",
  "version": "0.1.0",
  "skills": [
    {
      "path": "skills"
    }
  ],
  "hooks": {
    "path": "hooks/hooks.json"
  }
}
```

实际字段以对应工具版本为准，但核心思想是：插件负责“安装和启用能力”，技能负责“描述如何工作”。

### OpenCode 中的 skills、plugins、extensions

OpenCode 生态中经常会看到如下结构：

```text
opencode/
├── skills/          # Markdown 工作流说明
├── plugins/         # JavaScript / TypeScript 运行时扩展
└── extensions/      # 可分发的组合包
```

三者关系可以理解为：

| 维度 | skills | plugins | extensions |
| --- | --- | --- | --- |
| 形式 | Markdown 文档 | 运行时代码 | 目录或包 |
| 主要内容 | 工作流和规则 | hook、注册逻辑、工具集成 | skills + plugins + 资源 |
| 作用阶段 | 模型推理前后 | Agent 运行时 | 安装和分发 |

如果只是想让 Agent 按某种方法写文档，Skill 就够了；如果要在工具调用前后自动执行代码，就需要 Plugin 或 Hook；如果要把一整套能力分享给别人，就适合打包成 Extension。

## Hooks

Hook 是 Agent 生命周期中的自动触发点。它可以在用户提交 prompt、工具调用前后、上下文压缩前后、子代理开始或结束时运行。

### Hook 的作用

常见用途：

| 用途 | 示例 |
| --- | --- |
| 记录日志 | 每次工具调用后记录命令和退出码 |
| 安全检查 | 阻止读取 `.env` 或执行危险命令 |
| 自动验证 | 修改代码后自动提示运行测试 |
| 上下文注入 | 根据目录给 Agent 补充局部说明 |
| 工作流约束 | 在任务结束前检查是否完成验证 |

Hook 适合做确定性、重复性、边界清晰的事情。复杂推理仍然应该交给模型或 Skill。

### Hook 事件示例

不同工具事件名不同，但常见阶段类似：

| 阶段 | 可能事件 | 含义 |
| --- | --- | --- |
| 会话开始 | `SessionStart` | 初始化环境、读取额外说明 |
| 用户输入 | `UserPromptSubmit` | 检查 prompt、补充上下文 |
| 工具执行前 | `PreToolUse` | 审查命令或参数 |
| 权限请求 | `PermissionRequest` | 决定是否允许高风险操作 |
| 工具执行后 | `PostToolUse` | 记录结果、触发验证 |
| 压缩前后 | `PreCompact` / `PostCompact` | 保存或恢复关键上下文 |
| 子代理开始/结束 | `SubagentStart` / `SubagentStop` | 记录子任务输入和输出 |
| 任务停止 | `Stop` | 做收尾检查 |

### Codex Hook 配置示例

示意 `hooks.json`：

```json
{
  "hooks": [
    {
      "event": "PostToolUse",
      "matcher": {
        "tool": "shell_command"
      },
      "command": "powershell -File .codex/hooks/log-command.ps1"
    }
  ]
}
```

这个例子表达的是“shell 命令执行后运行一个日志脚本”。真实项目中要注意脚本权限和可信来源。

### Claude Hook 示例

Claude Code 的 Hook 可以是 shell 命令、HTTP endpoint 或 LLM prompt。命令类 Hook 通常从 stdin 接收 JSON 上下文，再输出决策。

一个概念性示例：

```bash
#!/usr/bin/env bash

input="$(cat)"

if echo "$input" | grep -q ".env"; then
  echo '{"decision":"block","reason":"Do not access secret files."}'
else
  echo '{"decision":"allow"}'
fi
```

这个脚本展示的是 Hook 的思路：把工具调用上下文交给一个确定性检查器，由检查器决定允许还是阻止。

### Hook 与 Skill 的区别

| 维度 | Hook | Skill |
| --- | --- | --- |
| 触发方式 | 生命周期自动触发 | Agent 判断或用户显式触发 |
| 内容形式 | 脚本、HTTP、prompt、配置 | Markdown 工作流 |
| 适合任务 | 安全检查、日志、自动验证 | 分析方法、编辑流程、复杂任务策略 |
| 风险 | 可能自动执行命令 | 主要影响模型行为 |

Hook 更接近“系统机制”，Skill 更接近“工作方法”。

## MCP：Model Context Protocol

MCP 是一种把外部工具和上下文暴露给模型的协议。它解决的问题是：Agent 不应该只会读本地文件和运行 shell，还应该能通过统一接口访问数据库、文档系统、浏览器、GitHub、内部平台等。

### MCP 的基本结构

```text
Agent Client
  ↓
MCP Client
  ↓
MCP Server
  ↓
外部系统：GitHub / Docs / Database / Browser / Internal API
```

MCP Server 会向 Agent 暴露：

| 能力 | 含义 |
| --- | --- |
| tools | 可调用动作，例如搜索 issue、读取文档 |
| resources | 可读取资源，例如文件、页面、数据表 |
| prompts | 预定义提示模板 |
| instructions | 使用这个服务时的额外说明 |

### MCP 与普通 API 的区别

普通 API 是给程序员调用的；MCP 是给 Agent 调用的。它需要让模型理解“有哪些工具、参数是什么、什么时候该用、结果如何进入上下文”。

示意配置：

```toml
[mcp_servers.docs]
command = "node"
args = ["./mcp-docs-server.js"]

[mcp_servers.github]
url = "https://example.com/mcp"
auth = "oauth"
```

### MCP、Plugin、Skill 的分工

| 需求 | 更适合 |
| --- | --- |
| 让 Agent 访问外部系统 | MCP |
| 打包并分发一整套能力 | Plugin / Extension |
| 教 Agent 做一类任务 | Skill |
| 在生命周期中自动检查 | Hook |

例如，”访问公司文档库”适合 MCP；”规定读文档后如何总结”适合 Skill；”安装时一起启用文档 MCP 和总结 Skill”适合 Plugin。

## Agent 配置中的常见数据格式

Agent 工具链中大量使用 JSON、YAML、TOML 三种格式。MCP Server 配置、Hook 定义、Skill frontmatter、权限声明、API 通信都会涉及它们。理解各自的设计倾向和坑，能减少配置错误和解析问题。

### JSON

```json
{
  “mcp_servers”: {
    “github”: {
      “command”: “npx”,
      “args”: [“-y”, “@modelcontextprotocol/server-github”],
      “env”: {
        “GITHUB_TOKEN”: “${GITHUB_TOKEN}”
      }
    }
  }
}
```

**设计倾向：** 机器间数据交换。几乎所有编程语言原生支持，无需第三方库。

**适合场景：**
- LLM API 请求/响应（OpenAI、Anthropic 等所有 API 都用 JSON）
- 工具调用参数（Function Calling 的 arguments）
- MCP Server 配置（`.claude/mcp.json`）
- Hook 配置（`hooks.json`）

**优点：**
- 生态最广，Schema 约束成熟（JSON Schema、OpenAPI）
- 流式解析友好（逐行 JSONL）
- 解析速度快

**缺点：**
- 不支持注释（JSONC 部分补救）
- 多行字符串只能 `\n` 转义，手写 Prompt 非常痛苦
- 人类手写体验差——花括号、引号冗余

### YAML

```yaml
mcp_servers:
  github:
    command: npx
    args:
      - “-y”
      - “@modelcontextprotocol/server-github”
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}

# 多行 prompt 可以直接写
system_prompt: |
  You are a coding assistant.
  Always read AGENTS.md first.
  Never use emojis.
```

**设计倾向：** 人类可读优先。缩进即结构，视觉噪音最小。

**适合场景：**
- Agent 的 System Prompt 定义
- CI/CD 工作流（GitHub Actions、GitLab CI）
- Docker Compose / Kubernetes 编排
- 需要大量注释和多行文本的配置

**优点：**
- 人类可读性最高，缩进层级直观
- 原生支持注释
- 多行字符串（`|` 保留换行、`>` 折叠换行）对 Prompt 工程极其友好
- 锚点和引用语法避免重复

**缺点（重要）：**
- **缩进敏感**——一个空格错误导致整个文件解析失败
- **隐式类型转换是著名的坑**：`country: NO` 会被解析为布尔 `false`（Norway 问题）；`version: 1.0` 变成浮点数而非字符串
- 解析速度比 JSON 慢一个数量级
- 不安全的 YAML 解析器可导致任意代码执行（`!!python/object` 等标签）

### TOML

```toml
[mcp_servers.github]
command = “npx”
args = [“-y”, “@modelcontextprotocol/server-github”]

[mcp_servers.github.env]
GITHUB_TOKEN = “${GITHUB_TOKEN}”

[prompts]
system = “””
You are a coding assistant.
Always read AGENTS.md first.
“””
```

**设计倾向：** 专为配置文件而生。比 YAML 更严谨，比 JSON 更可读。

**适合场景：**
- 项目元数据/构建配置（`pyproject.toml`、`Cargo.toml`）
- 需要类型明确、不会意外转换的配置
- 团队配置（降低 YAML 缩进事故率）

**优点：**
- 不含糊的类型系统：字符串就是字符串，数字就是数字，不会隐式转换
- 日期时间是一等公民（`2026-07-08` 直接解析为日期）
- 支持注释，不像 JSON 那么冰冷
- 没有 YAML 的缩进地狱

**缺点：**
- 深层嵌套写起来冗长（重复的节标题 `[a.b.c.d]`）
- 生态不如 JSON/YAML 广泛
- 不太适合 API 数据交换场景

### 三者横向对比

```
人类手写友好度:  YAML > TOML > JSON
  解析安全性:    TOML > JSON > YAML
  生态覆盖度:    JSON > YAML > TOML
  配置文件场景:  TOML ≥ YAML > JSON
  API/数据交换:  JSON > TOML > YAML
```

| 维度 | JSON | YAML | TOML |
| --- | --- | --- | --- |
| 注释 | 不支持 | 原生支持 | 原生支持 |
| 多行字符串 | 只能 `\n` | `|` / `>` | `”””...”””` |
| 隐式类型转换 | 无 | 有（常见坑） | 无 |
| 缩进敏感 | 否 | **是** | 否 |
| 解析速度 | 快 | 中-慢 | 快 |
| Agent 生态位置 | API 通信标准 | Prompt/CI 配置 | 构建/元数据配置 |

### 在 Agent 工具中的典型分布

实际看一个 Claude Code 项目就能看到三者的分工：

```
项目根目录/
├── .claude/
│   └── mcp.json          # JSON → MCP Server 配置
├── .github/
│   └── workflows/
│       └── ci.yml        # YAML → CI 工作流
├── pyproject.toml        # TOML → 项目构建配置
└── AGENTS.md             # Markdown → Agent 行为规则
```

**简单选型法则：** 机器间通信用 JSON；人写配置优先 TOML（规避 YAML 的隐式转换坑）；多行 Prompt 和 CI 流程用 YAML。

## Subagents

Subagent 是由主 Agent 派生出来的子代理。它通常有自己的上下文窗口，可以独立搜索、分析或执行某个子任务，最后把摘要返回给主 Agent。

### 为什么需要 Subagent

主 Agent 的上下文是有限的。很多任务中，大量搜索和阅读会污染主上下文。Subagent 可以把这些“读很多、产出少”的工作隔离出去。

```text
主 Agent
  ├── 子代理 A：搜索后端错误来源
  ├── 子代理 B：阅读前端调用链
  └── 子代理 C：梳理测试失败日志

主 Agent 收到三个摘要后，再决定最终修改方案。
```

这类似把“资料搜集”和“最终决策”分开。

### Subagent 的适用场景

| 场景 | 是否适合 |
| --- | --- |
| 多个目录可以独立阅读 | 适合 |
| 需要并行搜索不同方案 | 适合 |
| 需要隔离大量日志和上下文 | 适合 |
| 多个代理同时修改同一文件 | 谨慎 |
| 任务非常小，只需改一行 | 不需要 |
| 需要严格串行推理 | 不适合 |

Subagent 最适合“读和分析”，不一定适合“同时写”。多个代理并行写同一片代码，容易产生冲突。

### Codex 中的 Subagent 思路

Codex 的 Subagent 更强调显式触发：用户或主 Agent 明确要求创建子代理时，才会派生子任务。它适合并行探索、代码审阅、文档梳理等任务。

示例 prompt：

```text
请启动三个子代理并行阅读：
1. 一个只看 Agent/notes.md 的现状。
2. 一个只看 LLM/notes.md 的结构风格。
3. 一个只看官方 Codex/Claude Agent 文档。

每个子代理只返回结构化摘要，不要直接修改文件。
```

主 Agent 收到摘要后再统一编辑文件，这样可以减少冲突。

### Claude Code 中的 Subagent 思路

Claude Code 支持内置和自定义 subagent。它可以根据任务描述自动委派，也可以由用户明确要求。Subagent 有自己的工具权限和上下文窗口。

一个概念性 subagent 定义：

```markdown
---
name: repo-explorer
description: Use for read-only exploration of a repository before editing.
tools:
  - Read
  - Grep
---

# Repo Explorer

Read relevant files and return a concise map of modules, risks, and suggested edit points.
Do not edit files.
```

这里重点不是具体字段，而是把“只读探索”变成一个可复用代理角色。

### Subagent 与 Skill 的区别

| 维度 | Skill | Subagent |
| --- | --- | --- |
| 本质 | 工作流说明 | 独立执行者 |
| 上下文 | 通常进入当前 Agent 上下文 | 通常有自己的上下文 |
| 输出 | 指导当前 Agent 行动 | 返回子任务结果摘要 |
| 适合 | 稳定方法论 | 并行探索、隔离复杂上下文 |

Skill 告诉“怎么做”，Subagent 负责“去做一部分”。

## Memory

Memory 是跨对话保存的信息。它和当前上下文不同：上下文是这次任务中模型看到的内容，Memory 是工具或系统在更长时间里保存的事实、偏好和经验。

### Memory 适合保存什么

| 内容 | 是否适合 |
| --- | --- |
| 用户偏好 | 适合，例如“偏好中文 Markdown” |
| 项目长期规则 | 适合，例如“默认修改 notes.md” |
| 反复出现的环境问题 | 适合，例如“Windows PowerShell 环境” |
| 临时命令输出 | 不适合，容易过期 |
| 密钥、账号、私人数据 | 不适合 |
| 未验证猜测 | 不适合 |

Memory 的价值在于减少重复沟通，但它必须谨慎使用。越容易变化的信息，越应该在当前任务中重新验证。

### Memory 与项目指令的区别

| 维度 | Memory | 项目指令 |
| --- | --- | --- |
| 作用范围 | 跨项目或跨会话 | 当前仓库或目录 |
| 更新方式 | 系统或用户授权记录 | 文件编辑 |
| 适合内容 | 用户习惯、长期偏好 | 仓库规则、命令、目录结构 |
| 可见性 | 不一定在仓库中 | 仓库文件可见 |

如果某条规则只对当前仓库有效，优先写入 `AGENTS.md`；如果是用户长期偏好，才适合 Memory。

## Planning 与 Spec

Agent 在复杂任务中通常需要先形成计划。计划不是为了拖慢动作，而是为了避免在不知道边界时随意修改。

### 什么时候需要计划

| 任务类型 | 是否需要 |
| --- | --- |
| 修改一个错别字 | 不需要 |
| 重构整篇学习笔记 | 需要简短计划 |
| 多文件代码改造 | 需要明确步骤 |
| 牵涉测试、构建、发布 | 需要计划和验证方式 |
| 方向还不清楚的探索任务 | 先探索，再计划 |

### 一个简短计划示例

```markdown
## Plan

1. 读取现有 `Agent/notes.md` 和风格样本。
2. 确定新笔记结构：概念、机制、配置、示例、参考资料。
3. 重写 `Agent/notes.md`。
4. 更新 `readme.md` 中的 Agent 入口。
5. 检查 Markdown 代码块、标题层级和 git diff。
```

计划的粒度应该和任务风险匹配。计划太粗会失去约束，计划太细会变成额外负担。

## 安全边界与权限

Agent 能调用工具，所以必须有明确边界。越接近真实系统、真实数据和不可逆操作，越需要审批。

### 常见边界

| 边界 | 说明 |
| --- | --- |
| 文件边界 | 只能写工作区，不能随意改系统目录 |
| 命令边界 | 破坏性命令需要确认 |
| 网络边界 | 下载依赖、访问外网要受控 |
| Git 边界 | 不随意 reset、force push、覆盖用户改动 |
| 数据边界 | 不读取密钥、不连接生产数据库 |
| Hook 边界 | 自动脚本必须可信、可审计 |
| MCP 边界 | 外部工具需要明确授权和身份 |

### 高风险动作示例

```text
需要确认：
- 删除文件或目录
- reset / checkout 覆盖修改
- 推送远程分支
- 部署服务
- 连接生产数据库
- 安装或更新全局依赖
- 执行来自不可信来源的 hook 或 plugin
```

安全边界不是为了限制 Agent，而是为了让它在可控范围内更可靠地工作。

## 一个完整 Agent 工作流示例

以“重构学习笔记”为例，可以把前面的概念串起来：

```text
用户目标
  ↓
读取 AGENTS.md，确认仓库笔记规范
  ↓
读取目标 notes.md 和风格样本
  ↓
如果主题变化快，读取官方资料
  ↓
整理结构计划
  ↓
编辑 Markdown
  ↓
更新 readme.md 入口
  ↓
检查代码块、标题层级、git diff
  ↓
给出修改说明和资料来源
```

对应到 Agent 组件：

| 步骤 | 对应组件 |
| --- | --- |
| 读取仓库规则 | Project Instructions |
| 阅读目标文件 | Tools + Context |
| 使用官方资料 | Browser / Web / MCP |
| 套用笔记风格 | Skill / Memory / User Preference |
| 修改文件 | Edit Tool |
| 检查 diff | Shell Tool |
| 汇报结果 | Model |

这种拆解可以帮助理解：Agent 不是一个单一能力，而是一组机制组合起来的工作系统。

## 学习路线

Agent 相关概念很多，建议按下面顺序学习：

```text
1. Agentic Loop
   理解 Agent 为什么能多步行动。

2. Context
   理解模型到底看到什么，为什么要控制上下文。

3. Project Instructions
   学习 AGENTS.md / CLAUDE.md 如何约束 Agent。

4. Tools 与 Permissions
   理解工具调用和权限边界。

5. Skills
   学习如何沉淀可复用工作流。

6. Hooks
   学习如何在生命周期中自动执行检查。

7. MCP
   学习如何连接外部系统。

8. Plugins / Extensions
   学习如何分发整套能力。

9. Subagents
   学习如何并行探索和隔离上下文。

10. Loop / Evaluation
    学习如何让 Agent 反复验证并收敛。
```

如果只想先动手，可以从写一个好的 `AGENTS.md` 和一个简单 `SKILL.md` 开始。这两个文件最能体现 Agent 工具的核心思想：把隐含经验变成可执行规则。

## 参考资料

- [Claude Code Overview](https://code.claude.com/docs/en/overview)
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Hooks](https://code.claude.com/docs/en/hooks)
- [Claude Code Context Window](https://code.claude.com/docs/en/context-window)
- [Claude Code Settings](https://code.claude.com/docs/en/settings)
- [Codex AGENTS.md](https://developers.openai.com/codex/guides/agents-md)
- [Codex Skills](https://developers.openai.com/codex/skills)
- [Codex Plugins](https://developers.openai.com/codex/plugins/build)
- [Codex Hooks](https://developers.openai.com/codex/hooks)
- [Codex Subagents](https://developers.openai.com/codex/subagents)
- [Codex MCP](https://developers.openai.com/codex/mcp)
