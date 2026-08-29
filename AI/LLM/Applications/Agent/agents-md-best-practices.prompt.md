# AGENTS.md 编写最佳实践

`AGENTS.md` 是写给 AI 编程助手和自动化代理的仓库级协作说明。更准确地说，它应该是 AI agent 的“仓库操作地图”：帮助代理更安全、更一致地理解项目、修改代码、运行检查和提交变更，而不是替代 `README.md`、架构文档或人工入职手册。

一份好的 `AGENTS.md` 应该短、明确、可执行。它描述“代理在这个仓库里应该怎么工作”，而不是完整解释“这个项目是什么”。

## 仓库操作地图

把 `AGENTS.md` 当成操作地图时，重点不是把所有知识都搬进去，而是标出代理最容易走错、也最需要快速判断的路线：

- 入口：开始工作前应先读哪些文档、配置和关键源码。
- 路径：不同目录、包或服务分别负责什么，应该到哪里改；必要时写到三级目录，标出真正的代码职责边界。
- 命令：不同变更类型对应哪些构建、测试、格式化和验证命令。
- 步骤：新增模块、Provider、API、集成、迁移、发布流程时有哪些固定 checklist。
- 禁区：哪些文件、数据、服务和 Git 操作默认不能碰，哪些必须先确认。
- 文档：更细的开发、运维、架构和排障内容在哪里，不要把所有细节塞进 `AGENTS.md`。

这张地图不需要解释每一行代码，但要让代理知道下一步该去哪、该跑什么、什么地方不能猜。

### 地图化写法

写 `AGENTS.md` 时，可以把内容从“说明文”改成“路线图”：

- 不只写“项目使用 Go”，而是写“业务入口在 `cmd/`，核心逻辑在 `internal/`，修改后运行 `go test ./...`”。
- 不只写“项目有接口测试”，而是写“接口测试在 `tests/api/`，用 `uv run pytest tests/api`，需要先启动本地服务”。
- 不只写“遵循架构”，而是写“新增数据库表必须新增 migration、更新 repository、补充集成测试”。
- 不只写“注意安全”，而是写“不要读取 `.env`，不要连接生产数据库，部署和迁移必须先确认”。

判断一条规则是否值得写入，可以看它是否能帮助代理回答三个问题：我应该看哪里、我应该运行什么、我不能做什么。

## 文档地图与拆分

`AGENTS.md` 不应该无限膨胀。它应该是“地图”和“路标”，不是完整手册。更细的开发流程、运维 runbook、架构说明、排障步骤和 CI/CD 细节，应拆到 `docs/` 中，并通过索引串起来。

详细的文档体系设计、拆分边界和示例索引，见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)。`AGENTS.md` 中只保留摘要和跳转入口。

推荐的最小文档地图结构：

```text
.
├── AGENTS.md              # AI agent 的仓库操作地图：入口、路径、命令、禁区和关键跳转
├── README.md              # 项目根文档：项目简介、快速上手、常用入口和 docs 索引
└── docs
    ├── README.md          # docs 目录简介和文档索引
    ├── ARCHITECTURE.md    # 文档体系架构和拆分规则
    ├── development        # 面向开发人员：本地开发、测试、架构、编码规范、CI 说明
    │   ├── README.md      # development 文档索引
    │   └── ...
    └── ops                # 面向使用人员/运维人员：部署、监控、排障、备份恢复、发布
        ├── README.md      # ops 文档索引
        └── ...
```

保留在 `AGENTS.md` 的内容应满足以下条件：

- 如果一段内容是稳定规则、风险边界或常用命令，放在 `AGENTS.md`。
- 如果一段内容需要背景解释、步骤较长、包含多个场景或经常需要人类阅读，放到 `docs/`。
- 如果 `AGENTS.md` 中某节超过几段，优先保留摘要和链接，把细节移到 `docs/development/` 或 `docs/ops/`。
- 如果修改某类代码必须参考详细流程，在 `AGENTS.md` 写“先读哪篇 docs”，不要复制整篇流程。

## 核心原则

1. 面向操作，不面向宣传

   `AGENTS.md` 应写成仓库操作地图：读哪些文档、改哪些文件、跑哪些命令、避开哪些风险。不要写愿景、产品介绍或长篇背景。

2. 保持仓库特定

   只记录本仓库内会影响代理行为的约定。通用工程常识不必重复，例如“写清晰代码”“遵守最佳实践”这类表述通常没有执行价值。

3. 与人工文档分工

   `README.md` 负责项目介绍、安装、运行和常规使用。架构文档负责系统设计。`AGENTS.md` 负责代理协作边界、编辑规则、验证命令和安全注意事项。

4. 保持短小，链接细节

   `AGENTS.md` 只保留 agent 决策所需的地图、规则和入口。长流程、背景解释和 runbook 放进 `docs/`，并在 `AGENTS.md` 中提供明确链接。

5. 指令必须可验证

   优先写具体命令、路径、文件模式和审批条件。例如“修改后运行 `npm test`”比“确保测试通过”更有用。

6. 默认保护用户和生产环境

   对密钥、数据库、生产服务、远程部署、数据迁移、破坏性命令等高风险操作写清楚默认禁止、需要确认或必须 dry-run 的条件。

## 项目类型差异

这份最佳实践默认面向个人项目和团队内部项目。公开开源项目需要额外的社区协作规则，但这些内容不应成为默认模板，否则会让个人和团队项目变得过重。

| 内容 | 个人/团队项目 | 公开开源项目 |
| --- | --- | --- |
| `CONTRIBUTING.md` | 默认不需要；开发流程放 `docs/development/`。 | 需要时添加，用于外部贡献者入门。 |
| Issue / PR 模板 | 默认不需要；团队可用项目管理系统或内部规范。 | 常见需要，用于规范外部反馈和贡献。 |
| `CODE_OF_CONDUCT.md` | 默认不需要。 | 社区型项目通常需要。 |
| `SECURITY.md` / 漏洞披露政策 | 内部项目可放到安全或运维文档。 | 公开项目建议提供。 |
| `SUPPORT.md` | 默认不需要。 | 面向用户或社区支持时可添加。 |
| CLA / DCO / 贡献者授权 | 默认不需要。 | 有外部贡献和法律要求时使用。 |
| issue labeler / stale bot | 默认不需要。 | issue 量较大时可作为维护辅助。 |
| 维护者名单和治理规则 | 默认不需要；团队职责可放内部文档。 | 多维护者社区项目可添加。 |

写 `AGENTS.md` 时，不要把开源社区治理内容默认塞进去。只有当仓库真的接受外部贡献、公开发布包或需要社区维护流程时，才加入相关入口，并把细节放在专门文档中。

## 推荐结构

可以从下面结构开始，根据项目规模删减。这里的 `markdown` 模板保留中英文标题，方便直接复制到项目的 `AGENTS.md`。

```markdown
# Repository Guidelines / 仓库协作指南

## Project Overview / 项目概览

中文：用几句话说明项目类型、主要技术栈、关键入口文档，以及代理开始工作前应该先读什么。
English: Describe the project type, main technologies, key entry documents, and what an agent should read before starting work.

## Code Organization / 代码组织

中文：说明重要目录的职责，或指向已有文档。避免在这里维护一份容易过期的完整目录树。
English: Explain important directory responsibilities or link to existing docs. Avoid maintaining a full directory tree that will quickly become stale.

## Documentation Map / 文档地图

中文：说明 `README.md`、`docs/README.md`、`docs/development/`、`docs/ops/` 和关键详细文档的入口。
English: Point to `README.md`, `docs/README.md`, `docs/development/`, `docs/ops/`, and key detailed documents.

## Editing Guidelines / 编辑规范

中文：说明代码风格、配置修改原则、生成文件处理方式、兼容性要求、禁止触碰的区域。
English: Document code style, config editing rules, generated-file handling, compatibility requirements, and areas that should not be touched.

## Verification Guidelines / 验证规范

中文：列出常用检查命令、测试命令、格式化命令，以及不同变更类型对应的验证方式。
English: List common check, test, and formatting commands, plus verification steps for different change types.

## CI/CD Guidelines / CI/CD 工作流

中文：说明 CI 自动测试、CD 自动发布和辅助工作流的入口、触发条件、质量门禁和修改限制。
English: Describe CI testing, CD publishing, utility workflows, triggers, quality gates, and editing restrictions.

## Security & Safety Guidelines / 安全规范

中文：说明密钥、环境变量、生产数据、外部服务、部署、数据库和破坏性命令的处理规则。
English: Define rules for secrets, environment variables, production data, external services, deployments, databases, and destructive commands.

## Git Guidelines / Git 规范

中文：说明分支、提交、变基、推送、代码评审、生成文件和大文件处理约定。
English: Document branch, commit, rebase, push, review, generated-file, and large-file conventions.
```

小型仓库可以只保留 `Project Overview`、`Editing Guidelines`、`Verification Guidelines` 和 `Security & Safety Guidelines`。大型仓库可以在子目录放置更具体的 `AGENTS.md`，让规则就近生效。

## 各部分怎么写

### Project Overview / 项目概览

这一节回答代理开始前最需要知道的问题：

- 这个仓库是什么类型：前端应用、后端服务、库、基础设施配置、数据管道、移动端应用等。
- 主要技术栈是什么。
- 哪些文档是事实来源，例如 `README.md`、`docs/architecture.md`、`docs/development/README.md`。
- 代理在动手前应优先读取哪些文件。

示例：

```markdown
## Project Overview / 项目概览

中文：本仓库是基于 Next.js 和 PostgreSQL 构建的 TypeScript Web 应用。使用 `README.md` 了解安装和运行方式，使用 `docs/architecture.md` 了解服务边界，使用 `docs/api.md` 了解公共 API 行为。修改认证、计费或数据访问逻辑前，先阅读 `docs/adr/` 下的相关设计说明。

English: This repository contains a TypeScript web application built with Next.js and PostgreSQL. Use `README.md` for setup, `docs/architecture.md` for service boundaries, and `docs/api.md` for public API behavior. Before changing authentication, billing, or data access logic, read the relevant design note under `docs/adr/`.
```

### Code Organization / 代码组织

这一节只写会影响编辑判断的信息。不要复制完整目录树，避免文档频繁过期；也不要只停留在一级或二级目录。如果三级目录承载了稳定的架构边界，就应该写清楚。

推荐粒度：

- 一级目录说明大系统或大包，例如 `backend/`、`frontend/`、`infra/`。
- 二级目录说明模块或子项目，例如 `backend/api/`、`frontend/admin/`、`packages/sdk/`。
- 三级目录说明 agent 实际改代码时最容易混淆的职责边界，例如 `internal/order/handler/`、`internal/order/service/`、`tests/api/fixtures/`。

三级目录适合写这些内容：

- 分层职责：`handler/`、`service/`、`repository/`、`domain/`、`model/`、`schema/`、`client/`。
- 测试职责：`tests/unit/`、`tests/integration/`、`tests/api/fixtures/`、`tests/api/clients/`。
- 生成与契约：`api/proto/generated/`、`openapi/generated/`、`src/generated/`。
- 平台或部署：`infra/terraform/modules/`、`deploy/kubernetes/base/`、`deploy/kubernetes/overlays/`。

不需要展开每个三级目录；只写那些能帮助代理判断“这里该不该改、改完跑什么、是否生成或高风险”的目录。

示例：

```markdown
## Code Organization / 代码组织

- 中文：`src/app/` 包含路由处理和页面级 UI。
  English: `src/app/` contains route handlers and page-level UI.
- 中文：`src/app/api/` 包含 HTTP route handler；业务逻辑应放在 `src/lib/`。
  English: `src/app/api/` contains HTTP route handlers; keep business logic in `src/lib/`.
- 中文：`src/components/` 包含共享 UI 组件。
  English: `src/components/` contains shared UI components.
- 中文：`src/lib/services/` 包含业务流程。
  English: `src/lib/services/` contains business workflows.
- 中文：`src/lib/repositories/` 包含持久化访问；测试应使用 fixture，不要使用生产数据。
  English: `src/lib/repositories/` contains persistence access; tests should use fixtures instead of production data.
- 中文：`src/lib/clients/` 包含外部服务客户端。
  English: `src/lib/clients/` contains external service clients.
- 中文：`migrations/` 包含数据库迁移；除非用户明确要求，不要编辑已应用的迁移。
  English: `migrations/` contains database migrations; do not edit applied migrations unless explicitly requested.
```

如果项目已经有维护良好的架构文档，可以直接指向它：

```markdown
中文：目录归属和服务边界以 `docs/architecture.md` 为准；不要在 `AGENTS.md` 中重复维护完整目录地图。

English: Directory ownership and service boundaries are documented in `docs/architecture.md`; update that file instead of duplicating the directory map here.
```

### Editing Guidelines / 编辑规范

这一节应把“代理容易做错”的事写清楚。

适合写入：

- 使用现有框架、工具和本地抽象，不随意引入新依赖。
- 配置文件只改必要字段，避免整文件重排。
- 不编辑生成文件，除非说明生成命令。
- 数据库迁移、API schema、国际化文案、设计 token 等变更的专门规则。
- 保持向后兼容的要求。
- 哪些路径由其他系统管理，不应手改。

示例：

```markdown
## Editing Guidelines / 编辑规范

中文：遵循相邻文件的既有风格。优先使用 `src/lib/` 中的现有 helper，不要轻易新增工具函数。除非变更需要，不要重排整个 JSON、YAML 或 lockfile。

English: Follow the existing style in nearby files. Prefer existing helpers in `src/lib/` over introducing new utilities. Do not reformat entire JSON, YAML, or lock files unless the change requires it.

中文：不要直接编辑 `src/generated/` 下的生成文件。应运行 `npm run codegen` 重新生成，并在需要时同时提交源文件和生成结果。

English: Do not edit generated files under `src/generated/` directly. Regenerate them with `npm run codegen` and commit both source and generated output when required.
```

### Verification Guidelines / 验证规范

这一节是 `AGENTS.md` 最有价值的部分之一。它应该告诉代理改完以后跑什么，而不是只说“运行测试”。

建议按变更类型列出：

````markdown
## Verification Guidelines / 验证规范

中文：优先运行最小且相关的检查：

English: Use the narrowest relevant check first:

```bash
npm run lint
npm run typecheck
npm test
```

中文：UI 变更还要运行：

English: For UI changes, also run:

```bash
npm run test:e2e
```

中文：数据库变更应先在一次性本地数据库上运行迁移，再修改依赖新 schema 的应用代码。

English: For database changes, run migrations against a disposable local database before editing application code that depends on them.
````

如果某些检查很慢、不稳定或需要外部凭据，也应说明：

```markdown
中文：`npm run test:integration` 需要本地服务凭据。除非凭据已经配置好或用户明确要求，否则不要运行。

English: `npm run test:integration` requires local service credentials. Do not run it unless credentials are already configured or the user explicitly asks for it.
```

### CI/CD Guidelines / CI/CD 工作流

CI/CD 是项目质量保证的一部分，尤其适合写进 `AGENTS.md`。本地验证告诉代理“改完要跑什么”，CI/CD 地图告诉代理“代码评审、合并和发布真正由哪些自动化流程把关”。

建议把 CI/CD 拆成三类写：

- CI Workflows (automated testing)：PR/MR 和 push 上的自动测试、lint、typecheck、build、覆盖率、矩阵测试、必需状态检查。
- CD Workflows (automated publishing)：release、tag、手动 dispatch、部署、包发布、镜像发布、环境审批、OIDC 或密钥使用方式。
- Utility Workflows：文档索引检查、依赖更新、代码生成校验、安全扫描。`issue labeler`、`stale bot` 只适合公开开源项目或 issue 量很大的项目。

适合写入：

- workflow 文件名和用途。
- 触发条件，例如 `pull_request`、`push`、`release: published`、tag prefix、`workflow_dispatch`。
- 哪个 workflow 是 PR/MR 的总入口或必需状态检查。
- 修改某类代码时会触发哪些 workflow。
- 新增包、模块或服务时，需要把哪些 workflow、path filter、required check、发布路由一起更新。
- 哪些 workflow 文件名或 job 名称不能随意改，因为分支保护、registry trusted publishing 或外部系统依赖它们。

如果项目不用 GitHub Actions，也应写清对应平台的 pipeline、job、stage、required check、发布入口和审批规则，例如 GitLab CI、CircleCI、Buildkite 或 Jenkins。

示例：

```markdown
## CI/CD Guidelines / CI/CD 工作流

### CI Workflows (automated testing) / CI 工作流（自动化测试）

中文：`ci.yml` 在每个 pull request 或 merge request 上运行，是分支保护必需检查。包级 workflow 通过 path filter 调用；新增包时，同时更新 path filter 和聚合必需检查。

English: `ci.yml` runs on every pull request or merge request and is the required branch-protection check. Package-specific workflows are invoked through path filters; when adding a new package, update both the path filter and the aggregate required check.

### CD Workflows (automated publishing) / CD 工作流（自动化发布）

中文：`release.yml` 是唯一发布入口。它把 release tag 路由到包级发布 workflow。未经明确批准，不要修改 tag prefix、workflow 文件名、发布权限或 trusted-publisher 配置。

English: `release.yml` is the only publishing entry point. It routes release tags to package-specific publish workflows. Do not change tag prefixes, workflow filenames, publishing permissions, or trusted-publisher settings without explicit approval.

### Utility Workflows / 辅助工作流

中文：`docs-check.yml` 校验文档索引，`dependency-review.yml` 检查依赖变更。新增文档页、生成文件或依赖清单时，同步更新这些 workflow 的覆盖范围。

English: `docs-check.yml` validates documentation indexes. `dependency-review.yml` checks dependency changes. Keep these workflows in sync when adding docs pages, generated files, or new dependency manifests.
```

### Security & Safety Guidelines / 安全规范

这一节要写得具体、保守。代理需要知道哪些信息不能读取、不能输出、不能提交，哪些操作必须先确认。

建议覆盖：

- 密钥和环境变量文件。
- 生产数据库和真实用户数据。
- 证书、私钥、token、cookie、会话文件。
- 部署、重启、迁移、删除数据、重写历史等高风险操作。
- 日志和错误输出的脱敏规则。

示例：

```markdown
## Security & Safety Guidelines / 安全规范

中文：不要在对话、issue、提交说明或日志中输出密钥、token、私钥、cookie 或生产凭据。将 `.env*`、`secrets/`、证书私钥和服务账号文件视为敏感文件。

English: Do not print secrets, tokens, private keys, cookies, or production credentials in chat, issues, commit messages, or logs. Treat `.env*`, `secrets/`, certificate private keys, and service account files as sensitive.

中文：除非用户在当前对话中明确要求，否则不要执行生产部署、数据库迁移、破坏性清理命令或服务重启。外部变更前优先使用 dry-run 或状态检查。

English: Do not run production deploys, database migrations, destructive cleanup commands, or service restarts unless the user explicitly asks for that action in the current conversation. Prefer dry-run or status commands before making external changes.
```

对于确实需要代理编辑密钥模板的项目，应区分模板和真实值：

```markdown
中文：可以编辑 `.env.example` 来记录必需变量。真实 `.env` 文件不得被读取、输出或提交，除非用户明确要求本地编辑；回复中仍不得包含明文密钥值。

English: `.env.example` may be edited to document required variables. Real `.env` files must not be read, printed, or committed unless the user explicitly requests a local edit, and responses must not include raw secret values.
```

### Git Guidelines / Git 规范

Git 规则应贴合团队流程，不要假设所有项目都一样。

适合说明：

- 默认分支和分支命名。
- 是否允许自动提交。
- 是否允许 amend、rebase、force push。
- 是否允许推送远程。
- lockfile 和生成文件是否随变更提交。
- PR/MR 标题、说明和测试记录格式；个人项目可省略。

示例：

```markdown
## Git Guidelines / Git 规范

中文：除非用户要求，不要创建提交。提交前运行 `git status` 并审阅 diff。未经明确批准，不要使用 `git reset --hard`、`git clean`、force push 或重写共享历史。

English: Do not create commits unless the user asks. Before committing, run `git status` and review the diff. Never use `git reset --hard`, `git clean`, force push, or rewrite shared history without explicit user approval.

中文：依赖变更时提交 lockfile。不要包含无关格式化改动或本地环境文件。

English: Include lockfile changes when dependencies change. Do not include unrelated formatting churn or local environment files.
```

## 多层 AGENTS.md

对于大型仓库，可以在子目录放置更具体的 `AGENTS.md`。常见用法：

- 根目录 `AGENTS.md`：全仓库通用规则。
- `frontend/AGENTS.md`：前端框架、UI、测试和设计系统规则。
- `backend/AGENTS.md`：API、数据库、队列、迁移和服务边界规则。
- `infra/AGENTS.md`：Terraform、Kubernetes、部署和云资源安全规则。

子目录规则应补充而不是重复根目录规则。写子目录 `AGENTS.md` 时，重点说明该目录内的特殊风险和验证命令。

如果三级目录已经形成稳定职责边界，也可以写到根 `AGENTS.md` 的操作地图中，或在该三级目录放置局部 `AGENTS.md`。例如 `backend/internal/order/AGENTS.md` 可以说明 `handler/`、`service/`、`repository/` 的职责和测试命令；`tests/api/AGENTS.md` 可以说明 `clients/`、`fixtures/`、`test_<resource>.py` 的写法。

## 不建议写入的内容

避免把这些内容放进 `AGENTS.md`：

- 长篇产品背景、路线图或营销说明。
- 完整 API 文档、完整架构说明或完整目录树。
- 详细开发手册、部署 runbook、排障手册和长篇 CI/CD 设计说明。
- 会快速过期的任务清单。
- 与代理行为无关的团队文化口号。
- 明文密钥、内部账号、真实密码、访问 token。
- 过于宽泛的要求，例如“保证代码质量”“使用最佳实践”“不要犯错”。

这些内容应放在更合适的位置：`README.md`、`docs/`、ADR、runbook、issue 或项目管理系统。公开开源项目可另设 `CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md` 等社区文档；个人或团队内部项目默认不需要。

## 质量检查清单

提交或更新 `AGENTS.md` 前，可以用这份清单检查：

- 是否能让代理在 1 到 2 分钟内理解项目工作边界。
- 是否区分了个人/团队项目和公开开源项目，不把 `CONTRIBUTING.md` 等社区治理文件当成默认要求。
- 是否写清了关键二级目录和必要三级目录的职责边界。
- 是否把详细手册拆到 `docs/`，并在 `AGENTS.md` 中只保留摘要和入口链接。
- 是否存在 `README.md`、`docs/README.md`、`docs/development/README.md`、`docs/ops/README.md` 这类清晰索引。
- 是否列出了最常用、最可信的验证命令。
- 是否明确了敏感文件和高风险操作的处理方式。
- 是否说明了 CI/CD 的质量门禁、发布入口和辅助工作流。
- 是否避免复制 README、架构文档或目录树。
- 是否没有包含密钥、密码、token 或私钥。
- 是否删除了空泛、不可执行的口号式表述。
- 是否说明了哪些操作需要用户明确确认。
- 是否与当前项目实际工具链一致。

## 可复用模板

下面模板可以直接复制为项目的 `AGENTS.md` 初稿。模板使用中文和英文并列写法：中文方便本地团队快速理解，英文方便国际化协作和多数 AI 编程工具稳定识别。复制后应按真实目录、工具链和团队流程删改，不要保留不存在的路径或命令。

````markdown
# Repository Guidelines / 仓库协作指南

## Project Overview / 项目概览

中文：本仓库是基于 [主要技术栈] 构建的 [项目类型]。使用 `README.md` 了解安装和运行方式，使用 `docs/architecture.md` 了解系统设计。修改 [高风险区域] 前，先阅读 [具体文档]。

English: This repository contains [project type] built with [main technologies]. Use `README.md` for setup and `docs/architecture.md` for system design. Before changing [high-risk area], read [specific document].

## Operation Map / 仓库操作地图

- 中文：`[path]/` 负责 [职责]。
  English: `[path]/` contains [responsibility].
- 中文：`[path]/` 负责 [职责]。
  English: `[path]/` contains [responsibility].
- 中文：`[path]/[module]/[layer]/` 负责 [三级目录职责，例如 handler/service/repository/fixtures]；修改这里时需要同步 [测试或文档]。
  English: `[path]/[module]/[layer]/` contains [third-level responsibility, such as handlers/services/repositories/fixtures]; changes here require [tests or docs].
- 中文：`[path]/` 是生成文件或由外部系统管理，不要直接编辑。
  English: `[path]/` is generated or externally managed; do not edit it directly.

## Documentation Map / 文档地图

中文：`README.md` 是项目根文档，负责项目简介、快速上手和文档索引。

English: `README.md` is the root project document for overview, quick start, and documentation index.

中文：`docs/README.md` 是文档总索引；`docs/development/README.md` 面向开发人员；`docs/ops/README.md` 面向使用人员和运维人员。

English: `docs/README.md` is the documentation index; `docs/development/README.md` is for developers; `docs/ops/README.md` is for users and operators.

中文：详细开发流程、架构说明、测试策略、代码生成和 CI 质量门禁应放在 `docs/development/`，不要塞进 `AGENTS.md`。

English: Detailed development workflows, architecture notes, testing strategy, code generation, and CI quality gates should live under `docs/development/`, not inside `AGENTS.md`.

中文：部署、配置、监控、排障、发布、回滚和备份恢复应放在 `docs/ops/`。

English: Deployment, configuration, monitoring, troubleshooting, release, rollback, and backup/restore runbooks should live under `docs/ops/`.

中文：修改 [特定高风险区域] 前，先阅读 `[docs path]`。

English: Before changing [specific high-risk area], read `[docs path]`.

## Editing Guidelines / 编辑规范

中文：遵循相邻文件的既有模式。保持改动聚焦在用户请求的行为上。优先使用现有 helper、抽象和工具链，不要轻易引入新依赖。

English: Follow existing patterns in nearby files. Keep changes focused on the requested behavior. Prefer existing helpers, abstractions, and tooling over introducing new dependencies.

中文：不要重排整个配置文件或 lockfile，除非工具要求。不要直接编辑生成文件；应运行 `[generation command]` 重新生成。

English: Do not reformat entire config or lock files unless required by the tool. Do not edit generated files directly; use `[generation command]`.

## Verification Guidelines / 验证规范

中文：修改后优先运行最小且相关的检查：

English: Run the narrowest relevant checks after changes:

```bash
[lint command]
[typecheck command]
[test command]
```

中文：对于 [特定变更类型]，还需要运行：

English: For [specific change type], also run:

```bash
[specialized verification command]
```

中文：如果检查需要外部凭据或生产访问权限，运行前先询问用户。

English: If a check requires external credentials or production access, ask the user before running it.

## CI/CD Guidelines / CI/CD 工作流

### CI Workflows (automated testing) / CI 工作流（自动化测试）

中文：列出 PR/MR 和 push 会触发的自动测试入口、必需状态检查和包级 workflow。新增包、模块或服务时，同步更新 path filter、矩阵配置和必需检查。

English: List the automated testing entry points, required status checks, and package-level workflows triggered by PRs/MRs and pushes. When adding a package, module, or service, update path filters, matrix configuration, and required checks together.

中文：如果项目不使用 GitHub Actions，把下面的 `Workflow` 和 `.github/workflows/...` 替换为实际平台中的 pipeline、job、stage 或配置文件路径。

English: If the project does not use GitHub Actions, replace `Workflow` and `.github/workflows/...` below with the actual pipeline, job, stage, or configuration path used by the platform.

| Workflow | File | Trigger | Purpose |
| --- | --- | --- | --- |
| [CI gate name] | `.github/workflows/[ci-gate].yml` | `pull_request` | Aggregates required checks / 汇总必需检查 |
| [Package check] | `.github/workflows/[package-check].yml` | `workflow_call`, `push` | Runs package tests / 运行包级测试 |

### CD Workflows (automated publishing) / CD 工作流（自动化发布）

中文：列出发布入口、tag 规则、目标环境或包仓库、审批要求和安全凭据方式。未经明确批准，不要修改发布触发条件、workflow 文件名、OIDC/trusted publishing 配置或生产部署权限。

English: List publishing entry points, tag rules, target environments or registries, approval requirements, and credential model. Do not change publishing triggers, workflow filenames, OIDC/trusted-publishing settings, or production deployment permissions without explicit approval.

| Workflow | File | Trigger | Target |
| --- | --- | --- | --- |
| [Release router] | `.github/workflows/[release].yml` | `release: published` or `workflow_dispatch` | Routes package publishing / 路由包发布 |
| [Package publish] | `.github/workflows/[publish].yml` | `workflow_dispatch` | Publishes [package/image/service] / 发布目标 |

### Utility Workflows / 辅助工作流

中文：列出不直接发布但保障质量的工作流，例如文档索引检查、依赖审查、代码生成校验和安全扫描。公开开源项目可额外列出自动打标、stale 检查等社区维护工作流。

English: List quality-support workflows that do not publish directly, such as documentation index checks, dependency review, codegen validation, and security scanning. Public open-source projects may also list community-maintenance workflows such as issue labeling and stale checks. Keep their coverage in sync when changing related directories.

| Workflow | File | Purpose |
| --- | --- | --- |
| [Docs check] | `.github/workflows/[docs-check].yml` | Validates docs indexes / 校验文档索引 |
| [Dependency review] | `.github/workflows/[dependency-review].yml` | Reviews dependency changes / 检查依赖变更 |
| [Codegen check] | `.github/workflows/[codegen-check].yml` | Ensures generated files are current / 校验生成文件同步 |

## Security & Safety Guidelines / 安全规范

中文：不要读取、输出、提交或总结 `.env*`、`secrets/`、私钥、证书、服务账号文件、cookie 或 token store 中的明文密钥。除非用户在当前对话明确要求本地编辑，否则不要打开这些文件；即使编辑，也不要在回复中包含明文值。

English: Do not read, print, commit, or summarize raw secrets from `.env*`, `secrets/`, private keys, certificates, service account files, cookies, or token stores unless the user explicitly requests a local edit in the current conversation. Even then, do not include raw values in responses.

中文：未经用户明确批准，不要执行生产部署、破坏性数据操作、服务重启、远程基础设施修改或重写 Git 历史。优先使用 dry-run 和状态检查命令。

English: Do not run production deploys, destructive data operations, service restarts, remote infrastructure changes, or history-rewriting Git commands without explicit user approval. Prefer dry-run and status commands first.

## Git Guidelines / Git 规范

中文：暂存或提交前先运行 `git status`。除非用户要求，不要创建提交、推送分支、amend、rebase、force push 或删除分支。不要把无关的本地改动混入当前 diff。

English: Run `git status` before staging or committing. Do not create commits, push branches, amend commits, rebase, force push, or delete branches unless the user asks. Keep unrelated local changes out of the diff.
````

## 语言项目模板

### Go 项目模板

````markdown
# Repository Guidelines / 仓库协作指南

## Project Overview / 项目概览

中文：本仓库是 Go 服务或 Go 库。使用 `README.md` 了解安装和运行方式，使用 `docs/architecture.md` 了解服务边界。修改公共 API、持久化、认证、授权或后台任务前，先阅读 `docs/` 下的相关设计说明。

English: This repository contains a Go service/library. Use `README.md` for setup and `docs/architecture.md` for service boundaries. Before changing public APIs, persistence, authentication, authorization, or background jobs, read the relevant design notes under `docs/`.

## Operation Map / 仓库操作地图

- 中文：`cmd/` 包含可执行入口。
  English: `cmd/` contains executable entry points.
- 中文：`internal/` 包含私有应用包和核心业务逻辑。
  English: `internal/` contains private application packages and core business logic.
- 中文：`internal/<domain>/handler/` 负责协议入口和请求解析，`internal/<domain>/service/` 负责业务流程，`internal/<domain>/repository/` 负责持久化访问。
  English: `internal/<domain>/handler/` handles protocol entry points and request parsing, `internal/<domain>/service/` owns business workflows, and `internal/<domain>/repository/` owns persistence access.
- 中文：`pkg/` 包含允许外部复用的公共包。
  English: `pkg/` contains packages intended for external reuse.
- 中文：`api/`、`proto/` 或 `openapi/` 包含 API 源定义或生成文件。
  English: `api/`, `proto/`, or `openapi/` contain generated or source API definitions.
- 中文：`migrations/` 包含数据库迁移；除非用户明确要求，不要编辑已应用的迁移。
  English: `migrations/` contains schema migrations; do not edit applied migrations unless explicitly requested.

## Editing Guidelines / 编辑规范

中文：遵循现有包边界。`cmd/` 只负责装配、配置和进程启动，不要放核心业务逻辑。

English: Follow existing package boundaries. Keep business logic out of `cmd/`; use `cmd/` only for wiring, configuration, and process startup.

中文：除非项目已有其他工具链，否则使用标准 Go 工具。对修改过的 Go 文件运行 `gofmt`。引入新依赖前，先检查现有内部包是否已经提供能力。

English: Use the standard Go toolchain unless the project already uses another tool. Run `gofmt` on changed Go files. Do not introduce new dependencies without checking existing internal packages first.

中文：如果 API、protobuf、OpenAPI 或 mock 文件是生成的，应修改源定义并运行文档中的生成命令，不要直接编辑生成文件。

English: If API, protobuf, OpenAPI, or mock files are generated, update the source definition and run the documented generation command instead of editing generated files directly.

## Verification Guidelines / 验证规范

中文：优先运行最小且相关的检查：

English: Run the narrowest relevant checks first:

```bash
go test ./...
go vet ./...
```

中文：如果项目提供 Makefile，优先使用项目目标：

English: If the project uses a Makefile, prefer the project targets:

```bash
make test
make lint
```

中文：对于并发或竞态敏感代码，还要运行：

English: For race-sensitive or concurrent code, also run:

```bash
go test -race ./...
```

中文：涉及数据库变更时，先在一次性本地数据库上运行迁移，再修改依赖新 schema 的代码。

English: For database changes, run migrations against a disposable local database before changing code that depends on the new schema.

## CI/CD Guidelines / CI/CD 工作流

### CI Workflows (automated testing) / CI 工作流（自动化测试）

中文：Go 项目的 CI 至少应运行 `go test ./...`、`go vet ./...` 和格式检查。涉及并发代码时，CI 应包含 race 检查或有明确的手动触发入口。

English: Go CI should run at least `go test ./...`, `go vet ./...`, and formatting checks. For concurrency-sensitive code, CI should include race checks or provide a documented manual trigger.

| Workflow | File | Trigger | Purpose |
| --- | --- | --- | --- |
| Go CI | `.github/workflows/go-ci.yml` | `pull_request`, `push` | Runs Go tests, vet, and formatting checks / 运行 Go 测试、vet 和格式检查 |

### CD Workflows (automated publishing) / CD 工作流（自动化发布）

中文：如果项目发布 Go module、二进制或容器镜像，应写清 release tag 规则、发布目标和审批要求。未经明确批准，不要修改发布 token、OIDC/trusted publishing、镜像仓库或部署权限。

English: If the project publishes Go modules, binaries, or container images, document release tag rules, publishing targets, and approval requirements. Do not change publishing tokens, OIDC/trusted publishing, image registries, or deployment permissions without explicit approval.

| Workflow | File | Trigger | Target |
| --- | --- | --- | --- |
| Go release | `.github/workflows/go-release.yml` | `release: published` or `workflow_dispatch` | Publishes Go module, binary, or image / 发布 Go module、二进制或镜像 |

### Utility Workflows / 辅助工作流

中文：辅助 workflow 可覆盖代码生成校验、依赖审查、漏洞扫描和文档索引检查。新增 `proto/`、`openapi/` 或生成文件路径时，同步更新这些 workflow。

English: Utility workflows can cover codegen validation, dependency review, vulnerability scanning, and documentation index checks. When adding `proto/`, `openapi/`, or generated-file paths, update these workflows together.

## Task Completion Guidelines / 任务完成标准

- 中文：Bug 修复应新增或更新一个没有修复时会失败的回归测试。
  English: Bug fixes should add or update a regression test that fails without the fix.
- 中文：新增 handler 或 API 时，应添加 handler 测试并更新 API 文档。
  English: New handlers or APIs should add handler tests and update API documentation.
- 中文：新增存储行为时，应添加 repository 测试和迁移覆盖。
  English: New storage behavior should add repository tests and migration coverage.
- 中文：重构应保持公开行为不变，并运行完整包测试。
  English: Refactors should preserve public behavior and run the full package test suite.

## Security & Safety Guidelines / 安全规范

中文：不要输出、提交或总结 `.env*`、凭据、私钥、token 或服务账号文件中的明文密钥。除非用户明确要求，不要运行生产迁移、部署或服务重启。

English: Do not print, commit, or summarize raw secrets from `.env*`, credentials, private keys, tokens, or service account files. Do not run production migrations, deploys, or service restarts unless the user explicitly asks.

中文：执行外部操作前，优先使用本地 dry-run、状态检查或测试命令。

English: Prefer local dry-run, status, or test commands before external operations.

## Git Guidelines / Git 规范

中文：暂存或提交前先运行 `git status`。除非用户要求，不要创建提交、推送、rebase、amend、force push 或删除分支。不要把无关本地改动混入当前 diff。

English: Run `git status` before staging or committing. Do not create commits, push, rebase, amend, force push, or delete branches unless the user asks. Keep unrelated local changes out of the diff.
````

### Python 项目模板

````markdown
# Repository Guidelines / 仓库协作指南

## Project Overview / 项目概览

中文：本仓库是 Python 应用或库。使用 `README.md` 了解安装和运行方式，使用 `pyproject.toml` 查看工具配置，使用 `docs/architecture.md` 了解系统边界。修改公共 API、数据库模型、认证、授权或后台任务前，先阅读 `docs/` 下的相关文档。

English: This repository contains a Python application/library. Use `README.md` for setup, `pyproject.toml` for tool configuration, and `docs/architecture.md` for system boundaries. Before changing public APIs, database models, authentication, authorization, or background workers, read the relevant documentation under `docs/`.

## Operation Map / 仓库操作地图

- 中文：`src/` 或顶层包目录包含应用代码。
  English: `src/` or the top-level package directory contains application code.
- 中文：`src/<package>/api/` 或 `src/<package>/routes/` 负责接口入口，`src/<package>/services/` 负责业务流程，`src/<package>/repositories/` 负责持久化访问，`src/<package>/schemas/` 负责请求/响应模型。
  English: `src/<package>/api/` or `src/<package>/routes/` contains API entry points, `src/<package>/services/` owns business workflows, `src/<package>/repositories/` owns persistence access, and `src/<package>/schemas/` owns request/response models.
- 中文：`tests/` 包含单元测试和集成测试。
  English: `tests/` contains unit and integration tests.
- 中文：`tests/unit/` 放单元测试，`tests/integration/` 放集成测试，`tests/fixtures/` 放脱敏测试数据和 fixture。
  English: `tests/unit/` contains unit tests, `tests/integration/` contains integration tests, and `tests/fixtures/` contains anonymized test data and fixtures.
- 中文：`scripts/` 包含运维或维护脚本。
  English: `scripts/` contains operational or maintenance scripts.
- 中文：`migrations/` 或 `alembic/` 包含数据库迁移。
  English: `migrations/` or `alembic/` contains database migrations.
- 中文：`pyproject.toml` 是 formatter、linter、type checker 和测试配置的事实来源。
  English: `pyproject.toml` is the source of truth for formatter, linter, type checker, and test settings.

## Editing Guidelines / 编辑规范

中文：遵循现有包结构和 import 风格。优先使用本地 helper 和 service 抽象，不要轻易引入新依赖。保持 I/O、API 和持久化边界清晰。

English: Follow existing package layout and import style. Prefer local helpers and service abstractions over adding new dependencies. Keep I/O, API, and persistence boundaries explicit.

中文：不要直接编辑生成文件。应更新源 schema 或模型，并运行文档中的生成命令。

English: Do not edit generated files directly. Update the source schema or model and run the documented generation command.

中文：如果项目使用 `uv`，依赖管理和命令执行都使用 `uv`。如果项目使用 Poetry、Hatch、PDM 或 plain pip，则遵循现有工具链，不要混用包管理器。

English: If the project uses `uv`, use `uv` for dependency and command execution. If it uses Poetry, Hatch, PDM, or plain pip, follow the existing toolchain instead of mixing package managers.

## Verification Guidelines / 验证规范

中文：对于 `uv` 项目，运行：

English: For `uv` projects, run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

中文：如果配置了类型检查，也运行：

English: If type checking is configured, also run:

```bash
uv run mypy .
```

中文：非 `uv` 项目使用项目文档中的等价命令，例如：

English: For non-`uv` projects, use the project’s documented equivalents, such as:

```bash
pytest
ruff check .
ruff format --check .
mypy .
```

中文：涉及数据库变更时，先在一次性本地数据库上运行迁移，再修改依赖新 schema 的应用代码。

English: For database changes, run migrations against a disposable local database before changing dependent application code.

## CI/CD Guidelines / CI/CD 工作流

### CI Workflows (automated testing) / CI 工作流（自动化测试）

中文：Python 项目的 CI 应使用项目实际包管理器运行测试、lint、format check 和类型检查。`uv` 项目应优先使用 `uv run`，不要在 CI 中混用 Poetry、PDM、Conda 或裸 `pip`。

English: Python CI should use the project’s actual package manager to run tests, lint, format checks, and type checks. `uv` projects should prefer `uv run` and should not mix Poetry, PDM, Conda, or plain `pip` in CI.

| Workflow | File | Trigger | Purpose |
| --- | --- | --- | --- |
| Python CI | `.github/workflows/python-ci.yml` | `pull_request`, `push` | Runs pytest, Ruff, format check, and type check / 运行 pytest、Ruff、格式检查和类型检查 |

### CD Workflows (automated publishing) / CD 工作流（自动化发布）

中文：如果项目发布 Python package、Docker image 或内部服务，应写清发布入口、版本规则、目标仓库和凭据方式。未经明确批准，不要修改 PyPI trusted publishing、registry 权限或生产部署流程。

English: If the project publishes Python packages, Docker images, or internal services, document publishing entry points, version rules, target registries, and credential model. Do not change PyPI trusted publishing, registry permissions, or production deployment flow without explicit approval.

| Workflow | File | Trigger | Target |
| --- | --- | --- | --- |
| Python release | `.github/workflows/python-release.yml` | `release: published` or `workflow_dispatch` | Publishes package, image, or service / 发布包、镜像或服务 |

### Utility Workflows / 辅助工作流

中文：辅助 workflow 可覆盖 lockfile 校验、依赖审查、安全扫描、文档索引和生成文件校验。修改 `pyproject.toml`、`uv.lock` 或生成文件路径时，同步更新相关 workflow。

English: Utility workflows can cover lockfile validation, dependency review, security scanning, docs indexes, and generated-file checks. When changing `pyproject.toml`, `uv.lock`, or generated-file paths, update related workflows together.

## Task Completion Guidelines / 任务完成标准

- 中文：Bug 修复应新增或更新回归测试。
  English: Bug fixes should add or update a regression test.
- 中文：新增公共 API 时，应添加测试并更新文档或示例。
  English: New public APIs should add tests and update docs or examples.
- 中文：新增依赖时，应更新 lockfile，并说明为什么现有依赖不足。
  English: New dependencies should update the lockfile and explain why an existing dependency was insufficient.
- 中文：重构应保持公开行为和 import 稳定，除非用户明确要求破坏性变更。
  English: Refactors should preserve public behavior and keep imports stable unless the user requested a breaking change.

## Security & Safety Guidelines / 安全规范

中文：不要读取、输出、提交或总结 `.env*`、凭据、私钥、token store、notebook 或本地数据 dump 中的明文敏感信息。除非用户明确要求，不要连接生产数据库，也不要运行迁移、部署或破坏性清理命令。

English: Do not read, print, commit, or summarize raw secrets from `.env*`, credentials, private keys, token stores, notebooks, or local data dumps. Do not connect to production databases or run migrations, deploys, or destructive cleanup commands unless the user explicitly asks.

中文：测试应使用脱敏 fixture。不要把真实用户数据加入仓库。

English: Use anonymized fixtures for tests. Do not add real user data to the repository.

## Git Guidelines / Git 规范

中文：暂存或提交前先运行 `git status`。除非用户要求，不要创建提交、推送、rebase、amend、force push 或删除分支。不要把无关本地改动混入当前 diff。

English: Run `git status` before staging or committing. Do not create commits, push, rebase, amend, force push, or delete branches unless the user asks. Keep unrelated local changes out of the diff.
````

### Java 项目模板

````markdown
# Repository Guidelines / 仓库协作指南

## Project Overview / 项目概览

中文：本仓库是 Java 应用或库。使用 `README.md` 了解安装和运行方式，使用 `docs/architecture.md` 了解模块边界。运行命令前，先查看 `pom.xml`、`build.gradle` 或 `settings.gradle`，确认项目使用 Maven 还是 Gradle。

English: This repository contains a Java application/library. Use `README.md` for setup and `docs/architecture.md` for module boundaries. Check `pom.xml`, `build.gradle`, or `settings.gradle` to identify whether the project uses Maven or Gradle before running commands.

## Operation Map / 仓库操作地图

- 中文：`src/main/java/` 包含应用代码。
  English: `src/main/java/` contains application code.
- 中文：`src/main/java/.../controller/` 负责入口适配，`.../service/` 负责业务逻辑，`.../repository/` 或 `.../dao/` 负责持久化，`.../domain/` 负责领域模型，`.../config/` 负责框架配置。
  English: `src/main/java/.../controller/` contains entry adapters, `.../service/` owns business logic, `.../repository/` or `.../dao/` owns persistence, `.../domain/` owns domain models, and `.../config/` owns framework configuration.
- 中文：`src/test/java/` 包含测试代码。
  English: `src/test/java/` contains tests.
- 中文：`src/test/java/.../controller/` 放接口层测试，`.../service/` 放业务测试，`.../repository/` 放持久化测试。
  English: `src/test/java/.../controller/` contains controller tests, `.../service/` contains service tests, and `.../repository/` contains persistence tests.
- 中文：`src/main/resources/` 和 `src/test/resources/` 包含运行时资源和测试资源。
  English: `src/main/resources/` and `src/test/resources/` contain runtime and test resources.
- 中文：`src/main/resources/db/migration/`、`migrations/` 或项目指定目录包含数据库迁移。
  English: `src/main/resources/db/migration/`, `migrations/`, or the project-specific migration directory contains database migrations.
- 中文：`pom.xml` 是 Maven 构建配置；`build.gradle` 或 `build.gradle.kts` 是 Gradle 构建配置。
  English: `pom.xml` is Maven build configuration; `build.gradle` or `build.gradle.kts` is Gradle build configuration.

## Editing Guidelines / 编辑规范

中文：遵循现有包结构和框架约定。Controller 保持轻量；业务逻辑放在 service，持久化逻辑放在 repository 或 DAO，并匹配本地模式。

English: Follow the existing package structure and framework conventions. Keep controllers thin; put business logic in services and persistence logic in repositories or DAOs, matching local patterns.

中文：不要重排整个 XML、Gradle、YAML 或 properties 文件，除非工具要求。未经讨论，不要引入新框架或 annotation processor。

English: Do not reformat entire XML, Gradle, YAML, or properties files unless the tool requires it. Avoid introducing new frameworks or annotation processors without discussion.

中文：如果代码由 OpenAPI、protobuf、JOOQ、MapStruct 或其他生成器产生，应更新源定义并运行文档中的生成命令，不要直接编辑生成结果。

English: If code is generated from OpenAPI, protobuf, JOOQ, MapStruct, or another generator, update the source definition and run the documented generation command instead of editing generated output directly.

## Verification Guidelines / 验证规范

中文：Maven 项目运行：

English: For Maven projects, run:

```bash
./mvnw test
./mvnw verify
```

中文：Gradle 项目运行：

English: For Gradle projects, run:

```bash
./gradlew test
./gradlew check
```

中文：如果缺少 wrapper，仅在项目文档明确允许时使用 `mvn` 或 `gradle`。如果集成测试需要 Docker、Testcontainers 或外部服务，先确认本地服务可用。

English: If the wrapper is missing, use `mvn` or `gradle` only if that matches the project documentation. For integration tests that require Docker, Testcontainers, or external services, confirm the required local services are available before running them.

## CI/CD Guidelines / CI/CD 工作流

### CI Workflows (automated testing) / CI 工作流（自动化测试）

中文：Java 项目的 CI 应使用项目 wrapper 运行 Maven 或 Gradle 检查。不要在有 `./mvnw` 或 `./gradlew` 时改用系统级 `mvn` 或 `gradle`。如果 CI 依赖 Testcontainers 或 Docker，应写清服务要求。

English: Java CI should use the project wrapper to run Maven or Gradle checks. Do not switch to system `mvn` or `gradle` when `./mvnw` or `./gradlew` exists. If CI depends on Testcontainers or Docker, document the service requirements.

| Workflow | File | Trigger | Purpose |
| --- | --- | --- | --- |
| Java CI | `.github/workflows/java-ci.yml` | `pull_request`, `push` | Runs Maven/Gradle tests and checks / 运行 Maven 或 Gradle 测试与检查 |

### CD Workflows (automated publishing) / CD 工作流（自动化发布）

中文：如果项目发布 Maven artifact、Docker image 或部署 Java 服务，应写清版本规则、目标仓库、环境审批和凭据方式。未经明确批准，不要修改 signing、registry、OIDC/trusted publishing 或生产部署权限。

English: If the project publishes Maven artifacts, Docker images, or Java services, document version rules, target repositories, environment approvals, and credential model. Do not change signing, registry, OIDC/trusted publishing, or production deployment permissions without explicit approval.

| Workflow | File | Trigger | Target |
| --- | --- | --- | --- |
| Java release | `.github/workflows/java-release.yml` | `release: published` or `workflow_dispatch` | Publishes artifact, image, or service / 发布 artifact、镜像或服务 |

### Utility Workflows / 辅助工作流

中文：辅助 workflow 可覆盖依赖审查、漏洞扫描、许可证检查、生成代码校验和文档索引。修改 `pom.xml`、`build.gradle`、生成代码或迁移目录时，同步更新相关 workflow。

English: Utility workflows can cover dependency review, vulnerability scanning, license checks, generated-code validation, and docs indexes. When changing `pom.xml`, `build.gradle`, generated code, or migration paths, update related workflows together.

## Task Completion Guidelines / 任务完成标准

- 中文：Bug 修复应新增或更新回归测试。
  English: Bug fixes should add or update a regression test.
- 中文：新增 endpoint 时，应添加 controller 测试和 service 测试；如果是公开 API，还要更新 API 文档。
  English: New endpoints should add controller tests and service tests; update API docs if public.
- 中文：持久化变更应在项目支持的范围内添加 migration 测试或 repository 测试。
  English: Persistence changes should add migration tests or repository tests where the project supports them.
- 中文：重构应保持公开 API 不变，除非用户要求破坏性变更。
  English: Refactors should preserve public APIs unless a breaking change was requested.

## Security & Safety Guidelines / 安全规范

中文：不要输出、提交或总结 `.env*`、`application-*.yml`、keystore、truststore、私钥、token 或服务账号文件中的明文敏感信息。除非用户明确要求，不要运行生产迁移、部署或服务重启。

English: Do not print, commit, or summarize raw secrets from `.env*`, `application-*.yml`, keystores, truststores, private keys, tokens, or service account files. Do not run production migrations, deploys, or service restarts unless the user explicitly asks.

中文：使用本地 profile 和测试 fixture。不要把真实用户数据加入 `src/test/resources/`。

English: Use local profiles and test fixtures. Do not add real user data to `src/test/resources/`.

## Git Guidelines / Git 规范

中文：暂存或提交前先运行 `git status`。除非用户要求，不要创建提交、推送、rebase、amend、force push 或删除分支。不要把无关本地改动混入当前 diff。

English: Run `git status` before staging or committing. Do not create commits, push, rebase, amend, force push, or delete branches unless the user asks. Keep unrelated local changes out of the diff.
````

### Go 业务代码 + Python uv 接口测试模板

````markdown
# Repository Guidelines / 仓库协作指南

## Project Overview / 项目概览

中文：本仓库是 Go 后端服务，并使用 `uv` 管理 Python 接口测试。Go 代码负责生产服务；Python 只用于黑盒 API、契约和业务流程测试。使用 `README.md` 了解安装和运行方式，使用 `docs/api.md` 了解 API 行为，使用 `docs/architecture.md` 了解服务边界。

English: This repository contains a Go backend service with Python interface tests managed by `uv`. Go code owns the production service. Python is used for black-box API, contract, and workflow tests. Use `README.md` for setup, `docs/api.md` for API behavior, and `docs/architecture.md` for service boundaries.

## Operation Map / 仓库操作地图

- 中文：`cmd/` 包含 Go 服务二进制入口和本地工具。
  English: `cmd/` contains Go service binaries and local tools.
- 中文：`internal/` 包含 Go handler、service、repository 和领域逻辑。
  English: `internal/` contains Go handlers, services, repositories, and domain logic.
- 中文：`internal/<domain>/handler/` 负责 HTTP/gRPC 入口，`internal/<domain>/service/` 负责业务编排，`internal/<domain>/repository/` 负责数据访问；接口行为变化通常需要同步 Python 接口测试。
  English: `internal/<domain>/handler/` owns HTTP/gRPC entry points, `internal/<domain>/service/` owns business orchestration, and `internal/<domain>/repository/` owns data access; API behavior changes usually require Python interface test updates.
- 中文：`api/`、`proto/` 或 `openapi/` 包含 API 定义。
  English: `api/`, `proto/`, or `openapi/` contains API definitions.
- 中文：`migrations/` 包含数据库 schema 变更。
  English: `migrations/` contains schema changes.
- 中文：`tests/api/` 包含通过 `uv` 执行的 Python 接口测试。
  English: `tests/api/` contains Python interface tests executed with `uv`.
- 中文：`tests/api/clients/` 放 API 测试客户端，`tests/api/fixtures/` 放脱敏测试数据，`tests/api/test_<resource>.py` 覆盖公开接口行为。
  English: `tests/api/clients/` contains API test clients, `tests/api/fixtures/` contains anonymized test data, and `tests/api/test_<resource>.py` covers public API behavior.
- 中文：`pyproject.toml` 和 `uv.lock` 定义 Python 接口测试环境。
  English: `pyproject.toml` and `uv.lock` define the Python API test environment.
- 中文：`docker-compose.yml` 或 `compose.yml` 用于启动本地依赖服务。
  English: `docker-compose.yml` or `compose.yml` starts dependent services for local testing.

## Editing Guidelines / 编辑规范

中文：生产行为应保留在 Go 中。Python 只用于测试客户端、fixture、断言和 API workflow 覆盖。

English: Keep production behavior in Go. Use Python only for test clients, fixtures, assertions, and API workflow coverage.

中文：修改 endpoint 时，应同时更新 Go handler/service、API 契约和 Python 接口测试。Python 测试不应依赖 Go 私有实现细节，只应验证公开 HTTP/gRPC 行为。

English: When changing an endpoint, update the Go handler/service code, API contract, and Python interface tests together. Do not make Python tests depend on private Go implementation details; they should exercise public HTTP/gRPC behavior.

中文：修改 Go 文件后运行 `gofmt`。Python 测试依赖和命令使用 `uv`。除非项目已经使用其他工具，不要把 `pip`、Poetry 或 Conda 混入 Python 测试环境。

English: Use `gofmt` for changed Go files. Use `uv` for Python test dependencies and commands. Do not mix `pip`, Poetry, or Conda into the Python test environment unless the project already uses them.

中文：不要直接编辑生成的 API client、mock、protobuf 输出或 OpenAPI 输出。应更新源定义并运行文档中的生成命令。

English: Do not edit generated API clients, mocks, protobuf output, or OpenAPI output directly. Update the source definition and run the documented generation command.

## Verification Guidelines / 验证规范

中文：Go 代码变更运行：

English: For Go changes, run:

```bash
go test ./...
go vet ./...
```

中文：如果项目提供 Make 目标，优先运行：

English: If the project provides Make targets, prefer:

```bash
make test
make lint
```

中文：Python 接口测试需要先按项目文档启动本地服务和依赖，然后运行：

English: For Python API tests, first start the local service and dependencies using the documented command, then run:

```bash
uv run pytest tests/api
```

中文：如果测试需要 base URL，使用项目文档中的本地测试地址，例如：

English: If tests require a base URL, use the local test URL documented by the project, for example:

```bash
API_BASE_URL=http://localhost:8080 uv run pytest tests/api
```

中文：涉及 API 契约变更时，Go 单元测试和 Python 接口测试都要运行，之后才能认为任务完成。

English: For contract changes, run both the Go unit tests and the Python interface tests before considering the task complete.

## CI/CD Guidelines / CI/CD 工作流

### CI Workflows (automated testing) / CI 工作流（自动化测试）

中文：混合项目的 CI 应同时覆盖 Go 单元测试和 Python 接口测试。Go 变更至少运行 `go test ./...` 和 `go vet ./...`；接口契约、handler、service 或 OpenAPI/proto 变更必须触发 `uv run pytest tests/api`。

English: Mixed-project CI should cover both Go unit tests and Python interface tests. Go changes should run at least `go test ./...` and `go vet ./...`; API contract, handler, service, or OpenAPI/proto changes must trigger `uv run pytest tests/api`.

| Workflow | File | Trigger | Purpose |
| --- | --- | --- | --- |
| Go CI | `.github/workflows/go-ci.yml` | `pull_request`, `push` | Runs Go tests and vet / 运行 Go 测试和 vet |
| API test CI | `.github/workflows/api-test-ci.yml` | `pull_request`, `push` | Starts local services and runs Python API tests with uv / 启动本地服务并用 uv 运行 Python 接口测试 |

### CD Workflows (automated publishing) / CD 工作流（自动化发布）

中文：如果项目发布 Go 服务镜像或部署后端服务，应写清镜像 tag、环境审批、发布入口和回滚入口。Python 接口测试不应发布生产代码，但可作为发布前质量门禁。

English: If the project publishes Go service images or deploys backend services, document image tags, environment approvals, publishing entry points, and rollback entry points. Python interface tests should not publish production code, but they may act as a pre-release quality gate.

| Workflow | File | Trigger | Target |
| --- | --- | --- | --- |
| Service release | `.github/workflows/service-release.yml` | `release: published` or `workflow_dispatch` | Builds image and deploys service / 构建镜像并部署服务 |

### Utility Workflows / 辅助工作流

中文：辅助 workflow 应覆盖 API 契约生成校验、Go 依赖审查、`uv.lock` 校验、文档索引和安全扫描。修改 `api/`、`proto/`、`openapi/`、`go.sum`、`uv.lock` 或 `tests/api/` 时，同步更新相关 workflow。

English: Utility workflows should cover API contract codegen checks, Go dependency review, `uv.lock` validation, docs indexes, and security scanning. When changing `api/`, `proto/`, `openapi/`, `go.sum`, `uv.lock`, or `tests/api/`, update related workflows together.

## Task Completion Guidelines / 任务完成标准

- 中文：Go bug 修复应新增或更新 Go 单元测试；如果 bug 可通过 API 观察，也应补充 Python 接口覆盖。
  English: Go bug fixes should add or update Go unit tests; add Python interface coverage if the bug is visible through the API.
- 中文：新增 API endpoint 时，应添加 Go handler/service 测试、更新 API 文档或 schema，并添加 Python 接口测试。
  English: New API endpoints should add Go handler/service tests, update API docs or schema, and add Python interface tests.
- 中文：API 行为变更应更新契约文档、Python 测试；如果影响客户端，还应补充迁移说明。
  English: API behavior changes should update contract docs, Python tests, and migration notes if clients are affected.
- 中文：数据库变更应添加 migration，更新 repository 测试，并验证依赖新 schema 的 API 路径。
  English: Database changes should add migrations, update repository tests, and verify the API path that depends on the new schema.
- 中文：仅测试变更应保持 Python fixture 可重复，避免固定 sleep；异步流程优先使用带超时的轮询。
  English: Test-only changes should keep Python fixtures deterministic and avoid sleeps; prefer polling with timeouts for async workflows.

## Security & Safety Guidelines / 安全规范

中文：不要读取、输出、提交或总结 `.env*`、凭据、私钥、token、服务账号文件、本地数据库 dump 或捕获的 API 流量中的明文敏感信息。除非用户明确要求，不要把接口测试指向生产环境或共享 staging。

English: Do not read, print, commit, or summarize raw secrets from `.env*`, credentials, private keys, tokens, service account files, local database dumps, or captured API traffic. Do not point interface tests at production or shared staging unless the user explicitly asks.

中文：未经明确批准，不要运行生产迁移、部署、破坏性清理命令或服务重启。优先使用本地 Docker Compose 和一次性测试数据。

English: Do not run production migrations, deploys, destructive cleanup commands, or service restarts without explicit approval. Prefer local Docker Compose and disposable test data.

## Git Guidelines / Git 规范

中文：暂存或提交前先运行 `git status`。Python 测试依赖变更时提交 `uv.lock`；Go 依赖变更时提交 `go.sum`。除非用户要求，不要创建提交、推送、rebase、amend、force push 或删除分支。不要把无关本地改动混入当前 diff。

English: Run `git status` before staging or committing. Include `uv.lock` changes when Python test dependencies change and `go.sum` changes when Go dependencies change. Do not create commits, push, rebase, amend, force push, or delete branches unless the user asks. Keep unrelated local changes out of the diff.
````

## 维护建议

每当项目新增重要工具链、测试入口、安全边界或部署流程时，同步更新 `AGENTS.md`。如果某条规则长期不再适用，应删除而不是保留注释说明。文档越短、越贴近真实工作流，代理越容易正确执行。
