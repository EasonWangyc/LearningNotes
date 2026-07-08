# 文档体系架构

本文补充 [`agents-md-best-practices.prompt.md`](./agents-md-best-practices.prompt.md) 中的“文档地图与拆分”理念。最佳实践文档只保留摘要和模板；更细的文档组织原则、目录职责和拆分边界放在这里，按需查询。

## 目标

文档体系的目标不是把所有内容堆进一个文件，而是让人和 AI agent 都能快速判断：

- 我应该从哪里开始读。
- 当前任务应该改哪个目录。
- 更详细的开发、测试、部署、排障说明在哪里。
- 哪些操作需要先确认。
- 哪些内容是项目事实来源。

`AGENTS.md` 是操作地图，不是完整手册。它负责把 agent 带到正确入口；细节文档负责承载背景、流程和长说明。

## 推荐目录

```text
.
├── AGENTS.md
├── README.md
└── docs
    ├── README.md
    ├── ARCHITECTURE.md
    ├── development
    │   ├── README.md
    │   ├── architecture.md
    │   ├── local-development.md
    │   ├── testing.md
    │   ├── ci.md
    │   ├── code-generation.md
    │   └── database.md
    └── ops
        ├── README.md
        ├── deployment.md
        ├── configuration.md
        ├── monitoring.md
        ├── troubleshooting.md
        ├── release.md
        ├── rollback.md
        └── backup-restore.md
```

项目不需要一次性创建所有文件。先创建 `README.md`、`docs/README.md`、`docs/development/README.md`、`docs/ops/README.md` 这几个索引，再按真实需求增加细分文档。

## 文件职责

### `AGENTS.md`

面向 AI agent，回答“怎么在这个仓库里安全工作”。

适合放：

- 开始任务前应读的入口文档。
- 关键目录和必要三级目录职责。
- 常用验证命令。
- CI/CD 质量门禁和发布入口摘要。
- 敏感文件、高风险操作和 Git 边界。
- 指向详细文档的链接。

不适合放：

- 长篇架构说明。
- 完整开发手册。
- 完整部署 runbook。
- 完整 API 文档。
- 过长的 CI/CD 设计解释。

### `README.md`

面向首次进入项目的人，回答“这是什么、怎么快速开始、文档在哪里”。

建议包含：

- 项目一句话说明。
- 快速启动命令。
- 最小依赖说明。
- 常用任务入口。
- `docs/` 索引链接。
- 团队协作或支持入口；个人项目可省略。

公开开源项目可额外添加 `CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md`、`SUPPORT.md`、issue 模板和 PR 模板。个人项目或团队内部项目默认不需要这些社区治理文件；开发流程应优先放在 `docs/development/`。

### `docs/README.md`

面向所有读者，回答“文档体系怎么组织，我该看哪一篇”。

建议包含：

- `docs/development/` 索引。
- `docs/ops/` 索引。
- 架构、API、ADR、runbook 等主要文档入口。
- 文档维护规则。

### `docs/development/README.md`

面向开发人员，回答“怎么开发、测试、修改和验证代码”。

常见索引：

- 本地开发环境。
- 项目架构。
- 测试策略。
- 数据库迁移。
- 代码生成。
- CI 质量门禁。
- 编码规范。
- 调试和排障。

### `docs/ops/README.md`

面向使用人员和运维人员，回答“怎么部署、配置、监控、恢复和发布”。

常见索引：

- 部署流程。
- 配置说明。
- 监控和告警。
- 排障 runbook。
- 发布和回滚。
- 备份和恢复。
- 运维权限和审批要求。

## 拆分规则

判断内容放哪里，可以按下面规则：

| 内容类型 | 推荐位置 |
| --- | --- |
| Agent 必须遵守的安全规则 | `AGENTS.md` |
| Agent 每次常用的验证命令 | `AGENTS.md` |
| 详细本地开发流程 | `docs/development/local-development.md` |
| 长篇架构说明 | `docs/development/architecture.md` 或项目根 `ARCHITECTURE.md` |
| 测试分层、fixture、测试数据策略 | `docs/development/testing.md` |
| CI workflow 设计细节 | `docs/development/ci.md` |
| 数据库迁移流程 | `docs/development/database.md` |
| 部署和发布 runbook | `docs/ops/deployment.md`、`docs/ops/release.md` |
| 生产排障和回滚 | `docs/ops/troubleshooting.md`、`docs/ops/rollback.md` |
| 备份恢复 | `docs/ops/backup-restore.md` |

如果一段内容超过几段，并且不是 agent 每次都需要读的规则，优先拆出去。`AGENTS.md` 中保留一句摘要和链接。

## 索引示例

### 根 `README.md`

```markdown
## Documentation

- `docs/README.md`: full documentation index.
- `docs/development/README.md`: development setup, architecture, testing, CI, and database changes.
- `docs/ops/README.md`: deployment, configuration, monitoring, release, rollback, and troubleshooting.
```

### `docs/README.md`

```markdown
# Documentation

## Development

- `development/README.md`: development documentation index.
- `development/architecture.md`: system architecture.
- `development/testing.md`: testing strategy.
- `development/ci.md`: CI quality gates.

## Operations

- `ops/README.md`: operations documentation index.
- `ops/deployment.md`: deployment guide.
- `ops/release.md`: release process.
- `ops/troubleshooting.md`: troubleshooting runbooks.
```

### `AGENTS.md`

```markdown
## Documentation Map

- Use `README.md` for quick start.
- Use `docs/README.md` for the full documentation index.
- Use `docs/development/README.md` before changing code, tests, migrations, or CI.
- Use `docs/ops/README.md` before changing deployment, release, rollback, or production operations.
```

## 维护规则

- 新增详细文档时，同步更新同级 `README.md` 索引。
- 新增开发流程文档时，同步更新 `docs/development/README.md`。
- 新增运维 runbook 时，同步更新 `docs/ops/README.md`。
- 如果 `AGENTS.md` 中某节持续变长，优先拆到 `docs/`，并在 `AGENTS.md` 留链接。
- 如果某份文档成为事实来源，明确写出“以此文档为准”。
- 删除或移动文档时，更新所有引用入口，避免地图失效。

## 与 CI/CD 的关系

CI/CD 是质量地图的一部分，但设计细节不宜全部写进 `AGENTS.md`。

推荐做法：

- `AGENTS.md` 写 CI/CD 的入口、必需检查、发布入口和禁止随意修改的 workflow。
- `docs/development/ci.md` 写 CI workflow 设计、path filter、矩阵策略、缓存策略和 required checks。
- `docs/ops/release.md` 写发布流程、tag 规则、环境审批、回滚和故障处理。

这样 agent 能快速知道边界；需要修改 CI/CD 时，再按链接读取详细文档。
