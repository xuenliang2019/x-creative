<p align="center">
  <img src="open-x-creative.png" alt="X-Creative Logo" width="200">
</p>

<h1 align="center">X-Creative</h1>

<p align="center">一个基于<b>创造力理论驱动</b>的通用研究 Agent Workflow 系统。</p>

输入一个问题，系统自动从远域知识中发现结构同构，生成跨领域创新假说，经多模型验证与风险精炼，输出包含直接答案、可追溯证据和风险边界的完整研究报告。

```bash
x-creative answer -q "如何提高分布式系统的容错能力"
```

---

## 目录

- [项目简介](#项目简介)
- [工作原理](#工作原理)
- [文件夹结构](#文件夹结构)
- [安装配置](#安装配置)
- [命令行用法](#命令行用法)
- [Python API](#python-api)
- [配置系统](#配置系统)
- [目标领域配置](#目标领域配置)
- [输出格式](#输出格式)
- [开发](#开发)
- [源码架构](#源码架构)
- [致谢](#致谢)

---

## 项目简介

### 它解决什么问题

当我们面对一个研究或创新问题时，传统做法是在**已知领域内线性思考**——这容易陷入局部最优。X-Creative 的做法不同：它从**完全不相关的知识领域**中寻找结构上的相似性，用这种跨领域类比来生成创新假说。

例如，当你问"如何提高开源项目的用户留存"时，系统可能会从**排队论**中发现 Issue 响应延迟与服务队列拥塞的结构同构，从**生态学**中发现社区生态位竞争与物种共存的映射关系，从**热力学**中发现用户流失与熵增的类比——这些跨领域视角往往能激发出在单一领域内难以发现的创新方案。

### 背后的原理

系统的核心理论基础是 **Bisociation（双联想）**——Arthur Koestler (1964) 提出的创造力理论：

> **创造性行为的本质，是将两个此前毫不相关的认知框架连接起来。**

与日常联想（在单一框架内的线性思维）不同，bisociation 要求同时运行在两个独立的"思维平面"上，并发现它们之间的结构性同构。经典案例：

- 阿基米德在浴缸中发现浮力定律——将"洗澡时水位上升"与"物体体积度量"连接
- 达尔文的自然选择——将"人工育种"与"自然界物种变异"连接

除 Bisociation 外，系统还融合了三个互补的创造力理论：

| 理论 | 核心主张 | 在系统中的对应 |
|------|---------|---------------|
| **Boden 三类创造力** | 组合、探索、变形三种创造力形式 | SEARCH 阶段的 combine / refine / transform_space 算子 |
| **概念融合 (Fauconnier-Turner)** | 双向四空间融合产生涌现结构 | blend_expand 算子：从假说对中生成全新假说 |
| **C-K 理论 (Hatchuel-Weil)** | 概念空间与知识空间的交替扩展 | SAGA 编排的 C→K / K→C 阶段切换 |

> 详细的理论阐述和实现设计见 [`docs/theory.md`](docs/theory.md)。

### 核心流水线

系统通过四阶段流水线将上述理论转化为可计算的步骤：

```
问题输入
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ BISO（远域联想）                                      │
│ 从 18-30 个远距离源领域生成跨领域类比假说（~50-60 个） │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ SEARCH（图式搜索）                                    │
│ 基于 Graph of Thoughts 多策略扩展假说空间（~100-200+）│
│ 算子：refine / variant / combine / blend / transform │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ VERIFY（双模型验证）                                  │
│ 五维评分 + 逻辑验证 + 新颖性验证 + 映射质量门控       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ SOLVE（Talker-Reasoner 推理）                        │
│ 7 步多步推理 + 联网证据收集 + 自适应风险精炼循环       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
                研究报告输出
          (Markdown + JSON)
```

### 主要特性

- **通用领域支持**：通过 YAML 配置支持任意目标领域（科学研究、产品创新、开源开发等）
- **远域联想 (BISO)**：自动从多个源领域生成跨领域类比，源领域可行性过滤避免无效映射
- **映射质量门控**：抗凑数规则 + LLM 评分，低质量映射不会进入后续阶段
- **双模型验证**：逻辑验证 + 新颖性验证，支持置信度驱动的选择性评估与位置偏差防御
- **用户约束系统**：约束预检（矛盾检测）→ 约束编译（HardCore + ActiveSoftSet）→ 合规审计与补丁闭环
- **多模型编排**：通过 OpenRouter / 云雾 API 访问云端大模型，支持任务级路由和自动 fallback
- **SAGA 双进程认知架构**（实验性）：Fast Agent（系统1）+ Slow Agent（系统2）元认知监督
- **超图知识锚定 (HKG)**（实验性）：结构证据驱动的假说补全与桥接
- **概念混合 (Conceptual Blending)**（实验性）：Fauconnier-Turner 四空间融合
- **变换式创造力**（实验性）：Boden 变换式规则突破
- **MOME 质量-多样性档案**（实验性）：MAP-Elites 行为网格上的多样性维护
- **C-K 双空间调度**（实验性）：概念空间与知识空间的自动交替扩展

---

## 文件夹结构

```
x-creative/
├── x_creative/                      # 主包（源代码）
│   ├── answer/                      #   Answer Engine（单入口编排器）
│   ├── cli/                         #   CLI 命令定义
│   ├── config/                      #   配置与设置
│   │   └── target_domains/          #     内置目标领域 YAML
│   ├── core/                        #   核心类型（Hypothesis, ProblemFrame, Domain）
│   ├── creativity/                  #   创造力引擎（BISO, SEARCH, 搜索算子）
│   ├── verify/                      #   双模型验证系统
│   ├── saga/                        #   SAGA 双进程认知架构
│   ├── hkg/                         #   超图知识锚定
│   ├── llm/                         #   LLM 客户端与路由
│   ├── session/                     #   Session 管理
│   ├── domain_manager/              #   源领域 TUI 管理工具（xc-domain）
│   └── target_manager/              #   目标领域 TUI 管理工具（xc-target）
│
├── docs/                            # 文档
│   ├── theory.md                    #   设计文档（中文）
│   └── theory.en.md                 #   设计文档（英文）
│
├── local_data/                      # Session 数据存储（运行时生成）
│   ├── .current_session             #   当前活跃 session 标记
│   └── <session-id>/                #   各 session 的数据目录
│       ├── problem.json / .md       #     问题定义
│       ├── biso.json / .md          #     BISO 阶段结果
│       ├── search.json / .md        #     SEARCH 阶段结果
│       ├── verify.json / .md        #     VERIFY 阶段结果
│       ├── solve.json / .md         #     SOLVE 阶段结果
│       ├── answer.json / .md        #     Answer Engine 最终报告
│       └── saga/                    #     SAGA 内部状态与日志
│
├── log/                             # 应用日志（运行时生成）
│   └── output.log                   #   主日志文件
│
├── pyproject.toml                   # Poetry 项目配置
├── poetry.lock                      # 依赖锁定文件
├── .python-version                  # Python 版本（pyenv）
├── .env.example                     # 环境变量模板
├── .env                             # 实际环境变量（gitignored）
└── CLAUDE.md                        # Claude Code 指令
```

---

## 安装配置

### 前置条件

- **Python 3.12+**
- **Poetry 2.1+**
- **LLM Provider API Key**（OpenRouter 或 云雾，至少一个）

### 第 1 步：安装 Python（通过 pyenv）

如果你还没有 Python 3.12+，推荐使用 [pyenv](https://github.com/pyenv/pyenv) 管理 Python 版本：

```bash
# 安装 pyenv（如未安装）
curl https://pyenv.run | bash

# 安装 Python 3.12
pyenv install 3.12.12

# 进入项目目录后自动使用正确版本（项目已包含 .python-version 文件）
cd x-creative
python --version  # 应显示 3.12.12
```

### 第 2 步：安装 Poetry

```bash
# 安装 Poetry（如未安装）
curl -sSL https://install.python-poetry.org | python3 -

# 验证
poetry --version  # 需要 2.1+
```

### 第 3 步：克隆项目并安装依赖

```bash
git clone https://github.com/xuenliang2019/x-creative.git
cd x-creative

# 安装所有依赖
poetry install

# 验证安装
poetry run x-creative --version
```

### 第 4 步：配置 API Key

X-Creative 需要通过 LLM API 访问大模型。至少需要配置一个 LLM Provider。

#### 开通 OpenRouter 账号（推荐）

1. 访问 https://openrouter.ai/ 注册账号
2. 进入 https://openrouter.ai/keys 创建 API Key
3. 充值余额（按用量计费）

#### 设置环境变量

复制示例配置文件并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置以下必填项：

```bash
# LLM Provider（至少配置一个）
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 或者使用云雾（https://yunwu.ai/）
# YUNWU_API_KEY=your-yunwu-key-here
# X_CREATIVE_DEFAULT_PROVIDER=yunwu

# 可选：Brave Search API Key（用于新颖性验证的网络搜索）
# 注册地址：https://brave.com/search/api/
# BRAVE_SEARCH_API_KEY=your-brave-api-key
```

#### 验证配置

```bash
# 完整验证（静态检查 + API 连通性 + 模型可用性）
poetry run x-creative config check

# 仅静态检查（不发送 API 请求）
poetry run x-creative config check --quick
```

### 第 5 步：确认运行时目录

系统运行时会使用两个目录：

| 目录 | 用途 | 默认位置 | 配置方式 |
|------|------|----------|----------|
| `local_data/` | 存储 session 数据（问题定义、假说、报告） | 项目根目录下 | 环境变量 `X_CREATIVE_DATA_DIR` |
| `log/` | 存储运行日志 | 项目根目录下 | 自动创建 |

这两个目录会在首次运行时自动创建，无需手动操作。如需自定义数据目录：

```bash
# 在 .env 中设置
X_CREATIVE_DATA_DIR=/path/to/your/data
```

---

## 命令行用法

所有命令通过 `poetry run x-creative` 或激活虚拟环境后直接使用 `x-creative`：

```bash
# 方式一：通过 poetry run
poetry run x-creative <command>

# 方式二：先激活虚拟环境
poetry shell
x-creative <command>
```

以下示例均省略 `poetry run` 前缀。

### 最简单的开始：`answer`

`answer` 是系统的**主入口**——输入一个问题，自动完成全部流程：

```bash
x-creative answer -q "如何提高分布式系统的容错能力"
```

这一条命令会自动执行：问题框定 → 目标域推断 → 源域选择 → BISO → SEARCH → VERIFY → SOLVE → 输出报告。

#### answer 命令选项

| 选项 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--question` | `-q` | (必需) | 研究问题 |
| `--budget` | - | `60` | 认知预算单位 |
| `--target` | - | `auto` | 目标领域 ID（`auto` = 自动推断） |
| `--depth` | - | `3` | SEARCH 搜索深度 |
| `--breadth` | - | `5` | SEARCH 搜索广度 |
| `--mode` | - | `deep_research` | 模式：`quick` / `deep_research` / `exhaustive` |
| `--no-hkg` | - | 关 | 禁用超图知识锚定 |
| `--no-saga` | - | 关 | 禁用 SAGA 监督 |
| `--fresh` | - | 关 | 跳过预定义 YAML 域，由 LLM 从零生成 |
| `--output` | `-o` | - | 保存 Markdown 报告到文件 |

#### answer 使用示例

```bash
# 自动推断目标域
x-creative answer -q "如何提高分布式系统的容错能力"

# 指定目标域 + 导出报告
x-creative answer -q "如何提高分布式系统的容错能力" \
  --target open_source_development \
  --depth 2 \
  --output report.md

# 快速模式（减少搜索深度，更快出结果）
x-creative answer -q "探索用户留存提升方案" --mode quick

# 关闭 SAGA 监督（加快速度、降低成本）
x-creative answer -q "测试问题" --no-saga --no-hkg

# 全新生成模式（跳过预定义 YAML，由 LLM 从零生成目标域和源域）
x-creative answer -q "探索量子计算对密码学的影响" --fresh
```

#### answer 输出

`answer` 命令生成两种格式的报告（保存在 session 目录）：

- **Markdown 报告** (`answer.md`)：直接答案、关键证据、风险边界、假说排名、方法论附录
- **JSON 结构化数据** (`answer.json`)：所有元数据（session ID、目标域、源域数、搜索轮次、预算消耗）

如果系统无法确定问题领域（置信度 < 0.3），会交互式询问一个澄清问题，然后继续执行。

---

### 分阶段工作流：`session` + `run`

当你需要更精细的控制时，可以使用分阶段工作流：手动创建 session，逐步执行各阶段。

#### 管理 Session

```bash
# 创建新 session（自动设为当前）
x-creative session new "研究主题"

# 自定义 session ID
x-creative session new "研究主题" --id my-research

# 列出所有 session
x-creative session list

# 查看当前 session 状态
x-creative session status

# 切换当前 session
x-creative session switch my-research

# 删除 session
x-creative session delete my-research
```

#### 逐步执行流水线

工作流包含 4 个基础阶段（`problem` → `biso` → `search` → `verify`）和 1 个可选求解阶段（`solve`）：

```bash
# 1. 定义研究问题
x-creative run problem -d "探索提高用户留存的创新方法" \
  --target-domain open_source_development \
  --constraint "不增加运营成本" \
  --constraint "2周内可实施"

# 2. 运行 BISO（远域联想）
x-creative run biso --num-per-domain 3

# 3. 查看中间结果
x-creative show biso --top 5

# 4. 运行 SEARCH（假说扩展）
x-creative run search --depth 2

# 5. 运行 VERIFY（双模型验证）
x-creative run verify --threshold 6.0 --top 20

# 6. 查看最终验证结果
x-creative show verify --top 10

# 7. 运行 SOLVE（深度推理求解）
x-creative run solve --max-ideas 8 --auto-refine
```

也可以一次性运行所有基础阶段：

```bash
x-creative run all \
  -d "探索新的解决方案" \
  --target-domain general \
  --num-per-domain 2 \
  --depth 2 \
  --top 30
```

#### run 各阶段详细参数

**`run problem` — 定义研究问题**

| 选项 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--description` | `-d` | (交互式) | 问题描述 |
| `--target-domain` | `-t` | `general` | 目标领域 ID |
| `--context` | `-c` | `{}` | 领域上下文 (JSON) |
| `--constraint` | - | `[]` | 约束条件（可多次使用） |
| `--session` | `-s` | 当前 | 指定 session ID |
| `--force` | - | 关 | 强制重新执行 |

`--context` 是自由定义的 JSON 对象，用于向 LLM 提供领域背景信息（键名和值均可自由定义）：

```bash
x-creative run problem -d "设计一个开源 CLI 工具" \
  --target-domain open_source_development \
  --context '{"platform": "github", "language": "rust", "target_users": "developers"}' \
  --constraint "Must be implementable as a single-binary CLI tool"
```

**`run biso` — 远域联想**

| 选项 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-per-domain` | `-n` | `3` | 每个源领域生成的假说数 |

**`run search` — 假说扩展**

| 选项 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--depth` | `-d` | `3` | 搜索深度（迭代轮数） |
| `--breadth` | `-b` | `5` | 搜索广度（每轮扩展数） |

**`run verify` — 双模型验证**

| 选项 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--threshold` | `-t` | `5.0` | 最低评分阈值 |
| `--top` | - | `50` | 输出前 N 个假说 |

**`run solve` — Talker-Reasoner 深度推理求解**

前置条件：当前 session 的 `verify` 阶段已完成。

| 选项 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--max-ideas` | - | `8` | 从 verify 结果选取前 N 个假说 |
| `--max-web-results` | - | `8` | 每轮联网检索最大结果数 |
| `--auto-refine / --no-auto-refine` | - | 开 | 启用/关闭自适应风险精炼循环 |
| `--inner-max` | - | `3` | 内层循环最大轮次 |
| `--outer-max` | - | `2` | 外层循环最大轮次 |
| `--no-interactive` | - | 关 | 禁用交互式提问 |
| `--force` | - | 关 | 覆盖已有结果 |

所有 `run` 子命令均支持 `--session <id>` 和 `--force` 选项。

---

### 查看结果：`show`

```bash
# 查看各阶段结果
x-creative show problem [--session <id>] [--raw]
x-creative show biso [--top 10] [--raw]
x-creative show search [--top 10] [--raw]
x-creative show verify [--top 10] [--raw]

# 查看阶段 Markdown 报告
x-creative show report <stage>  # stage: problem, biso, search, verify

# 导出报告到文件
x-creative show report verify --output report.md
```

---

### 管理源领域：`domains` 和 `xc-domain`

```bash
# 列出源领域库
x-creative domains list --target open_source_development

# 查看特定源领域的详细信息
x-creative domains show thermodynamics --target open_source_development
```

`xc-domain` 是独立的 TUI 工具，用于交互式管理源领域配置：

```bash
poetry run xc-domain
```

支持三种操作：
- **手动添加领域**：输入领域名称 → Brave Search 搜索核心概念 → LLM 生成 structures 和 mappings → 审阅保存
- **自动探索领域**：输入研究目标 → 推荐 5-8 个候选源领域 → 智能去重（检测与现有领域的重合度）→ 批量生成
- **扩展现有结构**：为已有领域补充新的 structures

---

### 管理目标领域：`xc-target`

```bash
poetry run xc-target
```

TUI 工具，用于创建和管理目标领域配置：
- **创建向导**：基本信息 → LLM 并行生成元数据（constraints、evaluation_criteria、anti_patterns、terminology、stale_ideas）→ 可选复制已有源域 → 保存
- **查看与编辑**：分 Tab 查看各节详情，支持选择性重新生成

---

### 快速生成：`generate`

不需要 session，直接生成假说：

```bash
# 基本用法
x-creative generate "探索新的用户增长策略"

# 指定参数
x-creative generate "研究问题" \
  --num-hypotheses 30 \
  --search-depth 2 \
  --output hypotheses.json

# 快速测试
x-creative generate "测试问题" -n 5 -d 1
```

---

### 超图知识工具：`hkg`

构建和查询超图知识图谱（实验性功能）：

```bash
# 从目标领域 YAML 导入超图数据
x-creative hkg ingest --source yaml \
  --path x_creative/config/target_domains/open_source_development.yaml \
  --output local_data/hkg_store.json

# 构建索引（含可选 embedding 索引）
x-creative hkg build-index --store local_data/hkg_store.json
x-creative hkg build-index --store local_data/hkg_store.json --embedding

# 查询最短超路径
x-creative hkg traverse --store local_data/hkg_store.json \
  --start "entropy,temperature" --end "volatility" --K 3

# 查看超图统计
x-creative hkg stats --store local_data/hkg_store.json
```

---

### 配置管理：`config`

```bash
# 显示当前配置
x-creative config show

# 初始化用户配置文件（~/.config/x-creative/config.yaml）
x-creative config init

# 查看配置文件路径
x-creative config path

# 验证配置（三阶段：静态检查 → API 连通性 → 模型可用性）
x-creative config check

# 仅静态检查
x-creative config check --quick
```

---

### ConceptSpace 管理：`concept-space`

ConceptSpace 定义了变换式创造力的规则空间：

```bash
# 校验 ConceptSpace YAML
x-creative concept-space validate path/to/concept_space.yaml

# 比较两个版本的差异
x-creative concept-space diff old_space.yaml new_space.yaml
```

---

### 命令速查

| 命令 | 用途 |
|------|------|
| `x-creative answer -q "问题"` | 单入口深度研究（推荐，自动完成全流程） |
| `x-creative generate "问题"` | 快速生成假说（无需 session） |
| `x-creative session new "主题"` | 创建新的研究 session |
| `x-creative run problem -d "描述"` | 定义研究问题 |
| `x-creative run biso` | 远域联想生成 |
| `x-creative run search` | 假说空间扩展 |
| `x-creative run verify` | 双模型验证筛选 |
| `x-creative run solve` | 深度推理求解 |
| `x-creative run all -d "描述"` | 一次性执行全部基础阶段 |
| `x-creative show <stage>` | 查看某阶段结果 |
| `x-creative domains list` | 列出源领域库 |
| `x-creative config check` | 验证配置有效性 |
| `x-creative hkg ingest` | 导入超图数据 |
| `xc-domain` | 源领域 TUI 管理 |
| `xc-target` | 目标领域 TUI 管理 |

---

## Python API

### 单入口 AnswerEngine（推荐）

```python
import asyncio
from x_creative.answer.engine import AnswerEngine
from x_creative.answer.types import AnswerConfig

async def main():
    # 最简用法：一句话输入，自动完成全流程
    engine = AnswerEngine()
    pack = await engine.answer("如何利用跨学科视角发现新的解决方案")

    # 输出 Markdown 报告
    print(pack.answer_md)

    # 访问结构化数据
    print(f"目标域: {pack.answer_json['metadata']['target_domain']}")
    print(f"假说数: {pack.answer_json['metadata']['total_hypotheses_generated']}")

asyncio.run(main())
```

```python
# 自定义配置
config = AnswerConfig(
    budget=200,
    mode="exhaustive",          # quick / deep_research / exhaustive
    target_domain="auto",       # auto = 自动推断
    search_depth=4,
    search_breadth=8,
    hkg_enabled=True,
    saga_enabled=True,
    fresh=False,                # True = LLM 从零生成域
)

engine = AnswerEngine(config=config)
pack = await engine.answer("你的研究问题")

# 处理澄清请求
if pack.needs_clarification:
    print(f"需要澄清: {pack.clarification_question}")
    pack = await engine.answer("补充上下文后的问题")
```

### CreativityEngine（底层 API）

```python
import asyncio
from x_creative.core.types import ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine

async def main():
    problem = ProblemFrame(
        description="探索提高系统可靠性的创新方法",
        target_domain="engineering",
        constraints=["不增加硬件成本", "保持向后兼容"],
        context={"system_type": "distributed", "scale": "large"},
    )

    config = SearchConfig(
        num_hypotheses=20,
        search_depth=2,
        search_breadth=3,
    )

    engine = CreativityEngine()
    try:
        hypotheses = await engine.generate(problem, config)
        for hyp in hypotheses[:5]:
            score = hyp.final_score if hyp.final_score is not None else hyp.composite_score()
            print(f"[{score:.1f}] {hyp.description}")
            print(f"  Observable: {hyp.observable}")
    finally:
        await engine.close()

asyncio.run(main())
```

### 探索领域库

```python
from x_creative.core.domain_loader import DomainLibrary

library = DomainLibrary.from_target_domain("open_source_development")

for domain in library:
    print(f"{domain.id}: {domain.name}")

queueing = library.get("queueing_theory")
for structure in queueing.structures:
    print(f"  - {structure.name}: {structure.description}")
```

---

## 配置系统

配置按以下优先级加载（高优先级覆盖低优先级）：

1. **环境变量**（最高优先级）
2. **`.env` 文件**（项目目录）
3. **用户配置文件**（`~/.config/x-creative/config.yaml`）
4. **默认值**

> 推荐使用 `.env` 文件管理配置。复制 `.env.example` 即可开始。

### .env 文件配置（推荐）

```bash
cp .env.example .env
# 编辑 .env 填入 API Key 和其他配置
```

`.env` 文件内容示例：

```bash
# API Keys（必填）
OPENROUTER_API_KEY=sk-or-v1-your-key-here
X_CREATIVE_DEFAULT_PROVIDER=openrouter

# 可选 API Keys
# YUNWU_API_KEY=your-yunwu-key-here
# BRAVE_SEARCH_API_KEY=your-brave-api-key-here

# 基本配置
X_CREATIVE_DEFAULT_NUM_HYPOTHESES=50
X_CREATIVE_DEFAULT_SEARCH_DEPTH=3

# 评分权重（五项必须总和为 1.0）
X_CREATIVE_SCORE_WEIGHT_DIVERGENCE=0.21
X_CREATIVE_SCORE_WEIGHT_TESTABILITY=0.26
X_CREATIVE_SCORE_WEIGHT_RATIONALE=0.21
X_CREATIVE_SCORE_WEIGHT_ROBUSTNESS=0.17
X_CREATIVE_SCORE_WEIGHT_FEASIBILITY=0.15
```

### 用户配置文件

```bash
x-creative config init  # 创建 ~/.config/x-creative/config.yaml
```

```yaml
openrouter:
  api_key: "sk-or-v1-your-key-here"
default_provider: "openrouter"
brave_search:
  api_key: "your-brave-api-key-here"
default_num_hypotheses: 50
default_search_depth: 3
```

### 模型路由配置

不同任务使用不同的模型，以下为默认配置（可通过 `.env` 覆盖）：

| 任务 | 默认模型 | Temperature | 说明 |
|------|---------|-------------|------|
| creativity | anthropic/claude-sonnet-4 | 0.9 | 创意生成、远域联想 |
| analogical_mapping | anthropic/claude-sonnet-4 | 0.7 | 类比映射 |
| structured_search | openai/gpt-5.2 | 0.5 | Graph of Thoughts 搜索 |
| hypothesis_scoring | anthropic/claude-3-haiku | 0.3 | 假说评分 |
| logic_verification | openai/gpt-5.2 | 0.2 | 逻辑验证器 |
| novelty_verification | google/gemini-3-flash-preview | 0.3 | 新颖性验证器 |
| reasoner_step | anthropic/claude-sonnet-4 | 0.3 | Reasoner 多步推理 |
| talker_output | anthropic/claude-sonnet-4 | 0.2 | Talker 方案生成 |
| saga_adversarial | google/gemini-3-flash-preview | 0.4 | SAGA 对抗性评价 |
| blend_expansion | anthropic/claude-sonnet-4 | 0.8 | Conceptual Blending |
| transform_space | openai/gpt-5.2 | 0.6 | 变换式创造力 |

每个任务都有 fallback 模型列表，主模型失败时自动切换。

自定义模型配置（在 `.env` 中）：

```bash
# 修改特定任务的模型
X_CREATIVE_TASK_ROUTING__CREATIVITY__MODEL=anthropic/claude-3-opus
X_CREATIVE_TASK_ROUTING__CREATIVITY__TEMPERATURE=0.95

# 修改验证器模型
X_CREATIVE_VERIFIERS__LOGIC__MODEL=google/gemini-3-pro-preview
X_CREATIVE_VERIFIERS__NOVELTY__MODEL=google/gemini-3-pro-preview
```

> 模型名称必须使用 `provider/model` 格式。

### 实验性功能配置

以下功能默认关闭，需在 `.env` 中显式启用：

<details>
<summary><b>HKG 超图知识锚定</b></summary>

```bash
X_CREATIVE_HKG_ENABLED=true
X_CREATIVE_HKG_STORE_PATH=local_data/hkg_store.json
X_CREATIVE_HKG_K=3                              # 返回最短路径数
X_CREATIVE_HKG_IS=1                              # 相邻超边最小共享节点数
X_CREATIVE_HKG_ENABLE_STRUCTURAL_SCORING=true    # 启用 VERIFY 结构证据评分
X_CREATIVE_HKG_ENABLE_HYPERBRIDGE=false          # 启用 hyperbridge
```
</details>

<details>
<summary><b>MOME 质量-多样性档案</b></summary>

```bash
X_CREATIVE_MOME_ENABLED=true
X_CREATIVE_MOME_CELL_CAPACITY=10                 # 每个网格单元最大假说数
```
</details>

<details>
<summary><b>QD-Pareto 选择</b></summary>

```bash
X_CREATIVE_PARETO_SELECTION_ENABLED=true
X_CREATIVE_PARETO_NOVELTY_BINS=5
X_CREATIVE_PARETO_WN_MIN=0.15
X_CREATIVE_PARETO_WN_MAX=0.55
X_CREATIVE_PARETO_GAMMA=2.0
```
</details>

<details>
<summary><b>C-K 双空间调度</b></summary>

```bash
X_CREATIVE_CK_ENABLED=true
X_CREATIVE_CK_MIN_PHASE_DURATION_S=10.0          # 防振荡最小阶段持续秒数
X_CREATIVE_CK_MAX_K_EXPANSION_PER_SESSION=5      # 每会话最大 K-expansion 数
```
</details>

<details>
<summary><b>创造力搜索算子</b></summary>

```bash
X_CREATIVE_ENABLE_BLENDING=true                   # Conceptual Blending
X_CREATIVE_ENABLE_TRANSFORM_SPACE=true            # 变换式创造力
X_CREATIVE_MAX_BLEND_PAIRS=3
X_CREATIVE_MAX_TRANSFORM_HYPOTHESES=2
X_CREATIVE_RUNTIME_PROFILE=research               # interactive / research
```
</details>

<details>
<summary><b>SAGA 认知预算分配</b></summary>

7 项必须总和为 100%：

```bash
X_CREATIVE_COGNITIVE_BUDGET_EMERGENCY_RESERVE=10.0
X_CREATIVE_COGNITIVE_BUDGET_DOMAIN_AUDIT=9.0
X_CREATIVE_COGNITIVE_BUDGET_BISO_MONITOR=13.5
X_CREATIVE_COGNITIVE_BUDGET_SEARCH_MONITOR=13.5
X_CREATIVE_COGNITIVE_BUDGET_VERIFY_MONITOR=18.0
X_CREATIVE_COGNITIVE_BUDGET_ADVERSARIAL=22.5
X_CREATIVE_COGNITIVE_BUDGET_GLOBAL_REVIEW=13.5
```
</details>

<details>
<summary><b>VERIFY 置信度与异常检测</b></summary>

```bash
X_CREATIVE_MULTI_SAMPLE_EVALUATIONS=3             # 多次采样数 (1-9)
X_CREATIVE_POSITION_BIAS_CONFIDENCE_FACTOR=0.7    # 位置偏差衰减因子
X_CREATIVE_FINAL_SCORE_LOGIC_WEIGHT=0.4           # 逻辑验证权重（与下项总和 = 1.0）
X_CREATIVE_FINAL_SCORE_NOVELTY_WEIGHT=0.6         # 新颖性验证权重
X_CREATIVE_MAPPING_QUALITY_GATE_ENABLED=true      # 映射质量门控
X_CREATIVE_MAPPING_QUALITY_GATE_THRESHOLD=6.0     # 最低映射质量分
X_CREATIVE_MAX_CONSTRAINTS=15                      # 非用户约束库上限
```
</details>

> 完整配置项说明见 `.env.example` 文件。

---

## 目标领域配置

X-Creative 通过 YAML 文件定义目标领域。**文件名（不含 `.yaml`）即为领域 ID**。

### 配置文件查找顺序

| 优先级 | 目录 | 说明 |
|--------|------|------|
| 1 | `~/.config/x-creative/domains/` | 用户自定义 |
| 2 | `x_creative/config/target_domains/` | 内置 |

### 内置领域

| 领域 ID | 名称 | 说明 |
|---------|------|------|
| `open_source_development` | 开源软件开发 | 开源协作、维护与社区治理研究 |

### 创建自定义领域

```bash
# 创建配置目录
mkdir -p ~/.config/x-creative/domains

# 创建 YAML 文件（文件名 = 领域 ID）
vim ~/.config/x-creative/domains/product.yaml
```

YAML 格式：

```yaml
id: product
name: 产品创新
description: 互联网产品功能创新与用户体验优化

# 可选：领域约束
constraints:
  - name: user_value
    description: 功能必须为用户创造明确价值
    severity: critical       # critical | important | advisory
  - name: feasibility
    description: 方案必须在现有技术栈下可实现
    severity: important

# 可选：评估标准
evaluation_criteria:
  - 用户价值清晰度
  - 实现复杂度
  - 可衡量的成功指标

# 可选：反模式
anti_patterns:
  - 为技术而技术，忽略用户需求
  - 功能堆砌，缺乏聚焦

# 可选：术语表
terminology:
  DAU: 日活跃用户数
  留存率: 用户在一定时间后继续使用产品的比例

# 可选：已过时的想法（用于新颖性检查）
stale_ideas:
  - 过时想法 1

# 嵌入式源领域
source_domains:
  - id: queueing_theory
    name: 排队论
    name_en: Queueing Theory
    description: 研究排队现象的数学理论
    structures:
      - id: queue_dynamics
        name: 队列动态
        description: 到达率与服务率的关系
        key_variables: [arrival_rate, service_rate, queue_length]
        dynamics: 到达率超过服务率时队列增长
    target_mappings:
      - structure: queue_dynamics
        target: 用户等待体验
        observable: 平均等待时间 / 用户放弃率
```

使用自定义领域：

```bash
x-creative run problem -d "提高用户留存" --target-domain product
x-creative answer -q "如何提高用户留存" --target product
```

也可以使用 `xc-target` TUI 工具交互式创建目标领域。

---

## 输出格式

### 假说结构

每个生成的假说包含以下字段：

```json
{
  "id": "hyp_abc12345",
  "description": "基于排队论的 Issue 响应优先级模型",
  "source_domain": "queueing_theory",
  "source_structure": "queue_dynamics",
  "analogy_explanation": "Issue 提交和处理的到达率差异类似于排队系统中的服务压力...",
  "observable": "weekly_active_users / total_installs",
  "mapping_table": [...],
  "failure_modes": [...],
  "scores": {
    "divergence": 8.5,
    "testability": 9.0,
    "rationale": 8.0,
    "robustness": 7.5,
    "feasibility": 8.2
  },
  "final_score": 7.9
}
```

### 评分维度

| 维度 | 说明 | 高分特征 |
|------|------|---------|
| **Divergence** (发散度) | 与已知方法的语义距离 | 使用罕见的领域映射，概念新颖 |
| **Testability** (可检验性) | 能否转化为可测试的信号 | 公式明确，数据可得，可立即实现 |
| **Rationale** (领域合理性) | 是否有合理的领域机制 | 有清晰的因果逻辑，而非纯统计 |
| **Robustness** (稳健性) | 过拟合风险评估 | 条件简单，适用范围广，参数少 |
| **Feasibility** (可行性) | 数据和工程实现可达性 | 数据源明确、成本可控、落地路径清晰 |

### Session 数据结构

每个 session 的数据保存在 `{X_CREATIVE_DATA_DIR}/{session-id}/`：

```
local_data/<session-id>/
├── session.json              # session 元数据
├── problem.json / .md        # 问题定义
├── biso.json / .md           # BISO 结果
├── search.json / .md         # SEARCH 结果
├── verify.json / .md         # VERIFY 结果
├── solve.json / .md          # SOLVE 结果
├── answer.json / .md         # Answer Engine 最终报告
└── saga/                     # SAGA 数据（仅启用时）
    ├── events.jsonl           # Fast Agent 事件日志
    ├── directives.jsonl       # Slow Agent 指令日志
    └── reasoning_trace.jsonl  # Reasoner 推理步骤日志
```

---

## 开发

### 运行测试

```bash
# 运行所有单元测试
poetry run pytest tests/unit/ -v

# 运行测试并显示覆盖率
poetry run pytest tests/unit/ --cov=x_creative --cov-report=term-missing

# 运行集成测试（需要 API key）
OPENROUTER_API_KEY=your-key poetry run pytest tests/integration/ -v
```

### 代码检查

```bash
# 代码格式化
poetry run ruff format .

# 代码检查
poetry run ruff check .

# 类型检查
poetry run mypy x_creative/
```

---

## 源码架构

```
x_creative/
├── answer/             # Answer Engine（单入口编排器）
│   ├── engine.py       #   AnswerEngine - 顶层编排
│   ├── types.py        #   AnswerConfig, AnswerPack
│   ├── problem_frame.py #  ProblemFrameBuilder - 问题框定
│   ├── target_resolver.py # TargetDomainResolver - 目标域推断
│   ├── source_selector.py # SourceDomainSelector - 源域选择
│   └── pack_builder.py #   AnswerPackBuilder - 输出组装
├── core/               # 核心数据类型
│   ├── types.py        #   Domain, Hypothesis, ProblemFrame 等
│   ├── plugin.py       #   目标领域插件系统
│   └── domain_loader.py #  源领域库加载
├── llm/                # LLM 集成层
│   ├── client.py       #   OpenRouter / Yunwu 客户端
│   └── router.py       #   多模型路由器（任务级路由 + fallback）
├── creativity/         # 创造力引擎
│   ├── engine.py       #   主引擎（编排 BISO → SEARCH → VERIFY）
│   ├── biso.py         #   BISO 远域联想
│   ├── search.py       #   SEARCH 结构化搜索（Graph of Thoughts）
│   └── operators/      #   搜索算子：refine, variant, combine, blend, transform 等
├── verify/             # 双模型验证系统
│   ├── verifiers.py    #   LogicVerifier, NoveltyVerifier
│   ├── mapping_scorer.py # 映射质量门控（规则 + LLM 混合评分）
│   └── scoring.py      #   综合评分与最终分数计算
├── saga/               # SAGA 双进程认知架构
│   ├── coordinator.py  #   SAGACoordinator（Fast/Slow Agent 编排）
│   ├── solve.py        #   TalkerReasonerSolver（多步推理求解）
│   ├── reasoner.py     #   Reasoner（系统2，7步推理）
│   ├── belief.py       #   BeliefState 管理
│   ├── detectors/      #   统计异常检测器
│   ├── auditors/       #   领域约束审计器
│   └── memory/         #   跨 Session 模式记忆
├── hkg/                # 超图知识锚定
│   ├── store.py        #   HypergraphStore
│   ├── traversal.py    #   k-shortest hyperpaths
│   ├── matcher.py      #   NodeMatcher（exact/alias/embedding）
│   └── expand.py       #   hyperpath_expand + hyperbridge
├── config/             # 配置
│   ├── settings.py     #   Pydantic Settings（100+ 配置项）
│   ├── checker.py      #   配置验证（三阶段检查）
│   └── target_domains/ #   内置目标领域 YAML
├── session/            # Session 管理
│   ├── manager.py      #   SessionManager
│   └── report.py       #   阶段报告生成
├── domain_manager/     # xc-domain TUI 工具
├── target_manager/     # xc-target TUI 工具
└── cli/                # 命令行界面
    ├── main.py         #   Typer 入口
    ├── answer.py       #   answer 子命令
    ├── run.py          #   run 子命令
    ├── session.py      #   session 子命令
    ├── show.py         #   show 子命令
    ├── hkg.py          #   hkg 子命令
    └── concept_space_cli.py # concept-space 子命令
```

---

## 致谢

- [OpenRouter](https://openrouter.ai/) / [Yunwu](https://yunwu.ai/) — 统一的 LLM API 接口
- [Bisociation](https://en.wikipedia.org/wiki/Bisociation) — Arthur Koestler 的创造力理论
- [Graph of Thoughts](https://arxiv.org/abs/2308.09687) — 图结构思维探索
- [Conceptual Blending](https://en.wikipedia.org/wiki/Conceptual_blending) — Fauconnier & Turner 概念整合理论
- [Transformational Creativity](https://en.wikipedia.org/wiki/Computational_creativity) — Margaret Boden 变换式创造力理论
- [C-K Theory](https://en.wikipedia.org/wiki/C-K_theory) — Hatchuel & Weil 概念-知识设计理论
- [MAP-Elites](https://arxiv.org/abs/1504.04909) — 质量-多样性优化算法
- [Higher-Order Knowledge Representations](https://arxiv.org/abs/2601.04878) — MIT 高阶知识表示（HKG 理论基础）

## 许可证

MIT License
