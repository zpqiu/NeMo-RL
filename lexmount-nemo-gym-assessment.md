# Lexmount × NeMo-Gym/NeMo-RL 合作项目技术评估

> 内部文档 · 2026-07-20 · 基于 2026-07-14 对方简报及公开材料的初步评估
> 评估人: alexq · 用途: 团队对齐 + 答复对方确认项（A1–A4 / R1–R2）

## TL;DR

- **接入路径成立**：Lexmount 已向 NVIDIA-NeMo/Gym 提交 Draft PR #1865（浏览器 resources server，双后端 Playwright/Lexmount），结构符合 Gym 环境规范，可评审、需补课（无训练级验证）。
- **两个大断层**：① 对方材料里有两条互不相同的技术线，出训练曲线的那条（verifiers_agent + Stagehand + LLM judge）不是 PR 那条；② 我们的目标是 8B/9B VLM RL，但对方环境是纯文本 DOM 观察（全部代码零截图能力），且 NeMo-RL 的 nemo_gym 训练管线目前只支持文本。
- **关键简化已找到**：rollout 用 DOM 文本模式、仅 verify 时截图给 GPT-5.5 多模态 judge 打分——图像不进训练管线，胶水开发从 2–4 周缩到几天，且这正是原版 WebVoyager 的标准评法。
- **建议路线**：阶段一「文本 8B policy + DOM rollout + 截图 judge + Lexmount」全走已验证路径；阶段二再上真正的 VLM policy + 截图观察（需先打通 nemo_gym×VLM 管线，我们侧关键路径）。

---

## 1. 背景与材料

Lexmount（云端隔离浏览器基础设施，对标 BrowserBase）希望把远程浏览器接入 NeMo-Gym 作为 RL 训练环境，并与我们合作证明其对 browser-use RL 训练的价值。

| 材料 | 内容 | 状态 |
|---|---|---|
| 上游 PR | [NVIDIA-NeMo/Gym#1865](https://github.com/NVIDIA-NeMo/Gym/pull/1865)（回应 #644 BrowserGym 需求） | Draft，12 文件 +819 行 |
| 代码仓库 | [lexmount/lexmount-browser-lab](https://github.com/lexmount/lexmount-browser-lab)（训练脚本、WebVoyager 环境、8 节点交付包） | 公开，已 clone 分析 |
| 简报 | Lexmount × NeMo-Gym 研究简报（2026-07-14） | 已读（本地 HTML） |
| 已验证结果 | MiniWoB++ held-out reward 0.162→0.315（Qwen3-1.7B LoRA, 2×5090） | 仅证明闭环可工作 |
| 阻塞 | AWS p5 现货容量（InsufficientInstanceCapacity），8×H100 脚本从未实跑 | 对方侧 |

## 2. 两条技术线（评估的关键前提）

对方材料实际包含两条**互不相同**的技术栈，宣传数字与上游 PR 不能互相佐证：

| | 上游 PR 线 | Lab 训练线 |
|---|---|---|
| Agent harness | Gym `simple_agent` | Gym `verifiers_agent` + Verifiers 库 + Stagehand |
| 工具 | navigate/click/type/observe/finish | navigate/observe/act/extract（CDP 选择器 `[data-lex-id=lex-N]`） |
| 观察 | URL+标题+≤50 个交互元素（**无正文文本**） | URL+**1600 字符正文**+≤32 个元素；extract 另抓 3000 字符 |
| Reward | 规则（url/dom/answer 匹配） | GPT-5.5 文本 judge（经 dmxapi.cn 中转） |
| 验证程度 | 仅离线玩具站 rollout，**从未训练** | Qwen3-1.7B 真实训练跑通（2×5090） |

## 3. 可用性评估

### 3.1 PR #1865 评审意见（可修，非路线问题）

1. **无 scroll 工具**（docstring 提到但未实现）+ 50 元素截断 → 真实长页面不可用。
2. **观察无页面正文** → 信息查找类任务策略"看不见"答案，现状只够离线玩具站。
3. **observe() 逐元素 RPC** → 真实站点数百元素时每次观察极慢，云端 CDP 往返放大。
4. **会话回收只靠 verify()**，harness 崩溃泄漏云端 session；lab 线的 60s create-timeout/late-close 经验未进 PR；缺 TTL 兜底。
5. **Playwright 后端每 rollout 起完整 Chromium 进程**（对方自测 1000 并发 ~40–50GB RAM），未用 one-browser-N-contexts。
6. **离线示例数据用 `file://`**，切 `backend: lexmount` 即失效（云端浏览器读不到本地文件）——双后端"同一契约"在数据层不成立，示例站应改为可配 base_url 的 HTTP 服务。
7. 按 Gym 合入规范还缺：example_rollouts.jsonl、reward profiling 基线（多模型、方差<1%）、训练信号验证。

**R1 答复建议**：native SimpleResourcesServer 路径可接受（与现有 ~95 个环境一致），OpenEnv 适配留作后续。

### 3.2 Lab 训练线 pipeline 风险

1. **Pin 在 NeMo-RL v0.6 + monkey-patch**（GRPO 分组 patch、vLLM sleep patch、sitecustomize 运行时注入）。已核实 **GRPO 分组问题在当前 main 已修复**（`nemo_rl/environments/nemo_gym.py:456` 用 `input_message_log=[:1]` 做分组）→ **要求对方 rebase 到 main、去掉全部 patch**。
2. **Judge 走 dmxapi.cn 第三方中转** → NVIDIA 集群上不可接受（凭证与轨迹数据外流），需换内部 endpoint 或本地 judge。
3. **并发只验证过 8**（session create 60s 超时、Stagehand 15s DOM-ready），一个 64 轨迹 step 分 8 波收；多节点几百并发完全未验证。
4. 杂项：vLLM `served_model_name: gpt-4o` alias hack（为 Stagehand 兼容）；region 示例为 `beijing-1`（访问美国站点的延迟/连通性问题）。
5. **值得保留的经验**：infrastructure failure（session 失败、错误页）单独分桶 mask、不落 reward=0（`environment.py` 的 `infrastructure_browser_error_page`）；trajectory audit 可查。

## 4. WebVoyager 数据与可解性

- 600 条 × 14 个真实站点（Allrecipes/Amazon/Apple/ArXiv/BBC/Booking/Coursera/ESPN/GitHub/Google×3/HF/WolframAlpha），四字段 `{id, ques, web, web_name}`，**无 golden answer、无验收 spec** → reward 必须靠 judge。
- **任务时效性**：任务写死 2024 年日期/站点状态，2026 年相当比例过期或漂移 → 上训练前必须用**与训练相同的 DOM 观察管道**做一轮可解性筛查（勿用截图 agent 筛，那是 VLM 可解集）。
- **PR 线工具解不了 WebVoyager**（观察无正文）；lab 线 CDP 工具栈可以工作但有天花板（1600/3000 字符截断、32 元素、视觉化组件拿不到）。
- 原版 WebVoyager 是为"截图+GPT-4V"设计的 benchmark，纯文本 agent 跑它可解上限天然打折——跑通训练不需要截图，但把成功率顶上去、以及证明"对 VLM RL 有价值"需要。

## 5. Reward 设计（已定关键决策）

**决策：rollout 用 DOM 文本模式，仅 verify 时截图传 GPT-5.5 多模态 judge。**

- 图像不进 policy 上下文、不进训练 token 流 → nemo_gym×VLM 管线、harness 图像注入**全部不需要**，训练侧 100% 走现有文本路径。
- 这是原版 WebVoyager 的标准评法（GPT-4V 截图 judge）；对方现在的纯文本 transcript judge 是省钱降级版。
- 落点：PR 线 `verify()` 时浏览器还活着（close 在 finally），`_score()` 前 `page.screenshot()`；lab 线在 `judge_task_completion()`（`environment.py:205`）加图。
- **judge 证据 = 最终截图（宽 1024）+ 文本 transcript + agent 最终回答**，缺一不可（信息类任务答案在中途、只给截图不够）。
- 新增坑：①观察-奖励信息不对称（policy 看不见的任务 judge 正确判 0 → 死重任务，靠筛查解决）；②失败会话无截图 → 沿用 infrastructure mask；③截图 judge 残余 hacking 面 =“导航到看起来完成的页面”，可监控。
- **上训练前 judge 校准（不能省）**：temperature=0；同轨迹复判看方差；50–100 条人工 FP/FN；文本 judge vs 截图 judge 同批轨迹双跑、人工看分歧——顺带量化"加截图值多少"，本身是可发表的实验结论。
- judge 成本：~9600 次调用/150 步训练，量级可控。

**替代数据（无 judge 路线）**：MiniWoB++（JS 程序化奖励、站点近零成本、可复现对方 0.162→0.315——但其接入代码不在公开材料，**应作为 ask 要过来**）；WebArena-Lite（自托管 docker 站群 + 规则 evaluator，对方已有 setup 脚本但 runner 只支持本地 Playwright）。三者都必须有真实浏览器运行时——数据只是任务定义，换数据集不会架空 Lexmount 价值主张。

**网络拓扑坑**：云端浏览器访问自托管站群时网络方向反转（Lexmount 云 → 我们内网站点），对方自己也没解决（站群挂内网 IP、runner 拒绝 lexmount 后端）→ 列入对方确认项。

## 6. VLM 训练 gap（阶段二的前置工程）

已核实现状：

| 能力 | 状态 |
|---|---|
| NeMo-RL VLM GRPO（独立入口 `run_vlm_grpo.py`） | ✅ dtensor 与 megatron 均有 recipe（Qwen2.5-VL-3B 等），放大到 7–8B 属 config 级 |
| Gym 消息格式带图（`input_image` base64） | ✅ circle_click / labbench2_vlm 等已用于**评测** |
| **nemo_gym 路径 × VLM 训练** | ❌ **无先例**。`_postprocess_nemo_gym_to_nemo_rl_result` 只搬文本 token，无 pixel_values 回传/打包；入口走 tokenizer 非 processor |
| 对方环境截图能力 | ❌ PR + lab 仓库 screenshot 出现次数为零 |

阶段二需要：①NeMo-RL nemo_gym 多模态回传（postprocess/pixel_values/processor 路径/image placeholder token 与 `seen_token_ids` 连续性对齐——最大技术风险，估 2–4 周）；②环境侧截图观察（CDP 标准能力，量小但要设计分辨率/混合观察/历史帧裁剪，context 12K→24–32K）；③harness 多轮 `input_image` 注入。

## 7. 推进计划

| 阶段 | 内容 | 新变量 | 前置 |
|---|---|---|---|
| **M0a 管线冒烟** | PR 离线 5 任务 + Playwright + 8B 文本模型，1–2 GRPO step，只验管线 | 无 | 现在就能跑 |
| **M0b 训练信号** | MiniWoB++（首选，找对方要接入代码）或 WebArena-Lite，规则奖励、无 judge、全集群内网 | 数据 | `_score()` 扩展 + 数据转换 |
| **阶段一（原 M1 合并）** | 文本 8B + WebVoyager(筛后子集) + DOM rollout + **截图 judge** + Lexmount 后端 A/B | Lexmount 云、judge | judge 加图（几天）、judge 校准、任务筛查、对方配额承诺 |
| **阶段二（原 M2/M3）** | VLM policy（Qwen2.5-VL-7B / Qwen3-VL-8B 级）+ 截图观察 | VLM 管线、截图观察 | 第 6 节三项胶水；截图 judge 原样保留 |

原则：**每阶段只引入一个新变量**，不做"VLM+WebVoyager+Lexmount"三变量同时上线。

## 8. 清单

### 给 Lexmount 的 ask / 确认项

1. Rebase 到 NeMo-RL main，去掉全部 v0.6 monkey-patch 与 sitecustomize。
2. 把 MiniWoB++ 接入贡献到 PR 的 simple_agent 线（在上游可审路径上复现 0.162→0.315）。
3. PR 按 3.1 修改 + 补 Gym 合入规范要求的 baselining。
4. **并发配额与 SLA**：目标并发（几百级）下的 session 配额、create p95、数天训练窗口的成功率承诺（现状只验证过 8 并发）。
5. **计费**：万级 session/run 的单价与承担方——先谈好再跑。
6. Region 选择（贴近目标站点）+ 训练量级下反爬/封禁的风险说明（作为 V1 验收项）。
7. 云端浏览器访问客户私有站点的方案（隧道/白名单/专线）——决定自托管 benchmark 能否做 Lexmount A/B。
8. WebVoyager 数据再分发 license 确认。
9. Judge 一律不走 dmxapi 中转；由我们提供合规 GPT-5.5 endpoint。

### 我们侧 action items

1. M0a 冒烟（PR 线 + 8B 文本模型）。
2. WebVoyager 可解性筛查（DOM 观察管道 + 强模型，`gym eval run/profile`）。
3. Judge 校准实验（文本 vs 截图双跑 + 人工分歧核对）。
4. WebArena-Lite 站群集群适配（为无 judge 对照与阶段一备选）。
5. 立项 nemo_gym×VLM 多模态回传（阶段二关键路径，2–4 周）。
6. 答复 R1（native 路径可接受）/R2（验收标准按第 7 节阶段划分）及 A1–A4（入口/egress/资源/secrets 按集群实际情况）。

## 9. 参考：与 ServiceNow/BrowserGym 的关系

结论：**只在"可交互浏览器环境"层面类似，不能互相替代，本质是互补的两层。**

| | ServiceNow/BrowserGym | Lexmount PR #1865 |
|---|---|---|
| 本质 | benchmark 聚合框架（Gymnasium 接口） | 单个自研环境（Gym resources server） |
| 任务资产 | 9 个套件开箱即用：MiniWoB++、WebArena(+Verified)、VisualWebArena、WorkArena、AssistantBench、WebLINX、OpenApps、TimeWarp | 5 条离线示例任务，无 benchmark 套件 |
| Reward | 各套件自带成熟任务级 reward | 4 种规则 key |
| 观察空间 | DOM / AXTree / 截图可配 | 纯文本元素列表 |
| 动作空间 | 打磨过的原语（click(bid)/fill/scroll/tab/python action） | 5 工具（无 scroll） |
| 浏览器运行时 | 本地 Playwright，绑死训练进程旁 | 可插拔：本地 / Lexmount 云端 off-node |
| 训练接口 | Gymnasium 同步 API，接 NeMo-RL 需再包装 | NeMo-Gym Responses API 原生 |
| 维护 | 活跃（v0.14.3, 2026-01, 1.3k star） | Draft PR |

要点：

- **不能替代**：BrowserGym 的核心资产是 9 套件的任务+reward+标准化观察/动作空间（学术可比性锚点），Lexmount PR 在这层积累为零。且 Issue #644 字面诉求就是"integrate Browser Gym"，PR 交付的是自研环境——回应了精神、没回应字面。
- **反向也不能**：BrowserGym 没有浏览器 off-node 概念，浏览器就是训练节点旁的本地 Playwright——而这正是 Lexmount 的全部价值主张。
- **理想终态 = "BrowserGym on Lexmount"**：按 Gym external benchmark 规范在 agent 层包装 BrowserGym（先复现原 repo 数字、再复现 Gym 集成后数字），底层浏览器换成 Lexmount CDP（BrowserGym 底层即 Playwright，`connect_over_cdp` 理论上可直接对接）。同时满足 #644 字面需求与 Lexmount 价值展示，自带 MiniWoB++/WebArena 任务与规则 reward（覆盖 M0b 需求）。
- 待验证风险：各套件 reset/状态预置逻辑在远程浏览器上的兼容性；VisualWebArena 截图（CDP 支持，问题不大）；Gymnasium 同步 API 的高并发吞吐形态。

## 10. 第二次沟通问题清单（2026-07-23）

> 目标：把"能启动"变成"可被评审"。每个问题都附我们已掌握的事实（不必现场重新争论）和期望的答案形态。

### Q1 训练数据与 recipe reference —— 做 browser-use 任务验证，可用的训练数据有哪些？

**我们已知**：WebVoyager 600 条（无 golden answer、任务写死 2024 日期已过期漂移、reward 靠 judge）；PR 只有 5 条离线示例；MiniWoB++ 出过 0.162→0.315 但接入代码不在任何公开材料；WebArena-Lite 有自托管脚本但只用于 eval、runner 仅支持本地 Playwright。

要问的：
1. 你们计划用于**训练**（不是 eval）的数据集清单和优先级？各自配套的 reward 方案？
2. MiniWoB++ 那条 0.162→0.315 的**完整 recipe**（config + 数据 + 代码版本 + reward）能否交付/贡献到 PR 线，让我们独立复现？
3. WebArena-Lite 若做训练，任务集用什么（原 812 模板去除 Lite 测试集？WebRL 指令集？），evaluator 怎么接进 `verify()`？
4. WebVoyager 的任务过期问题你们如何处理？是否已有可解性筛查结果？数据再分发 license？
5. 是否考虑过"BrowserGym on Lexmount"路线（见第 9 节）——做 BrowserGym 的 Lexmount backend，任务与 reward 直接继承 9 个套件？

**期望答案形态**：每个候选数据集给出（任务数、reward 类型、已验证的训练 config、可复现 artifact 链接）。

### Q2 截图回传依赖 —— 你们的 recipe 依赖 gym 回传截图吗？

**我们已知（已核实）**：他们现有全部代码（PR + lab）零截图能力，所有 recipe 都是纯文本 DOM 观察 → **NeMo-RL 当前 nemo_gym 不支持图像回传这一点，不阻塞复现他们的任何已有结果**。

要问的：
1. 确认：你们所有已验证/计划中的 recipe 都不依赖截图回传？roadmap 里哪个阶段需要视觉观察？
2. 是否接受我们的中间态设计：**rollout 纯 DOM + 仅 verify 时截图给多模态 judge**（图像只进 judge、不进训练管线）？env 侧的截图抓取（CDP screenshot）你们负责实现吗？
3. 若走到 VLM 阶段：env 侧截图观察（分辨率/混合观察/历史帧策略）你们是否愿意设计实现？NeMo-RL 侧多模态回传由我们排期（2–4 周）。

### Q3 两条技术线 —— PR 和 lab 是不同的实验，你们想让我们验证哪个？

**我们已知**：PR 线（simple_agent + 规则 reward，从未训练过）与 lab 线（verifiers_agent + Stagehand + gpt-5.5 judge，1.7B 已跑通）互不相同，宣传数字来自后者、不能佐证前者。

要问的：
1. 你们希望 NVIDIA 集群验证的到底是哪条线？**我们的立场：上游可审的只有 PR 线，建议以 PR 线为准补训练验证**；lab 线的工程经验（session 超时处理、infrastructure 失败分桶）应回流进 PR。
2. 两条线是否有合流计划？Stagehand/verifiers 依赖在上游化时如何处理？
3. 硬性要求：rebase 到 NeMo-RL main、去掉全部 v0.6 monkey-patch（分组问题 main 已修复）；judge 不得走 dmxapi 第三方中转。
4. PR 按本文档 3.1 的 7 条意见修改 + 补 Gym 合入规范的 baselining，时间线？

### Q4 并发能力 —— Lexmount 能支撑多大并发的训练？

**我们已知**：只验证过 **8 并发**（session create 60s 超时、一个 64 轨迹 step 分 8 波收）；正式多节点训练需要几百并发；一次 150 步训练 ≈ 万级 session。

要问的：
1. 单项目/单账号的 session 并发配额上限？256 / 512 / 1024 并发下 session create 的 p95 延迟实测数据？
2. 数天训练窗口的会话成功率 SLA？失败会话的事后审计能力？
3. Region 布局（访问美国站点的推荐 region）？训练量级（几十万次页面访问）下的反爬/封禁风险评估？
4. 计费：万级 session/run、每条数分钟，单价和承担方？
5. 云端浏览器访问客户私有站点（自托管 WebArena 站群）的方案：隧道/IP 白名单/专线？——决定自托管 benchmark 能否做 Lexmount A/B 对照。

**期望答案形态**：并发-延迟-成功率的实测曲线（哪怕小规模），而非"理论上支持"。这直接决定可用 recipe 的 `num_prompts × group_size` 上限与 rollout 波数。

---

*来源：PR #1865 diff、lexmount-browser-lab@main（2026-07-14）、对方简报（2026-07-14）、ServiceNow/BrowserGym README（2026-07-21）、NeMo-RL main 与 Gym submodule 代码核查。所有"已核实"结论均对应具体文件行号，见正文。*
