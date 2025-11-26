# 实盘引擎

本页涵盖本地 QMT/模拟与远程 qmt-remote 两种运行方式，聚焦最小可用流程与关键截图。

## 场景对比（先选路径）
| 场景 | 命令 | 必填变量 | 适合谁 |
| --- | --- | --- | --- |
| 本地 QMT | `bullet-trade live ... --broker qmt` | `QMT_DATA_PATH`、`QMT_ACCOUNT_ID` | 已安装 QMT/xtquant，想直接盯盘 |
| 模拟券商 | `--broker simulator` | 无 | 风控演练/联调 |
| 远程 qmt-remote | `--broker qmt-remote` | `QMT_SERVER_HOST/PORT/TOKEN` | 云/局域网托管，策略在本地 |

> 更多变量说明见 [配置总览](config.md)。

## 本地 QMT / 模拟最小步骤
```bash
cp env.live.example .env
# 核心变量（其余看 config.md）
DEFAULT_BROKER=qmt          # 或 simulator
QMT_DATA_PATH=C:\国金QMT交易端\userdata_mini
QMT_ACCOUNT_ID=123456
LOG_DIR=logs
RUNTIME_DIR=runtime

bullet-trade live strategies/demo_strategy.py --broker qmt
```
- 建议先用 `--broker simulator` dry-run，再切换真实账号。

### 实盘效果与日志
QMT 本机持仓/下单：
![qmt-trade-live](assets/qmt-trade-live.png)

委托状态（限价/市价/撤单）：
![miniqmt-limit-market-rollback](assets/miniqmt-limit-market-rollback.png)

同步下单日志（16s 队列 vs 0.5s 成交示例）：
![sync-order](assets/sync-order.png)

运行态持久化目录（`live_state.json`、`g.pkl`）：
![run-time-dir](assets/run-time-dir.png)

## 远程实盘（qmt-remote）
1) 远程 Windows 主机启动（需放行防火墙）：
```bash
bullet-trade server --listen 0.0.0.0 --port 58620 --token secret \
  --enable-data --enable-broker \
  --accounts main=123456:stock
```
首次启动会弹防火墙，请选择放行：
![server-firewall](assets/server-firewall.png)

2) 本地 `.env`：
```bash
DEFAULT_BROKER=qmt-remote
QMT_SERVER_HOST=10.0.0.8
QMT_SERVER_PORT=58620
QMT_SERVER_TOKEN=secret
QMT_SERVER_ACCOUNT_KEY=main
```
3) 运行策略：
```bash
bullet-trade live strategies/demo_strategy.py --broker qmt-remote
```
4) 服务端日志示例：
![joinquant-server-qmt](assets/joinquant-server-qmt.png)

## 生命周期与状态
- `initialize`：策略首次加载。
- `process_initialize`：进程重启后执行，适合恢复 g/连接。
- `after_code_change`：代码热更新后执行。
- 持久化：`RUNTIME_DIR` 自动写入 `live_state.json`、`g.pkl`，可在回调里加载恢复。

## 安全与风控
- `.env` 不入库；远程服务务必带 `--token`，必要时限制到内网。
- 策略层保留风控钩子：停牌过滤、最大回撤、成交失败重试等。
- QMT server 更多参数见 [QMT server](qmt-server.md)。
- 若要让聚宽模拟盘接入远程实盘，请看 [trade-support](trade-support.md)。

### 风控参数配置（.env）
| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MAX_ORDER_VALUE` | `100000` | 单笔订单金额上限，超过即拒单。 |
| `MAX_DAILY_TRADE_VALUE` | `500000` | 单日累计成交金额上限，超过即拒单。 |
| `MAX_DAILY_TRADES` | `100` | 单日最大交易笔数，超过即拒单。 |
| `MAX_STOCK_COUNT` | `20` | 持仓标的数上限（仅买入检查）。 |
| `MAX_POSITION_RATIO` | `20` | 单标下单金额占总资产的最大比例（需在调用风控时传入账户总资产，否则不生效）。 |
| `STOP_LOSS_RATIO` | `5` | 止损阈值，仅供 `check_stop_loss` 辅助判断，需策略自行下撤单。 |
| `RISK_CHECK_ENABLED` | `false` | 开启后台风控巡检（间隔由 `RISK_CHECK_INTERVAL` 控制），拒单仍由下单路径内置检查触发。 |

> 提示：默认值偏宽，真实账户请按资金量收紧（例如把金额/笔数/仓位阈值调小）；先用 `LIVE_MODE=dry_run` 验证风控配置，再切 `LIVE_MODE=live`。需要用到单标仓位限制时，请在风控检查时传入账户总资产以触发 `MAX_POSITION_RATIO`。***

实盘风控情况
![risk-control](assets/real-risk-control.png)
