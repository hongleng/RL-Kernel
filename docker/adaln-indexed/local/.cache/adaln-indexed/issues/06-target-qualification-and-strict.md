# 06: Accept fresh target evidence and open indexed strict for the validated scope

**Status（2026-10-24，Asia/Shanghai）：目标验收材料与统一批次入口已准备；RunPod 未执行，完整目标验收与资格激活待完成。indexed strict 交付资格默认关闭；支持矩阵尚未最终冻结。**

**Parent:** [indexed spec](../spec.md)。依赖 03/04/05 的完整实现和证据审核。
本轮没有修改 production arithmetic、backend selection、launch、qualification 默认值、
当前 `_C.so`，没有 Git 写入、远端任务、上传或云费用。

## 清单

- [x] 读取 03–07/spec、ticket05 最新修复/fresh-run/review-summary/实际 manifest 文件，按当前源码和脚本 hash 核对历史证据。
- [x] 保存实际起始 HEAD、tracked/untracked/ignored 状态、独立普通复制快照；保留已有差异和历史证据。
- [x] 准备精确 correctness/L2/L3/shared、四 sanitizer、两个 backend 的大偏移与 H3072/S4096/6889/6032 清单、资源门槛和接收标准。
- [x] 锁定候选源码包、source/scripts hashes、精确 node list、性能计划；批次支持分阶段、失败日志及新 attempt 恢复，恢复前重新核对身份。
- [x] 复用已有公开 `IndexedAdaLNQualification` → resolver → execution 机制进行候选验收；不 patch 私有变量、不新增资格绕过。
- [ ] RunPod fresh target build、完整 correctness、sanitizer/生成代码与07采样实际执行。
- [ ] 独立审核全部目标证据、实际 instrumentation/kernel 覆盖及生成指令语义，接受所有必要资源项。
- [ ] 完整证据接受后限定 scope、最终冻结支持矩阵，并通过已有公开机制激活资格；重新验证 L3/相关 policy/shared。

## 当前现场与历史证据有效性

本轮起始 HEAD：`5490223a850fe9cb1efd5c3cceeb0fb964ee5c1c`；tracked/untracked
为空。票据、runner 和历史证据是 ignored 文件，不能据 Git clean 忽略它们。
起始独立快照：`/tmp/adaln-0607-start-20261024-122741`（4,562 个文件；普通复制，
含起始 hash/status/patch/ignored 清单）。

[ticket05 fresh review-summary](../ticket05-fresh-20261024-114051-882959/review-summary.md)
所述局部 SM86 证据保留；其实际 manifest 是 `status.json`、`source-sha256.json`、
逐 suite `identity.json`/`result.json`/`nodes.json`/`reports.json` 等组成，未找到单独
`manifest.json`，没有凭空创建或改写历史 manifest。

本轮现场核对报告：`artifacts/adaln-0607-local/historical-identity-audit.json`。
历史 931 个源码 hash 中，5 个当前文件字节不同：runtime registry、AdaLN helper、
activation/Triton/strict-policy tests。逐文件 diff/AST 核对为格式及 import 顺序整理；
不是新源码 fresh GPU PASS。三个 ticket05 runner/audit 脚本字节一致；13 个已执行
suite 的 loaded binary 和 7,778 个记录 cache/dump 文件 hash 均一致。
原始节点计数只描述历史快照，当前节点由新 collection 决定。

历史局部结果为 CUDA278/Triton319 correctness、integration1、shared41/17、四工具各
CUDA199/Triton153，零 skip/xfail；它们不能补齐四组 BLOCKED_RESOURCE，不能替代
新目标环境的 fresh build/执行，不能推断 full-extension 或更广硬件资格。
完整算术指令语义及 sanitizer 实际 kernel 覆盖仍须独立审核。

## 覆盖与接收标准

| 范围 | 当前材料/公开接口 | 接收要求/待执行项 |
| --- | --- | --- |
| L1 accuracy/bytes | 独立 CPU FP32 LayerNorm/autograd gold、独立 fixed-byte reference；已有 literal discriminators | accuracy 与 bytes 分开；finite/signed-zero/subnormal 不以 allclose 替代；仅 arithmetic NaN payload 例外 |
| L2 CUDA | public constructor，128/256/512 threads、dtype/domain/route/view/autograd/batch-layout | 当前锁定全部节点通过；不能用 metadata 替代实际正边界/大偏移 |
| L2 Triton | public constructor，warps4/8 × tile64/128/256 | 独立 correctness/bytes、各 specialization artifacts 与目标环境身份 |
| L3 | public resolver + qualification opt-in，strict/ordinary/unsupported/failed-trace | 已实现的交付路径；actual backend/profile/trace 正确；默认 gate 仍关闭 |
| integration/shared | 跨 CUDA/Triton bytes 前先各过 references；两 backend shared 回归 | 零 skip/xfail/xpass；shared scope 不扩大 |
| 大偏移 | CUDA/Triton 各 y/gate，真实 >INT_MAX upstream span | >5 GiB free GPU；实际在 copy 前拒绝，4节点均待目标执行 |
| 真实 workload | B3/H3072；FP32/BF16 × S4096/6889/6032 × segmented/dispersed；direct/L3/额外 Triton suite | >=8 GiB free GPU、>=16 GiB available host；精确节点以新 node-lock 为准 |
| sanitizer | CUDA/Triton 各 memcheck/racecheck/initcheck/synccheck 的完整已选小矩阵 | 零 errors/hazards/warnings；全部测试节点 PASS；实际 instrumented kernel 覆盖另外审核 |
| generated code | fresh binary/native PTX/SASS/resources、Triton cache/dump/specialization 与生成产物 hashes | 审核 RN/FTZ/FMA 修正、casts、selected-token folds、empty bank、不当原子/重关联；不以 flags 或自动提取代替审阅 |

每 suite 要求非空精确 collection、全部 call/setup/teardown PASS、零 skip/xfail/xpass、
无子进程/collection 失败。资源阻塞不是 PASS。H/B/INT_MAX metadata 计数不是 launch
计数；真实 INT_MAX 巨大正向工作负载仍不声明执行。支持范围须与实际通过 cells 一致。

## 执行、恢复与证据

完整命令、顺序、资源、计时含义和审核范围见
[统一 RunPod 说明](../runpod-0607.md)。
入口：`.cache/adaln-indexed/run_ticket0607.py`；本轮只执行 `--prepare` 的 CPU
collection。完整目标批次由用户在准备好的 RunPod checkout 手动运行：

```bash
python .cache/adaln-indexed/run_ticket0607.py --start --output /workspace/evidence/adaln-0607-new --sessions 3
```

顺序是资源核对 → fresh build/完整 correctness（含大偏移、target、shared）→
sanitizer/生成代码提取 → 07独立进程性能。correctness 任何失败停止正式采样。
源码包/archive、node list、性能计划、逐进程 import/binary/compiler/runtime/GPU 与
Triton cache/generated artifacts 身份均保留。恢复使用新 attempt，保留原失败；
身份变化必须新 candidate 全量执行，不拼接不同来源的 PASS。

本轮 CPU 证据：`artifacts/adaln-0607-local/`；新候选 collection/source package：
`artifacts/adaln-0607-candidate-final-v2/`。远端环境不同需要在目标重新 prepare，
当前源码一旦修改也须重新锁定。候选锁定不代表支持矩阵最终冻结。
独立 Standards/Spec 最终复核记录见本轮 CPU 证据目录的 `review.md`。

最终资格激活与07性能验收必须等远端返回后独立审核；本轮到“材料已准备”为止。


## 本轮最终 CPU 与清单记录

相关 CPU 检查 **163 passed / 12.36s**，无 skip/xfail；仅执行 benchmark/batch harness、
独立 reference 和既有 strict-policy CPU 文件。Ruff、两个脚本 scoped mypy 与
`git diff --check` 通过。没有运行 GPU、扩展构建、文档全站构建或远端任务。
17 套当前 collection：correctness CUDA278/Triton319/integration1；shared41/17；
large-offsets2/2；target24/36；四 sanitizer 每工具 CUDA199/Triton153。
这些计数是本轮实际 CPU collection，不是 GPU PASS。

初始 `candidate-local` 和 `candidate-final` 准备包保留原样；它们对应审查修复前的
控制脚本，已明确被 `candidate-final-v2` 取代，不允许拿它们与当前源码拼接验收。
Standards、Spec 各自独立复核后均无未解决源码问题；远端证据审核另行进行。

## 范围收敛复核（2026-10-24，覆盖此前“最终候选”指向）

以 HEAD `5490223` + 五个未跟踪文件独立复核后，正式新增收敛为
`.cache/adaln-indexed/benchmark_adaln_indexed.py` 及其 CPU 测试；
一次性批次入口、批次自检、运行说明迁至 `.cache/adaln-indexed/`，
分别为 `run_ticket0607.py`、`test_ticket0607.py`、`runpod-0607.md`。
ignored 材料明确进入可转移源码包，普通 Git checkout 不含本地批次材料。

审查修复：CUDA events 在 wall 区间前完成延迟初始化；session 汇总拒绝
不同 seed/warmup/repeat/计划。当前候选为
`artifacts/adaln-0607-candidate-scoped-v3/`；此前 final-v2 等包及其审查/
CPU结果保留为历史，不能当作当前控制脚本的锁定身份。
新复核记录：`artifacts/adaln-0607-scope-review/review.md`。

07 独立 warm kernel-only timing 仍是明确缺口，当前只交付 first-use 与完整
production/direct-dispatch 计时；不得标成原07全部性能要求完成。
06完整目标执行、指令/instrumentation审核、支持矩阵冻结与激活均待远端证据。
本轮无远端任务、上传、构建或 GPU 执行。

## 2026-10-25 当前交付范围（取代此前正式新增/候选指向）

本轮 benchmark 与其 CPU 自检已全部迁至 ignored 本地目录：
`.cache/adaln-indexed/benchmark_adaln_indexed.py`、
`.cache/adaln-indexed/test_benchmark_adaln_indexed.py`。
正式 benchmarks/tests/tools/docs 目录本轮无新增；已有 shared benchmark 不变。
批次 runner 的导入、子进程路径、候选源码包清单及运行说明同步更新。
最新候选 `artifacts/adaln-0607-candidate-local-v4/`，本地检查与身份记录
`artifacts/adaln-0607-local-v4-check/`；旧包/历史结果保留，不能替代当前身份。
独立 warm kernel-only timing、目标运行与资格激活仍待完成。
