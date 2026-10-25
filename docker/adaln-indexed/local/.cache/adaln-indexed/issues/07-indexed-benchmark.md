# 07: Reproducible indexed benchmarks against the original select01 fallback

**Status（2026-10-24，Asia/Shanghai）：本地 benchmark 实现完成，CPU 自检通过；RunPod/GPU 未执行，性能结论和性能验收尚不存在。indexed strict 交付资格未开放。**

**Parent:** [indexed spec](../spec.md)。03/04 实现是性能候选前提，benchmark 准备不依赖
06资格激活。本轮同时准备06验收材料，不把局部旧 GPU PASS 变成新目标资格。

## 清单

- [x] 新增 AdaLN-local `.cache/adaln-indexed/benchmark_adaln_indexed.py`，复用既有 benchmark hash/tool helpers 与 `tests.helpers.adaln_indexed.small_case/assert_bytes`。
- [x] 显式 CUDA、Triton、原 `NativeAdaLNModulationOp` select01 baseline；不使用自动 dispatch 选择 baseline；记录实际 backend/profile。
- [x] 同 x/M/index/dy/dgate；FP32/BF16、小诊断及 B3/H3072/S4096/6889/6032；all0/all1/alternating/fixed-random/真实连续分段；contiguous/strided。
- [x] forward、backward-only、forward+backward；独立 first-use/JIT-cache validation latency 与 warm production/direct-dispatch；不虚构 kernel-only 数据。
- [x] production 包含必需 validation/copies/临时分配；明确预分配、同步、warmup/repeat、平衡 backend 顺序、memory scope；保存原始样本与多进程 session 统计。
- [x] 每个候选先通过独立 accuracy gold 与 fixed-byte reference；旧 framework backward 不要求 indexed 固定字节相等。
- [x] CPU TDD 公开 CLI/callable 自检及独立 Standards/Spec 审查；无 registry/internal mock，无 GPU red→green 声明。
- [x] 准备顺序执行06→07的单批次、分阶段、失败保留与重新核对身份的恢复入口。
- [ ] RunPod 完整 correctness 后执行全部正式采样，核对 cold/warm cache/compiler/import/binary/环境身份与原始样本。
- [ ] 独立审核目标证据，接受性能报告；允许负收益或不稳定，不预设加速目标。

## 公开测试接口、反例与 TDD

用户已指定测试 seams：benchmark CLI 和公开 callable；CPU 自检验证 harness 行为，
不证明 CUDA/Triton correctness 或性能。变更不触及生产算术/资格 gate。

| 接口/行为 | 具体错误反例 | 本地结果 |
| --- | --- | --- |
| `make_inputs` | 同 seed 不同 x/M/dy/dgate；alternating route 错；strided fixture 偷变 contiguous | 先缺模块 RED → 实现 GREEN；所有 route/dtype/stride 再核对 |
| `prepare_call` | backward-only 把 forward 放进计时；图被二次 backward；forward+backward 提前建图 | 先缺 callable RED → 真实 Native public op GREEN；非零 gradients，叶 `.grad` 不累加 |
| `summarize`/CLI plan | 原始样本丢失；median/p95 错；空/NaN/负样本被接受；CPU 结果误标 GPU | 先缺 callable RED → 字面量统计 GREEN；plan=COMMAND_PREPARED，未创建输出 |
| `backend_order`/candidate check | backend 顺序永远固定；候选 y+1 仍通过；异常吞掉 | RED → 六 permutation 与真实 fixed/gold GREEN；错误候选拒绝 |
| CLI failure | 无 GPU 仍输出性能成功；结果失败却进程退出0 | CUDA_VISIBLE_DEVICES=''：FAIL、退出1、空results及失败报告保留 |
| `summarize_sessions` | 不同 binary/source/coverage 或失败 session 拼接 | RED → 字面量 parser fixtures GREEN；不作为真实 GPU 样本 |
| batch public helpers | skip/xfail/缺 call/变节点/变源码被计 PASS；身份篡改；失败日志覆盖；失败阶段无法恢复 | CPU 负例通过；恢复新attempt的真实 RED → GREEN；没有模拟GPU计时 |

完整 CPU red/green/最终检查日志、身份核对和独立审查：`artifacts/adaln-0607-local/`。
起始状态与独立快照见06；tracked/untracked 起始为空，ignored 历史证据单独核对保留。
本轮不运行 GPU、扩展构建、远端或长测试；不声称 GPU red→green。

## 覆盖、时序与标签

| 范围 | 实现定义 | 目标证据状态 |
| --- | --- | --- |
| baseline | 明确 Native select01；dispatch surface 包装同一个对象；actual_backend=pytorch、framework profile | COMMAND_PREPARED |
| candidates | public CUDA/Triton constructors；actual trace 验证；默认256 threads / warps4,tile128 | COMMAND_PREPARED；不以性能采样扩张资格launch矩阵 |
| inputs | seed386；两dtype × 4尺寸 × 5routes × 2layouts，80个 case；每case共用相同逻辑数值/upstreams | CPU生成自检PASS，目标未运行 |
| preflight | CPU FP32 accuracy gold与独立 fixed reference 先生成；每 backend 两 surface 候选核对后才正式采样 | CPU discriminator PASS；GPU未运行 |
| first-use/cold | 空cache独立session首次 validation forward/backward wall；含JIT/cache lookup/production；后续specialization可复用；保存cache前后metadata keys | COMMAND_PREPARED；不等同纯编译时间 |
| warm kernel | 未提供，公开接口没有预分配kernel API；不绕过validation调用私有kernel | 明确 NOT PROVIDED |
| warm production | direct public backend、完整 AdaLNDispatch，各自包含validation/copies/输出/partial分配 | COMMAND_PREPARED |
| forward | grad-enabled；含forward/建图；graph析构在计时后 | COMMAND_PREPARED |
| backward-only | 每样本新图；forward和prepare后同步不在区间内；backward必需upstream copies/分配仍在内 | CPU模式编排PASS；GPU未运行 |
| forward+backward | 含forward/建图/VJP；autograd.grad，不含leaf `.grad`累计 | CPU模式编排PASS；GPU未运行 |
| timing/order | 同步wall为主、CUDA event辅助；5 warmup/30 repeats；六种backend permutation与case/session偏移；保留原始order | CPU计划/编排检查PASS；无性能数据 |
| memory | allocated/reserved baseline/peak、allocated峰值差；含output/grad/copies/scratch；backward-only baseline含图 | COMMAND_PREPARED；不声称直接scratch-only |
| sessions | 默认3个独立进程/源码包/空cache；warmup后采样；无sanitizer/profiler；保留raw JSONL与wall/event samples、median/min/max/p95/stdev及session medians | CPU解析检查PASS；无波动结论 |

accuracy 与固定字节要求保持独立。原 framework backward 的求和次序可不同，故不强行
满足 indexed bytes；forward gate bits 仍检查。GPU dtype/forward/VJP accuracy 失败会
阻止其性能成功。统计不将 first-use 和 warm 采样混合，不用辅助event掩盖host开销。

## 执行与独立审核

[统一运行说明](../runpod-0607.md) 包含最低/建议资源、
单批次入口、分阶段命令、failure/identity/receipt/cache定义，以及06的接收标准。

```bash
# CPU计划，不build/JIT/GPU：
python .cache/adaln-indexed/benchmark_adaln_indexed.py --plan --output /tmp/unused
# 以下只准备，由用户在目标机执行：
python .cache/adaln-indexed/run_ticket0607.py --start --output /workspace/evidence/adaln-0607-new --sessions 3
```

阶段顺序：环境/资源→fresh build→完整correctness（含shared/large-offsets/target）→
sanitizer/generated-code→07。correctness失败停止正式采样；恢复前重核source/archive/
node/performance-plan/环境及全部先前receipt；失败attempt和原始日志保留。
远端返回后仍须独立审查实际 import/loaded binary、compiler/runtime/GPU、Triton
specializations/artifacts、cold/warm cache、全部原始样本、session波动及memory scope。

Standards、Spec 的最终复核与修复记录见 `artifacts/adaln-0607-local/review.md`。
实现/CPU通过、06材料已准备、RunPod未执行、硬件资格未开放、性能结论不存在这五项
分别记录；任何一次局部PASS都不能替代剩余验收。


## 最终本地结果

相关 CPU **163 passed / 12.36s**，无 skip/xfail；包含20个 benchmark/batch harness
CPU用例和143个既有reference/strict-policy用例。Ruff与两个脚本scoped mypy通过。
独立 Standards、Spec 的恢复/receipt命名空间/阶段顺序发现均已修复并复核，零未解决
源码问题；详见 `artifacts/adaln-0607-local/review.md`。最终候选包为
`artifacts/adaln-0607-candidate-final-v2/`，旧准备包保留并明确失效于当前脚本。
没有 GPU red→green、RunPod执行、资格激活或任何性能结论。

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
