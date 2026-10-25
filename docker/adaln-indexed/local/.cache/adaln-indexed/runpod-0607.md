# Indexed AdaLN tickets 06/07: user-run target batch

当前状态（2026-10-24，Asia/Shanghai）：07 benchmark 实现及 CPU 检查已准备；
06 目标验收材料已准备。RunPod 未创建、未上传、未执行。indexed strict 交付资格
默认关闭，性能结论不存在。候选运行清单是待验收范围，不是最终支持矩阵。

## 入口与资源

本文件、批次入口、benchmark 和全部配套自检均为 ignored 的本地验收材料，
不随普通 Git checkout 交付；本轮不向正式 benchmarks/tests/tools/docs 目录新增文件。

从包含当前修改和 `.cache/adaln-indexed/run_ticket05.py`、
`audit_ticket05_artifacts.py` 的 Git checkout 根目录运行。复用已有 Python 环境，
要求 torch、pytest、Triton、CUDA toolkit（nvcc/ptxas/cuobjdump）、C++ compiler、
Compute Sanitizer 和 NVIDIA driver 可用；此入口不安装依赖。
不依赖本地 checkout 的 `_C.so`，也不替换它。

最低资源门槛为 **8 GiB 空闲 GPU、16 GiB 可用主存**，包括 >5 GiB free 的大偏移测试。
建议 **24 GiB 或更大 GPU、64 GiB 主存**，并预留数十 GiB 证据盘空间。
B3/H3072 的独立 CPU references、保留多 suite fresh builds/cache/dumps 可能进一步
提高峰值；门槛不是 OOM 保证。OOM/资源不足是 FAIL/BLOCKED，不缩减矩阵换 PASS。
耗时取决于目标设备，可能超过十分钟；全部实际构建/GPU/采样只由用户启动。

一次批次顺序执行（以下仅准备命令，本轮未执行）：

```bash
cd /workspace/RL-Kernel-adaln-indexed
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
python .cache/adaln-indexed/run_ticket0607.py --start \
  --output /workspace/evidence/adaln-0607-new --sessions 3
```

选择目标机上正确的 Python/toolkit；输出必须是新目录。支持先准备、再分阶段：

```bash
python .cache/adaln-indexed/run_ticket0607.py --prepare --output /workspace/evidence/adaln-0607-new
python .cache/adaln-indexed/run_ticket0607.py --phase correctness --output /workspace/evidence/adaln-0607-new
python .cache/adaln-indexed/run_ticket0607.py --phase sanitizer --output /workspace/evidence/adaln-0607-new
python .cache/adaln-indexed/run_ticket0607.py --phase performance --output /workspace/evidence/adaln-0607-new
```

`--prepare` 只复制源码、生成压缩包和 CPU collection，不构建、不 JIT、不运行 GPU。
独立 benchmark 也有只打印计划的入口：

```bash
python .cache/adaln-indexed/benchmark_adaln_indexed.py --plan --output /tmp/unused
```

直接去掉 `--plan` 会执行 GPU；正式验收使用统一批次以强制前置条件。

## 阶段、精确清单与身份

1. 准备候选源码包、source/脚本 SHA256、Git HEAD/status、精确 `node-lock.json`、
   `performance-plan.json`；这些文件与 archive hash 绑定在 `candidate.json`。
   `candidate-source/` 和 `candidate-source.tar.gz` 是普通复制，不使用硬链接。
2. 核对资源、GPU UUID/name/capability、Python/Torch/CUDA/Triton、compiler/driver/
   sanitizer 路径、版本和 hash。恢复时重新核对相同环境。
3. fresh AdaLN-only build → CUDA correctness、真实无扩展 Triton correctness、
   CUDA/Triton integration、两 backend shared、两 backend large-offsets、两 backend
   target。复用 ticket05 runner；每个 native suite 都在新源码包 fresh build，
   不拿旧 `_C.so` 或另一 suite 的 binary 冒充新 build。
4. 全部 correctness 完成后，各 backend 的 memcheck/racecheck/initcheck/synccheck，
   独立进程、源码包、build/cache/dump；保留 PTX/SASS/resource 提取和 Triton audit。
5. 前四阶段全部成功后，07 三个独立 session，每个单独源码包/进程/空 cache，
   不在 sanitizer/profiler 下运行，复用本批 CUDA correctness 的明确 fresh binary。
   性能进程逐个核对实际 import/loaded binary hash。

本轮 CPU collection 的当前计数（全部为 COMMAND_PREPARED，无 GPU PASS）：

| suite | CUDA | Triton |
| --- | ---: | ---: |
| correctness | 278 | 319 |
| shared | 41 | 17 |
| large-offsets | 2 | 2 |
| target H3072 | 24 | 36 |
| memcheck / racecheck / initcheck / synccheck，各工具 | 199 | 153 |

integration correctness 另有 1 节点。完整 node IDs/参数保存在本地
`artifacts/adaln-0607-candidate-local-v4/collection/` 与 `node-lock.json`。
远端必须重新 collection 并锁定自己的候选；不能照抄本表计数代替实际清单。

`collection/<suite>/nodes.json` 和 `node-details.json` 是当前源码的精确节点/参数，
不以旧节点计数替代。执行后的节点必须与锁定列表逐项相同。
`source-sha256.json` 绑定所有相关源码/测试、benchmarks/build/tools 和 ignored runner。
每个 pytest 进程保留实际 import、binary、compiler/Triton identity、cache artifacts、
node set、JUnit、reports 和完整日志。native 提取文件与 binary 一起进入 receipt。
Triton cache 保留 specialization metadata/TTIR/TTGIR/LLVM/PTX/cubin/source、dump、
SASS/工具 hash。07 还记录每次 first-use 前后 cache metadata 名单及新增 keys；
最终保存整个 cache/dump 文件 hash。

恢复仍使用同一 `--output` 和阶段命令：已成功阶段的全量 receipt、source、node lock、
performance plan、archive、环境均须重新匹配。失败/中断阶段写新的 `*-attempt-N`
目录及独立日志；不覆盖旧失败、不重标旧 PASS，不自动重试。发生源码或环境变化时
必须新建 candidate/output，全量重跑，不允许跨来源拼接通过记录。

## 06 接收标准与待审范围

| 范围 | 当前锁定的选择/公开接口 | 接收标准 |
| --- | --- | --- |
| CUDA/Triton correctness | ticket05 `selection(backend, correctness)`；L2 public constructors、L3 resolver/trace、公开 qualification opt-in | 全节点 PASS；独立 accuracy gold + fixed bytes；真实 backend/profile；零 skip/xfail/xpass |
| L3 delivery path | `tests/runtime/test_adaln_indexed_activation.py` | 通过已有 `IndexedAdaLNQualification` 与公开 resolver；不 patch 私有变量；没有安装默认资格 |
| integration | `test_cuda_triton_indexed_small_bytes_and_independent_oracles` | 两 backend 各先通过独立 references，再跨 backend bytes |
| shared | 两 backend 原有 focused shared suites | 零 skip/xfail；shared arithmetic/support 不扩张 |
| large offsets | 两 backend `test_indexed_large_upstream_offset_rejected_before_copy[y/gate]` | 实际 >INT_MAX upstream span 在 copy 前拒绝；每 backend 2 节点，不以 metadata 替代 |
| H3072 workload | direct/L3 target nodes，另 Triton target launch suite | FP32/BF16 × S4096/6889/6032 × segmented/dispersed，B3；准确节点以 node-lock 为准 |
| launch/domain/autograd | correctness 中 threads128/256/512，Triton warps4/8 × tile64/128/256，H/B/eps/view/None/zero/bank cases | 原始字节、独立容差、dtype/classification、source leaf semantics 分开判断 |
| sanitizer | 两 backend 四工具独立小矩阵 | 零 errors/hazards/warnings、全部节点 PASS；另外审核实际 kernel instrumentation 范围 |
| generated code | fresh native binary、全部保留 indexed Triton specializations | 逐 specialization 审查 RN、subnormal、casts、selected-token fold、乘加舍入、无错误 atomic/reassociation；flags/提取本身不算验收 |

有限值含 signed zero/subnormal 要求固定 profile 原始 bytes；forward gate 保留原位宽
payload。仅 arithmetic NaN payload 豁免，分类、污染隔离、空 bank +0 仍必须验证。
metadata 节点数不代表 GPU launches，零 sanitizer summary 不自动证明所有 kernel
已被 instrument；审核还要核对工具选项、生成代码、实际执行节点及 kernel 范围。
RN sqrt/div 内部 FTZ/FMA 修正与 S1 BF16 等价特化必须按实测指令解释。

执行状态 `EXECUTION_COMPLETE_REVIEW_PENDING` 只说明命令结束，尚非支持资格。
远端返回后，独立 Standards/Spec 与证据审核需核对全部身份、实际节点/报告、
sanitizer kernel 覆盖、生成指令和资源项。完整证据接受后，才决定限定目标环境的
qualification 激活与最终支持矩阵冻结；本批不会安装默认资格。AdaLN-only build
不代表 full-extension、其他 GPU/runtime、ROCm/MUSA/NPU 或模型资格。

## 07 采样含义

| 维度 | 实现与标签 |
| --- | --- |
| baseline | 显式 `NativeAdaLNModulationOp` 的原 select01；不会通过自动 resolver 改成加速实现 |
| candidates | 显式 CUDA/Triton public constructors；另用 `AdaLNDispatch` 包装相同对象，核对实际 trace |
| 相同输入 | seed386，x/M/int32 index/random dy/dgate；FP32/BF16；B2/S8/H2 和 B3/H3072/S4096/6889/6032；all0/all1/alternating/fixed-random/连续分段；contiguous/strided |
| correctness preflight | 每个 case 先生成独立 CPU FP32 accuracy gold 和 fixed-byte reference；每个候选通过两者才进入正式 warm 采样；framework backward 只做 accuracy，不要求旧实现符合 indexed bytes |
| cold/JIT | 空 cache session 的首次候选 validation forward/backward 同步 wall latency；包含编译/cache lookup 与 production 开销，不声称纯 compiler 时间；共享 specialization 的后续 first-use 可能是 cache hit，以前后 metadata keys 区分 |
| warm kernel | 未提供：公开接口没有预分配 kernel-only API，不通过私有 kernel 绕过 validation 来制造结果 |
| warm production | `public-backend-production` 与 `dispatch-production`；全部必需 validation、index扫描、同步、copies、输出/temporary allocation 保留 |
| modes | forward（grad enabled，含建图）；backward-only（每样本 fresh graph，forward/pre-sync 在区间外）；forward+backward（含 forward/建图）；均用 autograd.grad，不含 leaf `.grad` 累加 |
| 计时 | wall 主指标：调用前同步、调用后同步；CUDA events 辅助，不用 event 值掩盖 CPU validation 开销 |
| 预分配 | inputs/index/dy/dgate 在外；public call 仍自己分配 outputs/dx/dm/partials/copies；event 对象在外；graph 析构在计时后 |
| warmup/repeat/order | 每 cell 5 warmup、30 repeats；6 种 backend permutation 周期平衡；case/session 偏移；逐样本保留执行 order |
| memory | 每样本 allocated/reserved baseline、peaks、allocated peak-minus-baseline；后者含输出/grad/copies/scratch，不是直接 scratch-only；backward-only baseline 已含 forward graph |
| sessions/statistics | 默认3个独立进程/空cache；原始 JSONL + wall/event samples；median/min/max/p95/stdev；session medians 独立汇总，身份/coverage 不一致拒绝合并 |

无预设 speedup 门槛。负收益、噪声和不稳定都是有效结果；尚无目标运行数据，不能声称
加速或性能验收。first-use validation 与 warm 样本分别保留，不将 CPU 自检里的字面量
统计样本当 GPU 性能。各 session 的温度/功耗与外部负载仍需目标执行记录和审核解释。

## 本地交付包与范围收敛（2026-10-24）

当前可转移源码包为 `artifacts/adaln-0607-candidate-local-v4/candidate-source.tar.gz`。
包内包括正式源码及本批次所需的 ignored benchmark/测试、runner、audit、
本说明和06/07/spec；不包含历史 GPU evidence、旧二进制或 Git 数据库。
目标须有同 HEAD `5490223a850fe9cb1efd5c3cceeb0fb964ee5c1c` 的干净 Git checkout
和已配置好的 Python/CUDA 环境；将此压缩包复制到目标（本轮不上传），
在该 checkout 中叠加当前候选文件：

```bash
cd /workspace/RL-Kernel-adaln-indexed
git rev-parse HEAD
tar -xzf /workspace/candidate-source.tar.gz --strip-components=1
python .cache/adaln-indexed/run_ticket0607.py --start --output /workspace/evidence/adaln-0607-new --sessions 3
```

目标会重新 collection/prepare/build。旧 `candidate-final-v2` 等准备包仅作历史，
不能与当前代码拼接验收。CPU自检命令：

```bash
python -m pytest .cache/adaln-indexed/test_benchmark_adaln_indexed.py .cache/adaln-indexed/test_ticket0607.py -q
```

本轮修正计时前 CUDA event 初始化和 session seed/warmup/repeat/计划一致性。
原07所要求的独立 warm kernel-only timing 仍未提供（公开 API 无预分配 kernel 接口）；
这是明确待办，不以完整 production timing 代替。硬件测量和接受仍待执行。

## 全部验收脚本本地化（2026-10-25）

参照上个 PR 的 `.cache/adaln-h100-image/docker/adaln-h100/acceptance/` 做法，
indexed benchmark 与 CPU 自检也归入本目录。计时定义保持本轮已审查实现；
没有沿用旧 shared 脚本的测量结果，也没有改变生产实现或现有 shared benchmark。
当前唯一候选为 `artifacts/adaln-0607-candidate-local-v4/`，此前 scoped-v3 等包
只作历史，不与当前脚本拼接。新复核记录在 `artifacts/adaln-0607-local-v4-check/`。
