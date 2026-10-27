# indexed AdaLN 本 PR 验收范围 v2（2026-10-27，Asia/Shanghai）

这是本 PR 新批次的范围修订，取代新批次的旧默认性能预算。旧 ticket06/07、
runpod-0607、expected-performance-plan 和历史批次保持历史身份；不能用 v2
给旧三 session 计划补 PASS。普通算子 Git checkout 不含这些 ignored 材料。
本次修改只涉及验收基础设施，不改生产算术、registry、资格 gate 或正式测试。

## 必须支持的结论与证据预算

| 义务/风险 | 本次覆盖 | 接受条件与结论边界 |
| --- | --- | --- |
| indexed FP32/BF16 accuracy、固定字节、L2/L3、支持域及 shared 回归 | 9 correctness suites，全部原节点，含两 backend 实际大偏移和 H3072 三尺寸 workload | 独立 gold/fixed refs、所有 call/setup/teardown PASS，零 skip/xfail/xpass；资源不足为失败 |
| kernel 安全性与算术契约 | 两 backend 各 memcheck/racecheck/initcheck/synccheck，8 suites，不删节点 | tool 零 errors/hazards/warnings 且测试全通过；实际 instrumentation/specialization/指令语义另行人工审核 |
| 对 select01 baseline 的可比生产调用性能 | 18 cases；两 dtype；small alternating contiguous/strided（4），B3/H3072 × S4096/6889/6032 × segmented/random contiguous（12），S6889 random strided（2） | 每 backend/surface 独立执行 correctness precheck；候选 FP32 gold + fixed bytes，baseline FP64 accuracy；同一逻辑输入/upstream |
| direct 与 dispatch 开销 | 所有 18 cases：direct forward/backward-only/forward+backward；dispatch forward+backward | 逐 case wall 主、event 辅，raw samples、峰值内存、first-use 与 warm 分开；dispatch 单独 F/B 开销不宣称 |
| 采样重复性 | 1 个独立进程，3 warmup/12 repeats；backend 六种排列，每排列 2 个正式 repeat | 只支持本进程内离散度、逐 case 相对延迟/负收益；不宣称进程间稳定性，不预设加速门槛 |
| 尚未提供/额外研究 | kernel-only 独立计时、全路由/布局性能交叉、跨进程稳定性 | 明确不作为本 PR 已证明结论；如 reviewer 要求，需另定有依据的范围；旧 ticket07 kernel-only 缺口仍存在 |

完整 correctness 不等于全性能交叉。all0/all1 等路由的正确性仍由原测试保证，
本性能范围不能推导未采样路由的加速。性能数据不能激活 qualification。

## 展开后的工作量

- 原计划：17 suites；9 次 native build；80 cases × 3 backends × 2 surfaces ×
  3 modes × 30 repeats × 3 sessions = **129,600 raw samples**；含 5 warmup 共
  **151,200 timed calls**；另有每 case/session 的 6 次验证（1,440 次 VJP 对比）。
- v2：CPU collection 从 17 次降为 **11 次**，相同 backend/sanitizer selection 只收集一次；各工具保留独立 node lock，执行后再次比对。**17 suites、节点不变**；1 次 batch fresh native build，8 次身份核对后复用，
  每 suite 独立源码包/进程/cache，Triton 保持 extension-free；native 架构带 `+PTX`，保留离线指令审核所需 PTX；sanitizer 是 runtime
  instrumentation，build flags 一致，不以复用绕过任一种工具或 backend。
- 性能：18 × 3 × (3 direct modes + 1 dispatch mode) = **216 cells**；
  12 repeats = **2,592 raw samples**；含 3 warmup = **3,240 timed calls**；
  另有 **108 次独立 VJP 对比**。timed calls 减少约 97.9%，不等于驻留成本减少 97.9%。
- native 指令提取从 9 次降为 **1 次**（同 binary/tool/options），9 个 native suite 各自记录引用与 hash；Triton 仍按各自实际 cache/specialization 提取。CUDA event pair 在首次 validation 后初始化一次，全部串行样本复用，同步和计时边界保持不变。
- 每 cell backward-only 仍逐样本准备新图，不把 graph/cache/PASS 当静态产物复用。
  单个 session 失败将保留 raw/checkpoint 并在新进程重跑；不跨进程拼接 cell。
- 性能 fixture/gold/fixed/FP64 baseline oracle 绑定源码/完整 seed 生成规则、输入及
  upstream、dtype/shape/route、eps、Torch/Python/线程/确定性环境和 payload hash。
  contiguous/strided 共逻辑值时共享数学 oracle：18 cases 最多 **14 个 oracle payloads**，
  而不是每 backend/surface 重算。实际 layout/leaf 在 CPU/GPU 重建并检查，每个后端仍比对。
  半写入 payload 保留 `.incomplete-N` 后重建；已发布 hash 不匹配则停止。
  correctness 测试内的独立 oracle 不改写、不注入缓存；其重算是现存测试契约的一部分。

尚无当前 H100 完整阶段耗时或本次实际费率，不作美元或总时长承诺。历史 SM86
耗时和 kernel 样本不能估算整单 Pod 费用；准备、CPU gold、JIT、sanitizer、PTX/SASS、
归档、传输和等待全部占驻留时间。64 GiB host RAM 建议保留，GPU >=8 GiB free / host
>=16 GiB available 是原下限而非峰值保证；oracle、cache、重复源码和解包审核需要充足磁盘
（当前锁定容量预算与模板设置见下节）。不要为估时额外创建 session/Pod。

## 用户下一次可直接执行的入口

本轮准备存储修复镜像；发布身份以成功 Actions 的 image-evidence/image-reference.txt 为准。
旧 `520e90e` 镜像缺少存储防护，不能用它运行下列新批次。
本地先检查并生成可审核 Docker context（无 Docker build、无 GPU、无网络下载）：

```bash
cd /home/dnd/personal-project/rl-kernel-adaln-h100-infra
python3 docker/adaln-indexed/prepare_context.py --out /tmp/adaln-indexed-pr-v2-context
```

Actions 发布成功后，使用新 image-evidence 中的 digest，在同一完整 H100 Pod、
镜像终端 `/opt/RL-Kernel-indexed` 运行；容器默认仍 `sleep infinity`：

```bash
cd /opt/RL-Kernel-indexed
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export ADALN_OUTPUT=/workspace/evidence/adaln-indexed-storage-v1
python .cache/adaln-indexed/run_ticket0607.py --start --profile pr-v2 --output "$ADALN_OUTPUT"
```

该入口依次 CPU collection/锁定 → `python -S` 隔离 CPU 回归预检（检查实际 package 路径及无扩展导入，禁用 site/.pth hooks） → target/compiler/资源核对 →
fresh build/完整 correctness → 四 sanitizer → 生成代码 → 性能。任何失败停止后续。
建议在 tmux 执行。`progress.json` 给当前 stage/attempt；`state.json` 给已完成 receipt；
各 command 的 `.timing.json` 记录起始时间/耗时/退出状态；性能日志中的 ORACLE/CASE COMPLETE、session 内 `checkpoint.json` 和逐条 flush 的
`samples.jsonl` 给进度。CPU 预检只证明 harness，不是 GPU PASS。Docker 的既有 CPU 检查仍显式 deselect
5 个依赖 native-selection 假设的参数组合（runner CPU selection 中 3 个 test 名）；不计为目标 GPU PASS。

保持相同源码/环境/输出；恢复时去掉 `--start`：

```bash
python .cache/adaln-indexed/run_ticket0607.py --profile pr-v2 --output "$ADALN_OUTPUT"
```

成功阶段按 receipt 跳过，失败阶段新 attempt；源码、锁、binary、compiler/runtime 或
已有证据变化即拒绝恢复。`--phase correctness|sanitizer|performance` 仍有前置检查；
局部完成是 PHASE_COMPLETE，只有所有阶段完成才是 EXECUTION_COMPLETE_REVIEW_PENDING。
旧范围只保留历史计划与起始脚本快照，不在当前入口提供执行选项；不要用 v2 恢复旧批次。
旧范围未跑完不构成 v2 的追加义务；v2 按上面的本 PR 结论单独审核。

## 同一路径回收，失败即停

运行/归档/发送始终使用同一个 `ADALN_OUTPUT`。先结束运行再打包，禁止边写边回收；输出目录独占锁会拒绝并行恢复或运行中打包。
标准库工具在 pack 前核对完整清单、receipts、report/raw cells；tar dereference 仅允许
batch 内 symlink；任何失败非零退出，不调用 sender。全文件 index 覆盖日志/失败
attempt/cache/generated code，tar/checksum 不排除 binary 或 JIT cache。

```bash
python .cache/adaln-indexed/evidence_io.py --pack "$ADALN_OUTPUT"
# 仅在已配置 runpodctl 时，可把上一条加 --send；先打包和隔离解包检查，再发送 archive/checksum。
```

记录输出的 `CANDIDATE` hash，下载 tar.gz 和 .sha256 后，在本地运行同版本脚本：

```bash
python3 /home/dnd/personal-project/RL-Kernel-adaln-indexed/.cache/adaln-indexed/evidence_io.py \
  --verify /path/to/adaln-indexed-storage-v1.tar.gz --candidate-sha256 COPY_CANDIDATE_HASH
```

verify 检查 checksum、成员安全性并隔离解包，核对批次身份、文件 index、原始状态、
套件/JUnit/日志、生成产物 receipt、性能计划及每个 raw cell/repeat 和 summary。
只有**本地** verify 成功才能宣布回收完整并建议结束资源；这不是证据接受或资格激活。
不会自动终止 Pod。空归档、错批次、缺文件和仅 checksum 正确都不能回收成功。
用户主动停止/失败时 pack 与 verify 都显式加 `--partial`，只证明已产生证据回收，
未完成义务仍可见；不为回收补跑 GPU。打包失败保留原目录和 `.tmp`，修复后保留失败
临时包再重试，不能发送 `.tmp`。


## 全部脚本与自动调用顺序

| 阶段/所在机器 | 入口 | 实际任务 |
| --- | --- | --- |
| GitHub Actions | `.github/workflows/adaln-h100-image.yml` | checkout infra/source 固定身份；调用 context 脚本；Buildx 构建 runtime 并推 GHCR；导出 image evidence；无 GPU 运行 |
| Actions/context | `docker/adaln-indexed/prepare_context.py` | 干净固定源码 Git checkout + 显式 acceptance overlay；校验 source.json overlay hash；写 source-profile，保留无凭据 Git 元数据 |
| Docker build | `docker/adaln-indexed/Dockerfile` → `image-acceptance/selfcheck.py` | 源码、Git、版本/依赖、时区、extension-free fallback 检查；不 build CUDA extension |
| Docker build | `run_ticket0607.py --cpu-preflight`，然后 `--prepare` | 隔离运行 6 个 harness CPU 文件和 reference/policy 两文件，明确 deselect 5 个 native-selection 参数；11 个 collection 进程锁 17 suite；与 image 锁逐项比较 |
| Pod，用户唯一验收启动入口 | `run_ticket0607.py --start --output "$ADALN_OUTPUT"` | prepare/CPU 预检 → target/资源 → 下列 17 suites → artifacts → benchmark；后续任何失败停止，不自动重试 |
| Pod，runner 自动调用 | `run_ticket05.py BACKEND SUITE` | 9 correctness suites：CUDA278、Triton319、integration1、shared41/17、large-offsets2/2、target24/36，共720节点调用；fresh native build 1 次，其他 native suite 复用；每个 suite 独立进程执行 |
| Pod，runner 自动调用 | `run_ticket05.py BACKEND sanitizer --sanitizer TOOL` | CUDA/Triton 各 memcheck/racecheck/initcheck/synccheck；每工具199/153节点，共1408节点调用；`--target-processes all`；不被其他 backend/tool 的 PASS 代替 |
| Pod，runner 自动调用 | native `cuobjdump` / `audit_ticket05_artifacts.py` | native PTX/SASS/resources 只提取1次并按身份共享；7个 indexed Triton suite 分别审核缓存/生成物，shared Triton单独保留；产物提取不代表人工语义/覆盖审核完成 |
| Pod，runner 自动调用 | `benchmark_adaln_indexed.py` | fresh process/空 Triton cache；18 cases、3 backends；direct三模式、dispatch完整F+B；216 cells；3 warmup/12 repeat；2592 raw samples，含warmup3240 calls |
| Pod，用户回收入口 | `evidence_io.py --pack "$ADALN_OUTPUT" [--send]` | 锁住已停止的输出目录，核对、归档、隔离内容检查；成功后才可 send；仍报告 LOCAL_VERIFICATION_PENDING |
| 本地，用户回收确认入口 | `evidence_io.py --verify ARCHIVE --candidate-sha256 HASH` | checksum、隔离解包、candidate、全文件 index、reports/raw samples/共享指令引用等完整检查；通过才报告回收完整 |

6 个 CPU harness 文件是 `test_ticket0607.py`、`test_benchmark_adaln_indexed.py`、
`test_acceptance_reuse.py`、`test_publish_preflight.py`、`test_disk_pressure.py`、`test_storage.py`。它们测试入口、复用/恢复、
归档负例、实际子进程导入隔离以及 event 调度；不是 GPU 测试通过证明。

`run_ticket01.py`～`run_ticket04.py`、`run_ticket05_all.py` 和旧
`followup_benchmark.py` 不在本批次调用链中；不需要逐个手动跑。
恢复仍使用相同 runner/output 并去掉 `--start`，不额外运行一套旧批次。
回收失败/主动停止批次时 pack/verify 均加 `--partial`，只确认已有材料回收。


## 满盘事故后的存储修复（2026-10-27）

性能范围与所有正确性节点不变。旧 `520e90e` 镜像未包含本节防护，采用新镜像 digest
并使用新输出目录；不拼接丢失的旧 Pod 证据。

### 容量模型与模板

14 个逻辑 oracle case（layout 数学参考复用），每份包含完整 inputs/upstreams、fixed、
FP32 gold、FP64 baseline。CPU 小样本实测旧 view 会连同未使用的 backing storage 保存；
新实现仅 clone 为紧凑 storage，保留 dtype/values/bytes，实际 strided backend 执行仍独立。
对 B/S/H，令 N=B*S*H、M=2*B*3*H、d=FP32的4或BF16的2字节，则紧凑 tensor 总量为
`(3*N+M)*(2*d+4+8) + B*S*4`。每文件另留1MiB序列化元数据上限，写后检查实际大小。

| 项 | 上限/预留 |
| --- | ---: |
| 14份完整紧凑 oracle | 31.574 GiB |
| 其余产物（native、JIT/cache/dump、源码副本、报告、attempt、临时文件）预算池 | 8 GiB |
| evidence 总量预算 | 39.574 GiB |
| 不可压缩归档预留（总量×1.02 + 64MiB） | 40.428 GiB |
| 同时隔离解包预留（总量×1.05 + 256MiB） | 41.803 GiB |
| 额外控制余量 | 1 GiB |
| 新空卷启动所需可用空间 | **122.805 GiB** |
| 加文件系统余量、向上取25GB档后的 Volume Disk 配置 | **150 GB** |

8GiB 是明确的运行预算池，不是实测 GPU 缓存峰值；超出会 BLOCKED_STORAGE，保留材料，
再依据实际情况审查预算。150GB 也是模板推荐容量，实际可用空间检查才决定能否启动。
已有批次恢复时按已占 evidence 扣除未来增量，归档和解包预留仍保留；旧 attempt 不自动删除。

模板：Private / NVIDIA GPU，Container Disk 40GB 用于系统，Volume Disk **150GB**，
Mount Path **/workspace**，Start Command **sleep infinity**，不配置 Network Volume。
实际 Pod 应显示该 Volume Disk；目录叫 `/workspace` 不能证明挂载。GPU/主存建议沿用完整
H100、至少64GiB主存；本轮没有用 GPU 或大 CPU workload 验证主存峰值。

本地可先看简短 CPU 预算，不创建目录或查询 GPU：

```bash
python .cache/adaln-indexed/run_ticket0607.py --storage-plan --output /tmp/unused
```

### 入口防护与停止行为

`--start` 在 collection/CPU预检/GPU查询前检查输出位于独立 `/workspace` 挂载，拒绝
root overlay、tmpfs、ramfs，以及覆盖输出的嵌套挂载；同时检查实际 free bytes/inodes。
挂载信息不独立证明云平台生命周期，实际 Pod 的 Volume Disk 配置仍须与模板核对。
恢复与各阶段会重新检查容量，结果保存在 `storage-checks/`，预算锁在 `storage-plan.json`。

GPU 子进程每秒由父进程做常数成本的可用空间/inode检查，不在采样期间遍历/hash目录。
超过本批未来增量配额时向独立进程组发 SIGINT，最多等主进程2秒，再对残留进程组发 SIGKILL；阻止依赖阶段
继续运行。此监督不是 agent 对远端长任务的反复轮询，也不自动 Stop/Terminate Pod。
控制阈值保留归档/解包及失败记录空间；无法保证其他进程瞬间占满同一文件系统时的写入。

每份新 oracle 写入前检查“payload上限+控制阈值”；先写 `oracle.pt.tmp`、核对大小，再
发布 payload 和原子 identity。已有失败/半写入产物保留，不当作可复用 oracle。
编译临时文件和 CUDA driver cache 也指向本批 Volume 输出内。

满盘时保留最初退出原因，失败记录/计时记录写失败仅输出短提示；旧 state/report
checkpoint 保留。`state.json` 的旧 RUNNING 不能证明进程仍存活。查看进度只读小型
`progress.json` 或最多2000字节的阶段日志，不再整份 cat receipt/state。

### 归档及本地验证

pack 前按实际文件数和体积检查归档+解包空间，verify 前按归档成员检查展开空间与inodes。
隔离解包在归档所在文件系统创建临时目录，不依赖可能很小的容器 `/tmp`。
目录 symlink 因无法直接界定展开量而拒绝；native file symlink 仍按真实目标保存并计入容量。

本地回收验证需要同一镜像版本的 **evidence_io.py 和 storage.py** 两个相邻文件，或使用
本仓库 `.cache/adaln-indexed/` 中的同版本工具。checksum 与关键证据验证通过后才回收完整；
仅成功执行、pack 或 send 都不等于本地回收完成。
