# AdaLN H100：Actions 镜像与手动验收

更新：2026-10-22。此目录是当前执行入口；旧 2026-09-25 transfer bundle 和
179d0a8 镜像准备包均不再使用。Actions 只构建并发布镜像，不创建 RunPod。

## 固定版本

- 验收源码：`69bd47f72a24d1177580d9b11b6efdfeb6c6bab5`。
- 算子比较基线：`16b81175874ecf11d5ff05c00de3b69e9457591b`。
- infra 分支：`infra/adaln-h100-image`；算子分支不包含镜像配置。
- CUDA 12.6.3 devel、Python 3.11.17、PyTorch 2.12.1+cu126、Triton 3.7.1。
- 完整依赖版本固定于 `requirements.txt`（来自首次成功构建39565761133），并执行 pip check；实际版本存于 `/opt/image-packages.txt`。
- 镜像构建完成后，以 `image-reference.txt` 的完整 `image@sha256:...` 固定整个环境。

用户已选择 CUDA12.6 + Python3.11。CUDA12.4 常规配套的 Triton3.2 不识别源码中的
`enable_reflect_ftz=False`，不能仅改基础镜像标签后沿用当前源码。没有改动生产算术。

## 发布与构建

workflow：`.github/workflows/adaln-h100-image.yml`，只有 `workflow_dispatch`。
GitHub 要求该 workflow 先在 fork 默认分支登记；准备脚本和 Dockerfile 位于 infra 分支。
默认分支只登记 workflow，不合入算子实现。

```bash
gh workflow run adaln-h100-image.yml --repo hongleng/RL-Kernel \
  --ref infra/adaln-h100-image -f infrastructure_ref=infra/adaln-h100-image
```

Actions 导出固定源码、加入此目录的验收工具并生成逐文件 hash；镜像中不带本地 .venv、
旧 .so 或密钥。构建执行 Python/torch/Triton/toolkit 检查、脚本自检及 CPU gold 回归测试。
Native 构建、真实 GPU 正确性和 sanitizer 在 H100 执行；Actions 成功不代表这些已通过。

下载 `adaln-h100-image-evidence` artifact：包括完整 digest、构建 metadata、源码清单、
环境及测试清单和源码/infra/workflow 三个提交身份。使用 digest，不依赖可移动的 tag。
GHCR package 应设为 public，或在 RunPod 配置只读拉取凭据；不把凭据写进镜像或证据。

## 当前镜像与拉取层优化（2026-10-22）

[构建39741435857](https://github.com/hongleng/RL-Kernel/actions/runs/39741435857)已成功，构建代码为
`0471725d928babe6e4c8fae329f5e2efdfa55c6b`。RunPod使用以下完整digest：

```text
ghcr.io/hongleng/rl-kernel/adaln-h100@sha256:682fae9afbb852c75c7b519e9b80645e6528c0d8d034d29ed4bd35f0a5073ce4
```

对应tag为`source-69bd47f-0471725d928babe6e4c8fae329f5e2efdfa55c6b`；
已核对tag/digest一致及GHCR匿名manifest访问，无需拉取凭据。
环境检查、38个固定依赖的pip check、脚本自检及1项CPU gold回归通过；H100验收尚未验证。

旧310f692镜像的Torch/CUDA Python依赖集中在单个3.35GiB层，RunPod日志显示下载约7分钟后重试。
新Dockerfile将依赖拆成7个安装层，Torch和cuDNN各自独立；BuildKit cache mount不随镜像发布。
安装阶段临时使用no-deps，最后按完整固定清单安装并执行pip check，38个依赖版本与原镜像一致。

| 新依赖层 | 压缩大小（MiB） |
| --- | ---: |
| cuBLAS | 377 |
| cuDNN | 679 |
| cuFFT / cuRAND / cuSOLVER | 398 |
| cuSPARSE / cuSPARSELt | 483 |
| NCCL / NVSHMEM | 411 |
| Torch | 805 |
| Triton及其余依赖 | 279 |

以上大小来自发布后的manifest，完整镜像压缩体积约7.285GiB，与旧版7.288GiB基本相同。
拆层减少单层失败后的重传量，没有显著减少总下载量；实际RunPod速度待验证，不保证低于10分钟。
CUDA基础层保持原样，节点已有这些层时可继续复用；旧digest不包含这次优化。

## 手动 RunPod

选择单张完整 H100、非 MIG，镜像填写上述 digest，启动命令保留 `sleep infinity`。
建议容器磁盘预留60GiB、CPU RAM至少32GiB；使用网页终端，SSH公钥由用户配置。
镜像已准备依赖，不在租GPU后下载模型或重装PyTorch。

```bash
cd /opt/RL-Kernel
export CUDA_VISIBLE_DEVICES=0
export OUT=/workspace/adaln-h100-evidence-20261021-01
mkdir "$OUT"
printf '%s\n' 'ghcr.io/hongleng/rl-kernel/adaln-h100@sha256:682fae9afbb852c75c7b519e9b80645e6528c0d8d034d29ed4bd35f0a5073ce4' > "$OUT/image-reference.txt"
printf '%s\n' 'https://github.com/hongleng/RL-Kernel/actions/runs/39741435857' > "$OUT/image-build-url.txt"
cp /opt/image-packages.txt "$OUT/"
python acceptance/selfcheck.py
tmux new -s adaln
```

在 tmux 中运行一键入口，`--out` 指定的子目录必须尚不存在：

```bash
python acceptance/run.py --out "$OUT/run" --cuda-home /usr/local/cuda-12.6 --jobs 4
```

RAM不足可用 `--jobs 2`。它重新构建整个 `_C`，固定通用 sm_90，关闭 fast-math 和额外
90a kernels。编译及全部验收耗时尚无此次环境实测，可能超过10分钟。
失败会停止后续阶段并归档已有证据；不要覆盖原目录或放宽数值门槛。

## 阶段与门禁

| 阶段 | 通过条件 |
| --- | --- |
| preflight / build | H100、CC9.0、单卡；环境匹配；新 .so 含 sm_90 及 AdaLN 符号；源码和二进制 hash 绑定 |
| correctness | 固定清单413项；含 indexed、runtime fallback、三个真实 S 和全部 launch 配置 |
| racecheck | 固定清单211项；所有 sanitizer summary 的 hazards/errors/warnings 均为0 |
| memcheck | 固定清单225项；所有 ERROR SUMMARY 均为0 |
| benchmark | 54项：2 dtype x 3 S x 3 backend x 3 mode；无指定 speedup 门槛 |

测试必须与 `acceptance/test-manifest.json` 的逐例清单一致，零 skip/xfail/xpass。
计数来自固定源码的 collect-only，不是旧笔记估计。sanitizer 使用限定子集，完整正确性先执行。
显式负例允许已声明的 H4097/H5000 或不可用后端回退，不允许在支持域静默替换 CUDA。

benchmark 使用相同输入及预分配随机 dy/dgate，平衡 backend 顺序，记录原始采样、CV、
前后半中位数、GPU遥测和峰值内存；比较 forward、backward-only、forward+backward。
CUDA/Triton 验证原始字节一致；三种 backend 都检查独立CPU FP32精度。
GPU PyTorch fixed reference 仅作为精度/性能基线，不宣称已通过跨后端字节资格。

结果包含 `status.json`、环境/源码清单、build参数及日志、.so及hash、逐例报告/JUnit XML、
sanitizer原文和benchmark JSON/CSV。`run.tar.gz` 排除中间 objects/lib/JIT cache，并附校验文件。

## 下载与结束

以下命令在Pod中执行，只会生成远端文件，不会自动下载到本地。示例对应前面的OUT；重复运行使用新目录名。

```bash
cd /workspace
tar --exclude='adaln-h100-evidence-20261021-01/run' \
  -czf adaln-h100-evidence-20261021-01.tar.gz adaln-h100-evidence-20261021-01
sha256sum adaln-h100-evidence-20261021-01.tar.gz > adaln-h100-evidence-20261021-01.tar.gz.sha256
```

优先使用用户确认过的`runpodctl send/receive`，不需要SSH IP、端口、私钥或RunPod API key。
两端先执行`command -v runpodctl`确认可用；自定义镜像没有预装此工具，缺少时按
[官方安装说明](https://github.com/runpod/runpodctl#installation)安装。不要仅为传文件重建验收镜像。

在 **Pod终端**发送压缩包，保持发送进程运行：

```bash
cd /workspace
runpodctl send adaln-h100-evidence-20261021-01.tar.gz
```

在 **本地电脑终端**进入接收目录，使用发送端当次显示的真实传输码：

```bash
runpodctl receive <传输码>
```

首个文件传输完成后，在Pod发送校验文件：

```bash
runpodctl send adaln-h100-evidence-20261021-01.tar.gz.sha256
```

本地再次执行`runpodctl receive <这次的新传输码>`。每次send会生成新码，两个文件分别接收；
发送和接收都须成功退出。用法见[官方文件传输说明](https://github.com/runpod/runpodctl#file-transfer)。

在 **本地电脑终端**校验：

```bash
sha256sum -c adaln-h100-evidence-20261021-01.tar.gz.sha256
```

显示OK后，检查归档内的运行状态和日志，确认失败证据也已保存，再手动终止Pod。
SCP可作为已配置完整SSH时的备选；脚本不自动创建、停止或删除RunPod。

## 更新源码后的维护

修改 `prepare_context.py` 的 SOURCE_SHA、workflow 的源码身份、此文档；使用对应源码重新
运行 `acceptance/collect_tests.py` 的三个 stage，审核完整节点清单再更新 test-manifest.json。
不要只改数量来通过门禁。环境变更同步 environment.json、Dockerfile 和 runtime preflight。

本地快速验证：prepare_context、自检、Ruff、YAML/脚本语法、CPU gold 回归；不下载依赖或
编译 GPU 扩展。本机Python3.12检查不冒充镜像Python3.11实测，后者以Actions结果为准。
