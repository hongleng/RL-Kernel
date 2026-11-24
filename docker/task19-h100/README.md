# Task19 / PR #707：预编译 H100 验证镜像

当前唯一运行说明，2026-11-20（Asia/Shanghai）。旧 `task19-h100-tools.tar.gz` 和官方镜像现场编译命令已停用。

源码固定 `2f685b8eb79722eae3a135341e881207cccca4cd`，baseline 固定 `49941801f942428fd52ac5b66a1b0c070418724e`。
基础镜像为官方 PyTorch 2.4.1 / CUDA 12.4 devel 的固定 digest，见 Dockerfile。
**当前状态：构建 [#44916076977](https://github.com/hongleng/RL-Kernel/actions/runs/44916076977) 已成功发布 GHCR；2026-11-21 已核对发布 artifact，并确认匿名 manifest 返回 HTTP 200。H100 GPU 验证尚未执行。**

```text
ghcr.io/hongleng/rl-kernel/task19-h100@sha256:4af7e74a5e6f71002806de8dc2c5fbd35926c75896c3488a72a1bba750ae28b3
```

当前构建：https://github.com/hongleng/RL-Kernel/actions/runs/44916076977

构建配置提交：`768aebb68f858dbbec01040b29b33110381e9c1e`；fork 默认分支 workflow 登记提交：`21d376d52d955633278f03d22ba35b3359dadcc0`。

## 与历史方案的关系

沿用 Actions → GHCR → RunPod、固定 source/infra/image 身份、CPU 自检、手动启动 GPU 和小证据包。
历史 shared 镜像只预装环境，native build 留到 H100；indexed runner 后续还曾按 suite 重建 9 次。
本方案在镜像构建时编译一次普通 `sm_90` 扩展；最终镜像包含源码、独立 Git HEAD/baseline、binary 和构建记录。
Pod 入口只校验和加载，任何身份不符均停止，没有重编译或安装依赖的恢复路径。
before/after 原生源码一致，所以两个 Python wrapper 和所有 GPU 测试共用同一 binary。

历史依据：

- [shared 镜像与运行指南](/home/dnd/personal-project/runpod-knowledge/sources/shared/adaln-h100-image/notes.md)。
- [9 次构建与复用分析](/home/dnd/personal-project/runpod-knowledge/sources/indexed/reuse-and-utilization-20261027.md)。
- [editable 安装污染事故](/home/dnd/personal-project/runpod-knowledge/sources/indexed/actions-fix-20261026.md)。
- `knowledge-RL-Kernel/README.md` 说明镜像/GPU 原始资料已归入 `runpod-knowledge`，不是第二套镜像入口。

`setup.py` 当前直接询问 GPU 可用性和 capability，忽略 GPU-less CUDA 构建的 FORCE_CUDA 意图。
`build_image.py` 仅在执行 setup.py 的进程中将这两个查询固定为 True / (9,0)，使用未修改的生产构建配置与原生源码。
`TORCH_CUDA_ARCH_LIST=9.0`、fast-math 关闭、实验 SM90 关闭、4 个 worker。
链接使用 toolkit 自带的 libcuda stub；stub 不加入 LD_LIBRARY_PATH，不随产物作为驱动库发布。
真实 GPU 测试使用真实设备检测，没有 mock。CPU 构建只检查 ELF 含 sm_90，不能声称 GPU PASS。

## 1. 准备与构建（有 Docker 的本地或 CI 主机，不在租用的 H100 上）

当前 infra worktree：`/home/dnd/personal-project/rl-kernel-task19-h100-infra`，分支 `infra/task19-h100-image`。
源码 worktree：`/home/dnd/personal-project/RL-Kernel-task19`，保持 PR 分支不变。

轻量 context 准备不下载依赖，也不编译：

```bash
cd /home/dnd/personal-project/rl-kernel-task19-h100-infra
python3 docker/task19-h100/prepare_context.py \
  --repo /home/dnd/personal-project/RL-Kernel-task19 \
  --out /tmp/task19-h100-context
```

`--out` 必须不存在。导出独立、无 remote/凭据的浅 Git 仓库，只取 HEAD 和 baseline，包含比较 runner 所需 Git 对象。
不复制原 worktree `.git` 指针、旧 `.so`、虚拟环境或本地缓存。Docker 专用 ignore 显式保留这个安全的 `.git`。

以下为用户执行的长构建命令（需要下载约 6.76 GiB 压缩基础镜像层，构建耗时未实测）：

```bash
docker buildx build --platform linux/amd64 --target runtime \
  -f /tmp/task19-h100-context/Dockerfile \
  -t ghcr.io/hongleng/rl-kernel/task19-h100:source-2f685b8-local \
  --push /tmp/task19-h100-context
```

需事先有 GHCR 发布权限并登录。记录 buildx 返回的 digest；部署使用完整 `ghcr.io/.../task19-h100@sha256:...`。
包应允许 RunPod 匿名拉取，或在平台配置只读拉取凭据，不向镜像写入凭据。
首次 build 一次，BuildKit 可命中缓存；不要额外执行 build_ext、pip install -e . 或第二轮 Docker evidence-target 构建。
最终镜像复用基础环境层，仅复制 `.so` 与必要构建记录，不携带 build objects/Ninja cache。

也已准备 `.github/workflows/task19-h100-image.yml`：单次 build-push，GHA cache，发布 GHCR，artifact 导出 digest 和基础设施提交。
要在 GitHub 使用，需先提交并推送本 infra 分支，并将该 workflow 登记到 fork 默认分支。本次已提交、推送并登记 workflow。首轮失败及修复重试见 publication.json；不要额外触发同一版本构建。
登记完成后：

```bash
gh workflow run task19-h100-image.yml --repo hongleng/RL-Kernel \
  --ref infra/task19-h100-image
```

不反复轮询；从完成后的 Actions artifact `task19-h100-image-identity/image-reference.txt` 取 digest。
镜像构建成功只表示 CPU 检查和编译成功；源码/工具链/ABI/flags/binary hash 的记录在 `/opt/task19-image-build/build.json`。

## 2. H100 运行（Pod 网页终端）

单张完整 H100 80GB、非 MIG、只暴露一张 GPU。建议 8 vCPU / 32GB RAM、80GB 容器磁盘；不是实测峰值门槛。
RunPod 镜像填新发布的 digest；启动命令 `sleep infinity`，不自动运行 GPU 任务。
不再上传工具包、git fetch、安装 Python 依赖或编译。所有入口和源码都在镜像 `/opt` 下，证据写到 `/workspace`。
需要跨 Stop 保留证据时，核实 `/workspace` 实际持久挂载；路径名称本身不是持久性证明。

把下面 IMAGE_REF 换成发布产物中的完整 digest（该值是外部部署身份记录，容器内无法自行验证平台实际使用的 digest）：

```bash
IMAGE_REF='ghcr.io/hongleng/rl-kernel/task19-h100@sha256:4af7e74a5e6f71002806de8dc2c5fbd35926c75896c3488a72a1bba750ae28b3'
python -I /opt/task19-h100/build_image.py --check
timeout --signal=INT --kill-after=30s 30m \
  python -I /opt/task19-h100/run.py run \
  --repo /opt/task19-source \
  --out /workspace/task19-h100-evidence \
  --image "$IMAGE_REF"
```

`--check` 只做 hash/源码/ABI 检查，不编译。输出目录必须不存在，失败/超时保留原 attempt，不自动重试。
30 分钟是上限，没有目标机实测耗时；镜像拉取、准备和回收另计。
计费按所选 Pod 实际费率及驻留时间；不为了低 GPU 利用率增加并发任务。

| 执行组 | 验收条件 |
| --- | --- |
| 预编译身份检查 | 源码、binary、ABI、Torch/CUDA 与 image build manifest 相符，实际 `_C` 来自固定源码目录 |
| 现有 GPU tests | 82 项全部执行，零 skip/failure/error，generic/deterministic/RL loss step |
| before/after 比较 | 3 dtype × 3 shape × 2 layout = 18 组，forward/梯度/8 variants 字节一致，另对独立 reference 检查精度 |
| 默认 SM90 wrapper fallback | BF16，generic native forward + PyTorch VJP 与 oracle 一致 |

82 项来自原固定源码的 collection：109 中选择 82、排除 27 个 CPU/reference/静态源码及实验 SM90 符号相关项。
fallback 检查绕过未使用的 TMA 构造符号检查，不证明实验 TMA kernel 或完整 SM90 构造路径。
MUSA 硬件、实验 SM90 TMA、性能、sanitizer 不在本次范围。已有 CPU ownership 证据不当作 GPU PASS。

## 3. 回收与本地验证

证据包含 build.log/build.json、native.so、runner、源码清单、环境、gpu.log/gpu.xml、comparison.log/comparison.json、fallback 和 status。
不回收对象文件、镜像层、虚拟环境、oracle 缓存或完整 workspace。SHA256SUMS 只绑定这份审核范围。
本地校验前保留 Pod 原证据目录；结束资源由用户操作。

Pod 端先检查 `command -v runpodctl`；本镜像没有额外预装它。缺少时按 [官方安装说明](https://github.com/runpod/runpodctl#install)安装。
沿用原先的传输方式，不为传文件再重建镜像：

```bash
cd /workspace
tar -cf task19-h100-evidence.tar task19-h100-evidence
sha256sum task19-h100-evidence.tar > task19-h100-evidence.tar.sha256
runpodctl send task19-h100-evidence.tar
# 第一份发送完成后发送 checksum，使用各自新码。
runpodctl send task19-h100-evidence.tar.sha256
```

本地进入新接收目录：

```bash
runpodctl receive <证据包传输码>
runpodctl receive <checksum传输码>
sha256sum -c task19-h100-evidence.tar.sha256
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import tarfile
out = Path('received')
out.mkdir(exist_ok=False)
with tarfile.open('task19-h100-evidence.tar') as archive:
    members = archive.getmembers()
    assert members, 'empty evidence archive'
    for member in members:
        p = PurePosixPath(member.name)
        assert p.parts[0] == 'task19-h100-evidence' and '..' not in p.parts
        assert not p.is_absolute() and (member.isfile() or member.isdir())
    archive.extractall(out, members=members, filter='data')
PY
python3 /home/dnd/personal-project/rl-kernel-task19-h100-infra/docker/task19-h100/run.py verify \
  --out received/task19-h100-evidence
```

本地解包使用支持 `tarfile` data filter 的 Python。verify 不需要 Torch/GPU。
仅 checksum OK 不代表 GPU PASS；verifier 通过才接受上述验证范围，失败包仍保留供诊断。
