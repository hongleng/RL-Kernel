# Task19 main integration：预编译 H100 验证镜像

当前运行说明，2026-11-23。状态：已获准推送 PR 集成分支和 infra 分支并触发 GHCR 构建；尚无新 digest。
这是新批次；旧 H100 包和旧镜像 digest 只证明旧源码，不作本批 PASS。

## 固定身份与资源

- 候选：`4c9ee6dcce483cf99a8385ef0a4da63fd3ed6b11`，DCO merge commit。
- baseline：`e9d52e95226b62d826f0a224bafad2cab3ed7538`，权威 upstream/main，含 #495 / #547。
- 基础镜像保持官方 PyTorch 2.4.1 / CUDA 12.4 devel 固定 digest，见 Dockerfile。
- 单张完整 H100（80GB / NVL 均可）、非 MIG；建议 8 vCPU / 32GB RAM、80GB 容器磁盘。
- 输出到实际挂载的 `/workspace`。建议 20GB Volume Disk；容量不是实测峰值。
- Container start command：`sleep infinity`。网页终端操作；无需 HTTP/TCP 暴露端口。

## 本批范围与复用

镜像构建执行 CPU collection、10 项 runner 自检及 ownership CPU 回归，然后仅编译一次普通 sm_90 扩展。
GPU-less 构建仅在 setup.py 进程中适配 availability/capability 查询；生产源码和 GPU 测试不做 mock。
保持 fast-math / 实验 SM90 关闭、4 个编译 worker。所有 suites 和 baseline/candidate 共用一份 binary。
Pod 上只检查 hash/ABI 并加载，binary 缺失或身份不符直接失败，不自动安装或重编译。

| 组 | 范围 |
| --- | --- |
| 原有 H100 回归 | 8 generic + 73 deterministic + 1 RL loss step |
| #495 原始 backward 测试 | 67 项，含 CPU chunked/reference、真实 CUDA fused/fallback 和三种 dtype；排除 6 个长序列 workspace case 和 2 个双卡 device/stream case |
| #547 一阶 autograd | 65 项；全局 not native 另排除一个缺失 native symbol 的负例 |
| before/after | 18 dtype/shape/layout 组，forward、native fused backward、8 variants 字节一致及独立 reference 精度 |
| SM90 wrapper 默认 fallback | 真实 generic native forward + native fused backward，计数验证实际调用；不证明 TMA 或构造器路径 |

共 **214 个锁定节点**，250 中排除 36。gpu-plan.json 保存实际 CPU collection 的完整 nodeids。
实际运行和本地 verifier 核对节点身份、数量和零 skip/failure/error；仅数量相同不足以通过。
不运行性能、sanitizer、TMA、多 GPU、长序列 workspace 或真实 MUSA，也不重新验证 #547 全部高阶接口。
未选择的 native 实现相对 baseline 不变；本批证明声明的 wrapper/backward 集成范围。

comparison.json 必须声明 #495 native backward；缺少新符号或导入其他 checkout 的扩展均拒绝。
保留 source/build/environment/binary/runner/plan hash。旧 82 项或 PyTorch-only 报告不能通过本批 verifier。

## 本地准备（不编译）

infra worktree：`/home/dnd/personal-project/rl-kernel-task19-main-h100-infra`；分支 `infra/task19-main-h100-image`。
原 infra worktree 的未提交发布记录保留原处，未覆盖。

```bash
cd /home/dnd/personal-project/rl-kernel-task19-main-h100-infra
python3 docker/task19-h100/prepare_context.py \
  --repo /home/dnd/personal-project/RL-Kernel-task19 \
  --out /tmp/task19-main-h100-context
python3 -I docker/task19-h100/test_run.py
```

输出目录必须不存在。context 包含无 remote/凭据的独立 Git HEAD/baseline，不拷贝 editable 安装或旧 binary。
Docker ignore 只排除 source/build 等缓存，保留 tracked tests/build。
CPU checks 不建立 H100 PASS；实际基础镜像内的 Torch/CUDA 兼容性须由 Docker 构建与 GPU 运行确认。

## 构建发布

用户已授权推送 `feat/task19-portable-logp` 与 `infra/task19-main-h100-image` 并触发 GHCR。
直接使用 PR 分支上的固定 SHA，不创建额外镜像源码分支；PR 正文和评论保持现状。
以下为发布入口，已触发的同版本批次不要重复提交，实际 run/digest 见 publication.json：

```bash
git -C /home/dnd/personal-project/RL-Kernel-task19 push origin feat/task19-portable-logp
git push --set-upstream origin infra/task19-main-h100-image
gh workflow run task19-h100-image.yml --repo hongleng/RL-Kernel \
  --ref infra/task19-main-h100-image
```

checkout、upload-artifact、build-push、login、setup-buildx 所用 release 的 action.yml 均已核对为 Node24。
单次 build-push，没有第二次 evidence-target build；读取旧/新 GHA cache，写入新 scope。
发布 tags 为 source-4c9ee6d-<infra-sha>；从成功 artifact 的 image-reference.txt 取新 digest，不复用旧 digest。

有 Docker 的主机也可使用同一 context（长构建/约 6.76GiB 基础镜像层下载，尚未执行）：

```bash
docker buildx build --platform linux/amd64 --target runtime \
  -f /tmp/task19-main-h100-context/Dockerfile \
  -t ghcr.io/hongleng/rl-kernel/task19-h100:source-4c9ee6d-local \
  --push /tmp/task19-main-h100-context
```

## Pod 运行（新 digest 发布后）

Template 镜像填新发布的完整 digest；启动命令 sleep infinity，连接这个新 Pod 自己的网页终端。
/opt/task19-source 和 /opt/task19-h100 必须存在。无需上传源码、安装依赖或运行 build_ext。

```bash
IMAGE_REF='ghcr.io/hongleng/rl-kernel/task19-h100@sha256:<新镜像的64位digest>'
git -C /opt/task19-source rev-parse HEAD
ls /opt/task19-h100/gpu-plan.json
python -I /opt/task19-h100/build_image.py --check
timeout --signal=INT --kill-after=30s 30m \
  python -I /opt/task19-h100/run.py run \
  --repo /opt/task19-source \
  --out /workspace/task19-main-h100-evidence \
  --image "$IMAGE_REF"
python -I /opt/task19-h100/run.py verify --out /workspace/task19-main-h100-evidence
```

先确认 HEAD 为上面的 4c9ee6dc 完整 SHA、gpu-plan.json 存在且 --check PASS，再执行 GPU 命令。输出目录必须不存在，失败保留原 attempt，不自动重跑。
30 分钟是上限，尚无新矩阵实测耗时；镜像拉取和回收另计。

## 回收与本地接受

审核范围为日志、JSON/XML、runner/builder/Dockerfile、gpu-plan 和实际 binary，不是完整 build cache。
不带 objects、虚拟环境、镜像层或 oracle 大缓存。本地验收前保留 Pod 原数据；结束资源由用户操作。
先确认两端有 runpodctl；缺少时按 https://github.com/runpod/runpodctl#install 安装。

Pod 中发送两个文件（上一份完成后再发下一份，每次使用各自新码）：

```bash
cd /workspace
tar -cf task19-main-h100-evidence.tar task19-main-h100-evidence
sha256sum task19-main-h100-evidence.tar > task19-main-h100-evidence.tar.sha256
runpodctl send task19-main-h100-evidence.tar
runpodctl send task19-main-h100-evidence.tar.sha256
```

本地进入新接收目录：

```bash
runpodctl receive <证据包传输码>
runpodctl receive <checksum传输码>
sha256sum -c task19-main-h100-evidence.tar.sha256
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import tarfile
out = Path('received')
out.mkdir(exist_ok=False)
with tarfile.open('task19-main-h100-evidence.tar') as archive:
    members = archive.getmembers()
    assert members, 'empty evidence archive'
    for member in members:
        p = PurePosixPath(member.name)
        assert p.parts[0] == 'task19-main-h100-evidence' and '..' not in p.parts
        assert not p.is_absolute() and (member.isfile() or member.isdir())
    archive.extractall(out, members=members, filter='data')
PY
python3 /home/dnd/personal-project/rl-kernel-task19-main-h100-infra/docker/task19-h100/run.py verify \
  --out received/task19-main-h100-evidence
```

解包使用支持 tarfile data filter 的 Python。checksum 只证明传输字节，本地 verifier PASS 才接受上述范围。
