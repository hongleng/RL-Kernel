# 当前PR只提取审核数据

唯一当前提取入口：`review_bundle.py extract`；配套本地验证：同版本脚本的 `verify`。
旧 `evidence_io.py --pack` 是全目录备份，已从当前操作命令移除；新版本必须显式 `--full-backup`。
旧full-backup验证器不适用于 `.review.tar.gz`，不得把新审核包误报为完整批次备份。
原镜像中的旧文件不会随本地修改自动消失；当前Volume、候选、已有archive保持不变。

## 提取清单与结论

| 保留 | 用途 |
| --- | --- |
| 原candidate/环境/源hash/节点与性能锁、一份candidate源码archive | 候选、环境及复现来源 |
| 17suite的独立reports/JUnit/identity/命令/console与sanitizer原始log | 实际节点、PASS/skip与工具错误核查 |
| 成功session的report/raw JSONL、first-use/traces/memory、summary | 18case/216cell/2592正式样本及性能口径核查 |
| oracle identity JSON：seed/shape/dtype/route/参考源码与payload hash | 复现参考计算来源，不声称重验缓存张量 |
| native binary与明确引用的PTX/SASS/IR/cubin/编译metadata | 加载身份与实际静态指令审核；按hash去重传输 |
| 原失败JSON和外层log、存储恢复policy及实际控制器 | 解释误停/独立重跑与执行控制器差异 |

**不读取、不提取 oracle.pt**。它保存运行时复用的输入/上游梯度/CPU gold/fixed/FP64答案，
不是原始性能样本；旧全量验证器也只检查其hash，没有重算这些答案。
还不复制重复checkout、build中间目录、整个cache/dump、失败session的raw samples/cache。
只选成功session与指令audit明确需要的编译产物，不能概括成“所有cache都删掉”。

每个不同源文件仍校验原receipt/hash；同inode且指纹相同才复用读取。
对象在包内去重；原state保留作来源，但明确不验证已省略的全部receipt成员。
这份包支持原执行报告/样本/指令的审核，不支持恢复完整运行目录或离线重算所有oracle/GPU输出。
反事实/数学正确性仍需看源码、独立测试与原报告，不能用缓存hash代替。

## 固定版本下载命令（当前已完成批次）

在能够访问保留Volume的终端运行；Pod仍Stop时不执行，不因此重跑验收。
当前工具source commit：`56b0163db8d4473273aa22109f1a3dcdfcdf481f`，SHA256：`3bd85836f2bc9c201362f52afcec1225f2d6e2e07205959a03029e74a28eeb8e`。
仅下载一个Python文件，无curl或第三方依赖；校验失败不安装、不执行。

```bash
python - <<'PYTHON'
import hashlib
from pathlib import Path
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/hongleng/RL-Kernel/56b0163db8d4473273aa22109f1a3dcdfcdf481f/docker/adaln-indexed/review_bundle.py"
expected = "3bd85836f2bc9c201362f52afcec1225f2d6e2e07205959a03029e74a28eeb8e"
with urlopen(url, timeout=30) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("下载SHA256校验失败，停止")
target = Path("/workspace/review-bundle.py")
temporary = target.with_suffix(".py.tmp")
temporary.write_bytes(data)
temporary.replace(target)
print("审核数据提取工具下载及校验成功")
PYTHON
```

下载成功后执行当前文档的 `extract`，不要执行旧 `--pack` 或重新运行benchmark。
包内candidate源码是原执行版本，里面的历史运行命令不作为当前操作说明；当前操作以此处extract/verify为准。

## 在可访问保留Volume的终端执行

准备期间保持Pod停止；该脚本是纯CPU文件处理，不调用CUDA、Torch、编译或GPU验收。
停止的Pod没有终端时不要以为命令能在停止状态执行；先准备好工具，再决定如何访问该Volume。
不要为取数据重跑17suite。若RunPod恢复时提供0 GPU数据访问选项，可用于取回此卷；它不是随时保证可选，不能另建CPU Pod就假定能挂上原Volume Disk。
依据：https://docs.runpod.io/pods/troubleshooting/zero-gpus 。若启动的是GPU Pod，即使此脚本只用CPU也仍有驻留计费。
只需下载一个固定版本、SHA256校验后的Python脚本，无curl依赖。
当次发布记录提供实际固定commit下载命令与SHA256；不使用未经校验的分支最新版。

```bash
python /workspace/review-bundle.py extract \
  --output /workspace/evidence/adaln-indexed-storage-v1
```

不依赖终端继承环境变量；直接使用完整输出路径。自动先显示实际选中上限，超过1024MiB就停，
列出大文件供检查；不靠默默删证据获得成功。可加 `--plan` 只看选择和大小。
只读选中文件，gzip等级1；不修改原输出、不远端解压、不覆盖任何旧包/临时包，不自动传输。
进度显示阶段与字节数。目标：

```text
/workspace/evidence/adaln-indexed-storage-v1.review.tar.gz
/workspace/evidence/adaln-indexed-storage-v1.review.tar.gz.sha256
```

只有命令退出码0且打印 `ARCHIVE ... CANDIDATE ...` 后才传输。留存打印的真实CANDIDATE hash。
若旧打包仍运行，先确认其进度/退出，不并行重复I/O；原数据与旧tmp都不需要删除。
源锁可阻止持锁的旧runner/pack，但不能由锁空闲断言旧入口的后续verify已经结束。
当前审核包不支持将未完成批次用 `--partial` 标为完整PR数据。

## 本地验证与审核

在本地把同版本脚本、审核包、checksum放在同一目录，执行：

```bash
read -r -p '粘贴提取输出的 CANDIDATE hash: ' CANDIDATE_SHA256
python3 review-bundle.py verify \
  --archive adaln-indexed-storage-v1.review.tar.gz \
  --candidate-sha256 "$CANDIDATE_SHA256" \
  --dest adaln-indexed-pr-review
```

验证以流式checksum读取archive、检查对象hash/大小、重建所需视图、重新核对选择范围、
原receipt、原始报告/2592样本覆盖、oracle身份和源码archive全部hash。
不依赖原Pod绝对路径、不执行包内Python、不加载.so或CUDA。
`--dest`保存可直接阅读的原相对路径目录、展开源码及REVIEW_BUNDLE_MANIFEST.json，拒绝覆盖已有目录。
只有 `REVIEW_BUNDLE_VERIFIED` 才能宣布这份PR审核数据取回成功；自动检查不替代
独立GPU/精度/指令语义及性能结论审核。未验证前保留Volume，不自动Terminate。

## 版本与实现边界

新工具放在infra，未来context复制到image-acceptance，source-profile记录hash；
当前旧镜像用外置脚本即可，不修改其candidate/helper字节，也不要求重建镜像后才取数据。
实际包大小由 `--plan` 决定；CPU合成样例不能当作真实Pod大小或速度承诺。
未实施或测量可信Pod quota遥测；文件系统pool free不等于200GB配置的剩余配额。
笔记和skill如何修改按用户后续讨论处理，本轮只修工具与有效操作说明。
