# Ubuntu / CUDA 迁移手册

## 迁移原则

服务器迁移时只传源码、经过确认的数据和结构化产物。不要迁移以下内容：

- Windows 的 `.venv/`、`venv/`
- Windows 本地 `qdrant_db/`，应在 Ubuntu 上重新建库
- `.cache/`，除非确认模型缓存完整且目录结构保持不变
- `project/.env` 和任何密钥文件
- 临时日志、爬虫状态和测试预测输出

推荐在服务器重新创建 Python 环境、重新下载模型，并从 `markdown_docs/` 与 `datasets/structured/syndrome_dictionary.jsonl` 重建索引。这能避免 Windows/Linux 路径、文件锁、二进制 wheel 和 Qdrant 本地存储兼容问题。

## Ubuntu 基础环境

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git locales
sudo locale-gen C.UTF-8 || true

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
```

检查 NVIDIA 驱动：

```bash
nvidia-smi
```

项目不要求单独安装完整 CUDA Toolkit 才能运行 PyTorch wheel，但 NVIDIA 驱动必须支持所选 wheel 携带的 CUDA runtime。

## 安装 CPU 环境

```bash
chmod +x setup_ubuntu.sh start.sh
./setup_ubuntu.sh
```

## 安装 CUDA 版 PyTorch

先在 PyTorch 官方安装页选择与服务器驱动匹配的 Linux CUDA wheel 地址，再显式传给安装脚本。不要把 CUDA wheel 地址永久写死在仓库里。

```bash
export INSTALL_CUDA_TORCH=1
export PYTORCH_INDEX_URL='从 PyTorch 官方页面取得的 CUDA wheel index URL'
export DOWNLOAD_RERANK_MODEL=1
./setup_ubuntu.sh
```

安装完成后必须通过：

```bash
.venv/bin/python scripts/check_runtime_environment.py \
  --require-cuda --require-models --check-qdrant --check-data-encoding
```

重点确认：

- `torch.cuda.is_available()` 为 true
- `cuda_build` 不是 null
- 显卡名称、数量和 compute capability 正常
- `compute_smoke` 为 `ok`；它会实际执行 CUDA 矩阵运算，能发现 wheel 不支持新显卡架构的问题
- `EMBEDDING_DEVICE`、`SYNDROME_RERANK_DEVICE` 没有指向不可用设备
- UTF-8 运行时和中文文件往返均为 OK

## 环境配置

```bash
cp project/.env.example project/.env
chmod 600 project/.env
```

服务器建议配置：

```dotenv
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

EMBEDDING_DEVICE=cuda
ENABLE_SYNDROME_RERANK=false
SYNDROME_RERANK_DEVICE=cuda

EMBEDDING_LOCAL_FILES_ONLY=true
SYNDROME_RERANK_LOCAL_FILES_ONLY=true
```

在独立排名盲测证明 rerank 有收益之前，保持 `ENABLE_SYNDROME_RERANK=false`。设备可提前设为 `cuda`，但关闭时不会加载 reranker。

API Key 只写入权限为 600 的 `project/.env` 或服务器 Secret Manager。不要写进 `.sh`、`.bat`、README、Dockerfile 或命令历史。

## Qdrant 模式

### 单进程演示

保持 `QDRANT_URL=`，使用 `QDRANT_DB_PATH` 指向 Ubuntu 本地绝对路径。嵌入式 Qdrant 适合单进程作品演示，不要让多个 Web 进程同时打开同一目录。

`QDRANT_DB_PATH` 必须位于服务器本地块存储，不能放在 NFS、SMB、网盘挂载或其他共享文件系统。Qdrant Local 使用 SQLite 持久化；共享文件系统可能出现 `database disk image is malformed` 或 `disk I/O error`。如果项目目录本身位于共享盘，应改用独立 Qdrant 服务，并把服务端 storage 放到本地磁盘。

### 独立 Qdrant 服务

```dotenv
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_PREFER_GRPC=false
```

独立服务适合多进程或容器部署。若服务不只监听回环地址，必须配置鉴权和防火墙。

## 重建索引

不要直接复制 Windows 的 `qdrant_db/`。在 Ubuntu 执行：

```bash
# 重建父子块主知识库
.venv/bin/python scripts/rebuild_qdrant_index.py --apply --wipe-storage

# 重建结构化方证库并写入 Qdrant
.venv/bin/python scripts/build_syndrome_dictionary.py --write-qdrant

# 验证结构化 JSONL、Qdrant 点数和冒烟查询
.venv/bin/python scripts/validate_syndrome_dictionary.py
```

如果使用远程 Qdrant，`--wipe-storage` 只负责本地目录备份；远程集合由重建逻辑删除并创建，执行前应确认 `QDRANT_URL` 指向正确实例。

## 启动与访问

```bash
./start.sh
```

`start.sh` 默认在服务器绑定 `0.0.0.0:7860`。公网环境应放在 Nginx/Caddy 反向代理之后，配置 HTTPS、访问控制和请求大小限制，不建议直接暴露 Gradio 端口。

## 乱码与换行

仓库已增加 `.editorconfig` 和 `.gitattributes`：

- Python、Shell、Markdown、JSON/JSONL 使用 UTF-8 + LF
- Windows `.bat` 保持 CRLF
- Python 文件读写继续显式指定 `encoding="utf-8"`

迁移后检查：

```bash
locale
.venv/bin/python -c "import sys,locale; print(sys.stdout.encoding); print(locale.getpreferredencoding(False)); print(sys.getfilesystemencoding())"
file -bi datasets/structured/syndrome_dictionary.jsonl
```

不要用 `iconv` 批量覆盖原数据。若个别外部 CSV 是 GBK/GB18030，应在导入脚本中检测编码，再统一输出 UTF-8。

## CUDA 验收门槛

GPU 迁移完成不等于 rerank 可以上线。至少需要：

1. CUDA 环境自检通过。
2. 全量单元测试通过。
3. 400 条似方证 hard-negative 零安全退化。
4. 独立排名盲测具备 `expected_entry_id` 或可接受方剂集合。
5. 对比 rerank 前后的 Recall@5、MRR@10、nDCG@10 和 P95 延迟。
6. 只有排序显著提升且安全门控不下降时，才设置 `ENABLE_SYNDROME_RERANK=true`。

当前 CPU A/B 已证明 rerank 链路可运行，但没有测得准确率增益，平均延迟明显增加。因此服务器迁移前置工作应先完成排名金标准，而不是提前开启 rerank。
