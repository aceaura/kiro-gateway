# 任务：安装并启动 kiro-gateway 代理服务

你的任务是在这台机器上部署 kiro-gateway：把 Kiro（Amazon Q Developer）账号里的 Claude 等模型，代理成 OpenAI / Anthropic 兼容 API，最终在 `http://localhost:8000` 提供服务。严格按下面步骤执行，每一步成功后才能进入下一步；任何一步失败就停下来向我报告原因，不要自作主张绕过。

## 第 0 步：选择启动方式（必须遵守）

- 先检查 Docker 是否**真正可用**：运行 `docker info`（只看 `docker --version` 不够，必须确认 daemon 在运行）。
- **如果 Docker 可用 → 必须用 Docker 方式（第 4A 步）。**
- **如果没有 Docker → 检查 Python ≥ 3.10（`python3 --version`），用本地 Python 方式（第 4B 步）。**
- 如果两者都没有：不要擅自安装 Docker 或 Python 等系统级软件，停下来告诉我缺什么，等我指示。

## 第 1 步：获取代码

```bash
git clone https://github.com/aceaura/kiro-gateway.git
cd kiro-gateway
```

- 克隆位置：如果这台机器有 `~/Documents/Workspace/<host>/<org>/<repo>` 的镜像目录约定，就克隆到 `~/Documents/Workspace/github.com/aceaura/kiro-gateway`；否则克隆到当前工作目录。
- 如果目录已存在：不要重新克隆，进入目录执行 `git pull` 更新即可。

## 第 2 步：检查 Kiro 凭证

```bash
ls ~/.aws/sso/cache/kiro-auth-token.json
```

- 文件存在 → 继续。
- 文件不存在 → **停下来**，告诉我：需要先在 Kiro IDE 登录，或运行 `kiro-cli login`（免费 Builder ID 即可）生成凭证文件，等我确认后再继续。不要尝试自己伪造或抓取凭证。

## 第 3 步：配置 .env

```bash
cp .env.example .env
```

然后生成一个随机密钥并写入 `.env`（至少包含这两项）：

```bash
openssl rand -hex 16   # 用输出作为 PROXY_API_KEY
```

```env
KIRO_CREDS_FILE="~/.aws/sso/cache/kiro-auth-token.json"
PROXY_API_KEY="<刚生成的随机密钥>"
```

- 如果这台机器需要代理才能访问 AWS/Kiro 服务，**先问我**代理地址，再写入 `VPN_PROXY_URL=http://127.0.0.1:7890` 这类配置；不要自己猜。

## 第 4A 步：Docker 方式启动（Docker 可用时）

优先用本地构建（这是 fork 仓库，不要直接 pull 上游镜像）：

```bash
docker build -t kiro-gateway:local .

# 如果已有同名容器在跑，先 docker rm -f kiro-gateway
docker run -d --name kiro-gateway --restart unless-stopped \
  -p 8000:8000 \
  -v ~/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro \
  -e KIRO_CREDS_FILE=/home/kiro/.aws/sso/cache/kiro-auth-token.json \
  -e PROXY_API_KEY="<.env 里的同一个密钥>" \
  kiro-gateway:local
```

- 如果 8000 端口被占用，改用 9000 并在最后告诉我。
- （备选：`docker-compose up -d`，但必须先在 `docker-compose.yml` 里取消 macOS/Linux 凭证卷挂载那一行的注释。）

## 第 4B 步：本地 Python 方式启动（没有 Docker 时）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
nohup .venv/bin/python main.py > gateway.log 2>&1 &
```

- 用 venv 隔离，不要往系统 Python 里装依赖；venv 也建不了的话（比如 pip/venv 缺失），停下来报告。
- 如果 8000 端口被占用：`.venv/bin/python main.py --port 9000`，并记住实际端口。

## 第 5 步：验证（全部通过才算完成）

把 `<key>` 换成 PROXY_API_KEY：

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer <key>"
curl -s http://localhost:8000/v1/usage
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"ping"}],"stream":false}'
```

- `claude-sonnet-4-5` 不可用就换 `claude-haiku-4-5` 再试。
- 如果是 Docker 方式且验证失败，先看 `docker logs kiro-gateway` 定位原因。
- 报网络超时/403 → 大概率是连不上 AWS，回来问我代理配置，不要反复重试。

## 第 6 步：向我汇报

报告以下内容：

1. 用了哪种启动方式（Docker / 本地 Python）
2. 服务地址和端口
3. PROXY_API_KEY（我需要用它连客户端）
4. `/v1/models` 返回的可用模型列表
5. 验证请求的实际输出
6. 日常管理命令：看日志 / 重启 / 停止 分别怎么操作

## 约束

- 凭证文件、token、.env 的内容**不要**发送到任何外部服务或打印到公开日志里。
- 不要修改本机上与本任务无关的文件和配置。
- 已存在的服务/容器优先复用并只做健康检查，不要盲目重建。
