# SSH 与内网穿透

> Secure Shell、远程管理、文件传输、端口转发与内网访问学习笔记

SSH（Secure Shell）是一组用于在不可信网络上建立安全通信的协议和工具。最常见的 OpenSSH 实现不仅提供远程登录，还提供：

- 远程命令执行。
- 公钥认证和身份管理。
- 加密文件传输。
- 跳板机和多级代理。
- 本地、远程和动态端口转发。
- X11、Unix Socket 等通道转发。
- 连接复用、保活和自动化运维。

## 内容概览

- SSH 的协议层和客户端/服务端模型
- `ssh`、`sshd`、`ssh-keygen`、`ssh-agent`
- 密钥认证、`known_hosts` 和主机身份校验
- `~/.ssh/config`、`/etc/ssh/sshd_config`
- 远程登录、命令执行、环境和 TTY
- `scp`、`sftp`、`rsync`
- 跳板机、`ProxyJump`、`ProxyCommand` 和连接复用
- `-L` 本地转发、`-R` 远程转发、`-D` SOCKS 转发
- 基于公网服务器的反向隧道和内网穿透
- 安全边界、权限、审计、保活和故障排查

## 1. SSH 的工作模型

### 1.1 客户端与服务端

```text
SSH Client                              SSH Server
┌──────────────────┐                    ┌──────────────────┐
│ ssh / scp / sftp │                    │ sshd             │
│ 私钥、配置       │ ── 加密连接 ─────> │ 公钥、配置、Shell │
└──────────────────┘                    └──────────────────┘
```

客户端主动连接服务端。服务端通常监听 TCP 22 端口，但也可以配置为其他端口：

```bash
ssh user@example.com
ssh -p 2222 user@example.com
```

### 1.2 SSH 的三个协议层

SSH-2 可以从三个功能层理解：

```text
Transport Layer Protocol
  ├── 协商算法
  ├── 密钥交换
  ├── 服务端身份确认
  └── 建立加密和完整性保护

User Authentication Protocol
  └── 证明客户端有权使用某个远程账号

Connection Protocol
  ├── 交互式 Shell
  ├── 远程命令
  ├── SFTP 子系统
  └── TCP / Unix Socket 转发通道
```

### 1.3 一次连接的基本过程

```text
1. TCP 连接到服务端
2. 双方交换 SSH 协议版本
3. 协商 KEX、Host Key、Cipher、MAC 等算法
4. 通过密钥交换建立会话密钥
5. 客户端验证服务端 Host Key
6. 客户端完成用户认证
7. 打开 Shell、命令、SFTP 或转发 Channel
```

SSH 使用非对称密码学完成身份和密钥交换，再使用对称加密保护大量会话数据。私钥通常不会直接发送到服务端。

### 1.4 SSH 的价值

SSH 的重要性来自“统一的安全传输通道”：

| 能力 | SSH 提供的基础 |
|------|----------------|
| 远程登录 | 加密 Shell 通道和用户认证 |
| 远程执行 | 在远端启动命令并传输标准输入输出 |
| 文件传输 | SFTP 子系统和安全复制工具 |
| 内网访问 | TCP 转发和 SOCKS 通道 |
| 自动化 | 非交互密钥认证、配置别名和复用连接 |
| 审计 | 服务端日志、用户身份和会话记录 |

## 2. 安装与服务管理

### 2.1 Linux 安装

Debian/Ubuntu：

```bash
sudo apt update
sudo apt install openssh-client openssh-server
```

RHEL/Fedora：

```bash
sudo dnf install openssh-clients openssh-server
```

检查版本：

```bash
ssh -V
sshd -V
```

### 2.2 管理 sshd 服务

```bash
sudo systemctl status ssh
sudo systemctl start ssh
sudo systemctl enable ssh
```

在部分发行版中服务名是 `sshd`：

```bash
sudo systemctl status sshd
```

修改服务端配置后先检查语法，再重载：

```bash
sudo sshd -t
sudo systemctl reload ssh
```

重启会中断已有连接，修改远程 SSH 配置时通常优先使用 `reload`，并保留当前会话用于回滚。

### 2.3 Windows OpenSSH

Windows 也可以使用 OpenSSH Client。PowerShell 中检查：

```powershell
ssh -V
Get-Service ssh-agent
```

配置文件通常位于：

```text
%USERPROFILE%\.ssh\config
%USERPROFILE%\.ssh\known_hosts
```

## 3. 密钥认证

### 3.1 密钥对

SSH 密钥认证由一对密钥组成：

```text
私钥：只保存在客户端，必须保密
公钥：复制到服务端账号的 ~/.ssh/authorized_keys
```

生成 Ed25519 密钥：

```bash
ssh-keygen -t ed25519 -C "ssh-key"
```

生成过程会询问：

- 私钥保存路径，默认是 `~/.ssh/id_ed25519`。
- 私钥口令短语，用于保护磁盘上的私钥。

常见文件：

```text
~/.ssh/id_ed25519          私钥
~/.ssh/id_ed25519.pub      公钥
~/.ssh/authorized_keys     服务端允许登录的公钥列表
```

### 3.2 把公钥安装到服务端

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@example.com
```

也可以手动追加：

```bash
cat ~/.ssh/id_ed25519.pub | ssh user@example.com \
  'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

服务端权限通常应满足：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

目录或文件权限过宽时，`sshd` 可能因为 Strict Modes 拒绝使用密钥。

### 3.3 指定私钥

```bash
ssh -i ~/.ssh/id_ed25519 user@example.com
scp -i ~/.ssh/id_ed25519 file.txt user@example.com:/tmp/
```

配置别名后可以避免反复输入：

```sshconfig
Host dev-server
    HostName 203.0.113.10
    User dev
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

连接：

```bash
ssh dev-server
```

### 3.4 ssh-agent

私钥可以设置口令短语。`ssh-agent` 在内存中暂存解锁后的私钥，避免每次连接重复输入：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh-add -l
ssh-add -D
```

Windows PowerShell 示例：

```powershell
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
ssh-add -l
```

使用 Agent Forwarding 时需要谨慎：远端进程可能借用转发的 Agent 请求客户端签名。除非确实需要，不要对不完全信任的服务器启用 `ForwardAgent yes`。

### 3.5 密钥类型与口令

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519
```

- `ed25519` 是现代 OpenSSH 中常用的短密钥类型。
- `-a` 增加私钥口令的 KDF 计算成本。
- 私钥仍应设置口令短语，避免磁盘文件泄露后直接被使用。

查看公钥指纹：

```bash
ssh-keygen -lf ~/.ssh/id_ed25519.pub
```

## 4. 主机身份与 known_hosts

### 4.1 为什么需要验证服务端

加密连接只能保证“连接内容被加密”，不能自动证明连接对象就是预期服务器。SSH 首次连接时会显示服务端 Host Key 指纹：

```text
The authenticity of host 'example.com' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting?
```

确认指纹来源可信后，接受的 Host Key 会写入：

```text
~/.ssh/known_hosts
```

### 4.2 查看和删除 Host Key

```bash
ssh-keygen -F example.com
ssh-keygen -F '[example.com]:2222'
ssh-keygen -R example.com
```

如果服务器重装导致 Host Key 变化，SSH 会提示 `REMOTE HOST IDENTIFICATION HAS CHANGED`。不要直接关闭校验，应先确认服务器身份，再删除旧记录并重新验证。

### 4.3 StrictHostKeyChecking

常见配置：

```sshconfig
Host *
    StrictHostKeyChecking ask
    UserKnownHostsFile ~/.ssh/known_hosts
```

临时调试可以使用：

```bash
ssh -o StrictHostKeyChecking=accept-new user@example.com
```

不建议长期使用：

```bash
ssh -o StrictHostKeyChecking=no user@example.com
```

关闭校验会削弱对中间人和错误主机的防护。

## 5. ssh_config

### 5.1 配置文件加载顺序

常见来源：

```text
命令行参数
  │
~/.ssh/config
  │
/etc/ssh/ssh_config
```

对同一个配置项，先匹配到的值通常优先生效，因此更具体的 `Host` 规则应放在通用规则之前。

### 5.2 常用配置项

```sshconfig
Host *
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ConnectTimeout 10
    TCPKeepAlive yes

Host dev-server
    HostName 192.0.2.10
    User dev
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ForwardAgent no
```

常用选项：

| 选项 | 作用 |
|------|------|
| `Host` | 配置匹配别名 |
| `HostName` | 实际连接的主机名或地址 |
| `User` | 远程用户名 |
| `Port` | 远程 SSH 端口 |
| `IdentityFile` | 指定私钥 |
| `IdentitiesOnly` | 只使用指定身份文件 |
| `ConnectTimeout` | 建立连接超时时间 |
| `ServerAliveInterval` | 客户端定期发送保活消息 |
| `ServerAliveCountMax` | 连续无响应后断开 |
| `ProxyJump` | 通过跳板机连接 |
| `LocalForward` | 配置本地端口转发 |
| `RemoteForward` | 配置远程端口转发 |
| `DynamicForward` | 配置 SOCKS 动态转发 |
| `ControlMaster` | 连接复用主连接 |

查看最终生效配置：

```bash
ssh -G dev-server
```

## 6. 远程登录与命令执行

### 6.1 交互式登录

```bash
ssh user@example.com
ssh -p 2222 user@example.com
ssh -i ~/.ssh/id_ed25519 user@example.com
```

用户名也可以写在配置文件中：

```bash
ssh dev-server
```

### 6.2 直接执行远程命令

```bash
ssh user@example.com 'uname -a'
ssh user@example.com 'df -h /'
ssh user@example.com 'systemctl status nginx'
```

多个命令建议交给远程 Shell：

```bash
ssh user@example.com 'cd /srv/app && git status && ./run.sh'
```

本地变量和远程变量的边界需要注意：

```bash
name="local"
ssh user@example.com "echo local=$name"
ssh user@example.com 'echo remote=$name'
```

双引号会先由本地 Shell 展开变量，单引号会把内容原样传给远程 Shell。

### 6.3 标准输入输出和退出码

```bash
cat local.txt | ssh user@example.com 'cat > /tmp/remote.txt'
ssh user@example.com 'command' > local-output.txt
ssh user@example.com 'command' 2> local-error.txt
```

远程命令的退出码可以被本地 Shell 使用：

```bash
if ssh user@example.com 'test -f /tmp/ready'; then
    echo "remote file exists"
fi
```

### 6.4 TTY

需要交互终端的程序使用 `-t`：

```bash
ssh -t user@example.com 'top'
ssh -t user@example.com 'sudo systemctl restart app'
```

批处理时通常禁用 TTY：

```bash
ssh -T user@example.com 'printf "%s\\n" ready'
```

### 6.5 常用连接参数

```bash
ssh -v user@example.com       # 一层调试日志
ssh -vv user@example.com      # 更详细
ssh -vvv user@example.com     # 最详细
ssh -4 user@example.com       # 强制 IPv4
ssh -6 user@example.com       # 强制 IPv6
ssh -F ./ssh_config host       # 使用指定配置文件
ssh -o ConnectTimeout=5 host   # 临时覆盖配置
```

## 7. 文件传输

### 7.1 scp

上传文件：

```bash
scp local.txt user@example.com:/tmp/
scp -P 2222 local.txt user@example.com:/tmp/
```

下载文件：

```bash
scp user@example.com:/tmp/remote.txt .
```

递归复制目录：

```bash
scp -r ./dist user@example.com:/srv/app/
```

不同路径的冒号含义：

```text
user@host:/remote/path    远程路径
./local/path               本地路径
```

现代 OpenSSH 的 `scp` 默认使用 SFTP 协议传输；如果必须兼容旧式 SCP 协议，可使用 `-O`，但应注意远程 Shell 路径解析和通配符行为。

### 7.2 sftp

启动交互式 SFTP：

```bash
sftp user@example.com
```

常用命令：

| 命令 | 作用 |
|------|------|
| `pwd` | 查看远程当前目录 |
| `lpwd` | 查看本地当前目录 |
| `ls` / `lls` | 查看远程 / 本地目录 |
| `cd path` | 切换远程目录 |
| `lcd path` | 切换本地目录 |
| `get file` | 下载文件 |
| `put file` | 上传文件 |
| `get -r dir` | 下载目录 |
| `put -r dir` | 上传目录 |
| `mkdir dir` | 创建远程目录 |
| `rm file` | 删除远程文件 |
| `rename a b` | 重命名远程文件 |
| `bye` / `exit` | 退出 |

批处理传输：

```bash
sftp user@example.com <<'EOF'
put local.txt /tmp/remote.txt
get /tmp/result.txt ./result.txt
bye
EOF
```

### 7.3 rsync over SSH

同步目录：

```bash
rsync -avz ./project/ user@example.com:/srv/project/
rsync -avz user@example.com:/srv/project/ ./project/
```

常用选项：

| 选项 | 作用 |
|------|------|
| `-a` | 归档模式，递归并保留常见属性 |
| `-v` | 显示详细过程 |
| `-z` | 传输时压缩 |
| `-n` | Dry Run，只预览不修改 |
| `--delete` | 删除目标端多余文件，使用前必须确认 |
| `--exclude` | 排除指定文件或目录 |
| `-e` | 指定 SSH 命令 |

先预览变化：

```bash
rsync -avzn --delete ./project/ user@example.com:/srv/project/
```

指定跳板机或端口：

```bash
rsync -av -e 'ssh -p 2222' ./project/ user@example.com:/srv/project/
```

路径末尾的 `/` 很重要：

```text
./project/  → 同步 project 目录中的内容
./project   → 可能在目标端创建 project 这一层目录
```

## 8. 跳板机与代理

### 8.1 ProxyJump

目标主机只能从跳板机访问时：

```bash
ssh -J jump-user@jump.example.com internal-user@10.0.0.20
```

配置方式：

```sshconfig
Host jump
    HostName jump.example.com
    User jump-user
    IdentityFile ~/.ssh/id_ed25519_jump

Host internal
    HostName 10.0.0.20
    User internal-user
    ProxyJump jump
    IdentityFile ~/.ssh/id_ed25519_internal
```

```bash
ssh internal
scp ./file.txt internal:/tmp/
sftp internal
```

`ProxyJump` 的关键点是：目标主机的 SSH 会话仍然由客户端建立，跳板机主要负责转发 TCP 连接。

### 8.2 多级跳板

```bash
ssh -J jump1,jump2 user@internal.example.com
```

配置：

```sshconfig
Host internal
    HostName 10.0.0.20
    User user
    ProxyJump jump1,jump2
```

### 8.3 ProxyCommand

`ProxyCommand` 允许使用一个本地命令提供到目标的标准输入输出通道：

```sshconfig
Host internal
    HostName 10.0.0.20
    User user
    ProxyCommand ssh jump.example.com -W %h:%p
```

通常优先使用 `ProxyJump`，只有需要特殊代理程序或自定义连接逻辑时才使用 `ProxyCommand`。

### 8.4 连接复用

重复建立 SSH 连接会产生认证和密钥交换开销，可以使用 ControlMaster：

```sshconfig
Host dev-server
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C
```

查看和关闭主连接：

```bash
ssh -O check dev-server
ssh -O exit dev-server
```

`ControlPath` 应使用不会与不同主机冲突的模板，并注意 Socket 路径长度限制。

## 9. SSH 端口转发

### 9.1 三类转发的方向

```text
-L local_port:destination:destination_port
    本地监听 → 通过 SSH 服务端访问目标

-R remote_port:destination:destination_port
    远程监听 → 通过 SSH 客户端访问目标

-D local_port
    本地监听 SOCKS → 由远端按请求访问不同目标
```

判断端口转发时，先回答三个问题：

1. 哪台机器上创建监听端口？
2. 哪台机器发起到最终目标的 TCP 连接？
3. 监听地址是 `127.0.0.1` 还是对外网卡开放？

### 9.2 本地转发 `-L`

客户端可以 SSH 到跳板机，但客户端不能直接访问内网数据库：

```text
本地程序                跳板机                 内网服务
127.0.0.1:15432 ─SSH─> jump ───────────────> db:5432
```

```bash
ssh -N -L 127.0.0.1:15432:db.internal:5432 user@jump.example.com
```

之后本地程序连接：

```bash
psql -h 127.0.0.1 -p 15432 -U appuser appdb
```

`db.internal:5432` 是由跳板机解析和访问的，不是由本地机器直接访问。

访问远程主机本地服务：

```bash
ssh -N -L 8080:127.0.0.1:8080 user@server.example.com
```

本地访问 `http://127.0.0.1:8080`，实际连接的是远程服务器本机的 8080 端口。

配置文件写法：

```sshconfig
Host db-tunnel
    HostName jump.example.com
    User user
    LocalForward 127.0.0.1:15432 db.internal:5432
    ExitOnForwardFailure yes
```

### 9.3 远程转发 `-R`

内网客户端可以主动连接公网服务器，但公网服务器无法主动访问内网客户端：

```text
内网客户端 ─────── SSH 主动连接 ───────> 公网服务器
    ▲                                      │
    └── 本地服务 127.0.0.1:8080            └── 监听 remote_port
```

```bash
ssh -N -R 127.0.0.1:18080:127.0.0.1:8080 user@public.example.com
```

此时访问公网服务器本机的 `127.0.0.1:18080`，流量会转发到内网客户端的 `127.0.0.1:8080`。

如果公网服务器需要让其他主机访问，服务端可能需要允许远程转发监听非回环地址：

```bash
ssh -N -R 0.0.0.0:18080:127.0.0.1:8080 user@public.example.com
```

对外监听会扩大暴露面，应优先使用服务端回环地址，再通过反向代理或访问控制暴露必要路径。

服务端相关配置项：

```text
AllowTcpForwarding
GatewayPorts
PermitOpen
PermitListen
```

### 9.4 动态转发 `-D`

动态转发在本地创建 SOCKS4/5 代理：

```bash
ssh -N -D 127.0.0.1:1080 user@jump.example.com
```

命令行测试：

```bash
curl --socks5-hostname 127.0.0.1:1080 http://internal.example.com:8080
```

`--socks5-hostname` 会让域名解析也交给 SOCKS 代理端，避免本地无法解析内网域名。

### 9.5 仅转发不打开 Shell

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:15432:db.internal:5432 user@jump.example.com
```

- `-N`：不执行远程命令。
- `-T`：不分配伪终端。
- `ExitOnForwardFailure=yes`：转发监听失败时立即退出。

## 10. 内网穿透

### 10.1 什么是内网穿透

内网穿透是指：在 NAT、防火墙或没有公网地址的网络环境中，通过具有公网可达性的中继节点，让外部访问内网中的服务。

```text
外部客户端 ───────────> 公网中继 / 跳板机
                              ▲
                              │ 长连接、反向连接
                              │
                       内网客户端
                              │
                         内网服务
```

关键条件是连接方向：

- 外部客户端无法主动连接内网服务。
- 内网客户端可以主动连接公网中继。
- 中继保存连接状态，把外部连接转发到内网连接。

内网穿透只是改变连接路径，仍然需要经过授权、认证、访问控制和审计。

### 10.2 NAT 为什么影响外部访问

```text
内网服务 192.168.1.20:8080
        │ 私有地址
        ▼
路由器公网地址 203.0.113.5
        │ NAT / 防火墙
        ▼
互联网客户端
```

外部客户端通常不知道内网服务的私有地址，路由器也不会默认把入站连接转发到某一台内网主机。内网主机主动建立到公网服务器的长连接后，公网服务器可以利用这条已建立的连接转发数据。

### 10.3 基于 SSH `-R` 的最小内网穿透

假设：

```text
内网客户端：可以访问 public.example.com
内网服务：127.0.0.1:8080
公网服务器：public.example.com
公网服务器端口：18080
```

内网客户端执行：

```bash
ssh -N -T \
    -R 127.0.0.1:18080:127.0.0.1:8080 \
    -o ExitOnForwardFailure=yes \
    tunnel@public.example.com
```

流量路径：

```text
访问 public:18080
        │
        ▼
SSH Server 远程转发端口
        │ SSH 加密通道
        ▼
内网客户端 127.0.0.1:8080
```

如果公网服务器需要让其他主机访问，服务端可能需要允许远程转发监听非回环地址：

```text
GatewayPorts clientspecified
```

然后使用：

```bash
ssh -N -T \
    -R 0.0.0.0:18080:127.0.0.1:8080 \
    tunnel@public.example.com
```

这种方式的风险较高，因为服务可能被公开暴露。更稳妥的方式是让 `sshd` 只监听回环地址，再在公网服务器上通过 Nginx、认证网关或 VPN 提供受控访问。

### 10.4 只允许隧道账号转发

为反向隧道建立独立的低权限账号，并限制其 Shell 和转发范围：

```text
Match User tunnel
    AllowTcpForwarding remote
    PermitTTY no
    X11Forwarding no
    AllowAgentForwarding no
    PasswordAuthentication no
    PermitOpen 127.0.0.1:8080
```

检查服务端最终生效配置：

```bash
sshd -T | grep -E 'allowtcpforwarding|permitopen|permitlisten|gatewayports'
```

### 10.5 SSH 反向隧道保活

网络短暂抖动会导致 SSH 连接断开，可以配置客户端保活：

```sshconfig
Host tunnel-server
    HostName public.example.com
    User tunnel
    IdentityFile ~/.ssh/id_ed25519_tunnel
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
    RemoteForward 127.0.0.1:18080 127.0.0.1:8080
```

服务需要长期运行时，可以使用 systemd 管理：

```ini
[Unit]
Description=SSH reverse tunnel
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/ssh -N -T tunnel-server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ssh-reverse-tunnel.service
systemctl status ssh-reverse-tunnel.service
```

复杂网络环境中也可以使用 `autossh` 监控并重启隧道：

```bash
autossh -M 0 -N -T tunnel-server
```

### 10.6 SSH 与专用穿透工具

| 方案 | 通道模型 | 特点 |
|------|----------|------|
| SSH `-R` | 单条加密反向隧道 | 系统已有、结构简单、适合少量服务 |
| frp | Client/Server 中继 | 支持多服务、配置化、协议类型较多 |
| ngrok | 托管中继 | 快速暴露 HTTP/TCP，依赖外部服务 |
| cloudflared Tunnel | 出站长连接到边缘网络 | 常与域名、访问策略和身份认证结合 |
| VPN | 网络层或三层互联 | 适合多个网段和长期网络访问 |

选择思路：

```text
单个临时 TCP 服务
    └── SSH -R

多个长期服务，需要连接管理
    └── frp 或专用隧道服务

需要域名、HTTPS 和身份策略
    └── 反向代理或 Tunnel 服务

需要访问整个网段
    └── VPN / Overlay Network
```

### 10.7 内网穿透与反向代理

对外提供 HTTP 服务时，不建议直接把应用端口暴露给互联网，可以采用：

```text
公网客户端
    │ HTTPS / 身份认证 / 限流
    ▼
公网 Nginx / API Gateway
    │ 仅访问回环端口
    ▼
SSH -R / frp / Tunnel
    │
    ▼
内网应用 127.0.0.1:8080
```

反向代理可以集中处理 TLS 证书、域名路由、身份认证、速率限制、访问日志和健康检查。

## 11. SSH 安全与权限

### 11.1 客户端安全

- 私钥文件不上传、不提交到 Git、不通过聊天工具传播。
- 私钥设置口令并使用 `ssh-agent` 管理。
- 核对首次连接的 Host Key 指纹。
- 不要长期使用 `StrictHostKeyChecking=no`。
- 不要对不可信主机开启 Agent Forwarding。
- 端口转发默认绑定 `127.0.0.1`，确需对外开放时再明确指定。

### 11.2 服务端安全

服务端配置通常位于 `/etc/ssh/sshd_config`。修改前备份并检查语法：

```bash
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
sudo sshd -t
```

常见方向：

```text
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
MaxAuthTries 3
X11Forwarding no
AllowTcpForwarding no 或按需限制
```

不要在不了解现有认证方式的情况下直接关闭密码认证。应先验证密钥登录成功，并保留一个可回滚的管理会话。

### 11.3 最小权限隧道账号

```text
专用账号
  ├── 仅允许公钥认证
  ├── 禁止 TTY
  ├── 禁止 Agent Forwarding
  ├── 限制可监听地址和端口
  ├── 限制可连接目标
  └── 单独记录日志
```

### 11.4 监听地址与暴露面

```text
127.0.0.1:port     仅本机访问
内网地址:port       同一网络可访问
0.0.0.0:port        所有 IPv4 网卡可能可访问
[::]:port           所有 IPv6 网卡可能可访问
```

检查监听：

```bash
ss -lntp
ss -lntp | grep 18080
```

公网端口还需要结合云防火墙、主机防火墙、反向代理和应用自身认证一起判断。

## 12. SSH 故障排查

### 12.1 连接失败

```bash
ssh -vvv user@example.com
```

按层定位：

```text
DNS
  └── getent hosts example.com / nslookup example.com

TCP
  └── nc -vz example.com 22

服务
  └── systemctl status sshd

防火墙
  └── ss -lntp、ufw status、firewall-cmd --list-all

认证
  └── ssh -vvv、authorized_keys 权限、sshd 日志
```

常见命令：

```bash
getent hosts example.com
nc -vz example.com 22
telnet example.com 22
```

### 12.2 公钥认证失败

客户端检查：

```bash
ssh -vvv -i ~/.ssh/id_ed25519 user@example.com
ssh-add -l
```

服务端检查：

```bash
ls -ld ~/.ssh
ls -l ~/.ssh/authorized_keys
sudo journalctl -u sshd -n 100 --no-pager
sudo sshd -T | grep -E 'pubkeyauthentication|authorizedkeysfile|strictmodes'
```

排查顺序：

1. 用户名是否正确。
2. 客户端是否使用了预期私钥。
3. 公钥是否完整写入 `authorized_keys`。
4. `~/.ssh` 和 `authorized_keys` 权限是否合理。
5. 服务端是否允许公钥认证。
6. SELinux/AppArmor 或家目录权限是否阻止读取。

### 12.3 Host Key 警告

```bash
ssh-keygen -F example.com
ssh-keygen -R example.com
```

只有在确认服务器确实更换过 Host Key 后，才删除旧记录并重新核对指纹。

### 12.4 端口转发失败

```bash
ssh -vvv -N -T -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:15432:db.internal:5432 user@jump.example.com
```

检查：

- 本地监听端口是否已被占用：`ss -lntp | grep 15432`。
- SSH 服务端是否允许 TCP Forwarding。
- 跳板机能否解析并访问最终目标。
- 目标服务是否监听正确地址和端口。
- 远程转发时 `GatewayPorts` 和防火墙是否允许。
- 转发绑定地址是否只监听了回环接口。

### 12.5 通过隧道访问服务失败

先分开验证两段连接：

```bash
# 在隧道发起端验证内网服务
curl http://127.0.0.1:8080

# 在中继服务器验证远程监听
curl http://127.0.0.1:18080
```

如果第一条失败，问题在内网服务；如果第一条成功而第二条失败，检查 SSH 转发、监听地址、服务端策略和防火墙。

### 12.6 连接频繁断开

检查网络、NAT 超时和保活：

```sshconfig
Host tunnel-server
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

保活不是性能优化，过短的间隔可能增加无意义的网络流量；应结合网络设备的连接超时设置调整。

## 13. 常用命令速查

### 13.1 连接和配置

```bash
ssh user@host
ssh -p 2222 user@host
ssh -i ~/.ssh/key user@host
ssh -G host
ssh -Q cipher
ssh -vvv user@host
```

### 13.2 密钥和指纹

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
ssh-keygen -lf ~/.ssh/id_ed25519.pub
ssh-keygen -F host
ssh-keygen -R host
ssh-add ~/.ssh/id_ed25519
ssh-add -l
```

### 13.3 文件传输

```bash
scp file user@host:/tmp/
scp user@host:/tmp/file .
sftp user@host
rsync -avz ./dir/ user@host:/srv/dir/
```

### 13.4 端口转发

```bash
ssh -N -L 127.0.0.1:8080:internal:80 user@jump
ssh -N -R 127.0.0.1:18080:127.0.0.1:8080 user@public
ssh -N -D 127.0.0.1:1080 user@jump
ssh -J jump user@internal
```

## 14. 学习实验路线

### 第一阶段：远程连接

1. 使用密码完成一次连接。
2. 生成 Ed25519 密钥并配置口令。
3. 使用 `ssh-copy-id` 安装公钥。
4. 编写 `~/.ssh/config` 别名。
5. 使用 `ssh -vvv` 观察认证过程。

### 第二阶段：文件和跳板

1. 使用 `ssh` 执行远程命令。
2. 使用 `scp` 和 `sftp` 传输文件。
3. 使用 `rsync --dry-run` 预览同步结果。
4. 配置 `ProxyJump` 访问内网主机。
5. 使用 ControlMaster 观察连接复用。

### 第三阶段：端口转发

1. 使用 `-L` 访问远程主机本地服务。
2. 使用 `-L` 通过跳板机访问内网数据库。
3. 使用 `-D` 建立 SOCKS5 代理。
4. 在授权环境中使用 `-R` 将测试服务转发到中继服务器。
5. 配置保活、日志和自动重启。

## 15. 参考资料

- [OpenSSH `ssh` manual](https://man7.org/linux/man-pages/man1/ssh.1.html)
- [OpenSSH `ssh_config` manual](https://man7.org/linux/man-pages/man5/ssh_config.5.html)
- [OpenSSH `scp` manual](https://man7.org/linux/man-pages/man1/scp.1.html)
- [OpenSSH `ssh-keygen` manual](https://man7.org/linux/man-pages/man1/ssh-keygen.1.html)
- [OpenBSD OpenSSH portable project](https://www.openssh.com/portable.html)
- [RFC 4251: The Secure Shell Protocol Architecture](https://www.rfc-editor.org/rfc/rfc4251)
- [RFC 4253: The Secure Shell Transport Layer Protocol](https://www.rfc-editor.org/rfc/rfc4253)
- [RFC 4254: The Secure Shell Connection Protocol](https://www.rfc-editor.org/rfc/rfc4254)
