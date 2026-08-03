# Shell 脚本与命令行

> Shell 脚本编程学习笔记（Bash 为主）

## 内容概览

- Shell脚本简介
- 基础语法（变量、字符串、数组）
- 控制流（条件判断、循环）
- 函数与脚本结构
- 输入/输出与管道
- 文本处理三剑客（grep、sed、awk）
- 进程管理与后台任务
- 实用脚本示例

## 1.Shell脚本简介
Shell 是一个命令行解释器，提供了用户与操作系统交互的接口。Shell 脚本是一种用来自动化执行任务的文本文件，包含了一系列的命令和控制结构。通过编写 Shell 脚本，可以简化重复性的工作，提高效率。

### Shell 与 Terminal 与 Kernel

想象你在一家**餐厅**吃饭：

- **Terminal (终端)** = **餐厅的座位和餐桌**。它提供了一个环境让你坐下来（输入）和看到菜品（输出）。它本身不做菜，也不记菜单，只是一个“窗口”。


- **Shell (壳)** = **服务员**。你把想吃的菜（命令）告诉服务员。服务员听懂你的话，把需求传达给厨房，厨房做好后，服务员再端给你。服务员有不同的“流派”，有的讲英语（Bash），有的讲法语（Zsh），有的讲方言（CMD）。


- **Kernel (内核)** = **厨房**。真正干活的地方（操作硬件、管理内存等）。你不能直接冲进厨房做饭，必须通过服务员（Shell）与厨房沟通。

#### **Terminal (终端)**

**关键词：窗口、输入输出设备**

**定义：** Terminal 是一个**界面程序**，它提供了一个窗口让你输入命令和查看输出。它负责处理显示、字体、颜色等视觉效果，但它不理解你输入的命令是什么，它只是把你的按键传给里面的 Shell。

例子：
* Windows 上的 **Windows Terminal**（那个现代化的多标签页窗口）、PuTTY。
* macOS 上的 **iTerm2**、系统自带的 Terminal.app。

**核心功能：** 它可以调整字体大小、背景颜色、复制粘贴。但它不懂你在打什么代码，它只负责把你的按键传给里面的 **Shell**。

#### **Shell (壳)**

**关键词：解释器、中间人**

**定义：** 它是一个**命令行解释器**。它运行在 Terminal 里面。它读取你输入的命令（比如 `ls` 或 `dir`），翻译成内核能听懂的指令，然后把结果返回给你。它像一层“壳”包裹着内核（Kernel），保护内核不被用户直接操作。

Bash, CMD, PowerShell三者都是 **Shell** 的具体实现，它们分别属于不同的阵营。

##### **Bash (Bourne Again Shell)**

用于Unix/Linux/macOS，特点如下：
* Linux 世界的**绝对标准**。
* 基于文本流：命令之间传递的是纯文本。
* 非常适合编写脚本来自动化管理服务器。
* *注：macOS 现在默认使用 **Zsh**，它是 Bash 的增强版，兼容 Bash 但更漂亮好用。*

##### **CMD (Command Prompt / 命令提示符)**

用于Windows (旧时代)，特点如下：
* 经典的黑色窗口，提示符通常是 `C:\>`。
* 它是从 MS-DOS 时代遗留下来的产物。
* **功能较弱**，命令难以记忆且语法古怪（比如 `dir` 列出文件）。
* 主要用于执行简单的系统维护，现在微软已经不再重点发展它，但为了兼容性一直保留。

##### **PowerShell**

用于Windows (新时代) & 跨平台，特点如下：
* 微软为了取代 CMD 并抗衡 Linux Shell 而开发的。
* **极其强大**：它不仅是 Shell，还是一门面向对象的脚本语言。
* **面向对象**：这是它与 Bash 最大的区别。在 Bash 里，`ls` 输出的是一串文本；在 PowerShell 里，`ls` 返回的是一个个“文件对象”（包含大小、日期等属性）。

**Terminal 可以运行不同的 Shell， 且Shell 可以互相调用。**

#### **Kernel (内核)**

**关键词：操作系统核心、资源管理**

**定义：** Kernel 是操作系统的核心部分，负责管理系统资源（CPU、内存、设备等），提供底层服务。用户不能直接与 Kernel 交互，必须通过 Shell 来发送指令。

**总结：**
* Terminal 是一个窗口，提供输入输出环境。
* Shell 是一个解释器，翻译用户命令并与内核沟通。
* Kernel 是操作系统的核心，管理资源和执行命令。

## 2. 基础语法

### Shebang 与脚本结构

```bash
#!/usr/bin/env bash          # 推荐写法：使用 env 找 bash（跨平台兼容）
set -euo pipefail            # 最佳实践：遇错退出(-e) 未定义变量报错(-u) 管道错误传递(-o pipefail)

# 注释用 #
echo "Hello, World!"
```

### 变量

```bash
# 赋值：= 两侧不能有空格
name="Alice"
age=30
readonly PI=3.14159          # 只读变量

# 引用：用 $var 或 ${var}（推荐后者，边界清晰）
echo "Name: ${name}, Age: ${age}"
echo "Name length: ${#name}"  # 变量长度

# 命令替换（两种等价写法，推荐 $()）
today=$(date +%Y-%m-%d)
files=`ls -1 | wc -l`

# 算术运算
result=$((age * 2 + 1))
echo "Result: ${result}"
```

### 字符串操作

```bash
str="Hello, World!"

echo "${str:0:5}"            # 截取：从位置0取5个字符 → Hello
echo "${str#Hello, }"        # 删除前缀 → World!
echo "${str%!}"              # 删除后缀 → Hello, World
echo "${str/World/Bash}"     # 替换第一个 → Hello, Bash!
echo "${str//l/L}"           # 替换全部 → HeLLo, WorLd!
echo "${str,,}"              # 全部转小写（bash 4+）
echo "${str^^}"              # 全部转大写（bash 4+）

# 默认值
unset var
echo "${var:-default}"       # var 未定义则用 default
echo "${var:=default}"       # var 未定义则赋值并返回 default
```

### 数组

```bash
# 定义数组
fruits=("apple" "banana" "cherry")
fruits+="orange"             # 追加

echo "${fruits[0]}"          # 第一个元素
echo "${fruits[@]}"          # 所有元素
echo "${#fruits[@]}"         # 数组长度
echo "${fruits[@]:1:2}"      # 切片（从索引1取2个）

# 关联数组（bash 4+）
declare -A colors
colors["red"]="#FF0000"
colors["green"]="#00FF00"
echo "${colors[red]}"
echo "${!colors[@]}"         # 所有键
```

## 3. 控制流

### 条件判断

```bash
# if / elif / else
x=10
if [[ $x -gt 5 ]]; then
    echo "x > 5"
elif [[ $x -eq 5 ]]; then
    echo "x == 5"
else
    echo "x < 5"
fi

# [[ ]] vs [ ]：推荐 [[ ]]（更安全，支持 && || 正则）
# 整数比较: -eq -ne -lt -le -gt -ge
# 字符串比较: == != < > -z(空) -n(非空)
# 文件测试: -f(文件) -d(目录) -e(存在) -r(可读) -w(可写) -x(可执行)

file="/etc/passwd"
if [[ -f "$file" && -r "$file" ]]; then
    echo "文件存在且可读"
fi

# 正则匹配
email="user@example.com"
if [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "邮箱格式合法"
fi

# case 语句
lang="python"
case "$lang" in
    python|py)  echo "Python 🐍" ;;
    bash|sh)    echo "Shell 🐚" ;;
    cpp|c++)    echo "C++ 🔷" ;;
    *)          echo "未知语言" ;;
esac
```

### 循环

```bash
# for 循环（列表）
for fruit in apple banana cherry; do
    echo "水果: ${fruit}"
done

# for 循环（C 风格）
for ((i=0; i<5; i++)); do
    echo -n "${i} "
done; echo

# for 循环（范围，bash 4+）
for i in {1..10..2}; do   # 1 3 5 7 9
    echo -n "${i} "
done; echo

# 遍历文件
for f in /etc/*.conf; do
    echo "配置文件: $(basename "$f")"
done

# while 循环
count=0
while [[ $count -lt 5 ]]; do
    echo "count = ${count}"
    ((count++))
done

# 读取文件每一行
while IFS= read -r line; do
    echo "行: ${line}"
done < /etc/hostname

# until（条件为假时执行）
n=0
until [[ $n -ge 3 ]]; do
    echo "n = ${n}"; ((n++))
done
```

## 4. 函数

```bash
# 定义函数
greet() {
    local name="$1"       # local 限制变量作用域（重要！）
    local greeting="${2:-Hello}"  # 第二个参数有默认值
    echo "${greeting}, ${name}!"
    return 0              # 返回值只能是 0-255（0=成功）
}

greet "Alice"             # Hello, Alice!
greet "Bob" "Hi"          # Hi, Bob!

# 返回字符串：用命令替换捕获输出
get_date() {
    date "+%Y-%m-%d %H:%M:%S"
}
now=$(get_date)
echo "当前时间: ${now}"

# 参数说明
usage() {
    cat <<EOF
用法: $0 [选项] <文件>
选项:
  -h, --help     显示帮助
  -v, --verbose  详细输出
  -n <N>         处理前 N 行（默认: 10）
EOF
}

# 使用 getopts 解析选项
parse_args() {
    local verbose=false
    local lines=10
    while getopts "hvn:" opt; do
        case $opt in
            h) usage; exit 0 ;;
            v) verbose=true ;;
            n) lines="$OPTARG" ;;
            ?) echo "未知选项"; exit 1 ;;
        esac
    done
    echo "verbose=${verbose}, lines=${lines}"
}
```

## 5. 输入/输出与管道

```bash
# 标准输入(0) 标准输出(1) 标准错误(2)
echo "正常输出"           # → stdout
echo "错误信息" >&2       # → stderr
ls /nonexistent 2>/dev/null         # 丢弃 stderr
ls /nonexistent 2>&1 | grep "No"    # stderr 重定向到 stdout 再管道

# 重定向
echo "写入文件" > file.txt     # 覆盖写
echo "追加内容" >> file.txt    # 追加写
cat < file.txt                 # 从文件读输入

# Here Document（多行字符串）
cat <<'EOF'
这里的 $变量 不会被展开
因为 EOF 加了引号
EOF

cat <<EOF
这里的 ${name} 会被展开
EOF

# 管道组合
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

# tee：同时写文件和传递给管道
ls -la | tee files.txt | wc -l

# xargs：将输入转换为参数
find . -name "*.pyc" | xargs rm -f
echo "a b c" | xargs -n1 echo   # 每个参数单独一行

# 进程替换
diff <(ls dir1/) <(ls dir2/)     # 不需要临时文件
```

## 6. 文本处理三剑客

### grep（搜索）

```bash
grep "pattern" file.txt           # 基本搜索
grep -i "pattern" file.txt        # 忽略大小写
grep -r "pattern" ./dir/          # 递归搜索
grep -n "pattern" file.txt        # 显示行号
grep -v "pattern" file.txt        # 反向匹配（不含 pattern 的行）
grep -c "pattern" file.txt        # 只显示匹配行数
grep -l "pattern" *.txt           # 只显示文件名
grep -E "pat1|pat2" file.txt      # 扩展正则（等价于 egrep）
grep -o "[0-9]\+" file.txt        # 只输出匹配部分

# 常用组合
ps aux | grep -E "python|java" | grep -v grep
```

### sed（流编辑器）

```bash
sed 's/old/new/' file.txt         # 替换每行第一个匹配
sed 's/old/new/g' file.txt        # 替换每行所有匹配
sed 's/old/new/gi' file.txt       # 全局且忽略大小写
sed -i 's/old/new/g' file.txt     # 原地修改文件（-i）
sed -i.bak 's/old/new/g' file.txt # 备份后修改

sed -n '3,7p' file.txt            # 打印第3-7行
sed '3,7d' file.txt               # 删除第3-7行
sed '/^#/d' file.txt              # 删除注释行
sed '/^$/d' file.txt              # 删除空行
sed -n '/start/,/end/p' file.txt  # 打印 start~end 之间的行

# 多命令
sed -e 's/foo/bar/' -e 's/baz/qux/' file.txt
```

### awk（数据处理）

```bash
awk '{print $1}' file.txt              # 打印第1列
awk '{print $NF}' file.txt             # 打印最后一列
awk -F: '{print $1}' /etc/passwd       # 以 : 为分隔符
awk '{sum+=$1} END{print sum}' nums.txt # 求第1列之和
awk 'NR>1{print}' file.txt             # 跳过第一行（表头）
awk 'length($0)>80' file.txt           # 打印长度>80的行

# 格式化输出
awk '{printf "%-10s %5d\n", $1, $2}' file.txt

# 条件过滤
ps aux | awk '$3>1.0 {print $0}'       # CPU 占用>1% 的进程

# 复杂示例：统计 nginx 访问日志中各 IP 访问次数
# awk '{count[$1]++} END{for(ip in count) print count[ip], ip}' access.log | sort -rn | head
```

## 7. 进程管理与后台任务

```bash
# 后台运行
./script.sh &              # 后台运行，返回 PID
nohup ./script.sh &        # 不受终端关闭影响，输出到 nohup.out
nohup ./script.sh > out.log 2>&1 &  # 自定义输出文件

# 作业控制
jobs                       # 列出后台任务
fg %1                      # 将作业1调到前台
bg %1                      # 将作业1在后台继续运行
Ctrl+Z                     # 暂停当前前台进程

# 进程信号
kill PID                   # 发送 SIGTERM（优雅停止）
kill -9 PID                # 发送 SIGKILL（强制终止）
killall python             # 按名称杀进程
pkill -f "script.sh"       # 按完整命令行匹配

# 等待子进程
wait $!                    # 等待最后一个后台进程
wait                       # 等待所有后台进程

# trap：捕获信号，实现清理
cleanup() {
    echo "清理临时文件..."
    rm -f /tmp/myapp_*
    exit 0
}
trap cleanup INT TERM EXIT  # Ctrl+C / SIGTERM / 脚本退出时触发

# 超时执行
timeout 30 ./long_running.sh || echo "超时了！"
```

## 8. 远程开发

### scp（安全复制）

```bash
# 从本地复制到远程
scp localfile.txt user@remote:/path/to/destination/
# 从远程复制到本地
scp user@remote:/path/to/source.txt ./
# 复制目录（加 -r）
scp -r localdir/ user@remote:/path/to/destination/
```

## 9. 实用脚本示例

### 批量重命名文件

```bash
#!/usr/bin/env bash
set -euo pipefail

# 将当前目录所有 .jpeg 重命名为 .jpg
rename_ext() {
    local old_ext="$1" new_ext="$2"
    local count=0
    for f in *."${old_ext}"; do
        [[ -f "$f" ]] || continue
        mv -- "$f" "${f%.${old_ext}}.${new_ext}"
        ((count++))
    done
    echo "已重命名 ${count} 个文件"
}
rename_ext "jpeg" "jpg"
```

### 检测服务健康

```bash
#!/usr/bin/env bash
check_service() {
    local url="$1" max_retry="${2:-5}" delay="${3:-3}"
    for ((i=1; i<=max_retry; i++)); do
        if curl -sf --max-time 5 "$url" > /dev/null; then
            echo "✅ 服务正常: ${url}"
            return 0
        fi
        echo "⚠️  第 ${i}/${max_retry} 次检测失败，${delay}s 后重试..."
        sleep "$delay"
    done
    echo "❌ 服务不可用: ${url}"
    return 1
}
```

### 日志轮转

```bash
#!/usr/bin/env bash
rotate_log() {
    local logfile="$1" max_size_mb="${2:-100}" keep="${3:-5}"
    local size_kb
    size_kb=$(du -k "$logfile" 2>/dev/null | cut -f1)
    local max_size_kb=$((max_size_mb * 1024))

    if [[ ${size_kb:-0} -ge $max_size_kb ]]; then
        for ((i=keep-1; i>=1; i--)); do
            [[ -f "${logfile}.${i}" ]] && mv "${logfile}.${i}" "${logfile}.$((i+1))"
        done
        mv "$logfile" "${logfile}.1"
        echo "$(date '+%Y-%m-%d %H:%M:%S') 日志已轮转" > "$logfile"
        echo "日志轮转完成: ${logfile}"
    fi
}
```

```python
import subprocess
import os
import tempfile
import sys

def run_bash(script, input_data=None):
    """在 Python 中执行 bash 脚本并返回输出"""
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True,
        input=input_data
    )
    if result.stdout: print(result.stdout, end="")
    if result.stderr: print("[stderr]", result.stderr, end="", file=sys.stderr)
    return result

# Windows 环境下 bash 可能不可用，用 Python 演示等效逻辑
if sys.platform == "win32":
    print("Windows 环境：以下展示 Shell 脚本等效的 Python 逻辑演示\n")
    BASH_AVAILABLE = False
else:
    BASH_AVAILABLE = True

# ===== 1. 变量与字符串 =====
print("=" * 50)
print("1. 变量与字符串操作")
print("=" * 50)

if BASH_AVAILABLE:
    run_bash(r"""
name="Alice"
age=30
echo "姓名: ${name}, 年龄: ${age}"
echo "字符串长度: ${#name}"

str="Hello, World!"
echo "截取: ${str:0:5}"
echo "替换: ${str/World/Bash}"
echo "转大写: ${str^^}"
echo "删后缀: ${str%!}"
    """)
else:
    # Python 等效演示
    name, age = "Alice", 30
    print(f"姓名: {name}, 年龄: {age}")
    s = "Hello, World!"
    print(f"截取[0:5]: {s[:5]}")
    print(f"替换: {s.replace('World', 'Bash')}")
    print(f"转大写: {s.upper()}")
    print(f"删后缀: {s.rstrip('!')}")

# ===== 2. 数组操作 =====
print("\n" + "=" * 50)
print("2. 数组操作")
print("=" * 50)

if BASH_AVAILABLE:
    run_bash(r"""
fruits=("apple" "banana" "cherry" "orange")
echo "所有元素: ${fruits[@]}"
echo "第一个: ${fruits[0]}"
echo "数组长度: ${#fruits[@]}"
echo "切片[1:2]: ${fruits[@]:1:2}"

# 关联数组
declare -A colors
colors[red]="#FF0000"
colors[green]="#00FF00"
colors[blue]="#0000FF"
echo "所有键: ${!colors[@]}"
echo "red = ${colors[red]}"
    """)
else:
    fruits = ["apple", "banana", "cherry", "orange"]
    print(f"所有元素: {fruits}")
    print(f"第一个: {fruits[0]}, 长度: {len(fruits)}, 切片: {fruits[1:3]}")
    colors = {"red": "#FF0000", "green": "#00FF00", "blue": "#0000FF"}
    print(f"所有键: {list(colors.keys())}, red={colors['red']}")

# ===== 3. 控制流 =====
print("\n" + "=" * 50)
print("3. 控制流")
print("=" * 50)

if BASH_AVAILABLE:
    run_bash(r"""
# for 循环
echo -n "数字: "
for i in {1..5}; do echo -n "$i "; done; echo

# while 循环
count=0
while [[ $count -lt 3 ]]; do
    echo "  count = $count"
    ((count++))
done

# case
for lang in python bash rust; do
    case "$lang" in
        python) echo "  $lang → 🐍 通用脚本" ;;
        bash)   echo "  $lang → 🐚 系统脚本" ;;
        *)      echo "  $lang → 🔧 系统编程" ;;
    esac
done
    """)
else:
    print("数字: " + " ".join(str(i) for i in range(1,6)))
    for i in range(3): print(f"  count = {i}")
    for lang in ["python", "bash", "rust"]:
        d = {"python": "🐍 通用脚本", "bash": "🐚 系统脚本"}
        print(f"  {lang} → {d.get(lang, '🔧 系统编程')}")

# ===== 4. 文本处理三剑客演示 =====
print("\n" + "=" * 50)
print("4. grep / sed / awk 演示")
print("=" * 50)

# 创建临时测试文件
sample_data = """\
alice 25 engineer python
bob 30 manager java
charlie 22 developer python
dave 35 architect golang
eve 28 engineer rust
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(sample_data)
    tmp_file = f.name

if BASH_AVAILABLE:
    run_bash(f"""
file="{tmp_file}"
echo "--- 原始数据 ---"
cat "$file"

echo ""
echo "--- grep: 搜索 python 开发者 ---"
grep "python" "$file"

echo ""
echo "--- grep -c: 统计 python 行数 ---"
grep -c "python" "$file"

echo ""
echo "--- sed: 将 engineer 替换为 SWE ---"
sed 's/engineer/SWE/g' "$file"

echo ""
echo "--- awk: 打印姓名和年龄，年龄>25 ---"
awk '\$2 > 25 {{printf "  %-10s 年龄: %s\\n", \$1, \$2}}' "$file"

echo ""
echo "--- awk: 统计各语言人数 ---"
awk '{{count[\$4]++}} END{{for(lang in count) printf "  %-10s %d人\\n", lang, count[lang]}}' "$file"
    """)
else:
    lines = sample_data.strip().split('\n')
    print("--- 原始数据 ---")
    print(sample_data)
    print("--- grep 等效: 含 python 的行 ---")
    for l in lines:
        if "python" in l: print(" ", l)
    print(f"--- 行数: {sum(1 for l in lines if 'python' in l)} ---")
    print("--- awk 等效: 年龄>25 ---")
    for l in lines:
        parts = l.split()
        if int(parts[1]) > 25: print(f"  {parts[0]:10s} 年龄: {parts[1]}")
    from collections import Counter
    langs = Counter(l.split()[3] for l in lines)
    print("--- 语言统计 ---")
    for lang, n in langs.items(): print(f"  {lang:10s} {n}人")

os.unlink(tmp_file)

# ===== 5. 进程管理 =====
print("\n" + "=" * 50)
print("5. 进程管理")
print("=" * 50)

if BASH_AVAILABLE:
    run_bash(r"""
# 后台运行并获取 PID
sleep 2 &
bg_pid=$!
echo "后台进程 PID: ${bg_pid}"
jobs
wait $bg_pid
echo "后台进程已结束"

# 超时执行
if timeout 1 bash -c 'sleep 0.5; echo "在时限内完成"'; then
    echo "任务成功"
fi

# 检查命令是否存在
check_cmd() {
    command -v "$1" &>/dev/null && echo "$1 已安装: $(command -v $1)" || echo "$1 未安装"
}
check_cmd python3
check_cmd docker
check_cmd nvcc
    """)
else:
    import shutil
    for cmd in ["python", "docker", "nvcc"]:
        path = shutil.which(cmd)
        print(f"  {cmd}: {'已安装 ' + path if path else '未安装'}")
```
