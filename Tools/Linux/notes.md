## 文件I/O

### C标准I/O库函数
1. `fopen`：打开文件，返回文件指针。
```c
FILE *fopen (const char *__restrict __filename,
            const char *__restrict __modes)
```
其中modes如下：
- "r": 只读模式 没有文件打开失败
- "w": 只写模式 存在文件写入会清空文件,不存在文件则创建新文件
- "a": 只追加写模式 不会覆盖原有内容 新内容写到末尾，如果文件不存在则创建
- "r+": 读写模式 文件必须存在 写入是从头一个一个覆盖
- "w+": 读写模式 可读取,写入同样会清空文件内容，不存在则创建新文件
- "a+": 读写追加模式 可读取,写入从文件末尾开始，如果文件不存在则创建

2. `fclose`：关闭文件。
```c
int fclose (FILE *__stream)
```
成功返回0否则返回EOF。

3. `fputc`：写入char字符到文件。
```c
int fputc (int __c, FILE *__stream)
```
其中：
- int __c: 写入的char按照AICII值写入 可提前声明一个char
- FILE *__stream: 要写入的文件,写在哪里取决于访问模式
- return: 成功返回char的值 失败返回EOF

4. `fputs`: 写入string字符串到文件。
```c
int fputs (const char *__restrict __s, FILE *__restrict __stream)
```
其中：
- char *__restrict __s: 需要写入的字符串
- FILE *__restrict __stream: 要写入的文件,写在哪里取决于访问模式
- return: 成功返回非负整数(一般是0,1) 失败返回EOF

5. `fgetc`: 从文件中读取一个字符。
6. `fgets`: 从文件中读取一行字符串。

### 系统调用
系统调用是操作系统内核提供给应用程序，使其可以间接访问硬件资源的接口。常见的文件I/O系统调用包括：
1. `open`: 打开文件，返回文件描述符。
2. `close`: 关闭文件描述符。
3. `read`: 从文件描述符中读取数据。
4. `write`: 向文件描述符中写入数据。
```c
#include <unistd.h>
ssize_t write(int fd, const void *buf, size_t count);
```
参数：
- fd (文件描述符)：指定要写入的文件的描述符（通过 `open()` 获得）。常见的有 0 (标准输入), 1 (标准输出), 2 (标准错误)。
- buf (缓冲区指针)：指向要写入的数据的指针，类型为 void * 。
- count (字节数)：要写入的最大字节数。
  
返回值：
- 成功：返回实际写入的字节数。
- 失败：返回 -1，并设置 errno 以指示错误。
  
特点：
- 这是不带缓存的 I/O 操作（相对于标准 C 库的 fwrite）。
- write 可能不会一次性写入所有 count 字节，需要根据返回值判断实际写入的量，或者循环调用。
它可以用于普通文件、管道、套接字等多种类型的文件描述符。
1. `lseek`: 移动文件指针位置。
2. `stat`: 获取文件状态信息。
3. `exit`: 终止进程。

## 进程

进程（Process）是正在运行的程序，是操作系统进行资源分配和调度的基本单位。程序是存储在硬盘或内存的一段二进制序列，是静态的，而进程是动态的。进程包括代码、数据以及分配给它的其他系统资源（如文件描述符、网络连接等。

进程与程序的区别：程序占用磁盘空间，而进程占用系统资源。

使用`system()`函数生成子进程，`system()`函数是标准库中执行shell指令的函数，可以使用`man 3 system`命令查看其声明，并通过`ps -ef`命令查看当前系统中的进程。

### 进程处理

#### fork()与vfork()
`fork()`函数用于创建一个新的子进程：
```c
pid_t fork(); //创建一个子进程，相当于复制包括内存空间
```
返回值`pid_t`本质上为int的别名，返回两次：子进程的返回值是0，父进程的返回值的新建子进程的ID，返回值为-1表示创建失败，数字越大，代表进程越晚创建。

相关函数：
- `getpid()`: 必然返回当前进程ID，不会失败
- `getppid()`: 必然返回父进程ID，不会失败

子进程是父进程的副本。子进程和父进程继续执行 *fork* 之后的指令。
  + 子进程获得父进程的 **数据空间、堆、栈的副本**
  + 共享的是：**文件描述符、mmap建立的映射区** 
  + 子进程和父进程共享的是 **代码段**，*fork* 之后各自执行。
  + 父进程和子进程的执行顺序谁先谁后是未知的，是竞争的关系。

*COW* 即写时复制(*Copy-On-Write*)， **数据空间、堆、栈的副本**在创建子进程时并不创建副本。而是在父进程或者子进程修改这片区域时，内核为修改区域的那块内存制作一个副本，以提高效率。

fork失败的原因：
1. 系统资源不足，无法创建新的进程。
2. 该实际用户ID的进程数超过了系统限制。

示例：
```c
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int globvar = 10;

int main(int argv, char* argc[]){

    int var;
    pid_t pid;

    var = 88;
    printf("before fork.\n");
    // 创建子进程后，后面的代码，父进程和子进程独立运行。
    if((pid=fork()) < 0){
        printf("fork() error.\n");
        exit(1);
    }
    else if(pid == 0){
        globvar++; //子进程运行不改变父进程的值
        var++;
    }
    else
        sleep(2);
    printf("ppid=%ld, pid=%ld, globvar=%d, var=%d.\n",(long)getppid(), (long)getpid(), globvar, var);
    return 0;
}
```
输出：
```bash
$ g++ -g fork.cpp -o fork
$ ./fork
before fork.
ppid=12832, pid=12833, globvar=11, var=89.
ppid=12430, pid=12832, globvar=10, var=88.
```
从中可以看到父进程和子进程的`globvar`和`var`值不同，说明它们拥有独立的内存空间。

`vfork()`函数也是用于创建子进程的，目的都是执行一个程序，但它与`fork()`不同，**它并不将父进程的地址空间复制到子进程中**。在子进程 `exec/exit` 之前，和父进程共享地址空间，提高了工作效率。但是在在子进程 `exec/exit`之前，子进程如果修改了数据、进行函数调用、返回都会带来未知的结果。因此，*vfork* 保证子进程比父进程先运行，在子进程 `exec/exit `之后父进程才会运行。

将上面例子中的`fork()`替换为`vfork()`，输出如下：
```bash
before fork.
ppid=13248, pid=13249, globvar=11, var=89.
ppid=12430, pid=13248, globvar=11, var=-1550865008.
```
可以看到，父进程的`globvar`值被子进程修改了，说明父子进程共享地址空间；且var出现了不可预知的值，说明子进程修改了父进程的栈空间。

#### execve()
`execve()` 是 Unix/Linux 系统中用于执行新程序的底层系统调用函数，属于 exec 函数族。它通过用新程序映像替换当前进程的数据段、代码段、堆和栈，使当前进程启动新程序，PID 保持不变。成功时不返回，失败则返回 -1 并设置 errno。 

函数原型
```c
#include <unistd.h>
int execve(const char *filename, char *const argv[], char *const envp[]);
```
参数说明:
`filename`: 要执行的可执行文件路径（例如 "/bin/ls"）。
`argv`: 传递给新程序的命令行参数数组指针。该数组必须以 NULL 结束（例如 {"ls", "-al", NULL}）。
`envp`: 传递给新程序的新环境变量数组指针。同样以 NULL 结束。 

特点：
- 进程替换：execve 替换当前进程。调用执行成功后，原程序的代码和数据全部被新程序覆盖，控制权不会返回给调用进程。
- PID 不变：新程序运行在原有的进程 ID 中。
- 返回值：只有在发生错误（如文件未找到、权限不足）时才会返回 -1。
- 适用场景：**常用于父进程在 fork() 子进程后，在子进程中执行不同的程序**。 

示例：
```c
#include <unistd.h>
#include <stdio.h>

int main() {
    char *argv[] = {"ls", "-l", "/", NULL};
    char *envp[] = {"PATH=/bin", NULL};
    
    // 用 /bin/ls 替换当前进程
    execve("/bin/ls", argv, envp);
    
    // 如果 execve 成功，下面的代码永远不会执行
    perror("execve");
    return -1;
}
```

常见错误 (errno)
- EACCES: 文件无执行权限或文件系统不允许。
- ENOENT: 文件路径不存在。
- ENOMEM: 内存不足。

#### waitpid()
Linux中父进程除了可以启动子进程，还要负责回收子进程的状态。如果子进程结束后父进程没有正常回收，那么子进程就会变成一个僵尸进程——即程序执行完成，但是进程没有完全结束，其内核中PCB结构体（下文介绍）没有释放。当父进程在子进程结束前就结束了，那么其子进程的回收工作就交给了父进程的父进程的父进程（省略若干父进程）。

相比`wait()`，`waitpid()`提供了更多的选项和控制。`waitpid()`允许父进程等待特定子进程的结束，并且可以选择是否阻塞等待。
```c
#include <sys/types.h>
#include <sys/wait.h>

pid_t waitpid(pid_t pid, int *status, int options);
```
- pid:
  - 返回 >0: 等待进程号为 pid 的指定子进程。
  - 返回-1: 等待任意一个子进程，等同于 wait()。
  - 返回0: 等待与当前进程在同一进程组的任意子进程。
- status: 指向存放子进程退出状态的整数指针，不需要可设为 NULL。
- options:
  - 0: 默认行为，若子进程未结束，父进程阻塞。
  - WNOHANG: 非阻塞模式，若子进程未结束，立即返回 0。

#### 进程树
Linux的进程是通过父子关系组织起来的，所有进程之间的父子关系共同构成了进程树（Process Tree）。进程树中每个节点都是其上级节点的子进程，同时又是子结点的父进程。一个进程的父进程只能有一个，而一个进程的子进程可以不止一个。

Linux 进程树是系统中所有进程的层次结构，以树状图形式显示进程间的父子关系。根节点通常是 `systemd` 或 `init (PID 1)` 进程。常用 `pstree` 命令查看，结合 `-p` (显示PID)、`-a` (命令行参数) 可快速定位、诊断及管理进程。 

- **父子关系**：每个进程（除PID 1）都有一个父进程（Parent Process, PPID），通过 fork() 创建子进程。
- **PID 1**：系统引导启动的第一个进程，是所有进程的祖先。现代 Linux 多为 systemd。
- **孤儿进程**：父进程死掉后，子进程被 init 接管。
- **僵尸进程**：子进程已死，但父进程未通过 wait() 回收其资源。
  
查看命令:
- pstree：最直观显示树状结构。
- pstree -p: 显示 PID (推荐)。
- pstree -a: 显示完整的命令行参数。
- pstree -u: 显示用户名。
- ps -ef --forest: 用 ASCII 字符表示树状结构。
- ps auxf: 同样是常用的树状查看方式。
- top/htop：动态查看进程。htop 中按 F5 可按树状排序。 

应用场景:
- 查找源头：确定异常进程（如高 CPU）是由哪个应用启动的。
- 批量终止：通过父进程一次性清理其产生的所有子进程。
- 分析系统启动过程：查看服务依赖树。
示例：
```bash 
$ pstree -p
systemd(1)─┬─sshd(1024)───sshd(1025)───bash(1026)───pstree(2000)
           ├─nginx(1050)─┬─nginx(1051)
           │             └─nginx(1052)
           └─{systemd}(2)
```
在该示例中，sshd 服务 fork 了子进程，nginx 是由 systemd 管理的主进程及其子进程。

### 进程间通信
进程间通信（Inter-Process Communication, IPC）是指不同进程之间交换数据和信息的机制。常见的 IPC 方式包括：
1. 管道（Pipe）：用于父子进程之间的通信，数据单向流动。
2. 命名管道（Named Pipe）：类似于管道，但可以在无亲子关系的进程之间通信，也叫 FIFO（先进先出）。
3. 消息队列（Message Queue）：通过消息队列进行进程间通信，允许进程以消息的形式发送和接收数据。
4. 共享内存（Shared Memory）：多个进程共享一块内存区域，读写效率高，但需要同步机制。
5. 信号量（Semaphore）：用于控制对共享资源的访问，常与共享内存结合使用。

#### System V IPC 和 POSIX IPC
System V IPC 和 POSIX IPC 是两种不同的进程间通信机制。

System V IPC（Inter-Process Communication，进程间通信）是System V操作系统引入的一组进程间通信机制，包括消息队列、信号量和共享内存。这些机制允许不同的进程以一种安全且高效的方式共享数据和同步操作。
- （1）消息队列：允许进程以消息的形式交换数据，这些消息存储在队列中，直到它们被接收。
- （2）信号量：主要用于进程间的同步，防止多个进程同时访问相同的资源。
- （3）共享内存：允许多个进程访问同一块内存区域，提供了一种非常高效的数据共享方式。
System V IPC是UNIX和类UNIX系统中常用的IPC方法之一，它通过关键字（key）来标识和访问IPC资源。

POSIX IPC是POSIX标准中的一部分，提供了一种更现代和标准化的进程间通信方式，同样包括消息队列、信号量和共享内存三种方式。
- （1）消息队列：类似于System V，但通常具有更简洁的API和更好的错误处理能力。
- （2）信号量：提供了更多的功能和更高的性能，支持更大范围的操作。
- （3）共享内存：提供了更多的控制和配置选项，以支持更复杂的应用场景。
POSIX IPC 使用名字（name）作为唯一标识。这些名字通常是以正斜杠（/）开头的字符串，用于唯一地识别资源如消息队列、信号量或共享内存对象。


#### 匿名管道
匿名管道，也称管道，是Linux下最常见的进程间通信方式之一。匿名管道在系统中没有实名，它只是进程的一种资源，会随着进程的结束而被系统清除。

**匿名管道**，也称**管道**，是Linux下最常见的进程间通信方式之一。匿名管道在系统中没有实名，它只是进程的一种资源，会随着进程的结束而被系统清除。

##### 管道的创建与关闭

Linux中使用**pipe()函数**创建一个匿名管道，其函数原型为：
```c
#include <unistd.h>
int pipe(int fd[2]);
```
创建成功返回0，出错返回1。参数`fd[2]`是一个长度为2的文件描述符数组，**fd[1]**是**写入端**的文件描述符，**fd[0]**是**读出端**的文件描述符。

可以使用文件I/O函数`read()`和`write()`读管道进行读写，使用`close()`函数关闭管道两端。

示例:

```c
#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>

int main(void)
{
    int fd[2];
    char str[256];
    if((pipe(fd))<0)
    {
        printf("create the pipe failed!\n");
        exit(1);
    }
    write(fd[1], "create the pipe sucessfully!\n", 31);
    read(fd[0], str, sizeof(str));
    printf("%s", str);
    printf("pipe file descriptors are%d,%d\n", fd[0], fd[1]);
    close(fd[0]);
    close(fd[1]);

    return 0;
}
```

编译后执行：

```bash
$ ./create_pipe
create the pipe sucessfully!
pipe file descriptors are3,4
```

单独对一个进程进行管道的读写是没有实际意义的，管道的应用体现在**父子进程**或**兄弟进程**之间的通信。
##### 父子进程间管道的读写
父进程利用管道向子进程发送消息，使用pipe函数建立管道，使用`fork()`函数创建子进程，在父进程中维护管道的数据方向，并在父进程中向子进程发送消息，示例:
```c
#include<unistd.h>
#include<stdio.h>
#include<sys/types.h>
#include<limits.h>
#include<stdlib.h>
#include<string.h>
#define BUFSIZE PIPE_BUF

void err_quit(char *msg)
{
    printf(msg);
    exit(1);
}

int main(void)
{
    int fd[2];
    char buf[BUFSIZE] = "hello my child!\n";
    pid_t pid;
    int len;

    // create a pipe
    if((pipe(fd)) < 0)
        err_quit("pipe failed\n");

    // create a son process
    pid = fork();
    if(pid<0)
        err_quit("fork failed\n");
    // parent process do this
    else if(pid>0)
    {
        close(fd[0]);// father process close read port (fd[0])
        write(fd[1], buf, strlen(buf));// write to pipe
        printf("I am parent process(ID:%d),write pipe ok\n", getpid());
        exit(0);
    }
    // son process do this
    else //(pid == 0)
    {
        close(fd[1]);// son process close itself write port (fd[1])
        len = read(fd[0], buf, BUFSIZE);// read from pipe
        if(len<0)
            err_quit("process failed when read a pipe\n");
        else
        {
            printf("I am son process(ID:%d),read pipe ok\n", getpid());
            write(STDOUT_FILENO, buf, len);
        }
        exit(0);
    }

}
```

编译后执行：

```bash
$ ./parent_pipe_child
I am parent process(ID:4953),write pipe ok
I am son process(ID:4954),read pipe ok
hello my child!
```

上述程序使用pipe加fork组合，实现父进程到子进程的通信，程序在**父进程段中关闭了管道的读出端**，并相应地在子进程中关闭了管道的输入端，从而实现**数据从父进程流向子进程**。

##### 兄弟进程间管道的读写
```c
#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<limits.h>
#include<string.h>
#define BUFSIZE PIPE_BUF

void err_quit(char *msg)
{
    printf(msg);
    exit(1);
}

int main(void)
{
    int fd[2];
    char buf[BUFSIZE] = "hello my brother!\n";
    pid_t pid;
    int len;

    // create pipe
    if((pipe(fd))<0)
        err_quit("pipe failed\n");

    // create the son process1
    pid = fork();
    if(pid < 0)
        err_quit("fork failed\n");
    else if(pid == 0)// son process1
    {
        close(fd[0]);// close read port
        write(fd[1], buf, strlen(buf));// write
        printf("I am son process1(ID:%d), write pipe ok\n", getpid());
        exit(0);
    }

    // create the son process2
    pid = fork();
    if(pid < 0)
        err_quit("fork failed\n");
    else if(pid > 0)
    {
        close(fd[0]);// close read port
        close(fd[1]);// close write port
        printf("I am parent process(ID:%d), close my pipe\n", getpid());
        exit(0);
    }
    else
    {
        close(fd[1]);// close write port
        len = read(fd[0], buf, BUFSIZE);// read
        printf("I am son process2(ID:%d), read pipe ok\n", getpid());
        write(STDOUT_FILENO, buf, len);
        exit(0);
    }

    return 0;
}
```

编译后执行：

```bash
$ ./brother_pipe
I am parent process(ID:4962), close my pipe
I am son process1(ID:4963), write pipe ok
I am son process2(ID:4964), read pipe ok
hello my brother!
```

上述程序中父进程分别建立了两个子进程，在**子进程1中关闭了管道的读出端**，在**子进程2中关闭了管道的输入端**，并在**父进程中关闭了管道的两端**，从而构成了**从子进程1到子进程2的管道**。另外，程序中父进程创建第1个子进程时并没有关闭管道两端，而是在创建第2个子进程时才关闭管道，这是为了在创建第2个进程时，子进程可以继承存活的管道。

#### 有名管道
Linux有名管道（FIFO）是一种特殊的文件类型，它在文件系统中作为一个实体存在，允许无亲缘关系的进程间进行双向通信。它克服了匿名管道只能在父子进程间通信的限制,**允许任何进程之间的通信**。使用` mkfifo` 命令或 `mkfifo()` 函数创建，遵循FIFO（先进先出）原则，读写方式与普通文件类似。 

FIFO和Pipe一样，提供了双向进程间通信渠道。但要注意的是，无论是有名管道还是匿名管道，同一条管道只应用于单向通信，否则可能出现通信混乱（进程读到自己发的数据）。

核心特点与用法：
- 文件系统体现： 管道以文件形式显示（类型为 p），但其数据存储在内核缓冲区，不在磁盘中。
- 通信方式： 一个进程打开进行写操作（open(fifo, O_WRONLY)），另一个进程打开进行读操作（open(fifo, O_RDONLY)）。
- 阻塞特性： 如果只有一端打开，读/写操作通常会阻塞，直到另一端也被打开。 

```c
int mkfifo(const char *pathname, mode_t mode);
```

示例操作：
```bash
# 终端1：创建并读取管道
mkfifo test_fifo
cat < test_fifo

# 终端2：向管道写入数据
echo "Hello from FIFO" > test_fifo
```

#### 共享内存
Linux共享内存是一种极高效的IPC机制，它通过映射同一块物理内存到不同进程的虚拟地址空间，实现无需数据拷贝的**零拷贝**通信。相比管道或消息队列，它速度最快，适用于大数据量、高频的进程间数据交互，但需要使用信号量等同步机制来防止数据覆盖。 

操作系统将物理内存映射到进程A和进程B的虚拟地址空间，A写入数据，B可直接读取，无需内核中转。通常与信号量（Semaphore）配合使用，因为共享内存本身不提供访问控制。
##### 开启和关闭共享内存
`shm_open()`:
```c
int shm_open(const char *name, int oflag, mode_t mode);
```
其中:
`const char *name`: 这是共享内存对象的名称，直接写一个文件名称，本身会保存在 /dev/shm 。名称必须是唯一的，以便不同进程可以定位同一个共享内存段。命名规则必须是以正斜杠/开头，以\0结尾的字符串，中间可以包含若干字符，但不能有正斜杠。

`mode_t mode`：当创建新共享内存对象时使用的权限位，类似于文件的权限模式,一般0644即可。

`shm_unlink()`:
```c
int shm_unlink(const char *name);
```
其中:
`const char *name`: 这是共享内存对象的名称，必须与 `shm_open()` 中使用的名称一致。
##### 文件的缩放
`truncate()`和`ftruncate()`都可以将文件缩放到指定大小，二者的行为类似：如果文件被缩小，截断部分的数据丢失，如果文件空间被放大，扩展的部分均为\0字符。缩放前后文件的偏移量不会更改。缩放成功返回0，失败返回-1。

不同的是，前者需要指定路径，而后者需要提供文件描述符；`ftruncate`缩放的文件描述符可以是通过`shm_open()`开启的内存对象，而`truncate`缩放的文件必须是文件系统已存在文件，若文件不存在或没有权限则会失败。
```c
int truncate(const char *path, off_t length);
int ftruncate(int fd, off_t length);
```

##### 内存映射
`mmap()`函数用于将文件或设备映射到内存中，返回一个指向映射区域的指针。对于共享内存对象，可以使用`mmap()`将其映射到进程的地址空间，从而实现进程间的共享数据访问。
```c
void *mmap(void *addr, size_t length, int prot, int flags,
           int fd, off_t offset);
```
将文件映射到内存区域,进程可以直接对内存区域进行读写操作,就像操作普通内存一样,但实际上是对文件或设备进行读写,从而实现高效的 I/O 操作。
`void *addr`: 指向期望映射的内存起始地址的指针,通常设为 NULL,让系统选择合适的地址
`size_t length`: 要映射的内存区域的长度,以字节为单位
`int prot`: 内存映射区域的保护标志,可以是以下标志的组合
`int flags`：映射选项标志
`int fd`: 文件描述符,用于指定要映射的文件或设备,如果是匿名映射,则传入无效的文件描述符（例如-1）
`off_t offset`: 从文件开头的偏移量,映射开始的位置
`return void *`: 成功时,返回映射区域的起始地址,可以像操作普通内存那样使用这个地址进行读写， 如果出错,返回 (void *) -1,并且设置 errno 变量来表示错误原因

示例：
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <string.h>

int main()
{
    char *share;
    pid_t pid;
    char shmName[100]={0};
    sprintf(shmName,"/letter%d",getpid());
    // 共享内存对象的文件标识符
    int fd;
    fd = shm_open(shmName, O_CREAT | O_RDWR, 0644);
    if (fd < 0)
    {
        perror("共享内存对象开启失败!\n");
        exit(EXIT_FAILURE);
    }
    // 将该区域扩充为100字节长度
    ftruncate(fd, 100);
    // 以读写方式映射该区域到内存，并开启父子共享标签 偏移量选择0从头开始
    share = mmap(NULL, 100, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    // 注意:不是p == NULL 映射失败返回的是((void *) -1)
    if (share == MAP_FAILED)
    { 
        perror("共享内存对象映射到内存失败!\n");
        exit(EXIT_FAILURE);
    }
    // 映射区建立完毕,关闭读取连接 注意不是删除
    close(fd);
    // 创建子进程
    pid = fork(); 
    if (pid == 0)
    {
        // 子进程写入数据作为回信 
        strcpy(share, "你是个好人!\n");
        printf("新学员%d完成回信!\n", getpid());
    }
    else
    {
        // 等待回信
        sleep(1);
        printf("老学员%d看到新学员%d回信的内容: %s", getpid(),pid,share);
        // 等到子进程运行结束
        wait(NULL);
        // 释放映射区
        int ret = munmap(share, 100); 
        if (ret == -1)
        {
            perror("munmap");
            exit(EXIT_FAILURE);
        }
    }
    // 删除共享内存对象
    shm_unlink(shmName);
    return 0;
}
```

#### 消息队列
消息队列是消息的链接表，存放在内核中并由消息队列标识符标识。我们将称消息队列为“队列”，其标识符为“队列ID”。

##### 数据类型
`mqd_t`：该数据类型定义在`mqueue.h`中，是用来记录消息队列描述符的。
```c
typedef int mqd_t;
```
实质上是int类型的别名。
`struct mq_attr`：记录消息队列的属性信息。
```c
struct mq_attr {
    long mq_flags;   //对于mq_open，忽略它，因为这个标记是通过前者的调用传递的
    long mq_maxmsg;  //队列可以容纳的消息的最大数量
    long mq_msgsize; //单条消息的最大允许大小，以字节为单位
    long mq_curmsgs; //当前队列中的消息数量，对于mq_open，忽略它
};
```
`struct timespec`：记录时间信息，提供了纳秒级的UNIX时间戳。
```c
struct timespec {
    time_t tv_sec;        /* seconds */
    long   tv_nsec;       /* nanoseconds */
};
```
##### 相关系统调用
`mq_open()`：创建或打开一个已存在的POSIX消息队列，消息队列是通过名称唯一标识的。
```c
#include <fcntl.h>    /* For O_* constants */
#include <sys/stat.h> /* For mode constants */
#include <mqueue.h>
mqd_t mq_open(const char *name, int oflag, mode_t mode, struct mq_attr *attr);
// name:命名规则必须是以正斜杠/开头，以\0结尾的字符串，中间可以包含若干字符，但不能有正斜杠
```
`mq_timedsend()`：将msg_ptr指向的消息追加到消息队列描述符mqdes指向的消息队列的尾部。如果消息队列已满，默认情况下，调用阻塞直至有充足的空间允许新的消息入队，或者达到abs_timeout指定的等待时间节点，或者调用被信号处理函数打断。需要注意的是，正如上文提到的，如果在`mq_open`时指定了O_NONBLOCK标记，则转而失败，并返回错误EAGAIN。
```c
#include <time.h>
#include <mqueue.h>

int mq_timedsend(mqd_t mqdes, const char *msg_ptr, size_t msg_len, unsigned int msg_prio, const struct timespec *abs_timeout);
```

`mq_timedreceive()`：从消息队列中取走最早入队且权限最高的消息，将其放入msg_ptr指向的缓存中。如果消息队列为空，默认情况下调用阻塞，此时的行为与mq_timedsend同理。
```c
#include <time.h>
#include <mqueue.h>

ssize_t mq_timedreceive(mqd_t mqdes, char *msg_ptr, size_t msg_len, unsigned int *msg_prio, const struct timespec *abs_timeout);
```
`mq_unlink()`：清除name对应的消息队列，mqueue文件系统中的对应文件被立即清除。消息队列本身的清除必须等待所有指向该消息队列的描述符全部关闭之后才会发生。
```c
#include <mqueue.h>
int mq_unlink(const char *name);
```
`clock_gettime()`：获取以struct timespec形式表示的clockid指定的时钟
```c
#include <time.h>

int clock_gettime(clockid_t clockid, struct timespec *tp);
```
示例：
```c
#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    // 创建消息队列
    struct mq_attr attr;
    // 有用的参数  表示消息队列的容量
    attr.mq_maxmsg = 10;
    attr.mq_msgsize = 100;
    // 被忽略的消息 在创建消息队列的时候用不到
    attr.mq_flags = 0;
    attr.mq_curmsgs = 0;
    
    char *mq_name = "/father_son_mq";
    mqd_t mqdes =  mq_open(mq_name,O_RDWR | O_CREAT,0664,&attr);

    if (mqdes == (mqd_t)-1)
    {
        perror("mq_open");
        exit(EXIT_FAILURE);
    }
    // 创建父子进程
    pid_t pid = fork();
    
    if (pid < 0)
    {
        perror("fork");
        exit(EXIT_FAILURE);
    }
    if (pid == 0)
    {
        // 子进程  等待接收消息队列中的信息
        char read_buf[100];
        struct timespec time_info;
        for (size_t i = 0; i < 10; i++)
        {
            // 清空接收数据的缓冲区
            memset(read_buf,0,100);
            // 设置接收数据的等待时间
            clock_gettime(0,&time_info);
            time_info.tv_sec += 15;
            // 接收消息队列的数据  打印到控制台
            if (mq_timedreceive(mqdes,read_buf,100,NULL,&time_info) == -1)
            {
                perror("mq_timedreceive");
            }
            printf("子进程接收到数据:%s\n",read_buf);
        }
        
    }else {
        // 父进程  发送消息到消息队列中
        char send_buf[100];
        struct timespec time_info;
        
        for (size_t i = 0; i < 10; i++)
        {
            // 清空处理buf
            memset(send_buf,0,100);
            sprintf(send_buf,"父进程的第%d次发送消息\n",(int)(i+1));
            // 获取当前的具体时间
            clock_gettime(0,&time_info);
            time_info.tv_sec += 5;
            // 发送消息
            if (mq_timedsend(mqdes,send_buf,strlen(send_buf),0,&time_info) == -1)
            {
                perror("mq_timedsend");
            }
            printf("父进程发送一条消息,休眠1s\n");
            sleep(1);
        }
        
       
    }
    // 最终不管是父进程还是子进程都需要释放消息队列的引用
    close(mqdes);
    // 清除消息队列只需要执行一次
    if (pid > 0)
    {
        mq_unlink(mq_name);
    }
    return 0;
}
```

## 线程

Linux 中的线程是指轻量级的执行单元，相比于进程，具有以下特点：
- 进程（Process）是正在执行的程序的实例。每个进程都有自己的地址空间、代码段、数据段和打开的文件描述符等资源。线程（Thread）是进程内的一个执行单元，它共享相同的地址空间和其他资源，包括文件描述符、信号处理等，但每个线程都有自己的栈空间。
- 由于共享地址空间和数据段，同一进程的多线程之间进行数据交换比进程间通信方便很多，但也由此带来线程同步问题。
- 同一进程的多线程共享大部分资源，除了每个线程独立的栈空间。这代表线程的创建、销毁、切换要比进程的创建、销毁、切换的资源消耗小很多，所以多线程比多进程更适合高并发。

### 线程控制

## Linux网络编程

## 常见Linux命令

### 文档编辑

#### vi、Vim 与 nano 概览

Linux 中常见的终端编辑器可以按交互模型分为两类：

| 编辑器 | 交互模型 | 适合场景 |
|--------|----------|----------|
| `vi` | 模态编辑器 | 最小系统、救援环境和 POSIX 基础操作 |
| `vim` | vi 的增强实现 | 代码、配置文件和长期编辑 |
| `nano` | 非模态编辑器 | 快速修改小文件、初次使用终端编辑 |

启动命令：

```bash
vi file.txt
vim file.txt
nano file.txt
```

编辑系统配置前可以先备份：

```bash
cp config.conf config.conf.bak
```

#### vi/Vim 的模式

Vim 的核心是“模式”：同一个按键在不同模式下含义不同。

| 模式 | 进入方式 | 作用 |
|------|----------|------|
| Normal | 启动 Vim 或按 `Esc` | 移动、删除、复制、粘贴、撤销、执行命令 |
| Insert | `i`、`a`、`o` 等 | 输入文本 |
| Visual | `v`、`V`、`Ctrl-v` | 选择字符、行或矩形区域 |
| Command-line | Normal 模式输入 `:`、`/`、`?` | 保存、退出、搜索、替换和配置 |
| Replace | `R` 或 `r` | 覆盖已有字符 |

不确定当前状态时，直接按一次或两次 `Esc` 回到 Normal 模式：

```text
任何模式 ── Esc ──> Normal 模式
```

进入 Insert 模式：

| 命令 | 作用 |
|------|------|
| `i` | 在光标前插入 |
| `a` | 在光标后追加 |
| `I` | 在行首插入 |
| `A` | 在行尾追加 |
| `o` | 在当前行下方新建一行并插入 |
| `O` | 在当前行上方新建一行并插入 |
| `s` | 删除当前字符并插入 |
| `C` | 删除光标到行尾并插入 |

离开 Insert 模式：

```text
Esc
```

#### Vim 的移动

##### 基础移动

| 按键 | 作用 |
|------|------|
| `h` / `l` | 左 / 右 |
| `j` / `k` | 下 / 上 |
| `0` | 行首 |
| `^` | 行首第一个非空白字符 |
| `$` | 行尾 |
| `gg` | 文件第一行 |
| `G` | 文件最后一行 |
| `nG` | 跳转到第 `n` 行 |
| `Ctrl-f` / `Ctrl-b` | 向下 / 向上翻页 |
| `Ctrl-d` / `Ctrl-u` | 向下 / 向上半页 |
| `H` / `M` / `L` | 屏幕顶部 / 中部 / 底部 |

##### 按单词移动

| 按键 | 作用 |
|------|------|
| `w` | 下一个单词开头 |
| `e` | 当前或下一个单词结尾 |
| `b` | 上一个单词开头 |
| `W` / `E` / `B` | 按空白分隔的大单词移动 |
| `f{char}` | 向右查找当前行中的字符 |
| `t{char}` | 向右移动到字符前 |
| `;` / `,` | 重复 / 反向重复 `f`、`t` |

数量可以与移动命令组合：

```text
5j       向下移动 5 行
3w       向前移动 3 个单词
10G      跳转到第 10 行
```

#### Vim 的操作符与动作

Vim 的编辑命令经常遵循：

```text
操作符 + Motion
```

常用操作符：

| 操作符 | 作用 |
|--------|------|
| `d` | 删除 |
| `c` | 删除并进入 Insert |
| `y` | 复制到寄存器 |
| `>` / `<` | 增加 / 减少缩进 |
| `g~` | 切换大小写 |

示例：

```text
dd       删除当前行
dw       删除到下一个单词开头
d$       删除到行尾
dG       删除到文件末尾
cc       重写当前行
yy       复制当前行
p        在光标后粘贴
P        在光标前粘贴
>>       增加当前行缩进
```

Vim 中 `d`、`c`、`y` 可以和很多 Motion 组合：

```text
diw      删除当前单词
ci"      修改一对双引号中的内容
ya{      复制包含花括号的文本
```

#### Vim 的撤销、重做和重复

```text
u        撤销
Ctrl-r   重做
.        重复上一次修改
U        恢复当前行的修改
```

`u`、`Ctrl-r` 和 `.` 应在 Normal 模式使用。

#### Visual 模式

| 命令 | 选择范围 |
|------|----------|
| `v` | 字符选择 |
| `V` | 行选择 |
| `Ctrl-v` | 矩形块选择 |
| `o` | 在选择区域两端切换光标 |
| `d` | 删除选择内容 |
| `y` | 复制选择内容 |
| `>` / `<` | 缩进选择内容 |
| `:` | 对选择区域执行 Ex 命令 |

批量缩进示例：

```text
Vjj>     选择当前行及下面两行并增加缩进
Vjj=     选择三行并自动格式化缩进
```

块插入示例：

```text
Ctrl-v  选择多行的同一列
I       输入前缀
Esc     把前缀应用到所有选中行
```

#### Vim 的搜索

```text
/pattern        向后搜索
?pattern        向前搜索
n               重复上一次搜索方向
N               反向重复搜索
*               搜索光标所在单词的下一个匹配
#               搜索光标所在单词的上一个匹配
```

命令行中可以使用正则表达式：

```text
/^server        搜索以 server 开头的行
/error$         搜索以 error 结尾的行
/foo.*bar       搜索 foo 和 bar 之间有任意内容的行
```

#### Vim 的替换

基本语法：

```vim
:s/old/new/          替换当前行第一个匹配
:s/old/new/g         替换当前行全部匹配
:%s/old/new/g        替换整个文件全部匹配
:%s/old/new/gc       替换前逐项确认
:5,10s/old/new/g     替换第 5 到 10 行
```

常用标志：

| 标志 | 作用 |
|------|------|
| `g` | 当前行全部匹配 |
| `c` | 每次替换前确认 |
| `i` | 忽略大小写 |
| `I` | 区分大小写 |

替换路径中的 `/` 可以换成其他分隔符：

```vim
:%s#/old/path#/new/path#g
```

#### Vim 的文件操作

| 命令 | 作用 |
|------|------|
| `:w` | 保存 |
| `:w file.txt` | 另存为 |
| `:q` | 退出 |
| `:wq` | 保存并退出 |
| `:x` | 有修改时保存并退出 |
| `ZZ` | Normal 模式下保存并退出 |
| `:q!` | 放弃修改并退出 |
| `:wq!` | 强制保存并退出 |
| `:e file.txt` | 打开另一个文件 |
| `:r file.txt` | 把另一个文件内容读入当前光标位置 |
| `:set readonly` | 设置只读 |

修改权限受限的文件时，`sudo vim file` 通常比普通用户打开后再处理更直接；不要把 `:w!` 当作权限绕过方式。

#### Vim 的分屏与标签页

```vim
:split file.txt       水平分屏
:vsplit file.txt      垂直分屏
:new                  新建水平窗口
:vnew                 新建垂直窗口
:tabnew file.txt      新建标签页
```

窗口移动：

```text
Ctrl-w h/j/k/l       移动到左/下/上/右窗口
Ctrl-w w             循环切换窗口
Ctrl-w q             关闭当前窗口
Ctrl-w =             平均分配窗口大小
Ctrl-w + / -         增大 / 减小高度
gt / gT              下一个 / 上一个标签页
```

#### Vim 的配置

用户配置通常位于：

```text
~/.vimrc
~/.config/nvim/init.vim
```

一个小型 `.vimrc`：

```vim
set number
set relativenumber
set expandtab
set tabstop=4
set shiftwidth=4
set autoindent
set smartindent
set ignorecase
set smartcase
set hlsearch
set incsearch
syntax on
```

临时设置：

```vim
:set number
:set nonumber
:set paste
:set nopaste
:set list
:set fileencoding?
```

查看帮助：

```vim
:help
:help motion
:help :substitute
:help E325
```

#### Vim 的恢复与 Swap 文件

Vim 编辑文件时可能创建 Swap 文件，用于崩溃恢复和并发编辑提示。出现交换文件提示时，先确认是否有另一个 Vim 进程正在编辑同一文件：

```bash
ps aux | grep '[v]im'
```

恢复文件：

```bash
vim -r file.txt
```

确认内容已经恢复并保存后，再删除不再需要的 `.swp` 文件。不要在未知情况下直接删除，因为它可能包含尚未保存的修改。

#### nano 的基本操作

`nano` 是非模态编辑器，启动后直接输入字符即可编辑，不需要在 Normal/Insert 模式之间切换。

```bash
nano file.txt
nano +20 file.txt
nano -l file.txt
```

`nano` 底部通常会显示快捷键。`^` 表示 Ctrl，`M-` 表示 Meta/Alt：

| 快捷键 | 作用 |
|--------|------|
| `Ctrl-g` | 打开帮助 |
| `Ctrl-o` | 写入/保存文件 |
| `Ctrl-x` | 退出 |
| `Ctrl-w` | 搜索 |
| `Ctrl-\\` | 搜索并替换 |
| `Ctrl-k` | 剪切当前行或选中内容 |
| `Ctrl-u` | 粘贴剪切缓冲区 |
| `Ctrl-^` | 设置或取消标记起点 |
| `Alt-6` | 复制当前行或选中内容 |
| `Ctrl-_` | 跳转到行号和列号 |
| `Ctrl-c` | 显示当前行列位置 |
| `Ctrl-j` | 对段落重新对齐 |

保存退出流程：

```text
Ctrl-o  输入或确认文件名
Enter   确认写入
Ctrl-x  退出
```

如果文件有未保存修改，`Ctrl-x` 会询问是否保存：

```text
Y       保存
N       放弃修改
Ctrl-c  取消退出
```

#### vi、Vim、nano 的选择

| 场景 | 推荐 |
|------|------|
| 最小化系统或救援 shell | `vi` |
| 需要批量修改、搜索替换和分屏 | `vim` |
| 只改几行配置，不需要记忆模态命令 | `nano` |
| 远程服务器中长期编辑代码 | `vim` 或通过 SSH 使用本地编辑器 |

练习 Vim 的最小路径：

```text
启动 → Esc 回 Normal → i 输入 → Esc
→ :w 保存 → /搜索 → dd 删除 → u 撤销 → :q 退出
```

##### sed

sed 是一种流编辑器，用于对文本进行过滤和转换。它可以执行各种文本操作，如替换、删除、插入等，常用于批处理文本文件。

### 文件处理类

#### 压缩与解压

##### dpkg
dpkg 是 Debian 系统中用于安装、构建、删除和管理 .deb 包的工具。它是 Debian 包管理系统的核心组件，提供了对软件包的底层操作功能。dpkg 主要用于处理本地 .deb 文件，而不涉及依赖关系的自动解决，这通常由更高级的包管理工具（如 apt）来完成。

##### bzip2
bzip2 是一种高效的压缩工具，使用 Burrows-Wheeler

##### tar
tar 是 Linux 中常用的归档工具，主要用于将多个文件和目录打包成一个文件，便于存储和传输。tar 本身不进行压缩，但可以与 gzip 或 bzip2 等压缩工具结合使用，实现归档和压缩的功能。
- 创建归档：`tar -cvf archive.tar /path/to/directory`

##### AppImage类型的解压
AppImage 是一种便携式的 Linux 应用程序格式，允许用户在不同的 Linux 发行版上运行应用程序，而无需安装。AppImage 文件本质上是一个自包含的压缩文件，包含了应用程序及其所有依赖项。要解压 AppImage 文件，可以使用以下命令：

```bash
# 首先，确保 AppImage 文件具有可执行权限
chmod +x your_appimage_file.AppImage
# 然后，使用以下命令解压 AppImage 文件
./your_appimage_file.AppImage --appimage-extract
```
执行上述命令后，AppImage 文件将被解压到当前目录下的 `squashfs-root` 目录中，可以在该目录中找到解压后的文件和资源。
