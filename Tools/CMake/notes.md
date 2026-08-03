## GNU中的GCC和G++
GNU Compiler Collection（GNU编译器套件）是一个由GNU项目开发的编译器集合，支持多种编程语言，其中最常用的是C和C++。GCC（GNU Compiler Collection）是其中的C编译器，而G++（GNU C++ Compiler）是C++编译器。

gcc/g++ 在执行编译工作的时候，总共需要4步：

- 1、预处理,生成 .i 的文件[预处理器cpp]
- 2、将预处理后的文件转换成汇编语言, 生成文件 .s [编译器egcs]
- 3、有汇编变为目标代码(机器代码)生成 .o 的文件[汇编器as]
- 4、连接目标代码, 生成可执行程序 [链接器ld]

### 预处理（Pre-processing）
在C语言编译过程中，预处理是其中的第一个阶段，它的主要目的是处理源代码文件中的预处理指令，将它们转换成编译器可以识别的形式。预处理主要包含宏替换、文件包含、条件编译、注释移除等几种任务：

- 输入：.c 源代码文件。

- 处理内容：编译器处理以 # 开头的指令。
  - 宏展开：将 #define 定义的宏展开。
  - 文件包含：将 #include 包含的头文件内容插入到 .c 文件中。
  - 条件编译：处理 #if、#ifdef、#endif 等，过滤不满足条件的指令。
  - 删除注释：删除所有注释。
  
- 输出：.i 文件（依然是C代码，但已完全展开）。 

预处理的输出通常是经过预处理后的源代码文件，它会被保存成一个临时文件，并作为编译器的输入。预处理器处理后的文件通常会比原始源文件大，因为它会展开宏和包含其他文件的内容。

假设cd到文件夹内有两个.c文件，用下面的命令对两个源文件进行预处理:
```bash
$ gcc -E hello.c -o hello.i
$ gcc -E main.c -o main.i
```
`-E`:Expand(展开)的缩写，该参数指定gcc执行预处理操作。
`-o`:Output(输出)的缩写，该参数指定gcc将预处理后的结果输出到指定文件中。

### 编译（Compiling）
编译阶段是将预处理后的源代码转换成汇编语言的过程。在这个阶段，编译器会进行语法分析、语义分析、优化等操作，最终生成汇编代码:
- 输入：.i 预处理文件。
- 处理内容：将C语言代码翻译成特定的汇编语言代码。
  - 进行词法分析、语法分析、语义分析。
  - 进行代码优化。
- 输出：.s 汇编文件。 

编译阶段是整个编译过程中最复杂和耗时的阶段之一，它对源代码进行了深入的分析和转换，确保了程序的正确性和性能。

使用下面的命令对刚刚生成的预处理文件进行编译:
```bash
$ gcc -S hello.i -o hello.s
$ gcc -S main.i -o main.s
```
`-S`:Source(源代码)的缩写，该参数指定gcc将预处理后的源码编译为汇编语言。

`.s`:Assembly Source(汇编源码)的缩写，通常编译后的汇编文件以.s作为后缀。

### 汇编（Assembling）
汇编阶段是C语言编译过程中的重要阶段，它将编译器生成的中间代码或汇编代码转换成目标机器的机器语言代码，也就是目标代码。这个阶段由汇编器(Assembler)完成，其主要任务是将汇编指令翻译成目标机器的二进制形式。主要包含以下几个任务:符号解析、指令翻译、地址关联、重定位、代码优化。最终，汇编器会将翻译和处理后的目标代码输出到目标文件中，用于后续的链接和生成可执行程序或共享库文件：

- 输入：.s 汇编文件。
- 处理内容：将汇编指令翻译成机器语言（二进制指令）。
- 输出：.o 或 .obj 目标文件（机器码，但不可直接运行）。

执行下面的指令对刚刚生成的汇编文件进行汇编:
```bash
$ gcc -c main.s -o main.o
$ gcc -c hello.s -o hello.o
```

### 链接（Linking）
链接阶段，由链接器完成。链接器将各个目标文件以及可能用到的库文件进行链接，生成最终的可执行程序。在这个阶段，链接器会解析目标文件中的符号引用，并将它们与符号定义进行匹配，以解决符号的地址关联问题。链接器还会处理全局变量的定义和声明，解决重定位问题，最终生成可执行文件或共享库文件：
- 输入：.o 目标文件和库文件。
- 处理内容：将多个目标文件以及所调用的库文件（如标准库）链接在一起，解决符号引用问题，生成最终的可执行文件。
- 输出：可执行文件（如 Windows 的 .exe，Linux 的无后缀文件）。 
#### 链接方式
假设定义了一个函数：
```c
void say_hello(){
    printf("Hello World!\n");
}
```
其中调用了`printf()`函数，这个函数是在\<stdio.h>中声明的，后者来源于glibc库，`printf()`的实现在glibc的二进制组件中，通常是在共享库(如libc.so)或静态库(如libc.a)文件中。因此，我们除了要链接main.o、hello.o，还需要和glibc库的文件链接。

通常，C语言的链接共有三种方式:静态链接、动态链接和混合链接。三者的区别就在于链接器在链接过程中对程序中库函数调用的解析。

##### 静态链接
静态链接将所有目标文件和所需的库在编译时一并打包进最终的可执行文件。库的代码被复制到最终的可执行文件中，使得可执行文件变得自包含，不需要在运行时查找或加载外部库。
```bash
$ gcc -static main.o hello.o -o mainstatic
```
`-static` 参数指示编译器进行静态链接，而不是默认的动态链接。使用这个参数，GCC会尝试将所有用到的库函数直接链接到最终生成的可执行文件中，包括C标准库(libc)、数学库(libm)和其他任何通过代码引用的外部库。
##### 动态链接
库在运行时被加载，可执行文件包含了需要加载的库的路径和符号信息。动态链接的可执行文件比静态链接的小，因为它们共享系统级的库代码。与静态链接不同，库代码不包含在可执行文件中。
```bash
$ gcc main.o hello.o -o maindynamic //默认动态链接
```
也可以将自己编写的部分代码处理为动态库：
```bash
$ gcc -fPIC -shared -o libhello.so hello.o
```
`-fPIC` 参数告诉编译器生成位置无关代码(Position Independent Code)，这是创建共享库所必需的。位置无关代码允许库在内存中的任何位置加载，而不需要修改代码中的地址引用。

`-shared` 参数指示编译器生成一个共享库，而不是一个可执行文件。使用这个参数，GCC会将输入的目标文件（如hello.o）打包成一个共享库文件（如libhello.so），该文件可以被其他程序动态加载和链接。

`.so` 是共享库文件的常见扩展名，表示这是一个动态链接库，可以在运行时被加载和使用。

##### 混合链接
某些库静态链接，而其他库动态链接。这种方式结合了静态链接和动态链接的优点。执行下面的指令可以将hello.o编译为静态链接库：
```bash
ar crv libhello.a hello.o
```
`ar`:归档命令，用于处理静态库文件。

`crv`:ar命令的选项，由三个字符组成，每个字符代表一个选项:
- c:创建归档文件。如果指定的归档文件不存在，a会创建它。
- r:替换归档文件中现有的文件或者向归档文件中添加新文件。如果hello.o已经在libhello.a中，它会被新版本替换;如果不存在，则会被添加。
- v:详细模式(verbosemode)，在处理文件时显示详细信息。使用这个选项，ar会列出它正在执行的操作，包括哪些文件被添加或替换。
  
`libhello.a`:要创建或更新的静态库文件的名称。按照惯例，Linux下的静态库文件名以lib开头，并以.a作为文件扩展名。

## CMake
使用CMake可以更方便地管理和构建C/C++项目。CMake是一个跨平台的构建系统，可以自动生成适合不同平台的构建文件（如Makefile、Visual Studio工程文件等）。通过编写CMakeLists.txt文件，您可以定义项目的结构、依赖关系和构建规则，从而简化编译和链接过程。CMake还支持自动处理库依赖、生成可执行文件和共享库，以及提供更好的跨平台支持，使得构建过程更加高效和灵活。

### CMake简介
CMake 是个一个开源的**跨平台**自动化建构系统，用来管理软件建置的程序，并不依赖于某特定编译器，并可支持多层目录、多个应用程序与多个函数库。

CMake 通过使用简单的配置文件 CMakeLists.txt，自动生成不同平台的构建文件（如 Makefile、Ninja 构建文件、Visual Studio 工程文件等），简化了项目的编译和构建过程。

CMake 本身不是构建工具，而是生成构建系统的工具，它生成的构建系统可以使用不同的编译器和工具链。
### CMake作用和优势
- 跨平台支持： CMake 支持多种操作系统和编译器，使得同一份构建配置可以在不同的环境中使用。
- 简化配置： 通过 CMakeLists.txt 文件，用户可以定义项目结构、依赖项、编译选项等，无需手动编写复杂的构建脚本。
- 自动化构建： CMake 能够自动检测系统上的库和工具，减少手动配置的工作量。
- 灵活性： 支持多种构建类型和配置（如 Debug、Release），并允许用户自定义构建选项和模块。

### CMakeLists.txt 文件结构
CMakeLists.txt 是 CMake 的配置文件，用于定义项目的构建规则、依赖关系、编译选项等。每个 CMake 项目通常都有一个或多个 CMakeLists.txt 文件。

必须项：

1.指定CMake的最低版本要求：
```cmake
cmake_minimum_required(VERSION version) 
//如：cmake_minimum_required(VERSION 3.10)
```
2.定义项目的名称和使用的编程语言：
```cmake
project(<project_name> [<language>...]) 
//如：project(MyProject CXX) 表示c++
```
3.指定要生成的可执行文件和其源文件：
```cmake
add_executable(<target> <source_files>...) 
//如：add_executable(MyExecutable main.cpp other_file.cpp)
```
4、创建一个库（静态库或动态库）及其源文件：
```cmake
add_library(<target> <source_files>...) 
//如：add_library(MyLibrary STATIC library.cpp)
```
5、链接目标文件与其他库：
```cmake
target_link_libraries(<target> <libraries>...)
// 如：target_link_libraries(MyExecutable MyLibrary)
```
6、添加头文件搜索路径：
```cmake
include_directories(<dirs>...)
//如：include_directories(${PROJECT_SOURCE_DIR}/include)
```
7、设置变量的值：
```cmake
set(<variable> <value>...)
//如：set(CMAKE_CXX_STANDARD 11)
```
8、设置目标属性：
```cmake
target_include_directories(TARGET target_name
                          [BEFORE | AFTER]
                          [SYSTEM] [PUBLIC | PRIVATE | INTERFACE]
                          [items1...])
```
9、安装规则：
```cmake
install(TARGETS target1 [target2 ...]
        [RUNTIME DESTINATION dir]
        [LIBRARY DESTINATION dir]
        [ARCHIVE DESTINATION dir]
        [INCLUDES DESTINATION [dir ...]]
        [PRIVATE_HEADER DESTINATION dir]
        [PUBLIC_HEADER DESTINATION dir])
```
10、条件语句 (if, elseif, else, endif 命令)
```cmake
if(expression)
  # Commands
elseif(expression)
  # Commands
else()
  # Commands
endif()
```

示例见Ubuntu上的hello文件夹：
```bash
cmake .
make 
./hello
```
当需要删除生成的一系列非原始文件时，比较麻烦：
```bash
make clean
rm ...
rm -rf ...
```
更加方便的做法时将所有非原始文件放在build文件夹中，这样就可以直接删除build文件夹来删除所有非原始文件了。
```bash
mkdir build
cd build
cmake ..
make
../ hello
rm -rf build
```
