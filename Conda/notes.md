## Conda

Conda是一款强大的命令行工具，用于软件包和环境管理，且适用于不同语言。Conda同时具有创建虚拟环境和下载包的功能，同时是package manager和environment manager。

以下列举了一些常见的Conda命令，详见：https://docs.conda.io/projects/conda/en/stable/commands/index.html

#### 1.管理环境

通用创建新环境：
```bash
conda create --name env-name
```

创建一个指定Python版本的环境：
```bash
conda create -n env-name python=3.13.1
```

创建一个带有指定包的环境：
```bash
conda create -n env-name scipy=0.17.3 pandas
```

从依赖文件创建环境：
```bash
conda env create -f environment.yml
```

激活/关闭环境：
```bash
conda activate/deactivate env-name
```

查看环境列表：
```bash
conda info --envs
conda env list
```

查看环境中的包列表：
```bash
conda list -n env-name  # 环境未激活
conda list              # 环境已激活
conda list package-name # 查看某个特定包
```

导出环境：
```bash
conda export > environment.yaml
```

删除该环境：
```bash
conda remove --name env-name --all
```

#### 2.管理包

查看某个特定的包是否可以下载，以及是否存在于特定channel：
```bash
conda search package-name
conda search --override-channels --channel defaults paackage-name
conda search --override-channels --channel http://conda.anaconda.org/mutirri iminuit
```

安装包：
```bash
conda install --name env-name package-name
conda install package-name
pip install package-name
```

使用pip安装包：

推荐只有在conda安装了尽可能多的包之后再使用pip：
```bash
conda install pip
pip install package-name
```

安装Anaconda.org的软件包：

无法通过conda安装的包可以从Anaconda.org获得，这是一个适用于公有和私有包仓库的包管理服务。首先进入 <http://anaconda.org>搜索包，找到对应的channel名后，执行如下命令：
```bash
conda install -c channel-name package-name
```

更新某个环境中的包：
```bash
conda update package-name
pip install --upgrade package-name
```

移除包：
```bash
conda remove package-name
```

查找包依赖关系：
```bash
conda search package-name --info
conda info
grep package-name ~/anaconda3/pkgs/*/info/index.json
```

删除缓存的下载包和压缩文件
```bash
conda clean --packages --tarballs -y
```