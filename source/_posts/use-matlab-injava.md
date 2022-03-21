---
title: 在Java的工程实现中使用Matlab计算能力
top: false
cover: false
toc: true
mathjax: true
date: 2022-03-21 18:55:01
password:
summary:
tags:
- matlab
categories:
- 工程化
---

Matlab丰富的模型资源、优化的算法能力以及对计算加速的硬件支持，使得其在计算核心的算法设计中成为主流开发方法，在传统实现中复杂的算法借助Matlab核心计算能力可以简单快速实现。
但是在工程化的实现过程中，实现Matlab能力集成踩坑众多，本文记录了一次在java中实现电网海量运行数据自定义模型计算引入matlab的踩坑及搞定方法

## 在算法运行设备安装完整的Matlab或者Matlab Compiler Runtime环境
[去这里]https://ww2.mathworks.cn/products/compiler/matlab-runtime.html

注意：版本选择很重要

## 安装对应的jdk版本
在matlab算法实现的console window，运行 version -java，查看内嵌的jdk版本，一定要选择相应的jdk版本独立安装

## 设置jdk环境变量
以macos为例
1. echo $SHELL 查看shell类型
2. bash:echo "export JAVA_HOME=<java_install_path>" >> ~/.bash_profile
        source ~/.bash_profile
3. zsh: echo "export JAVA_HOME=<java_install_path>" >> ~/.zprofile
        source ~/.zprofile

4. 设置DYLD_LIBRARY_PATH
   bash:echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<java_install_path>/bin" >> ~/.profile
        source ~/.profile
   zsh :echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<java_install_path>/bin" >> ~/.zshrc
        source ~/.zshrc
[参考这里]https://ww2.mathworks.cn/help/compiler_sdk/java/configure-your-java-environment.html?searchHighlight=java&s_tid=srchtitle_java_7

## 重启MacOS，并关闭SIP
1. 重启
2. command+R 进入recovery mode
3. 在recovery mode 打开terminal
4. command line: csrutil disable 关闭SIP

## 设置matlab运行时依赖
以macos为例
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:+${DYLD_LIBRARY_PATH}:}\
<MATLAB_RUNTIME_INSTALL_DIR>/runtime/maci64:\
<MATLAB_RUNTIME_INSTALL_DIR>/bin/maci64:\
<MATLAB_RUNTIME_INSTALL_DIR>/sys/os/maci64:\
<MATLAB_RUNTIME_INSTALL_DIR>/extern/bin/maci64"
[参考这里]https://ww2.mathworks.cn/help/compiler/mcr-path-settings-for-run-time-deployment.html


## 完成上述步骤，测试并可以开始工程化开发
