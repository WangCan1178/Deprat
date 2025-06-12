## README

![demo](https://github.com/W-caner/Deprat/blob/main/Resources/demo.gif)

***



### 语言

<p align="center">
    <a href = "./README.zh-CN.md">简体中文</a> | <a href = "./README.md">English</a>
</p>






### 项目结构

- Dataset：Deprat数据集，收集了3796 个 API 文档，构成包含122434个访问操作，403106 个参数信息的数据集。
  - gen_data.ipynb：数据爬取、提取、清理算法
  - oapi.json：接口粒度数据集
  - params.json：参数粒度数据集
- Experiments：论文中所提到的基准实验代码与结果
  - BART，GPT，LSTM，T5，DiffuSeq：所跑实验源码
  - result：结果文件夹
    - generate_txt：在测试集上生成的结果文件
    - log：训练过程中的输出
    - score.ipynb：应用传统指标以及SPDE进行评价的程序文件

- Platform：API参数自动注释生成平台源码
- Resources：项目所用其它资源

### Deprat数据集

- 快速下载：百度网盘链接https://pan.baidu.com/s/1fsfnSJNzcPvAQjfyBKmZOA?pwd=wcan 提取码：wcan
- 字段说明：

![image-20230802130427816](https://github.com/W-caner/Deprat/blob/main/Resources/Deprat%E5%AD%97%E6%AE%B5.png)

### API参数描述自动生成平台

- 平台运行于：[Welcome to API parameter descriptions automatic generation platform ](http://172.16.17.43:8501/)，点击可访问
- 使用方法：
  - 查看，编辑和管理API
  - 单击“Generate parameter descriptions”将自动为 **desc栏没有描述的参数** 生成参数描述
  - 使用“Save API information” 以持久化存储
  - 使用SPDE进行评估
  - 质量百分比基准是统计所得数据

### 其它

- 因大小限制，所有的需要的额外库，以及模型训练得到的checkpoint已经删除
- 目前装载的后端模型为T5，具体训练参数见论文
- 平台所显示的质量百分比基准是统计所得数据
- 该方法已成功应用于国家重点研发计划（批准号：2022yff0902703）的web API服务自动采集与评估中！
  ! [image-yingyong](https://github.com/WangCan1178/Deprat/blob/main/Resources/%E7%B3%BB%E7%BB%9F%E5%BA%94%E7%94%A8.jpg)
- 声明：本文研究得到国家重点研发计划项目（资助号：2022yff0902703）、国家自然科学基金项目（资助号：62472121）、国家自然科学基金项目（资助号：62306087）、国家自然科学基金项目（资助号：2022yff0902703）的部分资助。
山东省泰山学者项目专项资助。Quan zsheng的工作得到了澳大利亚研究委员会（ARC）的部分发现基金DP230100233的支持
