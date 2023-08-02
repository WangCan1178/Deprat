## README

### 项目结构

- Dataset：Deprat数据集，收集了3796 个 API 文档，构成包含122434个访问操作，403106 个参数信息的数据集。
  - gen_data.ipynb：数据爬取、提取、清理算法
  - oapi.json：接口粒度数据集
  - params.json：参数粒度数据集
- Experiments：论文中所提到的基准实验代码与结果
- Platform：API参数自动注释生成平台

### Deprat数据集

![image-20230802130427816](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230802130427816.png)

### API参数描述自动生成平台

- 平台运行于：[Welcome to API parameter descriptions automatic generation platform ](http://172.16.17.43:8501/)，点击可访问
- 



### 其它

- 因大小限制，所有的需要的额外库，以及模型训练得到的checkpoint已经删除。
- 声明：本项目所有