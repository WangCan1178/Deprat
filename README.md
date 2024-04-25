## README

![demo](https://github.com/W-caner/Deprat/blob/main/Resources/demo.gif)

***



### Language

<p align="center">
    <a href = "./README.zh-CN.md">简体中文</a> | <a href = "./README.md">English</a>
</p>








### Project structure

- Dataset: Deprat dataset, which collects 3796 API documents, constituting a dataset with 122434 access operations and 403,106 parameter information.
  - gen_data.ipynb: data crawling, extraction, and cleaning algorithms
  - oapi.json: Interface granularity dataset
  - params.json: parameter granularity dataset
- Experiments: The code and results of the benchmark experiments described in the paper
  - BART, GPT, LSTM, T5, DiffuSeq: source code of the experiments run
  - result: The results folder
    - generate_txt: The resulting file generated on the test set
    - log: The output during training
    - score.ipynb: Program file for evaluation using traditional metrics as well as SPDE

- Platform: API parameters are automatically annotated to generate platform source code
- Resources: Other resources used by the project

### Deprat Dataset

- Quick download: Baidu net disk linkhttps://pan.baidu.com/s/1fsfnSJNzcPvAQjfyBKmZOA?pwd=wcan Extract code: wcan
- Attribute Description:

![image-20230802130427816](https://github.com/W-caner/Deprat/blob/main/Resources/Deprat%E5%AD%97%E6%AE%B5.png)

### API parameter descriptions automatic generation platform

- The platform runs on：[Welcome to API parameter descriptions automatic generation platform ](http://58.59.92.190:54665/)，click to visit
- Usage:

  -  View, edit, and manage API
  - Clicking on "Generate parameter descriptions" will automatically generate parameter descriptions for **parameters not described in the desc column**
  - Use "Save API information" for persistence store
  - Evaluation using SPDE

  

### Others

- Due to size limitations, all additional libraries required and checkpoints for model training have been removed
- The backend model currently loaded is T5, see the paper for the specific training parameters
- The quality percentage benchmark displayed by the platform is the statistical data obtained
- Statement: Ownership of this project, which will be added later
