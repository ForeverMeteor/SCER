# Self-Consistency, Extract and Rectify: Knowledge Graph Enhance Large Language Model for Electric Power Question Answering

\[ [English](README.md) | 中文 \]

<div align="center">
[![Paper](https://img.shields.io/badge/Paper-Springer-red)](https://link.springer.com/chapter/10.1007/978-981-97-5615-5_40) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ForeverMeteor/SCER/blob/main/LICENSE)
</div> 

## 介绍

近年来，电力人工智能发展迅速，涵盖安全检测、辅助决策、优化调度等多个领域。随着大型语言模型（LLM）的兴起，基于知识的人工智能在各个领域日益普及。然而，在电力领域，大多数基于知识的人工智能以知识图谱（KG）技术为核心，而对电力LLM的研究较少。本文受自一致性（SC）的启发，提出了一个自一致性、提取和纠正框架——SCER，用于将知识图谱增强的LLM应用于电力运维问答场景。具体而言，我们将 SC 从通用领域迁移到电力领域，并用中文句子表示模型替换原模型，使其更加本地化。我们设计了一种提取机制，通过在 POMKG 上进行多次随机游走来生成证据链，并设计了一种纠正机制来纠正生成理由的得分。在 POMQA 数据集上进行的大量实验和具体案例研究证明了我们提出的 SCER 对于电力领域自一致性转移和改进的有效性。

## 环境

``` pip
pip install -r requirement.txt
```

## 运行

1. 将评估数据集放于`data/eval`下；将图数据集放于`data/graph`下

2. 将自己的镜像网站填入`CONSTANT.py`的`url`变量中，API_KEY填入`CONSTANT.py`的`api_key`变量中

3. 依次运行`SC_only.py`、`KB_only.py`，结果将储存至`result`下

**关于第2步：若要运行调用接口的模型则按原文操作；若要自行部署模型，则：**

1. 在`self_consistency`文件夹下增加新类并继承`SelfConsistency`类

2. 按照已有ChatGLM的模板自行实现调用自己的LLM

## 引用

欢迎引用本文：

```bibtex
@inproceedings{zhao2024self,
  title={Self-consistency, Extract and Rectify: Knowledge Graph Enhance Large Language Model for Electric Power Question Answering},
  author={Zhao, Jinxiong and Ma, Zhicheng and Zhao, Hong and Zhang, Xun and Liu, Qichuan and Zhang, Chentao},
  booktitle={International Conference on Intelligent Computing},
  pages={493--504},
  year={2024},
  organization={Springer}
}
```
