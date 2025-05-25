# Self-Consistency, Extract and Rectify: Knowledge Graph Enhance Large Language Model for Electric Power Question Answering

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Springer-red)](https://link.springer.com/chapter/10.1007/978-981-97-5615-5_40) [![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div> 

## Overview

Electric power artificial intelligence has rapidly advanced in recent years, encompassing safety detection, assistant decision-making, and optimal scheduling. With the rise of Large Language Models (LLMs), knowledge-based AI is becoming increasingly prevalent across various domains. However, in the field of electric power, most of the knowledge-based AI is centered on Knowledge Graph (KG) techniques, while lessresearch has been done on power LLMs. In this paper, we are inspired by Self-Consistency (SC) and propose a **S**elf-**C**onsistency, **E**xtraction and **R**ectify framework — SCER, for the usage of KG-enhanced LLM inpower operations and maintenance (O&M) question answering scenarios. Specifically, we transfer the SC from the general-purpose domain into the power domain and replace the original model with a Chinese sentence representation model to make it more localized. We design an Extract Mechanism to generate evidence chains through multiple random walks on the POMKG and a Rectify Mechanism to correct the score of the generated rationales. Extensive experiments and specific case studies on the POMQA dataset demonstrate the effectiveness of our proposed SCER for SC transfer and improvement in the power field.

## Environment

``` pip
pip install -r requirement.txt
```

## Run

1. Put the evaluation dataset in `data/eval`; put the graph dataset in `data/graph`.

2. Fill your own mirror website into the `url` variable of `CONSTANT.py`, and fill API_KEY into the `api_key` variable of `CONSTANT.py`.

3. Run `SC_only.py` and `KB_only.py` sequentially, and the results will be stored in `result`.

**About Step 2: If you want to run the model that calls the interface, follow the original operation; if you want to deploy the model yourself, then:**

1. Add a new class in the `self_consistency` folder and inherit the Class `SelfConsistency`.

2. According to the existing ChatGLM template, call your own LLM.

## Citation

If you find this work helpful, please cite our paper:

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