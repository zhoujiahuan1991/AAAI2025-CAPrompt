# CAPrompt: Cyclic Prompt Aggregation for Pre-Trained Model Based Class Incremental Learning

<div align="center">

<div>
      Qiwei Li&emsp; Jiahuan Zhou
  </div>
<div>

  Wangxuan Institute of Computer Technology, Peking University

</div>
</div>
<p align="center">
  <a href='https://arxiv.org/abs/2412.08929'><img src='https://img.shields.io/badge/Arxiv-2412.08929-A42C25.svg?logo=arXiv'></a>
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhoujiahuan1991%2FAAAI2025-CAPrompt&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</p>


Official implementation of "[CAPrompt: Cyclic Prompt Aggregation for Pre-Trained Model Based Class Incremental Learning](https://arxiv.org/abs/2412.08929)"


<p align="center"><img src="./files/pipeline-caprompt.jpg" align="center" width="750"></p>

Recently, prompt tuning methods for pre-trained models have demonstrated promising performance in Class Incremental Learning (CIL). These methods typically involve learning task-specific prompts and predicting the task ID to select the appropriate prompts for inference. However, inaccurate task ID predictions can cause severe inconsistencies between the prompts used during training and inference, leading to knowledge forgetting and performance degradation. Additionally, existing prompt tuning methods rely solely on the pre-trained model to predict task IDs, without fully leveraging the knowledge embedded in the learned prompt parameters, resulting in inferior prediction performance. 

To address these issues, we propose a novel **Cyclic Prompt Aggregation (CAPrompt)** method that eliminates the dependency on task ID prediction by cyclically aggregating the knowledge from different prompts. Specifically, rather than predicting task IDs, we introduce an innovative prompt aggregation strategy during both training and inference to overcome prompt inconsistency by utilizing a weighted sum of different prompts. Thorough theoretical analysis demonstrates that under concave conditions, the aggregated prompt achieves lower error compared to selecting a single task-specific prompt. Consequently, we incorporate a concave constraint and a linear constraint to guide prompt learning, ensuring compliance with the concave condition requirement. Furthermore, to fully exploit the prompts and achieve more accurate prompt weights, we develop a cyclic weight prediction strategy. This strategy begins with equal weights for each task and automatically adjusts them to more appropriate values in a cyclical manner. Experiments on various datasets demonstrate that our proposed CAPrompt outperforms state-of-the-art methods by 2\%-3\%.
## Requirements

### Environment
Python 3.9.0

pip install -r requirements.txt


### Dataset
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

### Pre-Trained Checkpoints
We incorporated the following supervised and self-supervised checkpoints as backbones:

Supervised

- [Sup-21K VIT](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)

Self-supervised 

- [iBOT](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
- [DINO](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)  
  
Please download the self-supervised checkpoints and put them in the ./checkpoints/{checkpoint_name} directory.
## Run commands
Training script for different experinments are provided in ./training_script/


## Acknowledgement

This project is mainly based on [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt).

## Citation

If you find this work helpful, please cite:
```
@inproceedings{li2024fcs,
  title={CAPrompt: Cyclic Prompt Aggregation for Pre-Trained Model Based Class Incremental Learning},
  author={Li, Qiwei and Zhou, Jiahuan},
  booktitle={AAAI},
  year={2025}
}

```