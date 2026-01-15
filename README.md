# VASMA: Manifold-Guided Dual-Domain Augmentation with Unified Cache Fusion

Official implementation of ['VASMA: Manifold-Guided Dual-Domain Augmentation with Unified Cache Fusion for Zero-/Few-Shot Recognition'](https://arxiv.org/abs/).

The paper has been submitted to **Information Fusion** 📝.

## News
* The code of VASMA has been released.
* VASMA achieves state-of-the-art results on 11 diverse benchmarks, improving over zero-shot CLIP by an average of +24.6 percentage points (+41.8% relative improvement).

## Introduction
We propose **VASMA** (Variational Augmentation with Semantic Manifold Anchoring), a unified information-fusion framework that bridges the modality gap through **manifold-guided geometry**. VASMA treats textual prototypes as **shared semantic anchors** that guide the generation and integration of multi-source evidence. 

Specifically, VASMA works by **`Semantic Anchoring, Dual-Domain Augmentation, then Unified Cache Fusion'**:
- **Semantic Anchoring**: We leverage GPT-3 to expand class names into rich textual descriptions, forming robust semantic prototypes that serve as anchors for all subsequent processes.
- **Dual-Domain Augmentation**: We propose a complementary strategy that couples pixel-space synthesis (via DALL-E) for semantic coverage with feature-space generation (via Conditional VAE) for manifold density.
- **Manifold-Guided Rectification**: By projecting generated features onto the tangent space of the class manifold, VASMA rectifies the prototype--real mismatch, ensuring synthetic evidence aligns with the intrinsic geometry of real visual clusters.
- **Unified Cache Fusion**: All evidence sources (Real, Pixel-Synthetic, Feature-Synthetic) are standardized into a single Key-Value memory format, enabling principled similarity computation and auditable decision attribution.

By such collaboration, VASMA achieves state-of-the-art performance across 11 diverse benchmarks, demonstrating superior data efficiency and robustness in zero- and few-shot recognition tasks.

<div align="center">
  <img src="VASMA.png"/>
</div>

## Requirements

### Installation
Create a conda environment and install dependencies:
```bash
git clone https://anonymous.4open.science/r/VASMA-8290/main.py
cd VASMA

conda create -n vasma python=3.7
conda activate vasma

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Please follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to download official ImageNet and other 10 datasets. VASMA evaluates on 11 diverse datasets:
- Generic objects: ImageNet
- Fine-grained categories: StanfordCars, OxfordPets, Flowers102, Food101, FGVC Aircraft
- Scenes: SUN397
- Textures: DTD
- Cross-domain tasks: EuroSAT, UCF101, Caltech101

### Foundation Models
* The pre-trained weights of **CLIP** (ResNet-50) will be automatically downloaded by running.
* The prompts produced by **GPT-3** have been stored at `gpt_file/`.
* Please download **DINO's** pre-trained ResNet-50 from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth), and put it under `dino/`.
* Please download **DALL-E 2's** generated images from [here](https://drive.google.com/drive/folders/[your-folder-id]), and organize them with the official datasets like:
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– ...
|–– dalle_imagenet/
|–– dalle_caltech-101/
|–– dalle_oxford_pets/
|–– ...
```
* The **Conditional VAE** for feature generation will be trained during the first run and saved for later use.

## Get Started
### Configs
The running configurations for different `[dataset]` with `[k]` shots can be modified in `configs/[dataset]/[k]shot.yaml`, including visual encoders, manifold projection dimension `k`, fusion temperature `beta`, and branch weights. We have provided the configurations for reproducing the results in the paper. You can edit the hyperparameters for fine-grained tuning and better results.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparameter tuning.

### Running
For 16-shot ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet/16shot.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/[dataset]/16shot.yaml
```

For zero-shot evaluation:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/[dataset]/0shot.yaml
```

### Numerical Results

We provide VASMA's numerical results on 11 datasets from 0 to 16 shots. VASMA achieves:
- **Zero-shot**: 61.87% on ImageNet (+1.15 points over CaFo)
- **1-shot**: 64.04% on ImageNet (+0.24 points over CaFo)
- **16-shot**: 69.28% on ImageNet (+0.49 points over CaFo)

Detailed results can be found in the paper and will be updated in the repository.

## Acknowledgement
This repo benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [CaFo](https://github.com/ZrrSkywalker/CaFo), [CLIP](https://github.com/openai/CLIP), [DINO](https://github.com/facebookresearch/dino), [DALL-E](https://github.com/borisdayma/dalle-mini) and related works. Thanks for their wonderful contributions.


## Citation
```bibtex
@article{vasma2024,
  title={VASMA: Manifold-Guided Dual-Domain Augmentation with Unified Cache Fusion for Zero-/Few-Shot Recognition},
  author={Anonymous},
  journal={Information Fusion},
  year={2024}
}
```

## Contributors
Anonymous

## Contact
If you have any question about this project, please feel free to contact the authors.
