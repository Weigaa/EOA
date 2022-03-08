# EOA

This is the repository for Ecosphere Optimization Algorithm (EOA)
The related paper is 《You also need to learn from the Ecosphere》

## Abstract：

Recently, the parameter size of deep learning models has reached 100 trillion. Large-scale distributed training using computer clusters has become the dominant approach to model training. During training, the learning rate schedule can seriously affect the speed and effectiveness of model convergence. Much research has been devoted to studying learning rate schedules in a single-computer environment. However, current methods for scaling single-machine learning rates to multi-machine often use intuitive, crude linear scaling or square root scaling, making it difficult to guarantee model convergence. Inspired by human learning from multiple populations in an ecosphere, we propose the Ecosphere Optimization Algorithm (EOA) to optimize the learning rate schedule for distributed training. Like humans learning from the experience of different species, EOA incorporates the strengths of multiple populations. Experimental results show that EOA improves the performance of the state-of-the-art vision models Vision Transformer (VIT) and Wide ResNet (WRN) in distributed training.

## Catalog Structure

### Directory EOA-SAM

The EOA Algorithm implementation on WRN (SAM) model.

### Directory EOA-ViT

The EOA Algorithm implementation on ViT model.


# USAGE

## use this code achieve EOA-SAM training:

```
python EOASAM-cifar100.py #code on Cifar-100
python EOASAM.py #code on Cifar-10
```
## use this code achieve EOA-ViT training:

### 1. Download Pre-trained model (Google's Official Checkpoint)

* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```
### 2.Train Model

```python
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --name cifar10-EOA --train_workers 8 --dataset cifar10  --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --gradient_accumulation_steps 32 #code on Cifar-10

python3 -m torch.distributed.launch --nproc_per_node=8 train.py --name cifar100-EOA --train_workers 8 --dataset cifar100  --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --gradient_accumulation_steps 32 #code on Cifar-100
```
CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`
