# EOA

This is the repository for Ecosphere Optimization Algorithm (EOA)
The related paper is 《You also need to learn from the Ecosphere》

## Abstract：

In the past three years, the parameter size of deep learning models has increased more than 1000x up to hundreds of billions of parameters, large-scale distributed training using computer clusters has become the mainstream way of model training. During model training, the setting of the learning rate schedule can seriously affect the convergence speed and the effectiveness of the model. There have been many search and decay algorithms dedicated to learning rate schedules in single-machine environments. However, current methods for scaling single-machine learning rates to multi-computer distributed environments often use intuitive, rough linear scaling or square root scaling, which makes it difficult to guarantee model convergence, especially in heterogeneous distributed environments. Inspired by human learning from multiple populations in an ecosphere, we propose the Ecosphere Optimization Algorithm (EOA) to determine the learning rate schedule for deep learning distributed training. We divide the distributed clusters into multiple populations, use different population optimization algorithms among the populations, and let the different populations communicate with each other periodically. Like humans learning from the experience of different species in nature, the ecosphere optimization algorithm incorporates the advantages of multiple populations to further improve model performance. As evidenced by our experiments, the EOA is able to further improve the performance of distributed training on state-of-the-art vision models.

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
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --name cifar10-dist --train_workers 8 --dataset cifar10  --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --gradient_accumulation_steps 16
```

The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`
