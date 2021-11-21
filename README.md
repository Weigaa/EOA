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

use this code achieve EOA-SAM training:

```
python EOASAM-cifar100.py #code on Cifar-100
python EOASAM.py #code on Cifar-10
```


