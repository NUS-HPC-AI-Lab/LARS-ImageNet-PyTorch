# Implementation of LARS for ImageNet with PyTorch

This is the code for the paper "[Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)", which implements a large batch deep learning optimizer called LARS using PyTorch. Although the optimizer has been released for some time and has an official TensorFlow version implementation, as far as we know, there is no reliable PyTorch version implementation, so we try to complete this work. We use [Horovod](https://github.com/horovod/horovod) to implement distributed data parallel training.

## Requirements

This code is validated to run with Python 3.6.10, PyTorch 1.5.0, Horovod 0.21.1, CUDA 10.0/1, CUDNN 7.6.4, and NCCL 2.4.7.

## Performance on ImageNet

We verified the implementation on the complete ImageNet-1K (ILSVRC2012) data set. The parameters and performance as follows. 

| Batch Size | GPU Numbers |  Base LR  | Warmup Epochs | Test Accuracy | TensorBoard Color |
| :--------: | :---------: | :-------: | :-----------: | :-----------: | :---------------: |
|    512     |      8      |   $2^2$   |  $10/2^{6}$   |  **76.95%**   |        Red        |
|    1024    |      8      | $2^{2.5}$ |  $10/2^{5}$   |  **77.06%**   |       Green       |
|    4096    |     32      | $2^{3.5}$ |  $10/2^{3}$   |  **76.78%**   |       Gray        |

Training process with TensorBoard

![Training process with TensorBoard](https://raw.githubusercontent.com/binmakeswell/LARS-ImageNet-PyTorch/main/Training%20process%20with%20TensorBoard.jpg)

We set epochs = 90, weight decay = 0.0001, model = resnet50 and use NVIDIA Tesla V100 GPU for all experiments. For parameters with other batch size, please refer to [Large-Batch Training for LSTM and Beyond](https://arxiv.org/abs/1901.08256) Table 4.

Thanks for computing resources from National Supercomputing Centre Singapore (NSCC)  and Texas Advanced Computing Center (TACC). 

## Usage

```
from lars import *
... 
optimizer = create_optimizer_lars(model=model, lr=args.base_lr,
                                momentum=args.momentum, weight_decay=args.wd,
                                bn_bias_separately=args.bn_bias_separately)
... 
lr_scheduler = PolynomialWarmup(optimizer, decay_steps=args.epochs * num_steps_per_epoch,
                                warmup_steps=args.warmup_epochs * num_steps_per_epoch,
                                end_lr=0.0, power=lr_power, last_epoch=-1)
... 
```

Note that We recommend using create_optimizer_lars and setting bn_bias_separately=True, instead of using class Lars directly, which helps LARS skip weight decay for parameters in BatchNormalization and bias, and has better performance in general. Polynomial Warmup learning rate decay is also helpful for better performance in general.

## Example Scripts

Example scripts for training with 8 GPU and 1024 batch size on ImageNet-1k are provided.

```
$ mpirun -np 8 \
python pytorch_imagenet_resnet.py  \
--epochs 90 \
--model resnet50 \
--batch-size 128 \
--warmup-epochs 0.3125 \
--train-dir=your path/ImageNet/train/ \
--val-dir=your path/ImageNet/val \
--base-lr 5.6568542494924 \
--base-op lars \
--bn-bias-separately \
--wd 0.0001
```

## Reference

[Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)

[Large-Batch Training for LSTM and Beyond](https://arxiv.org/abs/1901.08256) 

https://www.comp.nus.edu.sg/~youy/lars_optimizer.py

https://github.com/tensorflow/tpu/blob/5f71c12a020403f863434e96982a840578fdd127/models/official/efficientnet/lars_optimizer.py

