<h1 align='center'>SegNext-Pytorch</h1>

# [SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation](https://arxiv.org/pdf/2209.08575.pdf)
## This project is implemented in PyTorch, can be used to train your image-datasets for segmentation tasks.  

![image](https://github.com/Visual-Attention-Network/SegNeXt/blob/main/resources/flops.png)


## Preparation

### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### [Download Datasets(including VOC, ADE20K, COCO, COCOStuff, Hubmap, Synapse, CityScapes)](https://pan.baidu.com/s/1LLyIlP3sjuoFAwTBaYflRQ?pwd=0615)

## Project Structure
```
├── datasets: Load datasets
    ├── cityscapes.py: class of cityscapes
    ├── coco.py: class of coco
    ├── custom_transforms.py: image data aug methods
    ├── pascal.py: class of pascal voc
├── models: SegNext Models
    ├── bricks: Construct "drop_path", "conv_moudle", "trunc_normal_" layers.
    ├── mscan: Construct MSCAN backbone models.
    ├── segnext: Construct segnext models.
├── utils:
    ├── distributed_utils.py: Record various indicator information and output and distributed environment
    ├── losses.py: Define loss functions('CrossEntropy', 'OhemCrossEntropy', 'Dice', 'FocalLoss', etc)
    ├── metrics.py: Define Metrics (pixel_acc, f1score, miou)
├── engine.py: Function code for a training/validation process
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters.  

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Download the [pretrained-weights](https://github.com/Visual-Attention-Network/SegNeXt?tab=readme-ov-file#ade20k)  
Step 2: Write the ___pre-training weight path___ into the ___args.finetune___ in string format. Adjust ___args.input_size___ parameter based on the model pre-trained on images of different sizes.  
Step 3: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)  

### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. 

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```


## Citation
```
@article{guo2022segnext,
  title={SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Hou, Qibin and Liu, Zhengning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2209.08575},
  year={2022}
}


@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}


@inproceedings{
    ham,
    title={Is Attention Better Than Matrix Decomposition?},
    author={Zhengyang Geng and Meng-Hao Guo and Hongxu Chen and Xia Li and Ke Wei and Zhouchen Lin},
    booktitle={International Conference on Learning Representations},
    year={2021},
}
```