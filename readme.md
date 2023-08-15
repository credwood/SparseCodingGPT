## Autoregressive Transformer visualization via dictionary learning 

This is a reimplementation of [this](https://github.com/zeyuyun1/TransformerVis/blob/main/readme.md) project based on the paper [Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors](https://arxiv.org/pdf/2103.15949.pdf) by by Zeyu Yun*, Yubei Chen*, Bruno A Olshausen, and Yann LeCun.

 # Set up

You must specify which cupy version to install based on your your cuda version. To get this information, run: 

```
nvcc --version
```
or 

```
nvidia-smi
```

Edit cupy in requirements.txt based on your system requirements and the available versions [here](https://pypi.org/project/cupy/).

Create and activate a virtual environment before installing the requirements with the comand:

```
pip install -r requirements.tx
```

 # Training and Inference
 
 Please also see the [original repo](https://github.com/zeyuyun1/TransformerVis/blob/main/readme.md) for more training details and customizations. Here are the basics:

 Create dataset:

 ```
 python dataset.py
 ```

 Train a dictionaries (default is to train for each layer; in the paper they trained every other layer):

 ```
 python train.py
 ```

 For inference details and troubleshooting, please see the [original repo](https://github.com/zeyuyun1/TransformerVis/blob/main/readme.md). The methods have been adapted but remain untested