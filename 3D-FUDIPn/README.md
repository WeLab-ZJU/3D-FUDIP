## Install
To use 3D-FUDIPn, please create a virtual environment using conda/mamba and install PyTorch. 

```
mamba create -n fudipn python=3.12 -y
mamba activate fudipn
```


Please select the correct PyTorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/.

Install PyTorch using Mamba or Conda:

```
mamba install pytorch torchvision cudatoolkit -c pytorch 
```

Alternatively, install PyTorch using pip:

```
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Install other dependencies using the provided `requirements.txt`:

```
pip install -r requirements.txt
```

Navigate to /code/3D-FUDIPn and install `basicsr`


```shell
python setup.py develop --no-build-isolation
```
## Training 3D-FUDIPn

- Configure training by editing the appropriate `.yml` file in `FUDIPn/options/train/`. Specify the dataset paths, checkpoint settings, logging options, and other hyperparameters as needed.

- Run the `train.py` script with configuration file path to start training:

e.g. 

```
python basicsr/train.py -opt options/train/FUDIPn/train_FUDIPn_neuron_dataset.yml
```

## Inference

- Place the pretrained model (`.pth` file) in an appropriate directory, such as `./ckpt/model_best.pth`.

- Specify the model path in the inference configuration file or set `DEFAULT_CHECKPOINT` in the inference script.

- Run `inference_fudipn.py` to perform inference on the paired datasets:

```
python scripts/network_inference/inference_fudipn.py
```

- Alternatively, after completing the required configuration, run the script directly in an IDE such as PyCharm.
  
- Run the `apply_big_data_40xSR.py` to perform inference on the large size data. This script is used to process large size data in a batch manner.

## Training monitor
```shell
mamba activate sr
```

```shell
tensorboard --logdir=D:/Code/FUDIP_proj/fudipn-basicsr/tb_logger/train_FUDIPn_denosing_task_lyso/
```
