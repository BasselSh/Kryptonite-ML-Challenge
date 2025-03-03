# Install

1. Create conda environment and install dependencies

```bash
git clone git@github.com:BasselSh/Kryptonite-ML-Challenge.git

cd Kryptonite-ML-Challenge

conda create -n krypto python=3.10.12 -y

conda activate krypto

git clone https://github.com/OML-Team/open-metric-learning.git

cd open-metric-learning

conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia

Comment torch, torchvision in open-metric-learning/ci/requirements.txt

pip install -r ci/requirements.txt

pip install -e .

cd ..
```

2. In open-metric-learning/oml/inference/abstract.py comment the following for enabeling inference with **cuda**: 

(lines 30-31)

```python
    # if is_ddp():
    #     loader = patch_dataloader_to_ddp(loader)
```
(lines 50-52)

```python
    # data_to_sync = {"outputs": outputs, "ids": ids}
    # data_synced = sync_dicts_ddp(data_to_sync, world_size=get_world_size_safe())
    # outputs, ids = data_synced["outputs"], data_synced["ids"]
```

3. Change device to **cuda** in **train.py**, **predict.py**, and **make_submission.py**
