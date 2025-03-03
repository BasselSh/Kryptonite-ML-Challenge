```bash
conda create -n krypto python=3.10.12 -y

conda activate krypto

cd open-metric-learning

conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia

Comment torch, torchvision in open-metric-learning/ci/requirements.txt

pip install -r ci/requirements.txt

pip install -e .

cd ..
```

In open-metric-learning/oml/inference/abstract.py comment the following: (lines 30-31)

    # if is_ddp():
    #     loader = patch_dataloader_to_ddp(loader)

