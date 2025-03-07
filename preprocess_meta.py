import pandas as pd
import os
import json

def remove_zero_padding(img_index):
    folder = int(os.path.dirname(img_index))
    img_name = os.path.basename(img_index)
    img_index = os.path.join(str(folder), img_name)
    return img_index

with open("data/train/meta.json", "r") as f:
    meta = json.load(f)
indices = meta.keys()
indices = [remove_zero_padding(index) for index in indices]
real_fake = meta.values()
json_dict = dict(zip(indices, real_fake))
with open("data/train/meta_int.json", "w") as f:
    json.dump(json_dict, f, indent=4)