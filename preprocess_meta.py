import os
import json
from krypto import DATA_DIR

def remove_zero_padding(img_index):
    folder = int(os.path.dirname(img_index))
    img_name = os.path.basename(img_index)
    img_index = os.path.join(str(folder), img_name)
    return img_index

if __name__ == "__main__":
    with open(f"{DATA_DIR}/meta.json", "r") as f:
        meta = json.load(f)

    os.rename(f"{DATA_DIR}/meta.json", f"{DATA_DIR}/meta_old.json")
    indices = meta.keys()
    indices = [remove_zero_padding(index) for index in indices]
    real_fake = meta.values()
    json_dict = dict(zip(indices, real_fake))
    with open(f"meta.json", "w") as f:
        json.dump(json_dict, f, indent=4)