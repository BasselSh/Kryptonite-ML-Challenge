import os
from typing import List
import torch
import pandas as pd
from torch.nn import functional as F
from oml import datasets as d
from krypto.inference import inference
import argparse
from krypto.parameters import models_dict

device = "cuda"
OUTPUT_DIR = "output"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    return parser.parse_args()


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    last_work_dir_path_file = "last_work_dir.txt"
    last_work_dir_path = open(last_work_dir_path_file, 'r').read()
    model_path = os.path.join(last_work_dir_path, "best_model.pth")
    submission_path = os.path.join(last_work_dir_path, "submission.csv")

    model_dict = models_dict[model_name]
    model = model_dict["model"]()
    transform = model_dict["transform"]
    model.load_state_dict(torch.load(model_path))
    model = model.to(device).eval()

    test_path = "test.csv"
    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    embeddings = inference(model, test, batch_size=32, num_workers=0, verbose=True)

    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv(submission_path, index=False)
