import pandas as pd
import numpy as np
import os

columns = ["eer_real", "eer_fake"]
index = ["triplet loss", "triplet loss real + fake"]
values = np.array([[0.1041, 0.425], [0.1374, 0.025]])
df = pd.DataFrame(values, index=index, columns=columns)
print(str(df))
experiments_dir = "experiments"
os.makedirs(experiments_dir, exist_ok=True)
df.to_csv(os.path.join(experiments_dir, "experiments_with_withou_fake_triplet.csv"))
experiments_txt_file = os.path.join(experiments_dir, "experiments.txt")
with open(experiments_txt_file, "a") as f:
    f.write("default quad loss: margin 1, neg_real_weight 1, neg_fake_weight 1\n\n")
    f.write("With + without fake loss:\n")
    f.write(str(df))
