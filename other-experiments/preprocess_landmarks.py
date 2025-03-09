import pandas as pd
import os

def join_path(folder, image):
    return os.path.join(folder, image)

df = pd.read_csv("points_df.csv")
df['folder'] = df['folder'].astype(str)
df['image'] = df['image'].astype(str)
df['img_index'] = df.apply(lambda x: join_path(x.folder, x.image), axis=1)
df.drop(columns=['folder', 'image'], inplace=True)
df.to_csv("points_df_with_img_index.csv", index=False)