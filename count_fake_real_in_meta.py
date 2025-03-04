import json

with open('data/train/meta.json', 'r') as f:
    meta = json.load(f)
values = meta.values()
fake_len = sum(values)
all_len = len(values)
real_len = all_len - fake_len
print(f"Real len: {real_len}, Fake len: {fake_len}")
