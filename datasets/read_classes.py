import os

path = "D:/Research/VLM_project/dataset/Sketchy/photo"  # ví dụ: "./data"

labels = [
    name for name in os.listdir(path)
    if os.path.isdir(os.path.join(path, name))
]

print(labels)
