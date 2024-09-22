from datasets import load_dataset

from huggingface_hub import hf_hub_download

# 下载数据集 (确保文件名正确）
dataset_file = hf_hub_download(repo_id="Kelsy1031/imagenette", filename="imagenette.tgz")

# 下载模型 (确保文件名正确）
model_file = hf_hub_download(repo_id="Kelsy1031/mar", filename="pytorch_model.bin")


