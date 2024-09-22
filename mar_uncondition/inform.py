import torch

checkpoint_path = "/home/zeyuv1/syliu/mar/pretrained_models/vae/kl16.ckpt"
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Checkpoint loaded successfully")
except Exception as e:
    print(f"Error loading checkpoint: {e}")

