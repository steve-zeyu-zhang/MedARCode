torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=10.50.1.27 --master_port=49676 \
    main_mar.py \
    --img_size 256 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --model mar_large \
    --diffloss_d 3 \
    --diffloss_w 1024 \
    --epochs 400 \
    --warmup_epochs 100 \
    --batch_size 1 \
    --blr 1.0e-4 \
    --diffusion_batch_mul 4 \
    --output_dir 'output' \
    --resume 'output' \
    --data_path '/home/zeyuv1/syliu/mar/dataset/imagenette2'# MedARCode
