
conda activate self_forcing
CUDA_VISIBLE_DEVICES=1 python long.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd_long \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path myprompt/stick.txt \
    --use_ema \
    --num_blocks 85
