export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 scripts/generate_ode_pairs.py \
      --output_folder ode_pairs/ \
      --caption_path prompts/vidprom_filtered_extended.txt \
      --guidance_scale 6.0