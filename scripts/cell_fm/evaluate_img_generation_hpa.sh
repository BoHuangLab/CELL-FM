ulimit -c unlimited

[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

# Wandb
[ -z "${output_dir}" ] && output_dir=./output/hpa/img_generation/PT_CELL-Diff-LD_CELL-E2-split

# Dataset
[ -z "${data_path}" ] && data_path=/home/dihan.zheng/workspace/dihan.zheng/dataset/HPA/lmdb_dataset
[ -z "${img_crop_method}" ] && img_crop_method=center
[ -z "${img_crop_size}" ] && img_crop_size=1024
[ -z "${img_resize}" ] && img_resize=256
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.5
[ -z "${cell_image}" ] && cell_image='nucl,er,mt'
[ -z "${test_cell_image}" ] && test_cell_image='nucl,er,mt'
[ -z "${split_key}" ] && split_key=test

# DDPM
[ -z "${num_timesteps}" ] && num_timesteps=200
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=0.0001
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=0.02
[ -z "${ddpm_schedule}" ] && ddpm_schedule=squaredcos_cap_v2
[ -z "${diffusion_pred_type}" ] && diffusion_pred_type=noise

# Model
## VAE
[ -z "${num_down_blocks}" ] && num_down_blocks=3
[ -z "${latent_channels}" ] && latent_channels=4
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='128,256,512'

## CELL-Diff
[ -z "${block_out_channels}" ] && block_out_channels='320,640,1280,1280'
[ -z "${layers_per_block}" ] && layers_per_block=2
[ -z "${mid_num_attention_heads}" ] && mid_num_attention_heads=8
[ -z "${sample_size}" ] && sample_size=64
[ -z "${esm_embedding}" ] && esm_embedding=esm2
[ -z "${hidden_size}" ] && hidden_size=1280
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048
[ -z "${num_hidden_layers}" ] && num_hidden_layers=8
[ -z "${num_attention_heads}" ] && num_attention_heads=8
[ -z "${mlp_ratio}" ] && mlp_ratio=4
[ -z "${attn_drop}" ] && attn_drop=0.0
[ -z "${dit_patch_size}" ] && dit_patch_size=1
[ -z "${cell_image_ratio}" ] && cell_image_ratio=0.5

# Loss
[ -z "${sequence_loss_coeff}" ] && sequence_loss_coeff=1.0
[ -z "${image_loss_coeff}" ] && image_loss_coeff=1.0

# Training
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=/hpc/projects/group.huang/dihan.zheng/stable_diffusion_text2img/pretrain_hpa/PT_VAE_HPA_KL-1e-4_FP32_CELL-E2-split/checkpoint-50000/pytorch_model.bin
[ -z "${loadcheck_path}" ] && loadcheck_path=pretrain_hpa/PT_CELL-Diff-V4_R7-4_BOC320-1280_MNAH8_NHL8_DPS1_ZM0d5/checkpoint-30000
[ -z "${learning_rate}" ] && learning_rate=3e-4
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${per_device_train_batch_size}" ] && per_device_train_batch_size=12
[ -z "${per_device_eval_batch_size}" ] && per_device_eval_batch_size=128

[ -z "${num_train_epochs}" ] && num_train_epochs=5000
[ -z "${logging_dir}" ] && logging_dir=$output_dir
[ -z "${logging_steps}" ] && logging_steps=100
[ -z "${warmup_steps}" ] && warmup_steps=1000
[ -z "${max_steps}" ] && max_steps=50000
[ -z "${save_steps}" ] && save_steps=500

# Evaluation
[ -z "${timestep_respacing}" ] && timestep_respacing="ddim100"

python cell_fm/tasks/cell_fm/img_generation_hpa.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --img_crop_method $img_crop_method \
            --img_crop_size $img_crop_size \
            --img_resize $img_resize \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --cell_image $cell_image \
            --test_cell_image $test_cell_image \
            --split_key $split_key \
            --num_timesteps $num_timesteps \
            --ddpm_beta_start $ddpm_beta_start \
            --ddpm_beta_end $ddpm_beta_end \
            --ddpm_schedule $ddpm_schedule \
            --diffusion_pred_type $diffusion_pred_type \
            --num_down_blocks $num_down_blocks \
            --latent_channels $latent_channels \
            --vae_block_out_channels $vae_block_out_channels \
            --block_out_channels $block_out_channels \
            --layers_per_block $layers_per_block \
            --mid_num_attention_heads $mid_num_attention_heads \
            --sample_size $sample_size \
            --esm_embedding $esm_embedding \
            --hidden_size $hidden_size \
            --max_protein_sequence_len $max_protein_sequence_len \
            --num_hidden_layers $num_hidden_layers \
            --num_attention_heads $num_attention_heads \
            --mlp_ratio $mlp_ratio \
            --attn_drop $attn_drop \
            --dit_patch_size $dit_patch_size \
            --cell_image_ratio $cell_image_ratio \
            --sequence_loss_coeff $sequence_loss_coeff \
            --image_loss_coeff $image_loss_coeff \
            --vae_loadcheck_path $vae_loadcheck_path \
            --loadcheck_path $loadcheck_path \
            --learning_rate $learning_rate \
            --weight_decay $weight_decay \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --per_device_train_batch_size $per_device_train_batch_size \
            --per_device_eval_batch_size $per_device_eval_batch_size \
            --num_train_epochs $num_train_epochs \
            --logging_dir $logging_dir \
            --logging_steps $logging_steps \
            --warmup_steps $warmup_steps \
            --max_steps $max_steps \
            --save_steps $save_steps \
            --timestep_respacing $timestep_respacing \
            --seed 666666 \
            --infer \