ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

# Wandb
export WANDB_RUN_NAME=PT_OC3D_CELLFM3D_R1
export WANDB_PROJECT=CELL-FM-3D
[ -z "${output_dir}" ] && output_dir=pretrain_opencell_3d/$WANDB_RUN_NAME

# Dataset
[ -z "${data_path}" ] && data_path='/hpc/reference/opencell/opencell'
[ -z "${split_key}" ] && split_key='all'
[ -z "${phase}" ] && phase='train'
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.0

# Transport parameters
[ -z "${path_type}" ] && path_type=Linear
[ -z "${prediction}" ] && prediction=velocity

# VAE (3D, pretrained)
[ -z "${num_down_blocks}" ] && num_down_blocks=2
[ -z "${latent_channels}" ] && latent_channels=4
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='64,128'
[ -z "${input_spatial_size}" ] && input_spatial_size='48,192,192'
[ -z "${norm_num_groups}" ] && norm_num_groups=32
[ -z "${layers_per_block}" ] && layers_per_block=2

# 3D patch embedding
[ -z "${patch_d}" ] && patch_d=4
[ -z "${patch_size}" ] && patch_size=8

# Sequence / ESM
[ -z "${esm_embedding}" ] && esm_embedding='esmc_300m'
[ -z "${encoder_hidden_size}" ] && encoder_hidden_size=320
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048

# Image generator (SD3-3D)
[ -z "${img_generator_num_layers}" ] && img_generator_num_layers=18
[ -z "${attention_head_dim}" ] && attention_head_dim=64
[ -z "${num_attention_heads}" ] && num_attention_heads=16

# Checkpoint paths
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=pretrain_opencell_3d/PT_VAE3D_OC_192_KL1e-4/checkpoint-40000/pytorch_model.bin
[ -z "${loadcheck_path}" ] && loadcheck_path=.

# Training
[ -z "${learning_rate}" ] && learning_rate=3e-4
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${per_device_train_batch_size}" ] && per_device_train_batch_size=1
[ -z "${per_device_eval_batch_size}" ] && per_device_eval_batch_size=1

[ -z "${num_train_epochs}" ] && num_train_epochs=5000
[ -z "${logging_dir}" ] && logging_dir=$output_dir
[ -z "${logging_steps}" ] && logging_steps=100
[ -z "${warmup_steps}" ] && warmup_steps=1000
[ -z "${max_steps}" ] && max_steps=100000
[ -z "${save_steps}" ] && save_steps=10000

[ -z "${MASTER_PORT}" ] && MASTER_PORT=12325
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1

DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                  --master_port $MASTER_PORT \
                  --master_addr $MASTER_ADDR"

python -m torch.distributed.run $DISTRIBUTED_ARGS cell_fm/tasks/cell_fm_3d/pretrain_opencell_3d.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --split_key $split_key \
            --phase $phase \
            --data_aug \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --path_type $path_type \
            --prediction $prediction \
            --num_down_blocks $num_down_blocks \
            --latent_channels $latent_channels \
            --vae_block_out_channels $vae_block_out_channels \
            --input_spatial_size $input_spatial_size \
            --norm_num_groups $norm_num_groups \
            --layers_per_block $layers_per_block \
            --patch_d $patch_d \
            --patch_size $patch_size \
            --esm_embedding $esm_embedding \
            --encoder_hidden_size $encoder_hidden_size \
            --max_protein_sequence_len $max_protein_sequence_len \
            --img_generator_num_layers $img_generator_num_layers \
            --attention_head_dim $attention_head_dim \
            --num_attention_heads $num_attention_heads \
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
            --seed 6 \
            --wandb \

            # --ft \
            # --ifresume \
            # --bf16 \
