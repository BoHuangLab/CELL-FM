ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

# Wandb
export WANDB_RUN_NAME=PT_HPA_CELLFMC_R1
export WANDB_PROJECT=CELL-FM
[ -z "${output_dir}" ] && output_dir=pretrain_hpa/$WANDB_RUN_NAME

# Dataset
[ -z "${data_path}" ] && data_path=/hpc/reference/opencell/human_protein_atlas
[ -z "${split_key}" ] && split_key=cellfm_train
[ -z "${phase}" ] && phase='train'

[ -z "${img_resize}" ] && img_resize=256
[ -z "${img_crop_size}" ] && img_crop_size=1024
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.0
[ -z "${cell_image}" ] && cell_image='nucl,er,mt'

# Interaction context
[ -z "${n_context_proteins}" ] && n_context_proteins=474
[ -z "${context_embedding_dim}" ] && context_embedding_dim=1152

# Transport parameters
[ -z "${path_type}" ] && path_type=Linear
[ -z "${prediction}" ] && prediction=velocity

# Model
## VAE
[ -z "${num_down_blocks}" ] && num_down_blocks=3
[ -z "${latent_channels}" ] && latent_channels=4
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='128,256,512'

## CELL-FMC
[ -z "${cond_out_channels}" ] && cond_out_channels='32,64'
[ -z "${sample_size}" ] && sample_size=64
[ -z "${esm_embedding}" ] && esm_embedding='esmc_600m'
[ -z "${encoder_hidden_size}" ] && encoder_hidden_size=1152
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048

### Image generator
[ -z "${img_generator_num_layers}" ] && img_generator_num_layers=8
[ -z "${img_generator_patch_size}" ] && img_generator_patch_size=2
[ -z "${attention_head_dim}" ] && attention_head_dim=64
[ -z "${num_attention_heads}" ] && num_attention_heads=18

[ -z "${cell_image_ratio}" ] && cell_image_ratio=0.5

# Loss (only img_diff_loss)
[ -z "${img_diff_loss_coeff}" ] && img_diff_loss_coeff=1.0

# Training
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=pretrain_hpa/vae/checkpoint-50000/pytorch_model.bin
[ -z "${loadcheck_path}" ] && loadcheck_path=.
[ -z "${learning_rate}" ] && learning_rate=3e-4
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=2
[ -z "${per_device_train_batch_size}" ] && per_device_train_batch_size=32
[ -z "${per_device_eval_batch_size}" ] && per_device_eval_batch_size=128

[ -z "${num_train_epochs}" ] && num_train_epochs=5000
[ -z "${logging_dir}" ] && logging_dir=$output_dir
[ -z "${logging_steps}" ] && logging_steps=100
[ -z "${warmup_steps}" ] && warmup_steps=1000
[ -z "${max_steps}" ] && max_steps=50000
[ -z "${save_steps}" ] && save_steps=10000

[ -z "${MASTER_PORT}" ] && MASTER_PORT=11459
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1

DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                  --master_port $MASTER_PORT \
                  --master_addr $MASTER_ADDR"


python -m torch.distributed.run $DISTRIBUTED_ARGS cell_fm/tasks/cell_fmc/pretrain_hpa.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --split_key $split_key \
            --phase $phase \
            --data_aug \
            --img_resize $img_resize \
            --img_crop_size $img_crop_size \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --cell_image $cell_image \
            --n_context_proteins $n_context_proteins \
            --context_embedding_dim $context_embedding_dim \
            --path_type $path_type \
            --prediction $prediction \
            --num_down_blocks $num_down_blocks \
            --latent_channels $latent_channels \
            --vae_block_out_channels $vae_block_out_channels \
            --cond_out_channels $cond_out_channels \
            --sample_size $sample_size \
            --esm_embedding $esm_embedding \
            --encoder_hidden_size $encoder_hidden_size \
            --max_protein_sequence_len $max_protein_sequence_len \
            --img_generator_num_layers $img_generator_num_layers \
            --img_generator_patch_size $img_generator_patch_size \
            --attention_head_dim $attention_head_dim \
            --num_attention_heads $num_attention_heads \
            --cell_image_ratio $cell_image_ratio \
            --img_diff_loss_coeff $img_diff_loss_coeff \
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
            --seed 666666 \
            --wandb \

            # --ft \
            # --ifresume \
            # --bf16 \
            # --fp16 \
