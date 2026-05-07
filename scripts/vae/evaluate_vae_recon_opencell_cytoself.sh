ulimit -c unlimited

[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir='./output/opencell_cytoself/FT_VAE_OC_KL1e-4_FP32'

# Dataset
[ -z "${data_path}" ] && data_path='/hpc/reference/opencell/cytoself_dataset/packed_dataset'
[ -z "${split_key}" ] && split_key='test'

[ -z "${img_crop_size}" ] && img_crop_size=100
[ -z "${img_resize}" ] && img_resize=128

# Model
[ -z "${num_down_blocks}" ] && num_down_blocks=3
[ -z "${latent_channels}" ] && latent_channels=4
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='128,256,512'

# Loss
[ -z "${recon_loss_coeff}" ] && recon_loss_coeff=1.0
[ -z "${kl_loss_coeff}" ] && kl_loss_coeff=1.0

# Training
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=finetune_opencell/FT_VAE_OC_Cytoself_128_KL1e-4_FP32/checkpoint-50000/pytorch_model.bin
[ -z "${learning_rate}" ] && learning_rate=3e-4
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${per_device_train_batch_size}" ] && per_device_train_batch_size=64
[ -z "${per_device_eval_batch_size}" ] && per_device_eval_batch_size=128

[ -z "${num_train_epochs}" ] && num_train_epochs=5000
[ -z "${logging_dir}" ] && logging_dir=$output_dir
[ -z "${logging_steps}" ] && logging_steps=100
[ -z "${warmup_steps}" ] && warmup_steps=1000
[ -z "${max_steps}" ] && max_steps=50000
[ -z "${save_steps}" ] && save_steps=500


python cell_fm/tasks/vae/evaluate_vae_recon_opencell_cytoself.py \
    --output_dir $output_dir \
    --data_path $data_path \
    --split_key $split_key \
    --img_crop_size $img_crop_size \
    --img_resize $img_resize \
    --num_down_blocks $num_down_blocks \
    --latent_channels $latent_channels \
    --vae_block_out_channels $vae_block_out_channels \
    --recon_loss_coeff $recon_loss_coeff \
    --kl_loss_coeff $kl_loss_coeff \
    --vae_loadcheck_path $vae_loadcheck_path \
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
    --infer \

    # --bf16 \
    # --ifresume \