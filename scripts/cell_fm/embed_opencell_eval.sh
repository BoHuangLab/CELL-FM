ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir=./output/opencell/embedding/

# Dataset
[ -z "${data_path}" ] && data_path='/hpc/reference/opencell/opencell'
[ -z "${split_key}" ] && split_key=all

[ -z "${img_resize}" ] && img_resize=256
[ -z "${img_crop_size}" ] && img_crop_size=256
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.0
[ -z "${cell_image}" ] && cell_image='nucl'

# Transport parameters
[ -z "${path_type}" ] && path_type=Linear
[ -z "${prediction}" ] && prediction=velocity

# Model
## VAE
[ -z "${num_down_blocks}" ] && num_down_blocks=3
[ -z "${latent_channels}" ] && latent_channels=4
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='128,256,512'

## CELL-Diff
[ -z "${img_mask_ratio}" ] && img_mask_ratio=0
[ -z "${cond_out_channels}" ] && cond_out_channels='32,64'
[ -z "${sample_size}" ] && sample_size=64
[ -z "${esm_embedding}" ] && esm_embedding='esmc_300m'
[ -z "${encoder_hidden_size}" ] && encoder_hidden_size=960
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048
[ -z "${encoder_num_hidden_layers}" ] && encoder_num_hidden_layers=16
[ -z "${num_heads}" ] && num_heads=16
[ -z "${dim_head}" ] && dim_head=64
[ -z "${dropout}" ] && dropout=0
[ -z "${final_dropout}" ] && final_dropout=0
[ -z "${encoder_patch_size}" ] && encoder_patch_size=8

### Image generator
[ -z "${img_generator_num_layers}" ] && img_generator_num_layers=18
[ -z "${img_generator_patch_size}" ] && img_generator_patch_size=4
[ -z "${attention_head_dim}" ] && attention_head_dim=64
[ -z "${num_attention_heads}" ] && num_attention_heads=18

### Image decoder
[ -z "${img_decoder_num_hidden_layers}" ] && img_decoder_num_hidden_layers=8
[ -z "${img_decoder_hidden_size}" ] && img_decoder_hidden_size=512
[ -z "${img_decoder_num_heads}" ] && img_decoder_num_heads=8
[ -z "${img_decoder_dim_head}" ] && img_decoder_dim_head=64

[ -z "${cell_image_ratio}" ] && cell_image_ratio=0.5

# Training
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=finetune_opencell/FT_VAE_OC_256_KL1e-4_FP32/checkpoint-50000/pytorch_model.bin
[ -z "${loadcheck_path}" ] && loadcheck_path=finetune_opencell/old/FT_OC_CELLFM_S1_R1_all/checkpoint-100000

# Evaluation
[ -z "${num_steps}" ] && num_steps=100


python cell_fm/tasks/cell_fm/embed_opencell_eval.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --split_key $split_key \
            --img_resize $img_resize \
            --img_crop_size $img_crop_size \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --cell_image $cell_image \
            --path_type $path_type \
            --prediction $prediction \
            --num_down_blocks $num_down_blocks \
            --latent_channels $latent_channels \
            --vae_block_out_channels $vae_block_out_channels \
            --img_mask_ratio $img_mask_ratio \
            --cond_out_channels $cond_out_channels \
            --sample_size $sample_size \
            --esm_embedding $esm_embedding \
            --encoder_hidden_size $encoder_hidden_size \
            --max_protein_sequence_len $max_protein_sequence_len \
            --encoder_num_hidden_layers $encoder_num_hidden_layers \
            --num_heads $num_heads \
            --dim_head $dim_head \
            --dropout $dropout \
            --final_dropout $final_dropout \
            --encoder_patch_size $encoder_patch_size \
            --img_generator_num_layers $img_generator_num_layers \
            --img_generator_patch_size $img_generator_patch_size \
            --attention_head_dim $attention_head_dim \
            --num_attention_heads $num_attention_heads \
            --img_decoder_num_hidden_layers $img_decoder_num_hidden_layers \
            --img_decoder_hidden_size $img_decoder_hidden_size \
            --img_decoder_num_heads $img_decoder_num_heads \
            --img_decoder_dim_head $img_decoder_dim_head \
            --cell_image_ratio $cell_image_ratio \
            --vae_loadcheck_path $vae_loadcheck_path \
            --loadcheck_path $loadcheck_path \
            --seed 6 \
            --num_steps $num_steps \
            --infer \