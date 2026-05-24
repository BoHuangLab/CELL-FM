ulimit -c unlimited
[ -z "${output_dir}" ] && output_dir=./output/hpa/cellfmc_seq2img

# Dataset
[ -z "${data_path}" ] && data_path=/hpc/reference/opencell/human_protein_atlas
[ -z "${split_key}" ] && split_key=cellfm_test
[ -z "${phase}" ] && phase='test'

[ -z "${img_resize}" ] && img_resize=256
[ -z "${img_crop_size}" ] && img_crop_size=1024
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.0
[ -z "${cell_image}" ] && cell_image='nucl,er,mt'
[ -z "${test_cell_image}" ] && test_cell_image='nucl,er,mt'

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

# Checkpoints
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=pretrain_hpa/vae/checkpoint-50000/pytorch_model.bin
[ -z "${loadcheck_path}" ] && loadcheck_path=pretrain_hpa/PT_HPA_CELLFMC_R1/checkpoint-10000/pytorch_model.bin

# Evaluation
[ -z "${num_steps}" ] && num_steps=100


python cell_fm/tasks/cell_fmc/generate_img_hpa.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --split_key $split_key \
            --phase $phase \
            --img_resize $img_resize \
            --img_crop_size $img_crop_size \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --cell_image $cell_image \
            --test_cell_image $test_cell_image \
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
            --vae_loadcheck_path $vae_loadcheck_path \
            --loadcheck_path $loadcheck_path \
            --seed 6 \
            --num_steps $num_steps \
            --infer \
