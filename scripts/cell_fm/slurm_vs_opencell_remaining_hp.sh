#!/bin/bash

#SBATCH --job-name=predict
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --mem=128G
#SBATCH --constraint=h100|h200
#SBATCH --output=logs/vs_remaining_%A_%a.out
#SBATCH --error=logs/vs_remaining_%A_%a.err

module load anaconda
conda activate esm3

export PYTHONUNBUFFERED=1

cd /hpc/projects/group.huang/dihan.zheng/CELL-FM
mkdir -p logs

NUM_SHARDS=4   # keep in sync with --array upper bound + 1

nvidia-smi

# ── config (mirrors evaluate_virtual_staining_opencell_remaining_hp.sh) ──────
[ -z "${output_dir}" ] && output_dir=/hpc/reference/opencell/opencell/2d_proj_256_crop_virtual_staining_all_remaining_proteins/
[ -z "${data_path}" ]  && data_path='/hpc/reference/opencell/opencell'
[ -z "${split_key}" ]  && split_key=all
[ -z "${phase}" ]      && phase='test'

[ -z "${img_resize}" ]    && img_resize=256
[ -z "${img_crop_size}" ] && img_crop_size=256
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.0
[ -z "${cell_image}" ] && cell_image='nucl'

[ -z "${path_type}" ]   && path_type=Linear
[ -z "${prediction}" ]  && prediction=velocity

[ -z "${num_down_blocks}" ]          && num_down_blocks=3
[ -z "${latent_channels}" ]          && latent_channels=4
[ -z "${vae_block_out_channels}" ]   && vae_block_out_channels='128,256,512'

[ -z "${img_mask_ratio}" ]           && img_mask_ratio=0.5
[ -z "${cond_out_channels}" ]        && cond_out_channels='32,64'
[ -z "${sample_size}" ]              && sample_size=64
[ -z "${esm_embedding}" ]            && esm_embedding='esmc_600m'
[ -z "${encoder_hidden_size}" ]      && encoder_hidden_size=1152
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048
[ -z "${encoder_num_hidden_layers}" ] && encoder_num_hidden_layers=8
[ -z "${num_heads}" ]                && num_heads=8
[ -z "${dim_head}" ]                 && dim_head=64
[ -z "${dropout}" ]                  && dropout=0
[ -z "${final_dropout}" ]            && final_dropout=0
[ -z "${encoder_patch_size}" ]       && encoder_patch_size=4

[ -z "${img_generator_num_layers}" ]  && img_generator_num_layers=8
[ -z "${img_generator_patch_size}" ]  && img_generator_patch_size=2
[ -z "${attention_head_dim}" ]        && attention_head_dim=64
[ -z "${num_attention_heads}" ]       && num_attention_heads=18

[ -z "${img_decoder_num_hidden_layers}" ] && img_decoder_num_hidden_layers=4
[ -z "${img_decoder_hidden_size}" ]       && img_decoder_hidden_size=512
[ -z "${img_decoder_dim_head}" ]          && img_decoder_dim_head=64
[ -z "${img_decoder_num_heads}" ]         && img_decoder_num_heads=8

[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=finetune_opencell/vae/checkpoint-50000/pytorch_model.bin
[ -z "${loadcheck_path}" ]     && loadcheck_path=finetune_opencell/cellfm_vs/checkpoint-100000/pytorch_model.bin

[ -z "${num_steps}" ] && num_steps=100

# ─────────────────────────────────────────────────────────────────────────────

python cell_fm/tasks/cell_fm/virtual_staining_opencell_all_remaining_proteins.py \
    --output_dir $output_dir \
    --data_path $data_path \
    --split_key $split_key \
    --phase $phase \
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
    --vae_loadcheck_path $vae_loadcheck_path \
    --loadcheck_path $loadcheck_path \
    --seed 6 \
    --num_steps $num_steps \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS \
    --infer
