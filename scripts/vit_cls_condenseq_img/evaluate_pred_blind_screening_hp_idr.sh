ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir=.

# Dataset
[ -z "${data_path}" ] && data_path=/hpc/reference/opencell/condenseq
[ -z "${split_key}" ] && split_key=cellfm_test
[ -z "${phase}" ] && phase='test'  # 'train', 'test'

[ -z "${img_resize}" ] && img_resize=160
[ -z "${img_crop_size}" ] && img_crop_size=160
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.0
[ -z "${img_type}" ] && img_type='GFP'
[ -z "${cell_image}" ] && cell_image='nucl'

[ -z "${start_idx}" ] && start_idx=0
[ -z "${end_idx}" ] && end_idx=1000

# Model
[ -z "${in_channels}" ] && in_channels=2
[ -z "${patch_size}" ] && patch_size=4
[ -z "${num_layers}" ] && num_layers=8
[ -z "${num_heads}" ] && num_heads=8
[ -z "${hidden_dim}" ] && hidden_dim=512
[ -z "${mlp_dim}" ] && mlp_dim=2048

[ -z "${loadcheck_path}" ] && loadcheck_path=pretrain_condenseq_cellfm_split/PT_CondenSeq_img_ViT_cls_R1/checkpoint-10000/pytorch_model.bin

python cell_fm/tasks/vit_cls_condenseq_img/evaluate_blind_screening_hp_idr.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --split_key $split_key \
            --phase $phase \
            --img_resize $img_resize \
            --img_crop_size $img_crop_size \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --img_type $img_type \
            --cell_image $cell_image \
            --in_channels $in_channels \
            --patch_size $patch_size \
            --num_layers $num_layers \
            --num_heads $num_heads \
            --hidden_dim $hidden_dim \
            --mlp_dim $mlp_dim \
            --loadcheck_path $loadcheck_path \
            --seed 6 \
            --infer \
            --start_idx $start_idx \
            --end_idx $end_idx \

            # --ft \
            # --ifresume \
            # --bf16 \
            # --fp16 \