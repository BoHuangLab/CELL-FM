ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir=./output/opencell/embedding/

# Dataset
[ -z "${data_path}" ] && data_path='/hpc/reference/opencell/opencell'
[ -z "${phase}" ] && phase='test'  # 'train', 'val', 'test'

[ -z "${img_resize}" ] && img_resize=256
[ -z "${img_crop_size}" ] && img_crop_size=256

[ -z "${cell_image}" ] && cell_image='nucl'  # 'nucl', 'nucl,er', 'nucl,mt', 'nucl,er,mt'

# Model
[ -z "${patch_size}" ] && patch_size=4
[ -z "${num_layers}" ] && num_layers=12
[ -z "${num_heads}" ] && num_heads=8
[ -z "${hidden_dim}" ] && hidden_dim=512
[ -z "${mlp_dim}" ] && mlp_dim=2048

# Training
[ -z "${loadcheck_path}" ] && loadcheck_path=finetune_opencell/PT_OC_ViT_cls_R1/checkpoint-10000/pytorch_model.bin


python cell_fm/tasks/vit_cls/embed_opencell_compare.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --phase $phase \
            --normalize \
            --img_resize $img_resize \
            --img_crop_size $img_crop_size \
            --cell_image $cell_image \
            --patch_size $patch_size \
            --num_layers $num_layers \
            --num_heads $num_heads \
            --hidden_dim $hidden_dim \
            --mlp_dim $mlp_dim \
            --loadcheck_path $loadcheck_path \
            --seed 6 \
            --infer \
