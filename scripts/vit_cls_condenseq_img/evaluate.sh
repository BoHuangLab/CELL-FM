ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir=.

# Dataset
[ -z "${data_path}" ] && data_path=/hpc/reference/opencell/condenseq
[ -z "${split_key}" ] && split_key=test
[ -z "${img_resize}" ] && img_resize=160

# Model
[ -z "${in_channels}" ] && in_channels=2
[ -z "${patch_size}" ] && patch_size=4
[ -z "${num_layers}" ] && num_layers=8
[ -z "${num_heads}" ] && num_heads=8
[ -z "${hidden_dim}" ] && hidden_dim=512
[ -z "${mlp_dim}" ] && mlp_dim=2048

[ -z "${loadcheck_path}" ] && loadcheck_path=pretrain_condenseq_cellfm_split/PT_CondenSeq_img_ViT_cls_R1/checkpoint-10000/pytorch_model.bin

python -m torch.distributed.run $DISTRIBUTED_ARGS cell_fm/tasks/vit_cls_condenseq_img/evaluate.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --split_key $split_key \
            --img_resize $img_resize \
            --in_channels $in_channels \
            --patch_size $patch_size \
            --num_layers $num_layers \
            --num_heads $num_heads \
            --hidden_dim $hidden_dim \
            --mlp_dim $mlp_dim \
            --loadcheck_path $loadcheck_path \
            --seed 6 \
            --infer \

            # --ft \
            # --ifresume \
            # --bf16 \
            # --fp16 \