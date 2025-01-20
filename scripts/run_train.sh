# CUDA_VISIBLE_DEVICES=0 python main.py --mode "heart_nab_binary" \
#     --seed 1 \
#     --project_name "PhysioNet2022" \
#     --description "h_nab_wo_denoise" \
#     --num_classes 2 \
#     --data_dir "/work/chs/school/paper/denoise/data" \
#     --tarfile "PhysioNet2022_v1.tar" \
#     --epoch 200 \
#     --backbone 'resnet38' \
#     --nmels 64 \
#     --transform_type 'mel' \
#     --batch 8 \
#     --samplerate 8000 \
#     --wd 1e-06 \
#     --lr 1e-4 \
#     --use_h5 \
#     --split_mode "patient" \

CUDA_VISIBLE_DEVICES=0 python main.py --mode "heart_nab_binary" \
    --seed 1 \
    --no_logging \
    --project_name "PhysioNet2022_exp1" \
    --duration 5 \
    --description "h_nab_wo_denoise" \
    --num_classes 2 \
    --data_dir "/work/chs/school/paper/denoise/data" \
    --tarfile "PhysioNet2022_v1.tar" \
    --epoch 5 \
    --backbone 'ast' \
    --nmels 64 \
    --transform_type 'mel' \
    --batch 8 \
    --samplerate 8000 \
    --wd 1e-06 \
    --lr 1e-4 \
    --use_h5 \
    --split_mode "patient" \
    --imagenet_pretrain \
    # --audioset_pretrain \
