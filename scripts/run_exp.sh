SEEDS=(1 22 42 123 300 815 2025 3030 5555 8000)
MODES=("heart_nab_binary")
PROJECT_NAMES=("PhysioNet2022_nab_wo_denoise")
SPLIT_MODES=("patient")
DATA_DIRS=("/work/chs/school/paper/denoise/data")
TARFILES=("PhysioNet2022_v1.tar")
EPOCHS=(200)
NMELS=(64)
BATCHES=(8)
BACKBONES=("resnet38" "ast")
DURATIONS=(3 5 8)
TRANSFORM_TYPES=("mel" "fbank")
SAMPLERATES=(8000 16000)
LRS=(0.0001 5e-05)
DROPOUTS=(true false)
IMGNET=(true false)
AUDIOSET=(true false)

# Iterate through each tarfile
for SEED in "${SEEDS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for PROJECT_NAME in "${PROJECT_NAMES[@]}"; do
            for SPLIT_MODE in "${SPLIT_MODES[@]}"; do
                for DATA_DIR in "${DATA_DIRS[@]}"; do
                    for TARFILE in "${TARFILES[@]}"; do
                        for EPOCH in "${EPOCHS[@]}"; do
                            for NMEL in "${NMELS[@]}"; do
                                for BATCH in "${BATCHES[@]}"; do
                                    for BACKBONE in "${BACKBONES[@]}"; do
                                        for DURATION in "${DURATIONS[@]}"; do
                                            for TRANSFORM_TYPE in "${TRANSFORM_TYPES[@]}"; do
                                                for SAMPLERATE in "${SAMPLERATES[@]}"; do
                                                    for LR in "${LRS[@]}"; do
                                                        num_classes=""
                                                        if [ "$MODE" = "heart_nab_binary" ]; then
                                                            num_classes=2
                                                        fi
                                                        if [ "$MODE" = "sound" ]; then
                                                            num_classes=4
                                                        fi
                                                        if [ "$MODE" = "all" ]; then
                                                            num_classes=6
                                                        fi

                                                        dropout=""
                                                        if [ "$DROPOUTS" = true ]; then
                                                            dropout="--dropout"
                                                        fi

                                                        imagenet=""
                                                        if [ "$IMGNET" = true ]; then
                                                            imagenet="--imagenet_pretrain"
                                                        fi
                                                        
                                                        audioset=""
                                                        if [ "$AUDIOSET" = true ]; then
                                                            audioset="--audioset_pretrain"
                                                        fi

                                                        #CUDA_VISIBLE_DEVICES=0 
                                                        python main.py --mode "${MODE}" \
                                                            --project_name $PROJECT_NAME \
                                                            --description "${SEED}" \
                                                            --group "${BACKBONE}" \
                                                            --split_mode $SPLIT_MODE \
                                                            --data_dir $DATA_DIR \
                                                            --tarfile $TARFILE \
                                                            --epoch $EPOCH \
                                                            --backbone $BACKBONE \
                                                            --nmels $NMEL \
                                                            --transform_type $TRANSFORM_TYPE \
                                                            --batch $BATCH \
                                                            --samplerate $SAMPLERATE \
                                                            --lr $LR \
                                                            --use_h5 \
                                                            --num_classes $num_classes \
                                                            $dropout \
                                                            $imagenet \
                                                            $audioset \
                                                            
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done