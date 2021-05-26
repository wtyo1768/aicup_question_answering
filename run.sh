

export TOKENIZERS_PARALLELISM=false
export model_name ''
export fold_idx=1


python3 train.py \
    --model_name $model_name \
    --fold $fold_idx \
    --batch_size \
    --epoch \
    --lr  \
