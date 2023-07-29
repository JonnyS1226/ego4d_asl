yaml_name=$1
CUDA_VISIBLE_DEVICES=$2 python ./train.py ./configs/${yaml_name}.yaml