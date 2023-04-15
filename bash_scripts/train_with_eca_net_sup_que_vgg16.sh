shot=$1
split=$2
echo "../tools/train.py --config ../experiments/eca_net_sup_que_vgg16_config/${shot}_shot/uw_few_shot_training_config_eca_net_sup_que_vghg16_shot_${shot}_split_${split}.yaml"
python3 ../tools/train.py --config ../experiments/eca_net_sup_que_vgg16_config/${shot}_shot/uw_few_shot_training_config_eca_net_sup_que_vghg16_shot_${shot}_split_${split}.yaml