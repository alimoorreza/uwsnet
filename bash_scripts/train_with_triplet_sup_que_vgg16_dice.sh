shot=$1
split=$2
echo "../tools/train.py --config ../experiments/triplet_sup_que_vgg16_dice_config/${shot}_shot/uw_few_shot_training_config_triplet_sup_que_vgg16_dice_shot_${shot}_split_${split}.yaml"
python3 ../tools/train.py --config ../experiments/triplet_sup_que_vgg16_dice_config/${shot}_shot/uw_few_shot_training_config_triplet_sup_que_vgg16_dice_shot_${shot}_split_${split}.yaml

