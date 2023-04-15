shot=$1
split=$2
echo "../tools/train.py --config ../experiments/fpmms_config/${shot}_shot/uw_few_shot_training_config_fpmms_shot_${shot}_split_${split}.yaml"
python3 ../tools/train.py --config ../experiments/fpmms_config/${shot}_shot/uw_few_shot_training_config_fpmms_shot_${shot}_split_${split}.yaml

