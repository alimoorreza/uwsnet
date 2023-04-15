shot=$1
split=$2
gpu=$3

pwd=$(pwd)
replace_part="GPU: [0-9]"
replace_with="GPU: ${gpu}"

filename="${pwd}/../experiments/asnet_config/shot_${shot}/uw_few_shot_training_config_asnet_shot_${shot}_split_${split}.yaml"
sed -i "" -e "s/${replace_part}/${replace_with}/g" $filename
echo "../tools/train.py --config ${filename}"
python3 ../tools/train.py --config ${filename}

