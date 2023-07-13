import os


log_dir = '/kaggle/working/log/'
output_dir = '/kaggle/working/output/'
base_dir = '/kaggle/working/underwater_few_shot/'
dataset_root = '/kaggle/input/under-water-few-shot/'
vgg_model_path = '/kaggle/input/resnet-pretrained/vgg16-397923af.pth'

net_types = {
    'new_net': {
        'basic': [],
        'ecanet': [],
        'ecanet_vgg16': [],
        'triplet': [],
        'triplet_vgg16': [],
        'triplet_vgg16_dice': [],
        'triplet_dice': []
    },
    'new_net_3': {
        'basic': {'vgg16': [], 'resnet50': [], 'resnet101': []},
        'ecanet': {'vgg16': [], 'resnet50': [], 'resnet101': []},
        'triplet': {'vgg16': [], 'resnet50': [], 'resnet101': []},
        'triplet_dice': {'vgg16': [], 'resnet50': [], 'resnet101': []}
    },
    'new_net_4': {
        'basic': [],
        'ecanet': [],
        'ecanet_vgg16': [],
        'triplet': [],
        'triplet_vgg16': [],
        'triplet_vgg16_dice': [],
        'triplet_dice': []
    }
}

shots = ['1', '5']
splits = ['0', '1', '2', '3']

with open('template_config_new_nets.yaml', 'r') as f:
    template_data = f.read()

template_data = template_data.replace('__log_dir__', log_dir)
template_data = template_data.replace('__output_dir__', output_dir)
template_data = template_data.replace('__base_dir__', base_dir)
template_data = template_data.replace('__dataset_root__', dataset_root)
template_data = template_data.replace('__vgg_model_path__', vgg_model_path)

for net_type in net_types.keys():
    if net_types[net_type]:
        for key in net_types[net_type].keys():
            pth_r = os.path.join(f'{net_type}_config', key)
            if net_types[net_type][key]:
                for key2 in net_types[net_type][key].keys():
                    for s in shots:
                        shot = f'shot_{s}'
                        pth = os.path.join(pth_r, key2, shot)
                        if not os.path.exists(pth):
                            os.makedirs(pth)
                        for split in splits:
                            file_name = f'uw_few_shot_training_config_{net_type}_{key}_{key2}_{shot}_split_{split}.yaml'
                            full_path = os.path.join(pth, file_name)
                            file_data = template_data.replace('__split__', split)
                            file_data = file_data.replace('__shot__', s)
                            file_data = file_data.replace('__net_type__', f'{net_type}_{key}')
                            file_data = file_data.replace('__backbone_type__', f'{key2}')
                            with open(full_path, 'w') as f:
                                f.write(file_data)
            else:
                for s in shots:
                    shot = f'shot_{s}'
                    pth = os.path.join(pth_r, shot)
                    if not os.path.exists(pth):
                        os.makedirs(pth)
                    for split in splits:
                        file_name = f'uw_few_shot_training_config_{net_type}_{key}_{shot}_split_{split}.yaml'
                        full_path = os.path.join(pth, file_name)
                        file_data = template_data.replace('__split__', split)
                        file_data = file_data.replace('__shot__', s)
                        file_data = file_data.replace('__net_type__', f'{net_type}_{key}')
                        file_data = file_data.replace('__backbone_type__', f'')
                        with open(full_path, 'w') as f:
                            f.write(file_data)

