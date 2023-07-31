import numpy as np
from scipy.io import loadmat, savemat
import os
import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,IterableDataset
from torchvision.transforms import Compose
import random
from PIL import Image
from glob import glob
import torchvision.transforms.functional as tr_F
from scipy import ndimage
import itertools
import pickle5 as pickle
from utils.seed_init import place_seed_points
import time


class IUDataset(torch.utils.data.Dataset):

    def __init__(self, directory, class2labels, labels_split,
                 test_label_split_value, episodes, nways, nshots,
                 validation=False, transform=None, random_split=True):
        self.directory = directory
        self.class2labels = class2labels
        self.labels_split = labels_split
        self.test_label = test_label_split_value
        self.random_split = random_split
        self.train_classes = [self.labels_split[i] for i in range(len(self.labels_split)) if i != self.test_label]
        self.train_classes = list(itertools.chain(*self.train_classes))
        self.test_classes = self.labels_split[self.test_label]
        self.class2files = dict((class_, self.__getclassdata(class_)) for class_ in (self.train_classes+self.test_classes))
        #print(self.class2files)
        self.episodes = episodes
        self.nways = nways
        self.nshots = nshots
        self.transform = transform
        self.validation = validation

    def __getclassdata(self, class_):
        return glob(os.path.join(self.directory, f"dataset/{class_}/*.mat"))

    def __get_indices_file_dir(self):
        return os.path.join(self.directory, 'dataset/indices_files/')

    def __randomchoose(self):
        if not self.validation:
            class_ = np.random.choice(self.train_classes, size=self.nways, replace=False)[0]
        else:
            class_ = np.random.choice(self.test_classes, size=self.nways, replace=False)[0]
        class_list = self.class2files[class_]
        sample_images = np.random.choice(class_list, size=self.nshots+1, replace=False)
        query_images = sample_images[0]
        support_images = sample_images[1:]
        return support_images, query_images, class_

    def __predefined_choose(self, idx):
        if not self.validation:
            path_of_indices_file = os.path.join(self.__get_indices_file_dir(),
                                                f'train_indices_{self.nshots}_shot_for_split_{self.test_label}.pickle')
            b_file = open(path_of_indices_file, "rb")
            predefined_indices = pickle.load(b_file)
            b_file.close()
        else:
            path_of_indices_file = os.path.join(self.__get_indices_file_dir(),
                                                f'val_indices_{self.nshots}_shot_for_split_{self.test_label}.pickle')
            b_file = open(path_of_indices_file, "rb")
            predefined_indices = pickle.load(b_file)
            b_file.close()
        class_ = predefined_indices[idx]['class']
        query_images = os.path.join(self.directory, f"dataset/{class_}/{predefined_indices[idx]['query_image']}")
        support_images = [os.path.join(
            self.directory,
            f"dataset/{class_}/{predefined_indices[idx]['support_images'][rp_i]}") for rp_i in range(self.nshots)]
        return support_images, query_images, class_

    def __getmask(self, array, class_):
        # max_ = np.max(array)
        # fg = np.where(array==max_,1,0).astype("uint8")
        # bg = np.where(array!=max_,1,0).astype("uint8")
        fg = np.where(array == self.class2labels[class_], 1, 0).astype("uint8")
        bg = np.where(array != self.class2labels[class_], 1, 0).astype("uint8")
        return Image.fromarray(fg), Image.fromarray(bg)

    def __loadimages(self,class_,support_name,query_name):
        query_ = loadmat(query_name)
        # loadmat(self.directory+f"dataset/{class_}/{query_name}")
        query_image = Image.fromarray(query_["image_array"])
        query_mask,_ = self.__getmask(query_["mask_array"],query_["class"][0])

        support_image,support_fg,support_bg = [],[],[]

        for support in support_name:
            support_ = loadmat(support)
            # loadmat(self.directory+f"dataset/{class_}/{support}")
            support_image.append(Image.fromarray(support_["image_array"]))
            # plt.imshow(support_image[0])
            # plt.show()
            fg, bg = self.__getmask(support_["mask_array"],support_["class"][0])
            support_fg.append(fg)
            support_bg.append(bg)

        return support_image,support_fg,support_bg,query_image,query_mask

    def __getitem__(self, idx):
        if self.random_split:
            support_images, query_images, class_ = self.__randomchoose()
        else:
            support_images, query_images, class_ = self.__predefined_choose(idx)
        # print(support_images)
        # print(query_images)
        # print(class_)
        sample = {}

        support_image, support_fg, support_bg, query_image, query_mask = self.__loadimages(
            class_, support_images, query_images
        )
        sample['support_image'], sample['support_fg_mask'], sample['support_bg_mask'] = support_image, \
                                                                                        support_fg, \
                                                                                        support_bg
        sample['query_image'] = [query_image]
        sample["query_label"] = [query_mask]
        sample["class"] = class_
        sample["sup_name"] = support_images
        sample["que_name"] = query_images
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, idx

    def __len__(self):
        return self.episodes


class IUDatasetASGNet(torch.utils.data.Dataset):

    def __init__(self, directory, class2labels, labels_split,
                 test_label_split_value, episodes, nways, nshots,
                 validation=False, transform=None, max_sp=5, random_split=True):
        self.directory = directory
        self.class2labels = class2labels
        self.labels_split = labels_split
        self.test_label = test_label_split_value
        self.train_classes = [self.labels_split[i] for i in range(len(self.labels_split)) if i != self.test_label]
        self.train_classes = list(itertools.chain(*self.train_classes))
        self.test_classes = self.labels_split[self.test_label]
        self.class2files = dict((class_, self.__getclassdata(class_)) for class_ in (self.train_classes+self.test_classes))
        #print(self.class2files)
        self.episodes = episodes
        self.nways = nways
        self.nshots = nshots
        self.transform = transform
        self.validation = validation
        self.max_sp = max_sp
        self.random_split = random_split

    def __getclassdata(self, class_):
        return glob(os.path.join(self.directory, f"dataset/{class_}/*.mat"))

    def __get_indices_file_dir(self):
        return os.path.join(self.directory, 'dataset/indices_files/')

    def __randomchoose(self):
        if not self.validation:
            class_ = np.random.choice(self.train_classes, size=self.nways, replace=False)[0]
        else:
            class_ = np.random.choice(self.test_classes, size=self.nways, replace=False)[0]
        class_list = self.class2files[class_]
        sample_images = np.random.choice(class_list, size=self.nshots+1, replace=False)
        query_images = sample_images[0]
        support_images = sample_images[1:]
        return support_images, query_images, class_

    def __predefined_choose(self, idx):
        if not self.validation:
            path_of_indices_file = os.path.join(self.__get_indices_file_dir(),
                                                f'train_indices_{self.nshots}_shot_for_split_{self.test_label}.pickle')
            b_file = open(path_of_indices_file, "rb")
            predefined_indices = pickle.load(b_file)
            b_file.close()
        else:
            path_of_indices_file = os.path.join(self.__get_indices_file_dir(),
                                                f'val_indices_{self.nshots}_shot_for_split_{self.test_label}.pickle')
            b_file = open(path_of_indices_file, "rb")
            predefined_indices = pickle.load(b_file)
            b_file.close()
        class_ = predefined_indices[idx]['class']
        query_images = os.path.join(self.directory, f"dataset/{class_}/{predefined_indices[idx]['query_image']}")
        support_images = [os.path.join(
            self.directory,
            f"dataset/{class_}/{predefined_indices[idx]['support_images'][rp_i]}") for rp_i in range(self.nshots)]
        return support_images, query_images, class_

    def __getmask(self, array, class_):
        # max_ = np.max(array)
        # fg = np.where(array==max_,1,0).astype("uint8")
        # bg = np.where(array!=max_,1,0).astype("uint8")
        fg = np.where(array == self.class2labels[class_], 1, 0).astype("uint8")
        bg = np.where(array != self.class2labels[class_], 1, 0).astype("uint8")
        return Image.fromarray(fg), Image.fromarray(bg)

    def __loadimages(self,class_,support_name,query_name):
        query_ = loadmat(query_name)
        # loadmat(self.directory+f"dataset/{class_}/{query_name}")
        query_image = Image.fromarray(query_["image_array"])
        query_mask,_ = self.__getmask(query_["mask_array"],query_["class"][0])

        support_image,support_fg,support_bg = [],[],[]
        for support in support_name:
            support_ = loadmat(support)
            # loadmat(self.directory+f"dataset/{class_}/{support}")
            support_image.append(Image.fromarray(support_["image_array"]))
            # plt.imshow(support_image[0])
            # plt.show()
            fg,bg = self.__getmask(support_["mask_array"],support_["class"][0])
            support_fg.append(fg)
            support_bg.append(bg)

        return support_image,support_fg,support_bg,query_image,query_mask

    def __getitem__(self, idx):
        if self.random_split:
            support_images, query_images, class_ = self.__randomchoose()
        else:
            support_images, query_images, class_ = self.__predefined_choose(idx)

        # print(class_)
        sample = {}

        support_image, support_fg, support_bg, query_image, query_mask = self.__loadimages(
            class_, support_images, query_images
        )

        sample['support_image'], sample['support_fg_mask'], sample['support_bg_mask'] = support_image, \
                                                                                        support_fg, \
                                                                                        support_bg
        sample['query_image'] = [query_image]
        sample["query_label"] = [query_mask]
        sample["class"] = class_
        if self.transform is not None:
            sample = self.transform(sample)

        s_y = sample['support_fg_mask'][0].unsqueeze(0)
        for i in range(1, self.nshots):
            s_y = torch.cat([sample['support_fg_mask'][i].unsqueeze(0), s_y], 0)

        init_seed_list = []
        for i in range(0, self.nshots):
            mask = (s_y[i, :, :] == 1).float()
            # print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            # print(mask.shape)
            # print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            # im = Image.fromarray(np.array(mask.detach().cpu().numpy())*255)
            # im = im.convert("L")
            # if not os.path.exists('./img/'):
            #     os.makedirs('./img/')
            # cnt = 0
            # filename = f"./img/mask_{cnt}.png"
            # while os.path.exists(filename):
            #     cnt += 1
            #     filename = f"./img/mask_{cnt}.png"
            # im.save(filename)
            init_seed = place_seed_points(mask, down_stride=8, max_num_sp=self.max_sp, avg_sp_area=100)
            # print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            # print(init_seed)
            # print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            init_seed_list.append(init_seed.unsqueeze(0))

        s_init_seed = torch.cat(init_seed_list, 0)
        sample['s_init_seed'] = s_init_seed
        return sample, idx

    def __len__(self):
        return self.episodes


class IUDatasetPFENet(torch.utils.data.Dataset):

    def __init__(self, directory, class2labels, labels_split,
                 test_label_split_value, episodes, nways, nshots,
                 validation=False, transform=None, random_split=True):
        self.directory = directory
        self.class2labels = class2labels
        self.labels_split = labels_split
        self.test_label = test_label_split_value
        self.train_classes = [self.labels_split[i] for i in range(len(self.labels_split)) if i != self.test_label]
        self.train_classes = list(itertools.chain(*self.train_classes))
        self.test_classes = self.labels_split[self.test_label]
        self.class2files = dict((class_, self.__getclassdata(class_)) for class_ in (self.train_classes+self.test_classes))
        #print(self.class2files)
        self.episodes = episodes
        self.nways = nways
        self.nshots = nshots
        self.transform = transform
        self.validation = validation
        self.random_split = random_split

    def __getclassdata(self, class_):
        return glob(os.path.join(self.directory, f"dataset/{class_}/*.mat"))

    def __get_indices_file_dir(self):
        return os.path.join(self.directory, 'dataset/indices_files/')

    def __randomchoose(self):
        if not self.validation:
            class_ = np.random.choice(self.train_classes, size=self.nways, replace=False)[0]
        else:
            class_ = np.random.choice(self.test_classes, size=self.nways, replace=False)[0]
        class_list = self.class2files[class_]
        sample_images = np.random.choice(class_list, size=self.nshots+1, replace=False)
        query_images = sample_images[0]
        support_images = sample_images[1:]
        return support_images, query_images, class_

    def __predefined_choose(self, idx):
        if not self.validation:
            path_of_indices_file = os.path.join(self.__get_indices_file_dir(),
                                                f'train_indices_{self.nshots}_shot_for_split_{self.test_label}.pickle')
            b_file = open(path_of_indices_file, "rb")
            predefined_indices = pickle.load(b_file)
            b_file.close()
        else:
            path_of_indices_file = os.path.join(self.__get_indices_file_dir(),
                                                f'val_indices_{self.nshots}_shot_for_split_{self.test_label}.pickle')
            b_file = open(path_of_indices_file, "rb")
            predefined_indices = pickle.load(b_file)
            b_file.close()
        class_ = predefined_indices[idx]['class']
        query_images = os.path.join(self.directory, f"dataset/{class_}/{predefined_indices[idx]['query_image']}")
        support_images = [os.path.join(
            self.directory,
            f"dataset/{class_}/{predefined_indices[idx]['support_images'][rp_i]}") for rp_i in range(self.nshots)]
        return support_images, query_images, class_

    def __getmask(self, array, class_):
        # max_ = np.max(array)
        # fg = np.where(array==max_,1,0).astype("uint8")
        # bg = np.where(array!=max_,1,0).astype("uint8")
        # ignore_pix = np.where(array == 255)
        fg = np.where(array == self.class2labels[class_], 1, 0).astype("uint8")
        bg = np.where(array != self.class2labels[class_], 1, 0).astype("uint8")
        # fg[ignore_pix[0], ignore_pix[1]] = 255
        # fg = fg.astype("uint8")
        return Image.fromarray(fg), Image.fromarray(bg)
        # return fg, bg

    def __loadimages(self,class_,support_name,query_name):
        query_ = loadmat(query_name)
        # loadmat(self.directory+f"dataset/{class_}/{query_name}")
        query_image = Image.fromarray(query_["image_array"])
        # query_image = query_["image_array"]
        query_mask, _ = self.__getmask(query_["mask_array"],query_["class"][0])

        support_image,support_fg,support_bg = [],[],[]
        for support in support_name:
            support_ = loadmat(support)
            # loadmat(self.directory+f"dataset/{class_}/{support}")
            support_image.append(Image.fromarray(support_["image_array"]))
            # support_image.append(support_["image_array"])
            # plt.imshow(support_image[0])
            # plt.show()
            fg, bg = self.__getmask(support_["mask_array"],support_["class"][0])
            support_fg.append(fg)
            support_bg.append(bg)

        return support_image,support_fg,support_bg,query_image,query_mask

    def __getitem__(self, idx):
        if self.random_split:
            support_images, query_images, class_ = self.__randomchoose()
        else:
            support_images, query_images, class_ = self.__predefined_choose(idx)
        # print(support_images)
        # print(query_images)
        # print(class_)
        sample = {}

        support_image, support_fg, support_bg, query_image, query_mask = self.__loadimages(
            class_, support_images, query_images
        )
        sample['support_image'], sample['support_fg_mask'], sample['support_bg_mask'] = support_image, \
                                                                                        support_fg, \
                                                                                        support_bg
        sample['query_image'] = [query_image]
        sample["query_label"] = [query_mask]
        sample["class"] = class_
        if self.transform is not None:
            sample = self.transform(sample)

        # print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(torch.max(sample['query_image'][0]), torch.min(sample['query_image'][0]))
        # print(torch.max(sample['query_label'][0]), torch.min(sample['query_label'][0]))
        # print(torch.max(sample['support_image'][0]), torch.min(sample['support_image'][0]))
        # print(torch.max(sample['support_fg_mask'][0]), torch.min(sample['support_fg_mask'][0]))
        # print(torch.max(sample['support_bg_mask'][0]), torch.min(sample['support_bg_mask'][0]))
        # print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        return sample, idx

    def __len__(self):
        return self.episodes


