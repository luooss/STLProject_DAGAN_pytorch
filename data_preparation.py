import torch
# from torchvision import datasets, transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
import cv2
from scipy.io import loadmat
import os
import shutil


def find_classes(root_dir):
    retour = []
    # print('1', root_dir)
    for (root, dirs, files) in os.walk(root_dir):
        # print('origin file',files)
        files.sort()
        for f in files:
            if (f.endswith("jpg")):
                # if (f.endswith("jpg")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    # [('1.jpg', d/d, full_path),...]
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    # {'d/d': 0, '': 1, }
    return idx


class FIGR_Omniglot(data.Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.all_items = find_classes(self.root)
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0] # '1.jpg'
        img = str.join('/', [self.all_items[index][2], filename]) # 'path/1.jpg'
        target = self.idx_classes[self.all_items[index][1]] # label 0
        print(img,target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target # (width, height, channel), 0

    def __len__(self):
        return len(self.all_items)


def one_channel_preparation(data_dir, dataset, length, channels, batch_size):
    train_loader_1 = FIGR_Omniglot(data_dir, dataset,
                                   transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                                 lambda x: x.resize((length, length)),
                                                                 lambda x: np.reshape(x, (channels, length, length)),
                                                                 ]))
    train_loader = torch.utils.data.DataLoader(train_loader_1, batch_size=batch_size, shuffle=True)
    return train_loader


def three_channel_preparation(data_dir, length, batch_size):
    transform = transforms.Compose([
        transforms.Scale(length),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dset = datasets.ImageFolder(data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle=False)
    return train_loader


def three_channel_evaluation(data_dir, dataset, length, channels=1):
    dset = FIGR_Omniglot(data_dir, dataset, transform=transforms.Compose([lambda x: cv2.imread(x),
                                                                          lambda x: cv2.resize(x, (length, length),
                                                                                               interpolation=cv2.INTER_LINEAR)

                                                                          ]))
    return dset


def generate_image_label_pairs(data_set, store_path, image_size, each_class_total_samples=8):
    if data_set == 'Omniglot':
        data_dir = ['/home/PublicDir/luoshuai/Omniglot/python/images_background',
                    '/home/PublicDir/luoshuai/Omniglot/python/images_background_small1',
                    '/home/PublicDir/luoshuai/Omniglot/python/images_background_small2',
                    '/home/PublicDir/luoshuai/Omniglot/python/images_evaluation']
        channels = 1
        image_transform = transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                        lambda x: x.resize((image_size, image_size)),
                                                        lambda x: np.reshape(x, (image_size, image_size, channels)),
                                                        ])
        data_loader = []
        languages = []
        for dir in data_dir:
            for language in os.listdir(dir):
                if language in languages:
                    continue
                print(language)
                languages.append(language)
                for character in os.listdir(os.path.join(dir, language)):
                    images_under_this_category = []
                    for image in os.listdir(os.path.join(dir, language, character)):
                        images_under_this_category.append(image_transform(os.path.join(dir, language, character, image)))
                    data_loader.append(images_under_this_category)

    elif data_set == 'EMNIST':
        pass
    elif data_set == 'VGGFace':
        pass
    elif data_set == 'AnimalFaces':
        pass
    elif data_set == 'Flowers':
        pass
    elif data_set == 'NABbirds':
        pass

    npy_file_path = os.path.join(store_path, '{}.npy'.format(data_set))
    print('npy_file_path:', npy_file_path)

    data_loader = np.array(data_loader)
    shuffle_classes = np.arange(data_loader.shape[0])
    np.random.shuffle(shuffle_classes)
    data_loader = np.array([data_loader[i][:each_class_total_samples]
                            for i in shuffle_classes if data_loader[i].shape[0] >= each_class_total_samples])
    print('data_set shape:', data_loader.shape) # (1623, 8, 28, 28, 1)
    np.save(npy_file_path, data_loader)


def EMNIST():
    filename = [
        ["training_images", "./datasets/gzip/emnist-balanced-train-images-idx3-ubyte.gz"],
        ["test_images", "./datasets/gzip/emnist-balanced-test-images-idx3-ubyte.gz"],
        ["training_labels", "./datasets/gzip/emnist-balanced-train-labels-idx1-ubyte.gz"],
        ["test_labels", "./datasets/gzip/emnist-balanced-test-labels-idx1-ubyte.gz"]
    ]

    # filename = [
    # ["training_images","emnist-balanced-train-images-idx3-ubyte"],
    # ["test_images","emnist-balanced-test-images-idx3-ubyte"],
    # ["training_labels","emnist-balanced-train-labels-idx1-ubyte"],
    # ["test_labels","emnist-balanced-test-labels-idx1-ubyte"]
    # ]

    emnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            emnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            emnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("emnist.pkl", 'wb') as f:
        pickle.dump(emnist, f)
    print("Save complete.")

    x_train, y_train, x_test, y_test = emnist["training_images"], emnist["training_labels"], emnist["test_images"], \
                                       emnist["test_labels"]
    x_train = np.reshape(x_train, [np.shape(x_train)[0], 28, 28, 1])
    x_test = np.reshape(x_test, [np.shape(x_test)[0], 28, 28, 1])
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    save_file_data = './datasets/emnist.npy'
    final_data = []
    if not os.path.isfile(save_file_data):
        temp = dict()
        for i in range(np.shape(x)[0]):
            if y[i] in temp:
                temp[y[i]].append(x[i])
            else:
                temp[y[i]] = [x[i]]

        for classes in temp.keys():
            final_data.append(np.array(temp[list(temp.keys())[classes]][:6300]))
        final_data = np.array(final_data)
        temp = []
        # print('data',np.shape(final_data)) (47, 2800, 28, 28, 1)
    np.save(save_file_data, final_data)


def flowers(image_dir, label_dir, save_dir):
    label = loadmat(label_dir)
    flower_labels = list(label['labels'][0])

    for index_real, item in enumerate(flower_labels):
        index = index_real + 1
        if os.path.exists(save_dir + '/{}'.format(item)):
            flag = int(index / 10)
            if flag < 1:
                shutil.move(image_dir + 'image_0000{}.jpg'.format(index),
                            save_dir + '/{}/image_0000{}.jpg'.format(item, index))
            elif 1 <= flag < 10:
                shutil.move(image_dir + 'image_000{}.jpg'.format(index),
                            save_dir + '/{}/image_000{}.jpg'.format(item, index))
            elif 10 <= flag < 100:
                shutil.move(image_dir + 'image_00{}.jpg'.format(index),
                            save_dir + '/{}/image_00{}.jpg'.format(item, index))
            elif 100 <= flag < 1000:
                shutil.move(image_dir + 'image_0{}.jpg'.format(index),
                            save_dir + '/{}/image_0{}.jpg'.format(item, index))
        else:
            os.mkdir(save_dir + '/{}'.format(item))
            flag = int(index / 10)
            if flag < 1:
                shutil.move(image_dir + 'image_0000{}.jpg'.format(index),
                            save_dir + '/{}/image_0000{}.jpg'.format(item, index))
            elif 1 <= flag < 10:
                shutil.move(image_dir + 'image_000{}.jpg'.format(index),
                            save_dir + '/{}/image_000{}.jpg'.format(item, index))
            elif 10 <= flag < 100:
                shutil.move(image_dir + 'image_00{}.jpg'.format(index),
                            save_dir + '/{}/image_00{}.jpg'.format(item, index))
            elif 100 <= flag < 1000:
                shutil.move(image_dir + 'image_0{}.jpg'.format(index),
                            save_dir + '/{}/image_0{}.jpg'.format(item, index))


import argparse

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')

parser.add_argument('--data_set', nargs="?", type=str, default='Omniglot')
parser.add_argument('--store_path', nargs="?", type=str,
                    default='/home/PublicDir/luoshuai/data_preparation')
# the desired image_size after transformation
parser.add_argument('--image_width', nargs="?", type=int, default=28)
parser.add_argument('--each_class_total_samples', nargs="?", type=int, default=8)

args = parser.parse_args()

generate_image_label_pairs(data_set=args.data_set, store_path=args.store_path,
                           image_size=args.image_width, each_class_total_samples=args.each_class_total_samples)

