import os.path as osp
import torch.utils.data as util_data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image



class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(osp.join(data_path, val.split()[0]), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)
    
    def get_labels(self):
        labels = []
        for img_path, label in self.imgs:
            labels.append(label)
        labels = np.stack(labels, axis = 0)
        labels = torch.tensor(labels)
        return labels

        



def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transformations

                            

def get_loaders(config):
    dset_name = config.dataset_name
    root_dir = osp.join('./data', dset_name)
    db_path = osp.join(root_dir, 'database.txt')
    t_path = osp.join(root_dir, "test.txt")
    t_image_list =  ImageList(root_dir, open(t_path).readlines(), transform= get_transforms())
    d_image_list = ImageList(root_dir, open(db_path).readlines(), transform= get_transforms())
    d_loader =  util_data.DataLoader(d_image_list, batch_size= config.train_batch_size,
                                                      shuffle=True, num_workers=4)
    t_loader =  util_data.DataLoader(t_image_list, batch_size= config.test_batch_size,
                                                      shuffle=True, num_workers=4)
    return d_image_list, t_image_list, d_image_list.get_labels(), t_image_list.get_labels()
                                            