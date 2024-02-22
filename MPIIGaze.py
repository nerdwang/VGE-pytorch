import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2  # OpenCV for image processing
from scipy.ndimage import affine_transform
from scipy.stats import truncnorm
from util.gazemap import from_gaze2d


class MPIIGazeDataset(Dataset):
    def __init__(self, hdf_path, keys_to_use, mode='train', eye_image_shape=(36,60), data_format='NCHW'):
        self.hdf_path = hdf_path
        self.eye_image_shape = eye_image_shape
        self.data_format = data_format
        self.keys_to_use = keys_to_use
        self.mode = mode

        self.index_to_key = {}
        index_counter = 0
        with h5py.File(self.hdf_path, 'r') as f:
            for key in self.keys_to_use:
                n = f[f"{key}/eye"].shape[0]
                for i in range(n):
                    self.index_to_key[index_counter] = (key, i)
                    index_counter += 1
        self.num_samples = index_counter

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf_path, 'r') as f:
            key, index = self.index_to_key[idx]
            entry = {}
            entry['eye'] = f[f"{key}/eye"][index]
            entry['gaze'] = f[f"{key}/gaze"][index]
            entry = self.preprocess_entry(entry)
            
            # Converting all values to PyTorch tensors
            for key, value in entry.items():
                entry[key] = torch.tensor(value, dtype=torch.float32)

            return entry['eye'], entry['gaze'], entry['gazemaps']
    
    def _augment_training_images(self, eye):#这里数据增强函数中affine_transform用法错误，应该传入变换矩阵的逆，
                                            #而非变换矩阵，导致这里的数据增强严重破坏了训练数据。我已经把数据增强的代码注释掉，之后会修改这里的逻辑
        if self.mode == 'test':
            return eye 
        if self.mode == 'train':
            # 获取图像的高度和宽度
            _, h, w = eye.shape
    
            # 生成截断正态分布的平移量
            def truncated_normal(mean, stddev, low, high):
                return truncnorm((low - mean) / stddev, (high - mean) / stddev, loc=mean, scale=stddev).rvs()
    
            tx = truncated_normal(mean=0, stddev=0.05 * w, low=-0.05 * w, high=0.05 * w)
            ty = truncated_normal(mean=0, stddev=0.05 * h, low=-0.05 * h, high=0.05 * h)
    
            # 创建齐次坐标变换矩阵
            affine_matrix = np.array([[1, 0, tx],
                                      [0, 1, ty],
                                      [0, 0, 1]])
    
            # 应用仿射变换
            eye_transformed = affine_transform(eye, affine_matrix, order=1, mode='constant')
    
            return eye_transformed
        
    def preprocess_entry(self, entry):
        """Resize eye image and normalize intensities."""
        oh, ow = self.eye_image_shape
        eye = entry['eye']
        eye = cv2.resize(eye, (ow, oh))
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, axis=0 if self.data_format == 'NCHW' else -1)
        entry['eye'] = eye

       
        entry['gazemaps'] = from_gaze2d(
            entry['gaze'], output_size=(oh, ow), scale=1,
        ).astype(np.float32)

        if self.data_format == 'NHWC':
            entry['gazemaps'] = np.transpose(entry['gazemaps'], (1, 2, 0))

        #entry['eye']=self._augment_training_images(entry['eye'])
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)
        
        

        return entry


i=0
person_id = 'p%02d' % i
other_person_ids = ['p%02d' % j for j in range(15) if i != j]
train_data=MPIIGazeDataset(hdf_path="/home/ubuntu/VGE-Net/datasets/MPIIGaze.h5",keys_to_use=['train/' + s for s in other_person_ids])
test_data=MPIIGazeDataset(hdf_path="/home/ubuntu/VGE-Net/datasets/MPIIGaze.h5",keys_to_use=['test/' + person_id])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
# 创建一个数据加载器的迭代器
train_loader_iter = iter(train_loader)

# 从迭代器中获取一个批次的数据
batch = next(train_loader_iter)

# 打印这个批次的数据
eye_batch, gaze_batch, gazemaps_batch = batch
print("Eye batch shape:", eye_batch.shape)
print("Gaze batch shape:", gaze_batch.shape)
print("Gazemaps batch shape:", gazemaps_batch.shape)