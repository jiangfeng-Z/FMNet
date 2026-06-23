import numpy as np
import torch, torchvision
import torch.nn.functional as F
import random
from scipy import  io
from torch.utils.data import Dataset, DataLoader
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2,h5py,math, time
from os import listdir
from os.path import isfile, join
from augmentation import *
from tqdm import tqdm

def DDFF_vertical_flip(x,depth, random_val):
    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,0).copy()
    return x, depth

def DDFF_rotate(x,depth,degree):
    x=np.rot90(x,degree,axes=(1,2)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    return x,depth

def DDFF_horizontal_flip(x,depth, random_val):
    if(random_val>0.5):
        x=np.flip(x,2).copy()
        depth=np.flip(depth,1).copy()
    return x,depth

def DDFF_DFV_randcrop_3d(x, depth, x_seeds, y_seeds, interval_x, interval_y):
    x = x[:, y_seeds:y_seeds - interval_y, x_seeds:x_seeds - interval_x, :]  # [N, H, W, C]
    depth = depth[y_seeds:y_seeds - interval_y, x_seeds:x_seeds - interval_x]
    return x, depth

class FocalStackDDFFH5Reader_DFV(Dataset):
    def __init__(self, hdf5_filename, stack_key="stack_train", disp_key="disp_train"):
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        focal_length = 521.4052
        K2 = 1982.0250823695178
        flens = 7317.020641763665
        baseline = K2 / flens * 1e-3
        self.focus_dists = np.linspace(baseline * focal_length / 0.5, baseline * focal_length / 7, num=10)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=2).astype(np.float32)

        self.input_size = (383, 552)
        self.train_size = (224, 224)
        self.cropping = (self.input_size[0] - self.train_size[0], self.input_size[1] - self.train_size[1])

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        # Create sample dict
        FS = self.hdf5[self.stack_key][idx].astype(np.float32)
        gt = self.hdf5[self.disp_key][idx].astype(np.float32)
        if self.stack_key == "stack_train":
            y_crop, x_crop, contrast, brightness, gamma, flip_x, flip_y, angle = self.get_seeds()
            FS, gt = DDFF_DFV_randcrop_3d(FS, gt, x_crop, y_crop, self.cropping[1], self.cropping[0])
            FS = image_augmentation(FS, contrast, brightness, gamma)
            FS, gt = DDFF_horizontal_flip(FS, gt, flip_x)
            FS, gt = DDFF_vertical_flip(FS, gt, flip_y)
            FS, gt = DDFF_rotate(FS, gt, angle)
            self.Focus_Dists = torch.Tensor(np.tile(self.focus_dists, [1, self.train_size[0], self.train_size[1]]))
        elif self.stack_key == "stack_val":
            FS = FS / 127.5 - 1.0
            N, H, W, C = FS.shape
            if H % 32 != 0:
                pad_h = 32 - (H % 32)
            else:
                pad_h = 0
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
            else:
                pad_w = 0

            FS = np.pad(FS, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=(-1, -1))
            self.Focus_Dists = torch.Tensor(np.tile(self.focus_dists, [1, H + pad_h, W + pad_w]))

        # AiFDetphNet
        mask = gt > 0

        FS = torch.from_numpy(np.transpose(FS, (3, 0, 1, 2)))
        gt = torch.from_numpy(gt)

        return FS, gt, self.Focus_Dists, mask

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]

    def get_seeds(self):
        return (
            random.randint(0, self.cropping[0] - 1), random.randint(0, self.cropping[1] - 1),
            random.uniform(0.4, 1.6),
            random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0),
            random.uniform(0, 1.0), random.randint(0, 3))

# dataroot='/data/ddff-dataset-trainval.h5'
# dataroot = "/data/ddff_trainVal.h5"
# train_dataset=FocalStackDDFFH5Reader_DFV(dataroot, stack_key="stack_train", disp_key="disp_train")
# valid_dataset=FocalStackDDFFH5Reader_DFV(dataroot, stack_key="stack_val", disp_key="disp_val")
# dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)
# valid_dataloader=DataLoader(valid_dataset,1,shuffle=False)
# for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
#             train_input, train_gt_depth , train_focus_dists, train_mask = samples
#             print(train_input.shape)
#             exit()

class FS6_dataset(Dataset):
    def __init__(self,mode):
        self.root = "D:/Dataset/FoD500/" + mode + "/"
        self.imglist_all = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "Dpt.txt"]
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.max_depth = 3.0
        focus_dists = np.array([0.1,0.15,0.3,0.7,1.5])
        focus_dists = np.expand_dims(focus_dists,axis=1)
        focus_dists = np.expand_dims(focus_dists,axis=2).astype(np.float32)
        self.mode=mode
        self.Focus_Dists = torch.Tensor(np.tile(focus_dists,[1,256,256]))
    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, index):
        img_dpt=  np.loadtxt(self.root + self.imglist_dpt[index], delimiter=',')
        
        contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
        img_index = index * 5
        mats_input = np.zeros((256, 256, 3, 0))
        for i in range(5):
            img = cv2.imread(self.root + self.imglist_all[img_index + i])
            mats_input = np.concatenate((mats_input,np.expand_dims(img,axis=-1)), axis=3)
        
        if self.mode=="train":
            mats_input=image_augmentation(mats_input,contrast,brightness,gamma)
            mats_input,img_dpt = horizontal_flip(mats_input,img_dpt,flip_x)
            mats_input,img_dpt = vertical_flip(mats_input,img_dpt,flip_y)
            mats_input,img_dpt = rotate(mats_input,img_dpt,angle)
            img_dpt[img_dpt< 0.0] = 0.0
            img_dpt[img_dpt > 2.0] = 0.0

        elif self.mode == "test":
            mats_input = mats_input/127.5 -1.0
            img_dpt[img_dpt< 0.1] = 0.0
            img_dpt[img_dpt > 1.5] = 0.0

        mats_input = np.transpose(mats_input,(2,3,0,1))
        
        mask = img_dpt > 0
        img_dpt= torch.Tensor(img_dpt)
        mats_input=torch.Tensor(mats_input)

        return mats_input, img_dpt, self.Focus_Dists, mask
    
    def get_seeds(self):
        return (random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))

# train_dataset=FS6_dataset('train')
# valid_dataset=FS6_dataset('test')
# dataloader=DataLoader(train_dataset,batch_size=4,shuffle=True, pin_memory=True)
# num_train=len(dataloader)
# valid_dataloader=DataLoader(valid_dataset,1,shuffle=False,pin_memory=True)
# num_val=len(valid_dataloader)
# for idx, samples in enumerate(tqdm(dataloader,desc="valid")):
#     valid_input, test_gt_depth, test_mask, test_focus_dists = samples
#     print(valid_input.shape, test_gt_depth.shape, test_mask.shape, test_focus_dists.shape)

class FlyingThings3d(Dataset):
    def __init__(self,mode):
        self.mode = mode
        self.train_size=(256,256)
        self.num_imgs=15
        self.input_size=(540,960)
        self.cropping = (self.input_size[0] - self.train_size[0],self.input_size[1] - self.train_size[1])
        self.rgb_paths = [[] for i in range(self.num_imgs)]
        self.disp_paths = []
        self.low_bound = 10
        self.high_bound = 100
        self.focus_dists = np.linspace(self.low_bound,self.high_bound,self.num_imgs)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=2).astype(np.float32)
        with open (r"D:/Dataset/FlyingThings3D_FS/"+ mode + "/focal_stack_path.txt",'r') as f:
            for line in tqdm(f.readlines(),desc="flyingthings"):
                tmp = line.strip().split()
                for i in range(self.num_imgs):
                    self.rgb_paths[i].append(tmp[i])
                self.disp_paths.append(tmp[-1])
        
    def __len__(self):
        return len(self.disp_paths)

    def __getitem__(self,idx):#TEST/Train
        depth_path = self.disp_paths[idx]
        ext = os.path.splitext(depth_path)[1].lower()
        depth = None
        try:
            if ext in ('.txt', '.csv'):
                try:
                    depth = np.loadtxt(depth_path, delimiter=',')
                except Exception:
                    depth = np.loadtxt(depth_path)
            elif ext == '.npy':
                depth = np.load(depth_path)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is not None and depth.ndim == 3:
                    depth = depth[:, :, 0]
        except Exception as e:
            depth = None

        if depth is None:
            raise RuntimeError(f"Failed to read depth file: {depth_path}")

        imgs_list = []
        for x in self.rgb_paths:
            img_path = x[idx]
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Failed to read RGB file: {img_path}")
            imgs_list.append(np.expand_dims(img,axis=3))
        imgs = np.concatenate(imgs_list,3) #C*H*W
        if self.mode=="train":
            y_crop,x_crop,contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
            imgs,depth = randcrop_3d(imgs,depth,x_crop,y_crop,self.cropping[1],self.cropping[0])
            imgs=image_augmentation(imgs,contrast,brightness,gamma)
            imgs,depth = horizontal_flip(imgs,depth,flip_x)
            imgs,depth = vertical_flip(imgs,depth,flip_y)
            imgs,depth = rotate(imgs,depth,angle)    
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,self.train_size[0],self.train_size[1]]))

            imgs=torch.Tensor(np.transpose(imgs,(2,3,0,1)))
        elif self.mode == "val":
            imgs = imgs/127.5 -1.0
            imgs=torch.Tensor(np.transpose(imgs,(2,3,0,1)))
            C,N,H,W = imgs.shape        

            if H % 32 != 0:
                pad_h = 32 - (H % 32)
            else:
                pad_h = 0 
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
            else:
                pad_w =0

            imgs  = np.pad(imgs,
                        ( (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                        mode='constant',
                        constant_values=(-1, -1)
            )
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,self.input_size[0]+pad_h,self.input_size[1]+pad_w]))

        depth[depth< 0.0] = 0.0
        mask = depth > 0
        depth= torch.Tensor(depth)

        return imgs, depth, mask, Focus_Dists

    def get_seeds(self):
        return (random.randint(0,self.cropping[0]-1),random.randint(0,self.cropping[1]-1),
                random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),
                random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))

# train_dataset=FlyingThings3d('train')
# valid_dataset=FlyingThings3d('val')
# dataloader=DataLoader(train_dataset,batch_size=1,shuffle=False, pin_memory=True)
# num_train=len(dataloader)
# valid_dataloader=DataLoader(valid_dataset,1,shuffle=False,pin_memory=True)
# num_val=len(valid_dataloader)
# for idx, samples in enumerate(dataloader):
#     valid_input, test_gt_depth, test_mask, test_focus_dists = samples
    # print(valid_input.shape, test_gt_depth.shape, test_mask.shape, test_focus_dists.shape)
    # true_count = torch.sum(test_mask).item()
    # total_count = test_mask.numel()
    # print((true_count / total_count) * 100)

class HCI_dataset(Dataset):
    def __init__(self, hdf5_filename, stack_key="stack_train", disp_key="disp_train"):
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.input_size = (512,512)

        if stack_key == "stack_train":
            self.size = (224, 224)
        elif stack_key == "stack_val":
            self.size = (512, 512)
        self.cropping = (self.input_size[0] - self.size[0],self.input_size[1] - self.size[1])

        focus_dists = self.hdf5['focus_position_disp']
        focus_dists = np.squeeze(focus_dists,axis=0)
        focus_dists = np.expand_dims(focus_dists,axis=1)
        focus_dists = np.expand_dims(focus_dists,axis=2)
        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,self.size[0],self.size[1]]))

        self.min_dist = np.min(focus_dists) # -2.5
        self.max_dist = np.max(focus_dists)# 2.5

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        FS=self.hdf5[self.stack_key][idx].astype(np.float32)
        FS_re = np.zeros((512,512,3,10),dtype=np.float32)
        for i in range(0,10):
            FS_re[:,:,:,i] = FS[i,:,:,:]
        gt=self.hdf5[self.disp_key][idx].astype(np.float32)
        if self.stack_key == "stack_train":
            y_crop,x_crop,contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
            FS,gt = randcrop_3d(FS_re,gt,x_crop,y_crop,self.cropping[1],self.cropping[0])
            FS=image_augmentation(FS,contrast,brightness,gamma)
            FS,gt = horizontal_flip(FS,gt,flip_x)
            FS,gt = vertical_flip(FS,gt,flip_y)
            FS,gt = rotate(FS,gt,angle)
        elif self.stack_key == "stack_val":
            FS=FS_re/127.5 -1.0
            gt[gt< self.min_dist] = -3.0
            gt[gt > self.max_dist] = -3.0
        
        mask = gt > -3.0
        FS = torch.from_numpy(FS.transpose((2,3,0,1)))
        gt = torch.from_numpy(gt)
        return FS, gt , self.focus_dists, mask

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]
    def get_seeds(self):
        return (random.randint(0,self.cropping[0]-1),random.randint(0,self.cropping[1]-1),
                random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),
                random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))
    
class Smartphone(Dataset):
    def __init__(self, mode, num_imgs):
        self.mode = mode
        self.num_imgs = num_imgs
        self.input_size = (504,378)
        self.center_crop = (336,252)
        self.rand_crop = (224,224)
        self.cropping = (self.center_crop[0] - self.rand_crop[0],self.center_crop[1] - self.rand_crop[1])
        self.indexes = np.rint(np.linspace(0,48,num_imgs,endpoint=True)).astype(np.int8)
        self.focus_dists = []
        #https://storage.googleapis.com/cvpr2020-af-data/LearnAF%20Dataset%20Readme.pdf
        focus_dists = [3910.92,2289.27,1508.71,1185.83,935.91,801.09,700.37,605.39,546.23,486.87,447.99,407.40,379.91,350.41,329.95,307.54,
                            291.72,274.13,261.53,247.35,237.08,225.41,216.88,207.10,198.18,191.60,183.96,178.29,171.69,165.57,160.99,155.61,150.59,146.81,
                            142.35,138.98,134.99,131.23,127.69,124.99,121.77,118.73,116.40,113.63,110.99,108.47,106.54,104.23,102.01]
        for index in self.indexes:
            self.focus_dists.append(focus_dists[index])
        self.focus_dists = np.expand_dims(self.focus_dists,axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=2).astype(np.float32)
        self.focus_dists = self.focus_dists*0.001
        self.Fovs = (1/0.00444)-(1/np.array(self.focus_dists))#https://www.devicespecifications.com/en/model/121b4c25
        self.Fovs = self.Fovs/np.min(self.Fovs)
        self.Fovs = np.expand_dims(self.Fovs,axis=0)
        if mode == "train":
            self.focus_dists = torch.Tensor(np.tile(self.focus_dists,[1,self.rand_crop[0],self.rand_crop[1]]))
        elif mode == "test":
            self.focus_dists = torch.Tensor(np.tile(self.focus_dists,[1,self.center_crop[0]+16,self.center_crop[1]+4]))
        self.focus_dists=1/self.focus_dists
        self.max_depth = 1/0.10201  
        self.min_depth = 1/3.91092
        self.root= '/data/SmartPhone/'
        self.depths=[]
        self.confids=[]
        self.FS=[]
        if mode == "train":
            for i in tqdm(range(1,8),desc="trainset"):
                path = self.root + mode + str(i) + '/'
                scenes=os.listdir(path+'scaled_images/')
                for scene in scenes:
                    self.depths.append(path + 'merged_depth/'+ scene +'/' + 'result_merged_depth_center.png')
                    self.confids.append(path + 'merged_conf/'+ scene +'/' + 'result_merged_conf_center.exr')
                    FS_imgs=[]
                    for j in self.indexes:
                        FS_imgs.append(path + 'scaled_images/'+ scene +'/' + str(j)+ '/result_scaled_image_center.jpg')
                    self.FS.append(FS_imgs)
        elif mode == "test":
            path = self.root + mode  + '/'
            scenes=os.listdir(path+'scaled_images/')
            for scene in scenes:
                self.depths.append(path + 'merged_depth/'+ scene +'/' + 'result_merged_depth_center.png')
                self.confids.append(path + 'merged_conf/'+ scene +'/' + 'result_merged_conf_center.exr')
                FS_imgs=[]
                for j in self.indexes:
                    FS_imgs.append(path + 'scaled_images/'+ scene +'/' + str(j)+ '/result_scaled_image_center.jpg')
                self.FS.append(FS_imgs)
        
    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        
        FS = np.zeros((self.center_crop[0],self.center_crop[1],self.num_imgs,3),dtype=np.float32)
        for i in range(0,self.num_imgs):
            img = cv2.imread(self.FS[idx][i]).astype(np.float32)[:,:,:]
            FS[:,:,i,:] = img[84:-84,63:-63,:].astype(np.float32)
        img = cv2.imread(self.FS[idx][self.num_imgs-1]).astype(np.float32)[:,:,:]
        FS[:,:,(self.num_imgs-1),:] = img[84:-84,63:-63,:].astype(np.float32)

        gt =cv2.imread(self.depths[idx],cv2.IMREAD_UNCHANGED).astype(np.float32)[84:-84,63:-63]
        gt = gt/255.0
        gt = (20)/(100-(100-0.2)*gt)
        gt=1/gt
        conf = cv2.imread(self.confids[idx],cv2.IMREAD_UNCHANGED )[84:-84,63:-63,-1]
        conf [conf>1.0] = 1.0

        if self.mode == "train":
            y_crop,x_crop,contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
            FS,gt,conf = randcrop_3d_w_conf(FS,gt,conf,x_crop,y_crop,self.cropping[1],self.cropping[0])
            FS=image_augmentation(FS,contrast,brightness,gamma)
            FS,gt,conf = horizontal_flip_w_conf(FS,gt,conf,flip_x)
            FS,gt,conf = vertical_flip_w_conf(FS,gt,conf,flip_y)
            FS,gt,conf = rotate_w_conf(FS,gt,conf,angle)
            
        elif self.mode == "test":
            FS=FS/127.5 -1.0
        gt[gt< self.min_depth] = 0.0
        gt[gt > self.max_depth] = 0.0
        
        # mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))
        mask = gt > 0
        FS = torch.from_numpy(np.transpose(FS,(3,2,0,1)))
        N,C,H,W = FS.shape
        if H % 32 != 0:
            pad_h = 32 - (H % 32)
        else:
            pad_h = 0 
        if W % 32 != 0:
            pad_w = 32 - (W % 32)
        else:
            pad_w =0
        FS = F.pad(torch.Tensor(FS),(0,pad_w,0,pad_h))#top 4 padding

        gt = torch.from_numpy(gt)
        return FS, gt , self.focus_dists, mask, conf, torch.from_numpy(self.Fovs)

    def get_seeds(self):
        return (random.randint(0,self.cropping[0]-1),
                random.randint(0,self.cropping[1]-1),
                random.uniform(0.4,1.6),random.uniform(-0.1,0.1),
                random.uniform(0.5,2.0),random.uniform(0,1.0),
                random.uniform(0,1.0),random.randint(0,3))

# train_dataset=Smartphone('train',10)
# valid_dataset=Smartphone('test',10)
# dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)
# num_train = len(dataloader)
# valid_dataloader=DataLoader(valid_dataset,1,shuffle=False)
# num_val = len(valid_dataloader)
# print(num_train, num_val)
  
class NYUDepthV2_dataset(Dataset):
    def __init__(self, stack_key="stack_train", disp_key="disp_train"):
        self.hdf5 = h5py.File('/data/NYU_DepthV2.h5', 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.input_size = (460, 620)

        if stack_key == "stack_train":
            self.size = (224, 224)
        elif stack_key == "stack_val":
            self.size = (460, 620)

        self.cropping = (self.input_size[0] - self.size[0], self.input_size[1] - self.size[1])

        self.focus_dists = self.hdf5['focus_position_disp']#focus_dists focus_position_disp
        self.focus_dists = np.squeeze(self.focus_dists, axis=0)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=2)

        self.min_dist = np.min(self.focus_dists) #0.1
        self.max_dist = np.max(self.focus_dists) #3

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        FS = self.hdf5[self.stack_key][idx].astype(np.float32)
        FS_re = np.zeros((460, 620, 3, 3), dtype=np.float32)
        for i in range(0, 3):
            FS_re[:, :, :, i] = FS[i, :, :, :]
        gt = self.hdf5[self.disp_key][idx].astype(np.float32)

        if self.stack_key == "stack_train":
            y_crop, x_crop, contrast, brightness, gamma, flip_x, flip_y, angle = self.get_seeds()
            FS_re, gt = randcrop_3d(FS_re, gt, x_crop, y_crop, self.cropping[1], self.cropping[0])
            FS_re = image_augmentation(FS_re, contrast, brightness, gamma)
            FS_re, gt = horizontal_flip(FS_re, gt, flip_x)
            FS_re, gt = vertical_flip(FS_re, gt, flip_y)
            FS_re, gt = rotate(FS_re, gt, angle)
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists, [1, self.size[0], self.size[1]]))
        elif self.stack_key == "stack_val":
            FS_re = FS_re / 127.5 - 1.0
            gt[gt < self.min_dist] = -3.0
            gt[gt > self.max_dist] = -3.0
            H, W, _, _ = FS_re.shape
            if H % 32 != 0:
                pad_h = 32 - (H % 32)
            else:
                pad_h = 0
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
            else:
                pad_w = 0
            FS_re = np.pad(FS_re, ((0, pad_h), (0, pad_w), (0, 0), (0, 0)), mode='constant', constant_values=(-1, -1))
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists, [1, self.input_size[0] + pad_h, self.input_size[1] + pad_w]))

        mask = gt > -3.0
        FS_re = FS_re.transpose((3, 2, 0, 1))
        gt = torch.from_numpy(gt)
        return FS_re, gt, Focus_Dists, mask

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]

    def get_seeds(self):
        return (
            random.randint(0, self.cropping[0] - 1), random.randint(0, self.cropping[1] - 1), random.uniform(0.4, 1.6),
            random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0), random.uniform(0, 1.0),
            random.randint(0, 3))
    
# train_dataset = NYUDepthV2_dataset("stack_train", "disp_train")
# valid_dataset = NYUDepthV2_dataset("stack_val", "disp_val")
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# num_train = len(train_dataloader)
# valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False)
# for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
#     valid_input, valid_gt_depth, valid_focus_dists, valid_mask = samples
#     if torch.min(valid_gt_depth) < 0.1 or torch.max(valid_gt_depth) > 3.0:
#         print(torch.min(valid_gt_depth), torch.max(valid_gt_depth))
    # exit()
