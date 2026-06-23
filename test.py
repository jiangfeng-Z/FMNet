import os
import argparse
import time, cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py, matplotlib
import skimage.filters as skf
from imageio import imread, imwrite
from matplotlib import cm
from metrics import *
from test_Dataloader import *
import imageio.core.util
from NetWork import Network
from Fourier import *

def silence_imageio_warning(*args, **kwargs):
    pass


def load_testing_model(path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', None))
        if state_dict is not None:
            model = Network()
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                cleaned_key = key[len('module.'):] if key.startswith('module.') else key
                cleaned_state_dict[cleaned_key] = value
            try:
                model.load_state_dict(cleaned_state_dict, strict=False)
            except Exception:
                model.load_state_dict(cleaned_state_dict)
            if torch.cuda.is_available() and map_location is not None and 'cuda' in str(map_location):
                model = model.cuda()
            return model

    return checkpoint


def append_test_summary(dataset_name, summary_lines):
    summary_path = os.path.join(args.outdir, 'result.txt')
    with open(summary_path, 'a+', encoding='utf-8') as f:
        f.write(dataset_name + '\n')
        for line in summary_lines:
            f.write(line + '\n')
        f.write('\n')

# Arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pth', help='path to dumped .pth file', default=r"D:\Liyujie\AAAI\autolamda\DDFF05\DFV_best_mse.pth")
parser.add_argument('--outdir', default=r"D:/Liyujie/AAAI/testresult/DDFF/maelamda/", help='output directory')
parser.add_argument('--dataset', default='DDFF')  # DefocusNet, DDFF, ...
args = parser.parse_args()
args.dataset = args.dataset.upper()
os.makedirs(args.outdir, exist_ok=True)

def DDFF_testing(args):
    imageio.core.util._precision_warn = silence_imageio_warning

    focal_length = 521.4052
    K2 = 1982.0250823695178
    flens = 7317.020641763665
    baseline = K2 / flens * 1e-3
    Height=383
    Width=552
    max_depth = baseline * focal_length / 0.5
    min_depth = baseline * focal_length / 7
    
    dataroot= r'D:\Dataset\ddff-dataset-trainval.h5'
    valid_dataset=FocalStackDDFFH5Reader_DFV(dataroot, stack_key="stack_val", disp_key="disp_val")
    dataloader=DataLoader(valid_dataset,1,shuffle=False)
    num_val = len(dataloader)

    path = args.pth
    model = load_testing_model(path, map_location='cuda:0')
    
    model.eval()
    with torch.no_grad():
        Avg_abs_rel=0.0
        Avg_sq_rel=0.0
        Avg_bump=0.0
        Avg_accuracy_1=0.0
        Avg_accuracy_2=0.0
        Avg_accuracy_3=0.0
        Avg_mse = 0.0
        Avg_mae = 0.0
        Avg_rmse = 0.0
        Avg_rmse_log = 0.0
        val_time=0.0
        for idx, samples in enumerate(tqdm(dataloader,desc="Test")):
            valid_input, test_gt_depth , test_focus_dists, test_mask = samples

            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            test_gt_depth = np.squeeze(test_gt_depth.cpu().numpy())
            test_focus_dists=test_focus_dists.cuda()
            valid_input = valid_input.cuda()

            start= time.time()            
            _, _, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.cpu().numpy()
            test_pred3=np.squeeze(test_pred3)
            test_pred3 = test_pred3[:Height,:Width]
            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            imwrite(args.outdir + str(idx)+'.jpg', color_img[:, :], quality=100)

            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
            Avg_rmse += mask_rmse(test_pred3,test_gt_depth,test_mask)
            Avg_rmse_log += mask_rmse_log(test_pred3,test_gt_depth,test_mask)
            Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
            Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
            Avg_bump = Avg_bump +get_bumpiness(test_gt_depth, test_pred3,test_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)
        print("AVG_time:", val_time / num_val)
        print("Avg_mse: ", Avg_mse / num_val)
        print("Avg_mae: ", Avg_mae / num_val)
        print("Avg_rmse: ", Avg_rmse / num_val)
        print("Avg_rmse_log: ", Avg_rmse_log / num_val)
        print("Avg_abs_rel: ", Avg_abs_rel / num_val)
        print("Avg_sq_rel: ", Avg_sq_rel / num_val)
        print("Avg_bump: ", Avg_bump / num_val)
        print("Avg_accuracy_1: ", Avg_accuracy_1 / num_val)
        print("Avg_accuracy_2: ", Avg_accuracy_2 / num_val)
        print("Avg_accuracy_3: ", Avg_accuracy_3 / num_val)

        append_test_summary(
            'DDFF',
            [
                'AVG_time: ' + str(val_time / num_val),
                'Avg_mse: ' + str(Avg_mse / num_val),
                'Avg_mae: ' + str(Avg_mae / num_val),
                'Avg_rmse: ' + str(Avg_rmse / num_val),
                'Avg_rmse_log: ' + str(Avg_rmse_log / num_val),
                'Avg_abs_rel: ' + str(Avg_abs_rel / num_val),
                'Avg_sq_rel: ' + str(Avg_sq_rel / num_val),
                'Avg_bump: ' + str(Avg_bump / num_val),
                'Avg_accuracy_1: ' + str(Avg_accuracy_1 / num_val),
                'Avg_accuracy_2: ' + str(Avg_accuracy_2 / num_val),
                'Avg_accuracy_3: ' + str(Avg_accuracy_3 / num_val),
            ]
        )

def DefocusNet_testing(args):
    imageio.core.util._precision_warn = silence_imageio_warning
    max_depth = 1.5
    min_depth = 0.1
    Dataset=FS6_dataset()

    path = args.pth
    model = load_testing_model(path, map_location='cuda:0')
    dataloader=DataLoader(Dataset,1,shuffle=False,num_workers=4,pin_memory=True)
    num_test = len(dataloader)

    model.eval()
    with torch.no_grad():
        Avg_abs_rel=0.0
        Avg_sq_rel=0.0
        Avg_mse=0.0
        Avg_mae=0.0
        Avg_rmse=0.0
        Avg_bump = 0.0
        Avg_rmse_log=0.0
        Avg_accuracy_1=0.0
        Avg_accuracy_2=0.0
        Avg_accuracy_3=0.0
        Avg_bump = 0.0
        val_time=0.0

        for idx, samples in enumerate(tqdm(dataloader,desc="Test")):
            valid_input, test_gt_depth , test_focus_dists, test_mask = samples
            
            test_gt_depth= np.squeeze(test_gt_depth.numpy())
            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            test_focus_dists=test_focus_dists.cuda()   
            valid_input = valid_input.cuda()

            start= time.time()            
            _, _, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.data.cpu().numpy()#[0,29]
            test_pred3=np.squeeze(test_pred3)
            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            imwrite(args.outdir + str(idx) + '.jpg', color_img[:, :], quality=100)

            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
            Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
            Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
            Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
            Avg_bump = Avg_bump + get_bumpiness(test_gt_depth, test_pred3, test_mask)
            Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)

        print("Avg_abs_rel : " ,Avg_abs_rel/num_test)
        print("Avg_sq_rel : " ,Avg_sq_rel/num_test)
        print("Avg_mse : " ,Avg_mse/num_test)
        print("Avg_mae : " ,Avg_mae/num_test)
        print("Avg_bump : " ,Avg_bump/num_test)
        print("Avg_rmse : " ,Avg_rmse/num_test)
        print("Avg_rmse_log : " ,Avg_rmse_log/num_test)
        print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_test)
        print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_test)
        print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_test)
        print("AVG_time:",val_time/num_test)
        append_test_summary(
            'DefocusNet',
            [
                'AVG_time: ' + str(val_time / num_test),
                'Avg_abs_rel: ' + str(Avg_abs_rel / num_test),
                'Avg_sq_rel: ' + str(Avg_sq_rel / num_test),
                'Avg_mse: ' + str(Avg_mse / num_test),
                'Avg_mae: ' + str(Avg_mae / num_test),
                'Avg_bump: ' + str(Avg_bump / num_test),
                'Avg_rmse: ' + str(Avg_rmse / num_test),
                'Avg_rmse_log: ' + str(Avg_rmse_log / num_test),
                'Avg_accuracy_1: ' + str(Avg_accuracy_1 / num_test),
                'Avg_accuracy_2: ' + str(Avg_accuracy_2 / num_test),
                'Avg_accuracy_3: ' + str(Avg_accuracy_3 / num_test),
            ]
        )

def HCI_testing(args):
    imageio.core.util._precision_warn = silence_imageio_warning
    
    min_depth = -2.5
    max_depth = 2.5
    Dataset=HCI_dataset()
    dataloader=DataLoader(Dataset,1,shuffle=False,num_workers=4,pin_memory=True)
    num_test = len(dataloader)

    path = args.pth
    model = load_testing_model(path)
    
    model.eval()
    with torch.no_grad():
        Avg_abs_rel=0.0
        Avg_sq_rel=0.0
        Avg_mse=0.0
        Avg_mae=0.0
        Avg_rmse=0.0
        Avg_Bulmp=0.0
        Avg_accuracy_1=0.0
        Avg_accuracy_2=0.0
        Avg_accuracy_3=0.0
        val_time=0.0
        for idx, samples in enumerate(tqdm(dataloader,desc="Test")):
            valid_input, test_gt_depth , test_focus_dists, test_mask = samples

            test_gt_depth = np.squeeze(test_gt_depth.numpy())
            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            test_focus_dists=test_focus_dists.cpu()   
            valid_input = valid_input.cpu()

            start= time.time()            
            _, _, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.data.cpu().numpy()#[0,29]
            test_pred3=np.squeeze(test_pred3)
    
            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            imwrite(args.outdir+str(idx)+'.jpg', color_img[:, :], quality=100)
            
            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
            Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
            Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
            Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth, test_mask)
            Avg_Bulmp = Avg_Bulmp +get_bumpiness(test_gt_depth,test_pred3,test_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)


        print("Avg_abs_rel : " ,Avg_abs_rel/num_test)
        print("Avg_sq_rel : " ,Avg_sq_rel/num_test)
        print("Avg_mse : " ,Avg_mse/num_test)
        print("Avg_mae : " ,Avg_mae/num_test)
        print("Avg_rmse : " ,Avg_rmse/num_test)
        print("Avg_Bulmp : " ,Avg_Bulmp/num_test)
        print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_test)
        print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_test)
        print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_test)
        print("AVG_time:",val_time/num_test)
        append_test_summary(
            'HCI',
            [
                'AVG_time: ' + str(val_time / num_test),
                'Avg_abs_rel: ' + str(Avg_abs_rel / num_test),
                'Avg_sq_rel: ' + str(Avg_sq_rel / num_test),
                'Avg_mse: ' + str(Avg_mse / num_test),
                'Avg_mae: ' + str(Avg_mae / num_test),
                'Avg_rmse: ' + str(Avg_rmse / num_test),
                'Avg_Bulmp: ' + str(Avg_Bulmp / num_test),
                'Avg_accuracy_1: ' + str(Avg_accuracy_1 / num_test),
                'Avg_accuracy_2: ' + str(Avg_accuracy_2 / num_test),
                'Avg_accuracy_3: ' + str(Avg_accuracy_3 / num_test),
            ]
        )
 
def Middlebury_testing(args):
    imageio.core.util._precision_warn = silence_imageio_warning
    model = load_testing_model(args.pth)
    
    Dataset=Middlebury()
    dataloader=DataLoader(Dataset,1,shuffle=False,num_workers=4,pin_memory=True)
    num_test = len(dataloader)

    model.eval()
    with torch.no_grad():
        Avg_abs_rel=0.0
        Avg_sq_rel=0.0
        Avg_mse=0.0
        Avg_mae=0.0
        Avg_rmse=0.0
        Avg_rmse_log=0.0
        Avg_accuracy_1=0.0
        Avg_accuracy_2=0.0
        Avg_accuracy_3=0.0
        Avg_bump = 0.0
        val_time=0.0
        for idx, samples in enumerate(tqdm(dataloader,desc="Middlebury")):
            valid_input, test_gt_depth , test_focus_dists, test_mask = samples

            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            test_gt_depth = np.squeeze(test_gt_depth.numpy())
            test_focus_dists=test_focus_dists.cuda()   
            valid_input = valid_input.cuda()

            start= time.time()            
            _, _, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.data.cpu().numpy()#[0,29]
            test_pred3=np.squeeze(test_pred3)
            min_depth = 10
            max_depth = 60
            H,W = test_gt_depth.shape
            test_pred3=test_pred3[:H,:W]

            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            imwrite(args.outdir + str(idx)+'.jpg', color_img[:, :], quality=100)
       
            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
            Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
            Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
            Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
            Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)
            Avg_bump += get_bumpiness(test_gt_depth, test_pred3, test_mask)

        print("Avg_abs_rel : " ,Avg_abs_rel/num_test)
        print("Avg_sq_rel : " ,Avg_sq_rel/num_test)
        print("Avg_mse : " ,Avg_mse/num_test)
        print("Avg_mae : " ,Avg_mae/num_test)
        print("Avg_rmse : " ,Avg_rmse/num_test)
        print("Avg_rmse_log : " ,Avg_rmse_log/num_test)
        print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_test)
        print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_test)
        print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_test)            
        print("AVG_time:",val_time/num_test)

def Flying_testing(args):
    input_size=(540,960)
    min_depth = 10.0
    max_depth = 100.0

    valid_dataset=FlyingThings3d('val')
    valid_dataloader=DataLoader(valid_dataset,1,shuffle=False)
    num_val = len(valid_dataloader)

    path = args.pth
    model = load_testing_model(path, map_location='cuda:0')

    model.eval()
    with torch.no_grad():
        Avg_abs_rel=0.0
        Avg_sq_rel=0.0
        Avg_mse=0.0
        Avg_mae=0.0
        Avg_rmse=0.0
        Avg_rmse_log=0.0
        Avg_accuracy_1=0.0
        Avg_accuracy_2=0.0
        Avg_accuracy_3=0.0
        Avg_bump = 0.0
        val_time=0.0
        for idx, samples in enumerate(tqdm(valid_dataloader,desc="valid")):
            valid_input, test_gt_depth , test_mask, test_focus_dists = samples

            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            test_gt_depth = np.squeeze(test_gt_depth.numpy())
            test_focus_dists=test_focus_dists.cuda()
            valid_input=valid_input.cuda()

            start= time.time()            
            _, _, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.data.cpu().numpy()#[0,29]
            test_pred3=test_pred3[0,:input_size[0],:]
            test_pred3=np.squeeze(test_pred3)
            
            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            imwrite(args.outdir+str(idx)+'.jpg', color_img[:, :], quality=100)

            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
            Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
            Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
            Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
            Avg_bump += get_bumpiness(test_gt_depth, test_pred3, test_mask)
            Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)
        print("Avg_abs_rel : " ,Avg_abs_rel/num_val)
        print("Avg_sq_rel : " ,Avg_sq_rel/num_val)
        print("Avg_mse : " ,Avg_mse/num_val)
        print("Avg_mae : " ,Avg_mae/num_val)
        print("Avg_rmse : " ,Avg_rmse/num_val)
        print("Avg_rmse_log : " ,Avg_rmse_log/num_val)
        print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_val)
        print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_val)
        print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_val)
        print("AVG_time:",val_time/num_val)
        append_test_summary(
            'Middlebury',
            [
                'AVG_time: ' + str(val_time / num_val),
                'Avg_abs_rel: ' + str(Avg_abs_rel / num_val),
                'Avg_sq_rel: ' + str(Avg_sq_rel / num_val),
                'Avg_mse: ' + str(Avg_mse / num_val),
                'Avg_mae: ' + str(Avg_mae / num_val),
                'Avg_rmse: ' + str(Avg_rmse / num_val),
                'Avg_rmse_log: ' + str(Avg_rmse_log / num_val),
                'Avg_bump: ' + str(Avg_bump / num_val),
                'Avg_accuracy_1: ' + str(Avg_accuracy_1 / num_val),
                'Avg_accuracy_2: ' + str(Avg_accuracy_2 / num_val),
                'Avg_accuracy_3: ' + str(Avg_accuracy_3 / num_val),
            ]
        )

def NYU_testing(args):
    imageio.core.util._precision_warn = silence_imageio_warning
    max_Depth = 3.0
    min_Depth = 0
    input_size = (460, 620)

    path = args.pth
    model = load_testing_model(path, map_location='cuda')

    valid_dataset = NYUDepthV2_dataset("stack_val", "disp_val")
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=4, pin_memory=True)
    num_val = len(valid_dataloader)

    model.eval()
    with torch.no_grad():
        Avg_abs_rel=0.0
        Avg_sq_rel=0.0
        Avg_mse=0.0
        Avg_mae=0.0
        Avg_rmse=0.0
        Avg_Bulmp = 0.0
        Avg_rmse_log=0.0
        Avg_accuracy_1=0.0
        Avg_accuracy_2=0.0
        Avg_accuracy_3=0.0
        val_time=0.0
        for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
            valid_input, valid_gt_depth, valid_focus_dists, valid_mask = samples

            valid_input = valid_input.cuda()
            valid_focus_dists = valid_focus_dists.cuda()
            valid_gt_depth = np.squeeze(valid_gt_depth.numpy())
            valid_mask = np.squeeze(valid_mask.data.cpu().numpy())

            start = time.time()
            _, _, test_pred3 = model(valid_input, valid_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3 = np.squeeze(test_pred3.data.cpu().numpy())
            test_pred3 = test_pred3[:input_size[0], :input_size[1]]

            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_Depth) / (max_Depth - min_Depth)))[..., :3]
            imwrite(args.outdir+str(idx)+'.jpg', color_img[:, :], quality=100)
            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,valid_gt_depth,valid_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,valid_gt_depth,valid_mask)
            Avg_mse = Avg_mse + mask_mse(test_pred3,valid_gt_depth,valid_mask)
            Avg_mae = Avg_mae + mask_mae(test_pred3,valid_gt_depth,valid_mask)
            Avg_rmse = Avg_rmse + mask_rmse(test_pred3,valid_gt_depth, valid_mask)
            Avg_rmse_log += mask_rmse_log(test_pred3,valid_gt_depth,valid_mask)
            Avg_Bulmp = Avg_Bulmp +get_bumpiness(valid_gt_depth,test_pred3,valid_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,valid_gt_depth,1,valid_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,valid_gt_depth,2,valid_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,valid_gt_depth,3,valid_mask)
        print("Avg_abs_rel : " ,Avg_abs_rel/num_val)
        print("Avg_sq_rel : " ,Avg_sq_rel/num_val)
        print("Avg_mse : " ,Avg_mse/num_val)
        print("Avg_mae : " ,Avg_mae/num_val)
        print("Avg_rmse : " ,Avg_rmse/num_val)
        print("Avg_Bulmp : " ,Avg_Bulmp/num_val)
        print("Avg_rmse_log : " ,Avg_rmse_log/num_val)
        print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_val)
        print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_val)
        print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_val)
        print("AVG_time:",val_time/num_val)
        append_test_summary(
            'FlyingThings3D',
            [
                'AVG_time: ' + str(val_time / num_val),
                'Avg_abs_rel: ' + str(Avg_abs_rel / num_val),
                'Avg_sq_rel: ' + str(Avg_sq_rel / num_val),
                'Avg_mse: ' + str(Avg_mse / num_val),
                'Avg_mae: ' + str(Avg_mae / num_val),
                'Avg_rmse: ' + str(Avg_rmse / num_val),
                'Avg_rmse_log: ' + str(Avg_rmse_log / num_val),
                'Avg_accuracy_1: ' + str(Avg_accuracy_1 / num_val),
                'Avg_accuracy_2: ' + str(Avg_accuracy_2 / num_val),
                'Avg_accuracy_3: ' + str(Avg_accuracy_3 / num_val),
            ]
        )
        append_test_summary(
            'NYU',
            [
                'AVG_time: ' + str(val_time / num_val),
                'Avg_abs_rel: ' + str(Avg_abs_rel / num_val),
                'Avg_sq_rel: ' + str(Avg_sq_rel / num_val),
                'Avg_mse: ' + str(Avg_mse / num_val),
                'Avg_mae: ' + str(Avg_mae / num_val),
                'Avg_rmse: ' + str(Avg_rmse / num_val),
                'Avg_Bulmp: ' + str(Avg_Bulmp / num_val),
                'Avg_rmse_log: ' + str(Avg_rmse_log / num_val),
                'Avg_accuracy_1: ' + str(Avg_accuracy_1 / num_val),
                'Avg_accuracy_2: ' + str(Avg_accuracy_2 / num_val),
                'Avg_accuracy_3: ' + str(Avg_accuracy_3 / num_val),
            ]
        )

def Smart_testing(args):
    imageio.core.util._precision_warn = silence_imageio_warning
    
    Height = 336
    Width = 252
    path=args.pth
    model = load_testing_model(path)
    
    Dataset=Smartphone()
    dataloader=DataLoader(Dataset,1,shuffle=False,pin_memory=True)
    num_test = len(dataloader)

    model.eval()
    with torch.no_grad():
        Avg_mse=0.0
        Avg_mae=0.0
        val_time=0.0
        for idx, samples in enumerate(tqdm(dataloader,desc="Test")):
            valid_input, test_gt_depth , test_focus_dists, test_mask, test_conf = samples

            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            test_conf = np.squeeze(test_conf.data.cpu().numpy())
            test_gt_depth = np.squeeze(test_gt_depth.numpy())
            max_depth = np.max(test_gt_depth[test_conf==1.0])
            min_depth = np.min(test_gt_depth[test_conf==1.0])
            test_focus_dists=test_focus_dists.cuda()   
            valid_input = valid_input.cuda()

            start= time.time()            
            _,_, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.data.cpu().numpy()#[0,29]
            test_pred3=np.squeeze(test_pred3)
            test_pred3 = test_pred3[:Height,:Width]

            cmap = matplotlib.colormaps.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            imwrite(args.outdir+str(idx)+'.jpg', color_img[:, :], quality=100)

            Avg_mse = Avg_mse + mask_mse_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)
            Avg_mae = Avg_mae + mask_mae_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)

        print("Avg_mse: " ,Avg_mse/num_test)
        print("Avg_mae: " ,Avg_mae/num_test)
        print("AVG_time:",val_time/num_test)
        append_test_summary(
            'Smartphone',
            [
                'AVG_time: ' + str(val_time / num_test),
                'Avg_mse: ' + str(Avg_mse / num_test),
                'Avg_mae: ' + str(Avg_mae / num_test),
            ]
        )

if __name__ == '__main__':
    if args.dataset == 'DEFOCUSNET':
        DefocusNet_testing(args)
    elif args.dataset == 'DDFF':
        DDFF_testing(args)
    elif args.dataset == 'HCI':
        HCI_testing(args)
    elif args.dataset == 'MIDDLEBURY':
        Middlebury_testing(args)
    elif args.dataset == 'NYU':
        NYU_testing(args)
    elif args.dataset == 'FLYINGTHINGS3D':
        Flying_testing(args)
    elif args.dataset == 'SMARTPHONE':
        Smart_testing(args)
    else:
        raise NotImplementedError(
            "{} Dataset testing haven't implemented".format(args.dataset))