import numpy as np
import torch.nn as nn
import torch
import time
import gc
from NetWork import Network
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import *
import argparse
from train_Dataloader import Smartphone

def masked_MSE_loss(est,gt,conf,mask):
    out = torch.sum(conf[mask]*(torch.pow((est[mask]-gt[mask]),2)))/torch.sum(conf[mask])
    return out 

def mask_Mae_lossf(est_depth,gt_depth,conf,mask):
    return torch.sum(conf[mask]*(torch.abs(gt_depth[mask]-est_depth[mask])))/torch.sum(conf[mask])

def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--lr',default=0.001, type=float,help='learning rate')
    parser.add_argument('--max_epoch',default=1000,type=int,help='max epoch')
    parser.add_argument('--load_epoch',default=0,type=int,help='load epoch')
    parser.add_argument('--batch_size',default=4,type=int,help='batch size')
    parser.add_argument('--cpus',default=4,type=int,help='num_workers')
    args = parser.parse_args()
    load_epoch= args.load_epoch
    batch_size=args.batch_size
    max_epoch=args.max_epoch

    Weight1=0.3
    Weight2=0.7
    Weight3=1.0
    test_epoch=1 
    save_epoch=1
    min_depth = 1/3.91092
    max_depth = 1/0.10201
    Height = 336
    Width = 252
    avg_Loss=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    best_mse = 0.1
    best_mae = 0.145
    device_ids = [0]

    model=Network()
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    root = './'
    if(load_epoch>0):
        path= root + str(load_epoch)+'.pth'
        model = torch.load(path)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))
    train_dataset=Smartphone('train',10)
    valid_dataset=Smartphone('test',10)
    dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
    num_train = len(dataloader)
    valid_dataloader=DataLoader(valid_dataset,1,shuffle=False,num_workers=args.cpus,pin_memory=True)
    num_val = len(valid_dataloader)

    for epoch in range(load_epoch,max_epoch+1):#chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch%save_epoch==0 and epoch!=load_epoch):
            path= root + str(epoch)+'.pth'
            torch.save(model,path)
        if(epoch%test_epoch==0 and epoch !=load_epoch):
            model.eval()
            with torch.no_grad():
                Avg_mse=0.0
                Avg_mae=0.0
                val_time=0.0
                for idx, samples in enumerate(tqdm(valid_dataloader,desc="valid")):
                    valid_input, test_gt_depth , test_focus_dists, test_mask, test_conf, _= samples

                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_conf = np.squeeze(test_conf.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth.numpy())
                    test_focus_dists=test_focus_dists.cuda()   

                    start= time.time()            
                    _, _, test_pred3 = model(valid_input,test_focus_dists)
                    val_time = val_time+ (time.time() -start)

                    test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                    test_pred3 = test_pred3[0,:Height,:Width]

                    Avg_mse = Avg_mse + mask_mse_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)
                    Avg_mae = Avg_mae + mask_mae_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)
                if (Avg_mse/num_val) < best_mse:
                    best_mse = Avg_mse / num_val
                    path= root + 'best_mse.pth'
                    torch.save(model, path)
                if (Avg_mae/num_val) < best_mae:
                    best_mae = Avg_mae / num_val
                    path= root + 'best_mae.pth'
                    torch.save(model, path)

                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/num_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/num_val)
                print("AVG_time:",val_time/num_val)
                with open('./SmartPhone.txt', 'a+') as f:
                    f.write("Avg_mse:" + str(Avg_mse/num_val) + ' Avg_mae:' + str(Avg_mae/num_val)+ ' AVG_time:' + str(val_time/num_val) + '\n')
                f.close()

        model.train()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
            train_input, train_gt_depth , train_focus_dists, train_mask, train_conf,_ = samples

            train_input=train_input.cuda(non_blocking=True)
            train_gt_depth=train_gt_depth.cuda(non_blocking=True)
            train_focus_dists=train_focus_dists.cuda(non_blocking=True)
            train_mask=train_mask.cuda(non_blocking=True)
            train_conf=train_conf.cuda(non_blocking=True)

            pred1, pred2, pred3=model(train_input,train_focus_dists)
            pred1 = pred1[:,:,:]
            pred2 = pred2[:,:,:]
            pred3 = pred3[:,:,:]
            optimizer.zero_grad()
            Loss1 =masked_MSE_loss((pred1-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)#,gt_gradient,gt_sobel)
            Loss2 =masked_MSE_loss((pred2-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)#,gt_gradient,gt_sobel)
            Loss3 =masked_MSE_loss((pred3-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)#,gt_gradient,gt_sobel)            
            Total_Loss = (Weight1*Loss1) + (Weight2*Loss2) + (Weight3* Loss3) 
            Total_Loss = Total_Loss
            Total_Loss.backward()
            optimizer.step()

            avg_Loss=avg_Loss+Total_Loss.detach().data
            avg_DFF_1=avg_DFF_1+Loss1.detach().data
            avg_DFF_2=avg_DFF_2+Loss2.detach().data
            avg_DFF_3=avg_DFF_3+Loss3.detach().data
        print("Epoch:", epoch, "AVG_DFF_TotalLoss:", avg_Loss/num_train)
        avg_Loss=0.0
        avg_DFF_1=0.0
        avg_DFF_2=0.0
        avg_DFF_3=0.0

if __name__=="__main__":
    main()
