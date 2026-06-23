import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import gc
from torch.utils.data import DataLoader
from train_Dataloader import FS6_dataset
from tqdm import tqdm
from metrics import *
import argparse
from NetWork import Network

MSE_loss = nn.MSELoss()
def masked_MSE_loss(est,gt,mask):
    out = MSE_loss(est[mask],gt[mask])
    return out 

def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--lr',default=0.001, type=float,help='learning rate')
    parser.add_argument('--max_epoch',default=1000,type=int,help='max epoch')
    parser.add_argument('--load_epoch',default=0,type=int,help='load epoch')
    parser.add_argument('--batch_size',default=4,type=int,help='batch size')
    parser.add_argument('--cpus',default=4,type=int,help='num_workers')
    args = parser.parse_args()
    batch_size=args.batch_size
    max_epoch=args.max_epoch
    load_epoch=args.load_epoch

    device_ids = [0,]
    Weight1=0.3
    Weight2=0.7
    Weight3=1.0
    test_epoch=1 
    save_epoch=1
    avg_Loss=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    best_mse = 0.02
    best_mae = 0.050
    best_bulmp = 1.80
    best_bablance = 1.0
    
    model=Network()
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    root = './'
    if(load_epoch>0):
        path = root+str(load_epoch)+'.pth'
        model = torch.load(path)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))#hyperparmeters from AAAI2020 lr 0.00001 , betas=(0,0.9) #70 epoch 0.001->0.0001

    train_dataset=FS6_dataset('train')
    valid_dataset=FS6_dataset('test')
    dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
    num_train=len(dataloader)
    valid_dataloader=DataLoader(valid_dataset,1,shuffle=False,num_workers=args.cpus,pin_memory=True)
    num_val=len(valid_dataloader)
    print(num_train, num_val)

    for epoch in range(load_epoch,max_epoch+1):
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch%save_epoch==0 and epoch!=load_epoch):
            path=root + str(epoch)+'.pth'
            torch.save(model,path)

        if(epoch%test_epoch==0 and epoch !=load_epoch):
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
                val_time=0.0
                for idx, samples in enumerate(tqdm(valid_dataloader,desc="valid")):
                    valid_input, test_gt_depth, test_focus_dists, test_mask = samples

                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth.numpy())
                    test_focus_dists=test_focus_dists.cuda()   
                    
                    start= time.time()            
                    _, _, test_pred3 = model(valid_input,test_focus_dists)
                    val_time = val_time+ (time.time() -start)
                    test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                    test_pred3=np.squeeze(test_pred3)

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
                if (Avg_mse/num_val) < best_mse:
                    best_mse = Avg_mse / num_val
                    path=root + 'best_mse.pth'
                    torch.save(model, path)
                if (Avg_mae/num_val) < best_mae:
                    best_mae = Avg_mae / num_val
                    path = root + 'best_mae.pth'
                    torch.save(model, path)
                if (Avg_bump/num_val) < best_bulmp:
                    best_bulmp = Avg_bump / num_val
                    path = root + 'best_bulmp.pth'
                    torch.save(model, path)
                if ((Avg_mse + Avg_mae + Avg_rmse + Avg_bump + Avg_abs_rel +Avg_sq_rel )/(6*num_val)) < best_bablance:
                    best_bablance = (Avg_mse + Avg_mae + Avg_rmse + Avg_bump)/(4*num_val)
                    path = root + 'best_balance.pth'
                    torch.save(model, path)

                print("Avg_abs_rel(" +str(epoch)+") : " ,Avg_abs_rel/num_val)
                print("Avg_sq_rel(" +str(epoch)+") : " ,Avg_sq_rel/num_val)
                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/num_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/num_val)
                print("Avg_rmse(" +str(epoch)+") : " ,Avg_rmse/num_val)
                print("Avg_bump(" +str(epoch)+") : " ,Avg_bump/num_val)
                print("Avg_rmse_log(" +str(epoch)+") : " ,Avg_rmse_log/num_val)
                print("Avg_accuracy_1(" +str(epoch)+") : " ,Avg_accuracy_1/num_val)
                print("Avg_accuracy_2(" +str(epoch)+") : " ,Avg_accuracy_2/num_val)
                print("Avg_accuracy_3(" +str(epoch)+") : " ,Avg_accuracy_3/num_val)
                print("AVG_time:",val_time/num_val)
                with open('./DeFocus.txt', 'a+') as f:
                    f.write("Avg_abs_rel:" + str(Avg_abs_rel/num_val) + ' Avg_sq_rel:' + str(Avg_sq_rel/num_val)+
                            ' Avg_mse:' + str(Avg_mse/num_val) + ' Avg_mae:' + str(Avg_mae/num_val)+ ' Avg_bump:' + str(Avg_bump/num_val) +
                            ' Avg_rmse:' + str(Avg_rmse/num_val) + ' Avg_rmse_log:' + str(Avg_rmse_log/num_val)+
                            ' Avg_accuracy_1:' + str(Avg_accuracy_1/num_val) + ' Avg_accuracy_2:' + str(Avg_accuracy_2/num_val)+
                            ' Avg_accuracy_3:' + str(Avg_accuracy_3/num_val) + ' AVG_time:' + str(val_time/num_val) + '\n')
                f.close()

        model.train()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
            train_input, train_gt_depth, train_focus_dists, train_mask = samples

            train_input=train_input.to('cuda:0', non_blocking=True)
            train_gt_depth=train_gt_depth.to('cuda:0', non_blocking=True)
            train_focus_dists=train_focus_dists.to('cuda:0', non_blocking=True)
            train_mask=train_mask.to('cuda:0', non_blocking=True)

            pred1, pred2, pred3=model(train_input,train_focus_dists)
            optimizer.zero_grad()
            Loss1 =masked_MSE_loss(pred1,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)
            Loss2 =masked_MSE_loss(pred2,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)
            Loss3 =masked_MSE_loss(pred3,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)

            Total_Loss = (Weight1*Loss1) + (Weight2*Loss2) + (Weight3* Loss3)
            Total_Loss.backward()
            optimizer.step()

            avg_Loss=avg_Loss+Total_Loss.detach().data
            avg_DFF_1=avg_DFF_1+Loss1.detach().data
            avg_DFF_2=avg_DFF_2+Loss2.detach().data
            avg_DFF_3=avg_DFF_3+Loss3.detach().data
            
        print("Epoch:",epoch,"AVG_DFF_TotalLoss:",avg_Loss/(num_train))
        avg_Loss=0.0
        avg_DFF_1=0.0
        avg_DFF_2=0.0
        avg_DFF_3=0.0

if __name__=="__main__":
    main()
