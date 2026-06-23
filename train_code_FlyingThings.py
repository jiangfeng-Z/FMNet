import numpy as np
import torch.nn as nn
import torch
import time
import gc
from torch.utils.data import DataLoader
from train_Dataloader import FlyingThings3d
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
    parser.add_argument('--max_epoch',default=500,type=int,help='max epoch')
    parser.add_argument('--load_epoch',default=0,type=int,help='load epoch')#97
    parser.add_argument('--batch_size',default=4,type=int,help='batch size')
    parser.add_argument('--cpus',default=4,type=int,help='num_workers')
    args = parser.parse_args()
    batch_size=args.batch_size
    load_epoch=args.load_epoch
    max_epoch = args.max_epoch

    device_ids = [0]
    Weight1=0.3
    Weight2=0.7
    Weight3=1.0
    valid_epoch=1
    save_epoch=1
    valid_max_depth = 100.0
    valid_min_depth =10.0
    input_size=(540,960)
    avg_Loss=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    best_mse = 400.0

    model=Network()
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    if(load_epoch>=1):
        path = './'+str(load_epoch)+'.pth'
        model = torch.load(path)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))
    
    train_Flying=FlyingThings3d('train')
    dataloader=DataLoader(train_Flying,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
    num_train=len(dataloader)
    valid_Flying=FlyingThings3d('val')
    Flying_dataloader=DataLoader(valid_Flying,1,shuffle=False,num_workers=args.cpus,pin_memory=True)
    Flying_val = len(Flying_dataloader)

    for epoch in range(load_epoch,max_epoch+1):#chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch%save_epoch==0 and epoch!=load_epoch):
            path='./' + str(epoch)+'.pth'
            torch.save(model, path)

        if(epoch%valid_epoch==0 and epoch !=load_epoch):
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
                val_time=0.0
                for idx, samples in enumerate(tqdm(Flying_dataloader,desc="Flying_valid")):
                    valid_input, test_gt_depth , test_mask, test_focus_dists = samples

                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth.numpy())
                    test_focus_dists=test_focus_dists.cuda()   

                    start= time.time()            
                    _, _, test_pred3 = model(valid_input,test_focus_dists)
                    val_time = val_time+ (time.time() -start)

                    test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                    test_pred3=test_pred3[0,:input_size[0],:]
                    test_pred3=np.squeeze(test_pred3)

                    Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
                    Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
                    Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
                    Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
                    Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
                    Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
                    Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
                    Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
                    Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)
                    
                if (Avg_mse/Flying_val) < best_mse:
                    best_mse = Avg_mse / Flying_val
                    path='./' + 'best_mse.pth'
                    torch.save(model, path)

                print("Avg_abs_rel(" +str(epoch)+") : " ,Avg_abs_rel/Flying_val)
                print("Avg_sq_rel(" +str(epoch)+") : " ,Avg_sq_rel/Flying_val)
                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/Flying_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/Flying_val)
                print("Avg_rmse(" +str(epoch)+") : " ,Avg_rmse/Flying_val)
                print("Avg_rmse_log(" +str(epoch)+") : " ,Avg_rmse_log/Flying_val)
                print("Avg_accuracy_1(" +str(epoch)+") : " ,Avg_accuracy_1/Flying_val)
                print("Avg_accuracy_2(" +str(epoch)+") : " ,Avg_accuracy_2/Flying_val)
                print("Avg_accuracy_3(" +str(epoch)+") : " ,Avg_accuracy_3/Flying_val)
                print("AVG_time:",val_time/Flying_val)
                with open('./FlyingThings.txt', 'a+') as f:
                    f.write("Avg_abs_rel:" + str(Avg_abs_rel/Flying_val) + ' Avg_sq_rel:' + str(Avg_sq_rel/Flying_val)+
                            ' Avg_mse:' + str(Avg_mse/Flying_val) + ' Avg_mae:' + str(Avg_mae/Flying_val)+
                            ' Avg_rmse:' + str(Avg_rmse/Flying_val) + ' Avg_rmse_log:' + str(Avg_rmse_log/Flying_val)+
                            ' Avg_accuracy_1:' + str(Avg_accuracy_1/Flying_val) + ' Avg_accuracy_2:' + str(Avg_accuracy_2/Flying_val)+
                            ' Avg_accuracy_3:' + str(Avg_accuracy_3/Flying_val) + ' AVG_time:' + str(val_time/Flying_val) + '\n')
                f.close()

        model.train()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
            train_input, train_gt_depth , train_mask, train_focus_dists = samples

            train_input=train_input.to('cuda:0', non_blocking=True)
            train_gt_depth=train_gt_depth.to('cuda:0', non_blocking=True)
            train_focus_dists=train_focus_dists.to('cuda:0', non_blocking=True)
            train_mask=train_mask.to('cuda:0', non_blocking=True)

            pred1, pred2, pred3=model(train_input,train_focus_dists)
            optimizer.zero_grad()
            pred1 = (pred1 - valid_min_depth) / (valid_max_depth - valid_min_depth)
            pred2 = (pred2 - valid_min_depth) / (valid_max_depth - valid_min_depth)
            pred3 = (pred3 - valid_min_depth) / (valid_max_depth - valid_min_depth)
            train_gt_depth = (train_gt_depth - valid_min_depth) / (valid_max_depth - valid_min_depth)

            Loss1 =masked_MSE_loss(pred1,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)
            Loss2 =masked_MSE_loss(pred2,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)
            Loss3 =masked_MSE_loss(pred3,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)

            Total_Loss = (Weight1*Loss1) + (Weight2*Loss2) + (Weight3* Loss3)
            Total_Loss = Total_Loss
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
