import numpy as np
import torch.nn as nn
import torch
import time
import gc
import os
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


def get_adaptive_lamda_info(model):
    net = model.module if hasattr(model, "module") else model

    lamda_values = []
    lamda_details = []

    for name, module in net.named_modules():
        if hasattr(module, "last_lamd") and module.last_lamd is not None:
            value = float(module.last_lamd)
            lamda_values.append(value)
            lamda_details.append(name + ":" + str(round(value, 6)))

    if len(lamda_values) == 0:
        for name, module in net.named_modules():
            value = None

            if hasattr(module, "lamda"):
                raw = getattr(module, "lamda")
                if isinstance(raw, (int, float)):
                    value = float(raw)
                elif torch.is_tensor(raw) and raw.numel() == 1:
                    value = float(raw.detach().item())

            if value is None and hasattr(module, "lamd"):
                raw = getattr(module, "lamd")
                if isinstance(raw, (int, float)):
                    value = float(raw)
                elif torch.is_tensor(raw) and raw.numel() == 1:
                    value = float(raw.detach().item())

            if value is not None:
                lamda_values.append(value)
                lamda_details.append(name + ":" + str(round(value, 6)))

    if len(lamda_values) == 0:
        return None, "None"

    lamda_avg = sum(lamda_values) / len(lamda_values)
    lamda_detail = "; ".join(lamda_details)

    return lamda_avg, lamda_detail

def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--lr',default=0.001, type=float,help='learning rate')
    parser.add_argument('--max_epoch',default=500,type=int,help='max epoch')
    parser.add_argument('--batch_size',default=4,type=int,help='batch size')
    parser.add_argument('--cpus',default=4,type=int,help='num_workers')
    parser.add_argument('--root', default='./', type=str, help='directory for checkpoints and logs')
    args = parser.parse_args()
    batch_size=args.batch_size
    max_epoch = args.max_epoch
    root = args.root

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
    best_mse_epoch = -1
    best_mae = float('inf')
    best_mae_epoch = -1
    best_bulmp = float('inf')
    best_bulmp_epoch = -1
    best_balance = float('inf')
    best_balance_epoch = -1

    model=Network()
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))
    print("[Run] from scratch, no checkpoint migration.")
    
    train_Flying=FlyingThings3d('train')
    dataloader=DataLoader(train_Flying,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
    num_train=len(dataloader)
    valid_Flying=FlyingThings3d('val')
    Flying_dataloader=DataLoader(valid_Flying,1,shuffle=False,num_workers=args.cpus,pin_memory=True)
    Flying_val = len(Flying_dataloader)

    for epoch in range(0,max_epoch+1):#chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch%save_epoch==0 and epoch!=0):
            path=os.path.join(root, str(epoch)+'.pth')
            torch.save({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }, path)

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
                for idx, samples in enumerate(tqdm(Flying_dataloader,desc="Flying_valid", dynamic_ncols=True, mininterval=1.0, leave=False)):
                    valid_input, test_gt_depth , test_mask, test_focus_dists = samples

                    valid_input = valid_input.to('cuda:0', non_blocking=True)
                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth.numpy())
                    test_focus_dists=test_focus_dists.to('cuda:0', non_blocking=True)   

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
                    best_mse_epoch = epoch
                    path=os.path.join(root, 'best_mse.pth')
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
                    print("[Best] best_mse updated at epoch", best_mse_epoch, ":", best_mse)

                if (Avg_mae/Flying_val) < best_mae:
                    best_mae = Avg_mae / Flying_val
                    best_mae_epoch = epoch
                    path=os.path.join(root, 'best_mae.pth')
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
                    print("[Best] best_mae updated at epoch", best_mae_epoch, ":", best_mae)

                if (Avg_rmse/Flying_val) < best_bulmp:
                    best_bulmp = Avg_rmse / Flying_val
                    best_bulmp_epoch = epoch
                    path=os.path.join(root, 'best_bulmp.pth')
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
                    print("[Best] best_bulmp updated at epoch", best_bulmp_epoch, ":", best_bulmp)

                if ((Avg_mse + Avg_mae + Avg_rmse + Avg_rmse_log + Avg_abs_rel + Avg_sq_rel) / (6 * Flying_val)) < best_balance:
                    best_balance = (Avg_mse + Avg_mae + Avg_rmse + Avg_rmse_log) / (4 * Flying_val)
                    best_balance_epoch = epoch
                    path=os.path.join(root, 'best_balance.pth')
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
                    print("[Best] best_balance updated at epoch", best_balance_epoch, ":", best_balance)

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
                lamda_avg, lamda_detail = get_adaptive_lamda_info(model)
                print("lamda_avg(", epoch, ") : ", lamda_avg)
                print("lamda_detail(", epoch, ") : ", lamda_detail)
                with open(os.path.join(root, 'FlyingThings.txt'), 'a+') as f:
                    f.write("Avg_abs_rel:" + str(Avg_abs_rel/Flying_val) + ' Avg_sq_rel:' + str(Avg_sq_rel/Flying_val)+
                            ' Avg_mse:' + str(Avg_mse/Flying_val) + ' Avg_mae:' + str(Avg_mae/Flying_val)+
                            ' Avg_rmse:' + str(Avg_rmse/Flying_val) + ' Avg_rmse_log:' + str(Avg_rmse_log/Flying_val)+
                            ' Avg_accuracy_1:' + str(Avg_accuracy_1/Flying_val) + ' Avg_accuracy_2:' + str(Avg_accuracy_2/Flying_val)+
                            ' Avg_accuracy_3:' + str(Avg_accuracy_3/Flying_val) + ' AVG_time:' + str(val_time/Flying_val) +
                            ' lamda_avg:' + str(lamda_avg) + ' lamda_detail:' + str(lamda_detail) +
                            ' best_mse:' + str(best_mse) + ' best_mse_epoch:' + str(best_mse_epoch) +
                            ' best_mae:' + str(best_mae) + ' best_mae_epoch:' + str(best_mae_epoch) +
                            ' best_bulmp:' + str(best_bulmp) + ' best_bulmp_epoch:' + str(best_bulmp_epoch) +
                            ' best_balance:' + str(best_balance) + ' best_balance_epoch:' + str(best_balance_epoch) + '\n')

        model.train()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train", dynamic_ncols=True, mininterval=1.0, leave=False)): #check variable ranges, images
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
        lamda_avg, lamda_detail = get_adaptive_lamda_info(model)
        print("Train lamda_avg(", epoch, ") : ", lamda_avg)
        print("Train lamda_detail(", epoch, ") : ", lamda_detail)
        with open(os.path.join(root, 'FlyingThings.txt'), 'a+') as f:
            f.write("Train_Epoch:" + str(epoch) + ' AVG_DFF_TotalLoss:' + str(avg_Loss/(num_train)) +
                ' lamda_avg:' + str(lamda_avg) + ' lamda_detail:' + str(lamda_detail) + '\n')
        avg_Loss=0.0
        avg_DFF_1=0.0
        avg_DFF_2=0.0
        avg_DFF_3=0.0

if __name__=="__main__":
    main()
