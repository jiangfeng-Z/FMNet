import numpy as np
import torch.nn as nn
import torch, time
from metrics import *
from train_Dataloader import NYUDepthV2_dataset
from NetWork import Network
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

MSE_loss = nn.MSELoss()
def masked_MSE_loss(est, gt, mask):
    out = MSE_loss(est[mask], gt[mask])
    return out

def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=3000, type=int, help='max epoch')
    parser.add_argument('--load_epoch', default=0, type=int, help='load epoch')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    load_epoch = args.load_epoch

    max_Depth = 3.0
    min_Depth = 0
    device_ids = [0]
    Weight1=0.3
    Weight2=0.7
    Weight3=1.0
    avg_Loss=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    input_size = (460, 620)
    best_mse = float('inf')
    best_mae = float('inf')
    best_bulmp = float('inf')
    best_bablance = float('inf')

    model=Network()
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    root = './'
    if(load_epoch > 0):
        path = root+str(load_epoch)+'.pth'
        model = torch.load(path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    train_dataset = NYUDepthV2_dataset("stack_train", "disp_train")
    valid_dataset = NYUDepthV2_dataset("stack_val", "disp_val")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    num_train = len(train_dataloader)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=4, pin_memory=True)
    num_val = len(valid_dataloader)

    # amp
    for epoch in range(load_epoch, max_epoch + 1):  # chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch % 1==0 and epoch != load_epoch):
            path=root + str(epoch)+'.pth'
            torch.save(model, path)

        if epoch != load_epoch:
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
                print("Avg_abs_rel(" +str(epoch)+") : " ,Avg_abs_rel/num_val)
                print("Avg_sq_rel(" +str(epoch)+") : " ,Avg_sq_rel/num_val)
                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/num_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/num_val)
                print("Avg_rmse(" +str(epoch)+") : " ,Avg_rmse/num_val)
                print("Avg_Bulmp(" +str(epoch)+") : " ,Avg_Bulmp/num_val)
                print("Avg_rmse_log(" +str(epoch)+") : " ,Avg_rmse_log/num_val)
                print("Avg_accuracy_1(" +str(epoch)+") : " ,Avg_accuracy_1/num_val)
                print("Avg_accuracy_2(" +str(epoch)+") : " ,Avg_accuracy_2/num_val)
                print("Avg_accuracy_3(" +str(epoch)+") : " ,Avg_accuracy_3/num_val)
                print("AVG_time:",val_time/num_val)
                with open('./NYU.txt', 'a+') as f:
                    f.write("Avg_abs_rel:" + str(Avg_abs_rel/num_val) + ' Avg_sq_rel:' + str(Avg_sq_rel/num_val)+
                            ' Avg_mse:' + str(Avg_mse/num_val) + ' Avg_mae:' + str(Avg_mae/num_val)+ ' Avg_bump:' + str(Avg_Bulmp/num_val) +
                            ' Avg_rmse:' + str(Avg_rmse/num_val) + ' Avg_rmse_log:' + str(Avg_rmse_log/num_val)+
                            ' Avg_accuracy_1:' + str(Avg_accuracy_1/num_val) + ' Avg_accuracy_2:' + str(Avg_accuracy_2/num_val)+
                            ' Avg_accuracy_3:' + str(Avg_accuracy_3/num_val) + ' AVG_time:' + str(val_time/num_val) + '\n')
                f.close()

                if (Avg_mse/num_val) < best_mse:
                    best_mse = Avg_mse / num_val
                    path = root + 'best_mse.pth'
                    torch.save(model, path)
                if (Avg_mae/num_val) < best_mae:
                    best_mae = Avg_mae / num_val
                    path = root + 'best_mae.pth'
                    torch.save(model, path)
                if (Avg_Bulmp/num_val) < best_bulmp:
                    best_bulmp = Avg_Bulmp / num_val
                    path = root + 'best_bulmp.pth'
                    torch.save(model, path)
                if ((Avg_mse + Avg_mae + Avg_rmse + Avg_Bulmp + Avg_abs_rel +Avg_sq_rel )/(6*num_val)) < best_bablance:
                    best_bablance = (Avg_mse + Avg_mae + Avg_rmse + Avg_Bulmp)/(4*num_val)
                    path = root + 'best_balance.pth'
                    torch.save(model, path)

        model.train()
        for idx, samples in enumerate(tqdm(train_dataloader, desc="Train")):  # check variable ranges, images
            train_input, train_gt_depth, train_focus_dists, train_mask = samples

            train_input = train_input.cuda(non_blocking=True)
            train_focus_dists = train_focus_dists.cuda(non_blocking=True)
            train_gt_depth = train_gt_depth.cuda(non_blocking=True)
            train_mask = train_mask.cuda(non_blocking=True)

            pred1, pred2, pred3 = model(train_input, train_focus_dists)

            pred1 = (pred1 - min_Depth) / (max_Depth - min_Depth)
            pred2 = (pred2 - min_Depth) / (max_Depth - min_Depth)
            pred3 = (pred3 - min_Depth) / (max_Depth - min_Depth)
            train_gt_depth = (train_gt_depth - min_Depth) / (max_Depth - min_Depth)

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

if __name__ == "__main__":
    main()
