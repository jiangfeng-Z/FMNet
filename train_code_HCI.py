import os
import numpy as np
import torch.nn as nn
import torch
import time
import gc
from NetWork import Network
from torch.utils.data import DataLoader
from train_Dataloader import HCI_dataset
from tqdm import tqdm
from metrics import *
import argparse

MSE_loss = nn.MSELoss()


def masked_MSE_loss(est, gt, mask):
    out = MSE_loss(est[mask], gt[mask])
    return out


def unravel_index(indices, shape):
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def miss_MSE_loss(est, gt, mask, miss=0.2):
    residuals = torch.abs(est - gt)
    sorted_indices = torch.argsort(residuals)
    twenty_percent = int(miss * len(residuals))
    top_percent_indices = sorted_indices[:twenty_percent]
    top_percent_coords = unravel_index(top_percent_indices, gt.shape)
    mask_update = torch.zeros_like(mask, dtype=torch.bool).to(gt.device)
    mask_update[top_percent_coords] = True
    mask = torch.where(mask_update, torch.tensor(False).to(gt.device), mask)
    out = MSE_loss(est[mask], gt[mask])
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
        return None, "None"

    lamda_avg = sum(lamda_values) / len(lamda_values)
    lamda_detail = "; ".join(lamda_details)

    return lamda_avg, lamda_detail


def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=10000, type=int, help='max epoch')
    parser.add_argument('--load_epoch', default=0, type=int, help='load epoch')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--cpus', default=0, type=int, help='num_workers')
    args = parser.parse_args()

    batch_size = args.batch_size
    max_epoch = args.max_epoch
    load_epoch = args.load_epoch

    device_ids = [0]

    Weight1 = 0.3
    Weight2 = 0.7
    Weight3 = 1.0

    test_epoch = 1
    save_epoch = 1

    max_Depth = 2.5
    min_Depth = -2.5

    avg_Loss = 0.0
    avg_DFF_1 = 0.0
    avg_DFF_2 = 0.0
    avg_DFF_3 = 0.0

    best_mse = 0.023
    best_mae = 0.050
    best_bulmp = 1.80
    best_bablance = 1.0

    model = Network()
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    dataroot = 'D:/Dataset/HCI_FS_trainval.h5'
    train_dataset = HCI_dataset(dataroot, "stack_train", "disp_train")
    valid_dataset = HCI_dataset(dataroot, "stack_val", "disp_val")

    root = 'D:/Liyujie/AAAI/autolamda/HCI/adalamdav3/'
    os.makedirs(root, exist_ok=True)

    if load_epoch > 0:
        path = root + str(load_epoch) + '.pth'
        model = torch.load(path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.cpus,
        pin_memory=True
    )

    num_train = len(dataloader)

    valid_dataloader = DataLoader(
        valid_dataset,
        1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    num_val = len(valid_dataloader)

    for epoch in range(load_epoch, max_epoch + 1):
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if epoch % save_epoch == 0 and epoch != load_epoch:
            path = root + str(epoch) + '.pth'
            torch.save(model, path)

        if epoch % test_epoch == 0 and epoch != load_epoch:
            model.eval()

            with torch.no_grad():
                Avg_abs_rel = 0.0
                Avg_sq_rel = 0.0
                Avg_mse = 0.0
                Avg_mae = 0.0
                Avg_rmse = 0.0
                Avg_Bulmp = 0.0
                Avg_accuracy_1 = 0.0
                Avg_accuracy_2 = 0.0
                Avg_accuracy_3 = 0.0
                val_time = 0.0

                for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
                    valid_input, test_gt_depth, test_focus_dists, test_mask = samples

                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth.numpy())
                    test_focus_dists = test_focus_dists.cuda()

                    start = time.time()
                    _, _, test_pred3 = model(valid_input, test_focus_dists)
                    val_time = val_time + (time.time() - start)

                    test_pred3 = test_pred3.data.cpu().numpy()
                    test_pred3 = np.squeeze(test_pred3)

                    Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3, test_gt_depth, test_mask)
                    Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3, test_gt_depth, test_mask)
                    Avg_mse = Avg_mse + mask_mse(test_pred3, test_gt_depth, test_mask)
                    Avg_mae = Avg_mae + mask_mae(test_pred3, test_gt_depth, test_mask)
                    Avg_rmse = Avg_rmse + mask_rmse(test_pred3, test_gt_depth, test_mask)
                    Avg_Bulmp = Avg_Bulmp + get_bumpiness(test_gt_depth, test_pred3, test_mask)
                    Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3, test_gt_depth, 1, test_mask)
                    Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3, test_gt_depth, 2, test_mask)
                    Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3, test_gt_depth, 3, test_mask)

                if (Avg_mse / num_val) < best_mse:
                    best_mse = Avg_mse / num_val
                    path = root + 'best_mse.pth'
                    torch.save(model, path)

                if (Avg_mae / num_val) < best_mae:
                    best_mae = Avg_mae / num_val
                    path = root + 'best_mae.pth'
                    torch.save(model, path)

                if (Avg_Bulmp / num_val) < best_bulmp:
                    best_bulmp = Avg_Bulmp / num_val
                    path = root + 'best_bulmp.pth'
                    torch.save(model, path)

                if ((Avg_mse + Avg_mae + Avg_rmse + Avg_Bulmp + Avg_abs_rel + Avg_sq_rel) / (6 * num_val)) < best_bablance:
                    best_bablance = (Avg_mse + Avg_mae + Avg_rmse + Avg_Bulmp) / (4 * num_val)
                    path = root + 'best_balance.pth'
                    torch.save(model, path)

                lamda_avg, lamda_detail = get_adaptive_lamda_info(model)

                print("Avg_mse(" + str(epoch) + ") : ", Avg_mse / num_val)
                print("Avg_mae(" + str(epoch) + ") : ", Avg_mae / num_val)
                print("Avg_Bulmp(" + str(epoch) + ") : ", Avg_Bulmp / num_val)
                print("Avg_rmse(" + str(epoch) + ") : ", Avg_rmse / num_val)
                print("AVG_time:", val_time / num_val)
                print("Adaptive_lamda(" + str(epoch) + ") : ", lamda_avg)
                print("Adaptive_lamda_detail(" + str(epoch) + ") : ", lamda_detail)

                with open(root + 'HCI.txt', 'a+') as f:
                    f.write(
                        "mse:" + str(Avg_mse / num_val) +
                        ' mae:' + str(Avg_mae / num_val) +
                        ' bulmp:' + str(Avg_Bulmp / num_val) +
                        ' rmse:' + str(Avg_rmse / num_val) +
                        ' time:' + str(val_time / num_val) +
                        ' lamda_avg:' + str(lamda_avg) +
                        ' lamda_detail:' + str(lamda_detail) +
                        '\n'
                    )

        model.train()

        for idx, samples in enumerate(tqdm(dataloader, desc="Train")):
            train_input, train_gt_depth, train_focus_dists, train_mask = samples

            train_input = train_input.to('cuda:0', non_blocking=True)
            train_gt_depth = train_gt_depth.to('cuda:0', non_blocking=True)
            train_focus_dists = train_focus_dists.to('cuda:0', non_blocking=True)
            train_mask = train_mask.to('cuda:0', non_blocking=True)

            pred1, pred2, pred3 = model(train_input, train_focus_dists)

            pred1 = (pred1 - min_Depth) / (max_Depth - min_Depth)
            pred2 = (pred2 - min_Depth) / (max_Depth - min_Depth)
            pred3 = (pred3 - min_Depth) / (max_Depth - min_Depth)
            train_gt_depth = (train_gt_depth - min_Depth) / (max_Depth - min_Depth)

            optimizer.zero_grad()

            Loss1 = masked_MSE_loss(pred1, train_gt_depth, train_mask)
            Loss2 = masked_MSE_loss(pred2, train_gt_depth, train_mask)
            Loss3 = masked_MSE_loss(pred3, train_gt_depth, train_mask)

            Total_Loss = (Weight1 * Loss1) + (Weight2 * Loss2) + (Weight3 * Loss3)

            # lamda_reg = 0.0

            # net = model.module if hasattr(model, "module") else model

            # for name, module in net.named_modules():
            #     if hasattr(module, "current_lamd") and module.current_lamd is not None:
            #         lamda_reg += (module.current_lamd - 0.8).pow(2)

            # Total_Loss = Total_Loss + 0.01 * lamda_reg
            Total_Loss = Total_Loss
            Total_Loss.backward()
            optimizer.step()

            avg_Loss = avg_Loss + Total_Loss.detach().data
            avg_DFF_1 = avg_DFF_1 + Loss1.detach().data
            avg_DFF_2 = avg_DFF_2 + Loss2.detach().data
            avg_DFF_3 = avg_DFF_3 + Loss3.detach().data

        print("Epoch:", epoch, "AVG_DFF_TotalLoss:", avg_Loss / num_train)

        avg_Loss = 0.0
        avg_DFF_1 = 0.0
        avg_DFF_2 = 0.0
        avg_DFF_3 = 0.0


if __name__ == "__main__":
    main()