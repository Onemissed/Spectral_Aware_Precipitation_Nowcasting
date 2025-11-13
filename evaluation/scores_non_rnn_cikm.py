import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

def calculate_metrics(output, target, thresh, args):
    # Ensure tensors are of float type for calculations
    output, target = output.float(), target.float()

    # 设置阈值（此处设置dBZ阈值为20）
    threshold = torch.tensor((thresh + 10.0) * 255.0/ 95.0).to(args.device)
    output_binary = (output > threshold).float()
    target_binary = (target > threshold).float()
    # TP
    hits = torch.sum((output_binary == 1) & (target_binary == 1)).float()
    # FN
    misses = torch.sum((output_binary == 0) & (target_binary == 1)).float()
    # FP
    false_alarms = torch.sum((output_binary == 1) & (target_binary == 0)).float()
    # TN
    correct_negatives = torch.sum((output_binary == 0) & (target_binary == 0)).float()

    return hits, misses, false_alarms, correct_negatives


def calculate_scores(metrics):
    """
    Given a dictionary with keys "hits", "misses", "false_alarms", and "correct_negatives",
    compute the metrics: POD, FAR, CSI, and HSS.
    """
    a = metrics["hits"]
    b = metrics["false_alarms"]
    c = metrics["misses"]
    d = metrics["correct_negatives"]

    pod = a / (a + c) if (a + c) > 0 else 0
    far = b / (a + b) if (a + b) > 0 else 0
    csi = a / (a + b + c) if (a + b + c) > 0 else 0
    n = a + b + c + d
    # aref is the reference (expected hits by chance)
    aref = ((a + b) / n * (a + c)) if n > 0 else 0
    denom = (a + b + c - aref)
    gss = (a - aref) / denom if denom != 0 else 0
    hss = 2 * gss / (gss + 1) if (gss + 1) != 0 else 0
    return {"pod": pod, "far": far, "csi": csi, "hss": hss}


class Model_eval_nonRNN(object):
    def __init__(self, args):
        self.args = args
        self.minMSE = 5000
        self.minMSE_epoch = -1
        self.minMAE = 1000
        self.minMAE_epoch = -1
        self.minMSE_MAE = 3000
        self.minMSE_MAE_epoch = -1
        self.maxSSIM = 0
        self.maxSSIM_epoch = -1
        self.maxPSNR = 0
        self.maxPSNR_epoch = -1

        self.maxAvgCSI = -0.5
        self.maxAvgCSI_epoch = -1
        self.maxAvgHSS = -99
        self.maxAvgHSS_epoch = -1

        self.max_metrics = {
            5: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            20: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            40: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
        }
        # self.thresholds = dBZ_to_pixel(np.array([5.0, 20.0, 40.0]))

        self.metrics = {
            5: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            20: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            40: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
        }

    def eval_update(self, gt, pred, threshold):
        # Calculate the metrics for a given threshold.
        hits, misses, false_alarms, correct_negatives = calculate_metrics(pred, gt, threshold, self.args)
        m = self.metrics[threshold]
        m["hits"] += hits
        m["misses"] += misses
        m["false_alarms"] += false_alarms
        m["correct_negatives"] += correct_negatives

    def eval(self, dataloader, model, epoch):
        mse_loss = 0
        mae_loss = 0
        ssim_total = 0
        psnr_total = 0
        count = 0

        self.metrics = {
            5: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            20: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            40: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
        }

        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, X in enumerate(dataloader):
                    ims = X.numpy()
                    # 对图像进行归一化（除255）
                    target = ims[:, 5:]
                    B, T, H, W, C = target.shape
                    ims = torch.FloatTensor(ims).to(self.args.device)
                    ims /= 255

                    if self.args.model in {'pastnet', 'earthfarseer'}:
                        # For PastNet and EarthFarseer the input needs to be zero-padding to 128×128
                        ims = F.pad(ims.permute(0, 1, 4, 2, 3), pad=(13, 14, 13, 14), mode='constant', value=0)
                    else:
                        ims = F.pad(ims.permute(0, 1, 4, 2, 3), pad=(1, 2, 1, 2), mode='constant', value=0)

                    if self.args.model == 'earthformer':
                        # For earthformer, change the tensor size to [B, T, H, W, C]
                        ims = ims.permute(0, 1, 3, 4, 2)

                    if self.args.model in {'simvp', 'tau', 'pastnet', 'earthfarseer'}:
                        pred_y = model(ims[:, :5])
                        pred_y_next = model(pred_y)
                        pred = torch.cat((pred_y, pred_y_next), dim=1)
                    else:
                        pred = model(ims[:, :5])

                    if self.args.model != 'earthformer':
                        pred = pred.permute(0, 1, 3, 4, 2)

                    pred *= 255
                    pred.clamp_(min=0, max=255)

                    if self.args.model in {'pastnet', 'earthfarseer'}:
                        img_out = pred[:, :, 13:-14, 13:-14, :].cpu().numpy()
                    else:
                        img_out = pred[:, :, 1:-2, 1:-2, :].cpu().numpy()

                    mse = np.mean(np.square(target - img_out))
                    mae = np.mean(np.abs(target - img_out))

                    ssim_temp = 0.0
                    psnr_temp = 0.0
                    epsilon = 1e-10
                    for b in range(B):
                        for f in range(T):
                            # Remove the channel dimension as `ssim` expects 2D images
                            output_frame = img_out[b, f, :, :, 0]
                            target_frame = target[b, f, :, :, 0]

                            # Compute SSIM for the single frame pair and add to the total
                            ssim_value = ssim(output_frame, target_frame, data_range=255.0)
                            ssim_temp += ssim_value

                            mse_temp = np.mean((output_frame - target_frame) ** 2)
                            psnr_value = 20 * np.log10(255 / np.sqrt(mse_temp + epsilon))
                            psnr_temp += psnr_value

                    ssim_mean = ssim_temp / (B * T)
                    psnr_mean = psnr_temp / (B * T)
                    ssim_total += ssim_mean
                    psnr_total += psnr_mean

                    mse_loss = mse_loss + mse
                    mae_loss = mae_loss + mae
                    count = count + 1

                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=5)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=20)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=40)

                    pbar.update(1)

            mse_score = mse_loss / count
            mae_score = mae_loss / count
            ssim_score = ssim_total / count
            psnr_score = psnr_total / count

            scores = {}
            for thresh, m in self.metrics.items():
                scores[thresh] = calculate_scores(m)

            info = 'Test EPOCH INFO: epoch:{} \nMSE:{:.4f}  MAE:{:.4f}  SSIM:{:.4f}  PSNR:{:.4f}  MSE_MAE:{:.4f}  CSI_5:{:.4f}  HSS_5:{:.4f}  CSI_20:{:.4f}  HSS_20:{:.4f}  CSI_40:{:.4f}  HSS_40:{:.4f}\n'. \
                format(epoch + 1, mse_score, mae_score, ssim_score, psnr_score, mse_score + mae_score, scores[5]['csi'], scores[5]['hss'], scores[20]['csi'], scores[20]['hss'], scores[40]['csi'], scores[40]['hss'])
            print(info)

            if mse_score < self.minMSE:
                self.minMSE = mse_score
                self.minMSE_epoch = epoch + 1
            if mae_score < self.minMAE:
                self.minMAE = mae_score
                self.minMAE_epoch = epoch + 1
            if mse_score + mae_score < self.minMSE_MAE:
                self.minMSE_MAE = mse_score + mae_score
                self.minMSE_MAE_epoch = epoch + 1
            if ssim_score > self.maxSSIM:
                self.maxSSIM = ssim_score
                self.maxSSIM_epoch = epoch + 1
            if psnr_score > self.maxPSNR and psnr_score < 1000:
                self.maxPSNR = psnr_score
                self.maxPSNR_epoch = epoch + 1

            if scores[5]['csi'] > self.max_metrics[5]['maxCSI']:
                self.max_metrics[5]['maxCSI'] = scores[5]['csi']
                self.max_metrics[5]['maxCSI_epoch'] = epoch + 1
            if scores[5]['hss'] > self.max_metrics[5]['maxHSS']:
                self.max_metrics[5]['maxHSS'] = scores[5]['hss']
                self.max_metrics[5]['maxHSS_epoch'] = epoch + 1

            if scores[20]['csi'] > self.max_metrics[20]['maxCSI']:
                self.max_metrics[20]['maxCSI'] = scores[20]['csi']
                self.max_metrics[20]['maxCSI_epoch'] = epoch + 1
            if scores[20]['hss'] > self.max_metrics[20]['maxHSS']:
                self.max_metrics[20]['maxHSS'] = scores[20]['hss']
                self.max_metrics[20]['maxHSS_epoch'] = epoch + 1

            if scores[40]['csi'] > self.max_metrics[40]['maxCSI']:
                self.max_metrics[40]['maxCSI'] = scores[40]['csi']
                self.max_metrics[40]['maxCSI_epoch'] = epoch + 1
            if scores[40]['hss'] > self.max_metrics[40]['maxHSS']:
                self.max_metrics[40]['maxHSS'] = scores[40]['hss']
                self.max_metrics[40]['maxHSS_epoch'] = epoch + 1

            avgcsi = (scores[5]['csi'] + scores[20]['csi'] + scores[40]['csi']) / 3
            avghss = (scores[5]['hss'] + scores[20]['hss'] + scores[40]['hss']) / 3
            if avgcsi > self.maxAvgCSI:
                self.maxAvgCSI = avgcsi
                self.maxAvgCSI_epoch = epoch + 1
            if avghss > self.maxAvgHSS:
                self.maxAvgHSS = avghss
                self.maxAvgHSS_epoch = epoch + 1

            print(
                "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_5: {:.4f}  epoch:{}\nmaxHSS_5: {:.4f}  epoch:{}\nmaxCSI_20: {:.4f}  epoch:{}\nmaxHSS_20: {:.4f}  epoch:{}\nmaxCSI_40: {:.4f}  epoch:{}\nmaxHSS_40: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                    self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE, self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                    self.max_metrics[5]['maxCSI'], self.max_metrics[5]['maxCSI_epoch'], self.max_metrics[5]['maxHSS'], self.max_metrics[5]['maxHSS_epoch'],
                    self.max_metrics[20]['maxCSI'], self.max_metrics[20]['maxCSI_epoch'], self.max_metrics[20]['maxHSS'], self.max_metrics[20]['maxHSS_epoch'],
                    self.max_metrics[40]['maxCSI'], self.max_metrics[40]['maxCSI_epoch'], self.max_metrics[40]['maxHSS'], self.max_metrics[40]['maxHSS_epoch'],
                    self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))

            with open(os.path.join(self.args.record_dir, 'record.txt'), 'a') as f:
                f.write(info)
                f.write(f"Avg_CSI: {avgcsi:.4f}\tAvg_HSS: {avghss:.4f}\n\n")
                if epoch + 1 == self.args.epoch:
                    f.write(
                        "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_5: {:.4f}  epoch:{}\nmaxHSS_5: {:.4f}  epoch:{}\nmaxCSI_20: {:.4f}  epoch:{}\nmaxHSS_20: {:.4f}  epoch:{}\nmaxCSI_40: {:.4f}  epoch:{}\nmaxHSS_40: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                        self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE, self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                        self.max_metrics[5]['maxCSI'], self.max_metrics[5]['maxCSI_epoch'], self.max_metrics[5]['maxHSS'], self.max_metrics[5]['maxHSS_epoch'],
                        self.max_metrics[20]['maxCSI'], self.max_metrics[20]['maxCSI_epoch'], self.max_metrics[20]['maxHSS'], self.max_metrics[20]['maxHSS_epoch'],
                        self.max_metrics[40]['maxCSI'], self.max_metrics[40]['maxCSI_epoch'], self.max_metrics[40]['maxHSS'], self.max_metrics[40]['maxHSS_epoch'],
                        self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))