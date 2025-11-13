import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(output, target, thresh, args):
    # Ensure tensors are of float type for calculations
    output, target = output.float(), target.float()

    # Set threshold
    threshold = torch.tensor(thresh).to(args.device)
    output_binary = (output > threshold).float()
    target_binary = (target > threshold).float()

    output_binary[torch.isnan(output_binary)] = 0
    target_binary[torch.isnan(target_binary)] = 0
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
        self.minMSE = 2000
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
            12: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            18: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            24: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            32: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
        }

        self.metrics = {
            12: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            18: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            24: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            32: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
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
            12: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            18: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            24: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            32: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
        }

        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, X in enumerate(dataloader):
                    ims = X.numpy()
                    target = ims[:, 5:, :, :]
                    B, T, H, W = target.shape
                    ims = torch.FloatTensor(ims).to(self.args.device)

                    # Normalization
                    ims /= 80
                    if self.args.model == 'earthformer':
                        # [B, T, H, W, C] for earthformer
                        ims = ims.unsqueeze(dim=4)
                    else:
                        # [B, T, C, H, W]
                        ims = ims.unsqueeze(dim=2)

                    if self.args.model in {'simvp', 'tau', 'pastnet', 'earthfarseer'}:
                        # Obtain the output results through autoregressive way
                        pred_1 = model(ims[:, :5])
                        pred_2 = model(pred_1)
                        pred_3 = model(pred_2)
                        pred = torch.cat((pred_1, pred_2, pred_3), dim=1)
                    else:
                        pred = model(ims[:, :5])

                    # Denorm
                    pred *= 80
                    pred.clamp_(min=0, max=80)

                    if self.args.model == 'earthformer':
                        img_out = pred.squeeze(dim=4)
                    else:
                        img_out = pred.squeeze(dim=2)
                    img_out = img_out.cpu().numpy()

                    mse = np.mean(np.square(target - img_out))
                    mae = np.mean(np.abs(target - img_out))

                    ssim_temp = 0.0
                    psnr_temp = 0.0
                    epsilon = 1e-10
                    for b in range(B):
                        for f in range(T):
                            output_frame = img_out[b, f, :, :]
                            target_frame = target[b, f, :, :]

                            # Compute SSIM for the single frame pair and add to the total
                            ssim_value = ssim(output_frame, target_frame, data_range=80.0)
                            ssim_temp += ssim_value

                            mse_temp = np.mean((output_frame - target_frame) ** 2)
                            psnr_value = 20 * np.log10(80 / np.sqrt(mse_temp + epsilon))
                            psnr_temp += psnr_value

                    ssim_mean = ssim_temp / (B * T)
                    psnr_mean = psnr_temp / (B * T)
                    ssim_total += ssim_mean
                    psnr_total += psnr_mean

                    mse_loss = mse_loss + mse
                    mae_loss = mae_loss + mae
                    count = count + 1

                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=12)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=18)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=24)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=32)

                    pbar.update(1)

            mse_score = mse_loss / count
            mae_score = mae_loss / count
            ssim_score = ssim_total / count
            psnr_score = psnr_total / count

            scores = {}
            for thresh, m in self.metrics.items():
                scores[thresh] = calculate_scores(m)

            info = 'Test EPOCH INFO: epoch:{} \nMSE:{:.4f}  MAE:{:.4f}  SSIM:{:.4f}  PSNR:{:.4f}  MSE_MAE:{:.4f}\nCSI_12:{:.4f}  CSI_18:{:.4f}  CSI_24:{:.4f}  CSI_32:{:.4f}\nHSS_12:{:.4f}  HSS_18:{:.4f}  HSS_24:{:.4f}  HSS_32:{:.4f}\n'. \
                format(epoch + 1, mse_score, mae_score, ssim_score, psnr_score, mse_score + mae_score,
                       scores[12]['csi'], scores[18]['csi'], scores[24]['csi'], scores[32]['csi'],
                       scores[12]['hss'], scores[18]['hss'], scores[24]['hss'], scores[32]['hss'])
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
            if scores[12]['csi'] > self.max_metrics[12]['maxCSI']:
                self.max_metrics[12]['maxCSI'] = scores[12]['csi']
                self.max_metrics[12]['maxCSI_epoch'] = epoch + 1
            if scores[12]['hss'] > self.max_metrics[12]['maxHSS']:
                self.max_metrics[12]['maxHSS'] = scores[12]['hss']
                self.max_metrics[12]['maxHSS_epoch'] = epoch + 1

            if scores[18]['csi'] > self.max_metrics[18]['maxCSI']:
                self.max_metrics[18]['maxCSI'] = scores[18]['csi']
                self.max_metrics[18]['maxCSI_epoch'] = epoch + 1
            if scores[18]['hss'] > self.max_metrics[18]['maxHSS']:
                self.max_metrics[18]['maxHSS'] = scores[18]['hss']
                self.max_metrics[18]['maxHSS_epoch'] = epoch + 1

            if scores[24]['csi'] > self.max_metrics[24]['maxCSI']:
                self.max_metrics[24]['maxCSI'] = scores[24]['csi']
                self.max_metrics[24]['maxCSI_epoch'] = epoch + 1
            if scores[24]['hss'] > self.max_metrics[24]['maxHSS']:
                self.max_metrics[24]['maxHSS'] = scores[24]['hss']
                self.max_metrics[24]['maxHSS_epoch'] = epoch + 1

            if scores[32]['csi'] > self.max_metrics[32]['maxCSI']:
                self.max_metrics[32]['maxCSI'] = scores[32]['csi']
                self.max_metrics[32]['maxCSI_epoch'] = epoch + 1
            if scores[32]['hss'] > self.max_metrics[32]['maxHSS']:
                self.max_metrics[32]['maxHSS'] = scores[32]['hss']
                self.max_metrics[32]['maxHSS_epoch'] = epoch + 1

            avgcsi = (scores[12]['csi'] + scores[18]['csi'] + scores[24]['csi'] + scores[32]['csi']) / 4
            avghss = (scores[12]['hss'] + scores[18]['hss'] + scores[24]['hss'] + scores[32]['hss']) / 4
            if avgcsi > self.maxAvgCSI:
                self.maxAvgCSI = avgcsi
                self.maxAvgCSI_epoch = epoch + 1
            if avghss > self.maxAvgHSS:
                self.maxAvgHSS = avghss
                self.maxAvgHSS_epoch = epoch + 1

            print(
                "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_12: {:.4f}  epoch:{}\nmaxCSI_18: {:.4f}  epoch:{}\nmaxCSI_24: {:.4f}  epoch:{}\nmaxCSI_32: {:.4f}  epoch:{}\nmaxHSS_12: {:.4f}  epoch:{}\nmaxHSS_18: {:.4f}  epoch:{}\nmaxHSS_24: {:.4f}  epoch:{}\nmaxHSS_32: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                    self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE, self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                    self.max_metrics[12]['maxCSI'], self.max_metrics[12]['maxCSI_epoch'], self.max_metrics[18]['maxCSI'], self.max_metrics[18]['maxCSI_epoch'], self.max_metrics[24]['maxCSI'], self.max_metrics[24]['maxCSI_epoch'],
                    self.max_metrics[32]['maxCSI'], self.max_metrics[32]['maxCSI_epoch'],
                    self.max_metrics[12]['maxHSS'], self.max_metrics[12]['maxHSS_epoch'], self.max_metrics[18]['maxHSS'], self.max_metrics[18]['maxHSS_epoch'], self.max_metrics[24]['maxHSS'], self.max_metrics[24]['maxHSS_epoch'],
                    self.max_metrics[32]['maxHSS'], self.max_metrics[32]['maxHSS_epoch'],
                    self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))

            # Save evaluation results
            with open(os.path.join(self.args.record_dir, 'record.txt'), 'a') as f:
                f.write(info)
                f.write(f"Avg_CSI: {avgcsi:.4f}\tAvg_HSS: {avghss:.4f}\n\n")
                if epoch + 1 == self.args.epoch:
                    f.write(
                        "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_12: {:.4f}  epoch:{}\nmaxCSI_18: {:.4f}  epoch:{}\nmaxCSI_24: {:.4f}  epoch:{}\nmaxCSI_32: {:.4f}  epoch:{}\nmaxHSS_12: {:.4f}  epoch:{}\nmaxHSS_18: {:.4f}  epoch:{}\nmaxHSS_24: {:.4f}  epoch:{}\nmaxHSS_32: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                            self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE, self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                            self.max_metrics[12]['maxCSI'], self.max_metrics[12]['maxCSI_epoch'], self.max_metrics[18]['maxCSI'], self.max_metrics[18]['maxCSI_epoch'], self.max_metrics[24]['maxCSI'], self.max_metrics[24]['maxCSI_epoch'],
                            self.max_metrics[32]['maxCSI'], self.max_metrics[32]['maxCSI_epoch'],
                            self.max_metrics[12]['maxHSS'], self.max_metrics[12]['maxHSS_epoch'], self.max_metrics[18]['maxHSS'], self.max_metrics[18]['maxHSS_epoch'], self.max_metrics[24]['maxHSS'], self.max_metrics[24]['maxHSS_epoch'],
                            self.max_metrics[32]['maxHSS'], self.max_metrics[32]['maxHSS_epoch'],
                            self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))