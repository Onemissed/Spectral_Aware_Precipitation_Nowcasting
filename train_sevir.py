import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import random
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from generator_sevirlr import DataGenerator
from util.preprocess import reshape_patch, schedule_sampling
from load_earthformer_cfg import load_earthformer_config

from model.convlstm import ConvLSTM_model
from model.predrnn import PredRNN_model
from model.simvpv2 import SimVP_model
from model.tau_model import TAU_model
from model.earthformer import CuboidTransformer_model
from model.PastNet_Model import PastNet_model
from model.earthfarseer import Earthfarseer_model
from model.afno import AFNO_model
from model.mafno import mAFNO_model
from evaluation.scores_rnn_sevir import Model_eval_RNN
from evaluation.scores_non_rnn_sevir import Model_eval_nonRNN


def load_model_config(model_name: str, cfg_root='config/sevir') -> dict:
    path = os.path.join(cfg_root, f'{model_name.lower()}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config for model '{model_name}' not found at {path!r}")
    with open(path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    return full_cfg['model']

def DoTrain(args):
    # Data index
    TrainSetFilePath = './data_index/sevir_training_periods.txt'
    TestSetFilePath = './data_index/sevir_test_periods.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            train_list.append(line.rstrip('\n').rstrip('\r\n'))
    test_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            test_list.append(line.rstrip('\n').rstrip('\r\n'))

    # Dataloader
    train_data = DataGenerator(train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    test_data = DataGenerator(test_list)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    MODEL_REGISTRY = {
        'convlstm': ConvLSTM_model,
        'predrnn': PredRNN_model,
        'simvp': SimVP_model,
        'tau': TAU_model,
        'earthformer': CuboidTransformer_model,
        'pastnet': PastNet_model,
        'earthfarseer': Earthfarseer_model,
        'afno': AFNO_model,
        'm_afno': mAFNO_model,
    }
    ModelClass = MODEL_REGISTRY.get(args.model.lower())
    if ModelClass is None:
        raise ValueError(f'Unknown model: {args.model!r}. '
                         f'Available models: {list(MODEL_REGISTRY)}')

    # Load the model configuration
    if args.model == 'earthformer':
        model_kwargs = load_earthformer_config('sevir')
    else:
        model_kwargs = load_model_config(args.model.lower(), 'config/sevir')
        model_kwargs['args'] = args
    model = ModelClass(**model_kwargs).to(args.device)
    model = torch.nn.DataParallel(model)

    # Loss function
    MSE_criterion = nn.MSELoss(reduction='mean')

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epoch)
    # eta for schedule sampling
    eta = args.sampling_start_value

    # eval
    if args.model in {'convlstm', 'predrnn'}:
        model_eval_testdata = Model_eval_RNN(args)
    else:
        model_eval_testdata = Model_eval_nonRNN(args)

    for epoch in range(args.epoch):
        print("epoch:", epoch + 1)
        with tqdm(total=len(train_loader)) as pbar:
            model.train()
            for i, X in enumerate(train_loader):
                # For RNN models
                if args.model in {'convlstm', 'predrnn'}:
                    ims = X.numpy().astype(np.float32)
                    # [B, T, H, W]
                    ims = np.transpose(ims, (0, 3, 1, 2))
                    ims = np.expand_dims(ims, axis=4)
                    # Split into patch
                    ims = reshape_patch(ims, args.patch_size)
                    # Normalization
                    ims /= 255
                    ims = torch.FloatTensor(ims).to(args.device)
                    B, T, H, W, C = ims.shape

                    itr = epoch * len(train_loader) + i + 1
                    # Schedule sampling
                    eta, real_input_flag = schedule_sampling(eta, itr, args, B)
                    real_input_flag = torch.FloatTensor(real_input_flag).to(args.device)

                    next_frames = model(ims, real_input_flag)

                    loss = MSE_criterion(next_frames, ims[:, 1:])

                # For Non-RNN models
                else:
                    ims = X.float().to(args.device)
                    # [B, T, H, W]
                    ims = ims.permute(0, 3, 1, 2)
                    # Normalization
                    ims /= 255
                    if args.model == 'earthformer':
                        # For earthformer, change the tensor shape to [B, T, H, W, C]
                        ims = ims.unsqueeze(dim=4)
                    else:
                        # [B, T, C, H, W]
                        ims = ims.unsqueeze(dim=2)
                    pred = model(ims[:, :7])

                    if args.model == 'm_afno':
                        # Compute Frequency-domain Loss for our model
                        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
                        gt_freq = torch.fft.fft2(ims[:, 7:], dim=(-2, -1))
                        freq_loss = torch.mean(torch.abs(pred_freq - gt_freq))

                        loss = MSE_criterion(pred, ims[:, 7:]) + 0.57 * freq_loss
                    else:
                        loss = MSE_criterion(pred[:, :6], ims[:, 7:])

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                pbar.update(1)

        model_weights_path = args.model_weight_dir + f'model_weight_epoch_{epoch + 1}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # 'loss': loss,
        }
        torch.save(checkpoint, model_weights_path)

        model.eval()
        model_eval_testdata.eval(test_loader, model, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='m_afno',
                        choices=['m_afno', 'afno', 'convlstm', 'predrnn', 'simvp', 'tau', 'earthformer', 'pastnet', 'earthfarseer'],
                        help='The model used for training')
    parser.add_argument('--dataset', type=str, default='sevir', help='dataset name')
    parser.add_argument('--patch_size', type=int, default=4, help='')
    parser.add_argument('--img_width', type=int, default=128, help='image size')
    parser.add_argument('--img_channel', type=int, default=1, help='')

    # Schedule sampling
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_stop_iter', type=int, default=50000)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

    # Training hyperparameters
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='use cpu/gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpu device ID')
    parser.add_argument('--epoch', type=int, default=100, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_length', type=int, default=7, help='')
    parser.add_argument('--output_length', type=int, default=6, help='')
    parser.add_argument('--total_length', type=int, default=13, help='')
    parser.add_argument('--record_dir', type=str, default=None, help='')
    parser.add_argument('--model_weight_dir', type=str, default=None, help='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    current_file = args.dataset + '/' + args.model
    # Create directory
    folder_path = './record/' + current_file
    weight_path = './model_weight/' + current_file
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    args.record_dir = folder_path + '/'
    args.model_weight_dir = weight_path + '/'

    DoTrain(args)
