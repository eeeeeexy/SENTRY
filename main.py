import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import ConfusionMatrix
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
import random

import argparse
import models.utils as utils
import models.Models_Sequence
import models.Models_Image
import models.Models_Fusion
import models.SupCon_models
from models.logger import CompleteLogger
from models.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('walk', 'bike', 'car&taxi', 'bus', 'subway', 'train')

def main(args: argparse.Namespace):

    logger = CompleteLogger(args.log, args.phase)
    print(args)

    cudnn.benchmark = True

    if args.backbone == 'Estimator':

        train_loader, test_loader = utils.load_data_Seq_Img(args)
        seca_model = models.Models_Sequence.SECA(input_dims=7, output_dims=128, depth=10).to(device)
        print(f'--> Build Estimator defense model')
        seq_model_wo_softmax = models.Models_Sequence.SECA_wo_softmax(input_dims=7, output_dims=128, depth=10).to(device)
        map_model_wo_softmax = models.Models_Image.CNN_map19_wo_softmax(input_channel=args.input_img_channel, ker_size=args.kernel_size_image).to(device)
        align_model = models.Models_Fusion.Align_Model().to(device)
        defense_model = models.SupCon_models.GPS_Pixel_Align(seq_model_wo_softmax, map_model_wo_softmax, align_model, args.num_classes).to(device)

    optimizer_merge = torch.optim.Adam(defense_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    # Train the defense model
    print("Training model...")
    for epoch in range(args.epochs):
        
        acc_clean, acc_adv, epoch_loss = train(train_loader, seca_model, defense_model, optimizer_merge, args)
        print(f'epoch {epoch+1} / {args.epochs}, loss = {epoch_loss}, acc clean = {acc_clean:.4f}, acc adv = {acc_adv:.4f}')

        print("--> Testing clean accuracy:")
        acc_clean_test = test(test_loader, seca_model, defense_model, test_mode='clean', do_plot=False, args=args)

        print("--> Testing adv accuracy:") 
        acc_adv_test = test(test_loader, seca_model, defense_model, test_mode='adv', do_plot=False, args=args)

    logger.close()


def train(train_loader, seq_model, defense_model, optimizer_merge, args):

    losses = AverageMeter()

    n_correct_clean = 0
    n_correct_adv = 0
    n_samples = 0
    n_class_correct_clean = [0 for i in range(args.num_classes)]
    n_class_correct_adv = [0 for i in range(args.num_classes)]
    n_class_samples = [0 for i in range(args.num_classes)]

    label_list = []
    pred_list_clean = []
    pred_list_adv = []

    losses = []
    epoch_loss = 0

    defense_model.train()

    for (clean_traj, map_img_sample, map_extra_sample, labels, index) in tqdm(train_loader):

        clean_traj = clean_traj.to(device)
        map_img_sample = map_img_sample.to(device)
        map_extra_sample = map_extra_sample.to(device)
        labels = labels.to(device)
        index = index.to(device)
        clean_map19 = torch.concat((map_img_sample, map_extra_sample), dim=1)

        adv_traj, pert = attack(clean_traj, seq_model, labels)
        adv_map19 = attack_activate_gaussian_noise(clean_map19, std=args.map_attack_std)

        clean_seq_emb, clean_img_emb, clean_fusion_emb, clean_outputs, re_clean_output = defense_model(clean_traj=clean_traj, adv_traj=None, clean_map19=clean_map19, adv_map19=None, pixel_index=index, compressed_size=args.compressed_size, args=args)
        adv_seq_emb, adv_img_emb, adv_fusion_emb, adv_outputs, re_adv_output = defense_model(clean_traj=None, adv_traj=adv_traj, clean_map19=None, adv_map19=adv_map19, pixel_index=index, compressed_size=args.compressed_size, args=args)

        loss_normal = F.cross_entropy(clean_outputs, labels)
        # SupCon loss
        criterion_supcon = SupConLoss(temperature=0.1)
        features = torch.stack([clean_fusion_emb, adv_fusion_emb], dim=0)
        loss_con = criterion_supcon(features, labels)
        # loss_supcon = F.cross_entropy(clean_outputs, labels) + 0.5 * loss_con
        loss_supcon = loss_normal + loss_con

        # KL loss
        loss_kl_global = F.kl_div(
            F.log_softmax(adv_outputs, dim=1),
            F.softmax(clean_outputs, dim=1),
            reduction='batchmean'
        )
        # Compressed data loss
        loss_re_cls = F.cross_entropy(re_clean_output, labels)
        loss_re_kl = F.kl_div(
            F.log_softmax(re_adv_output, dim=1),
            F.softmax(re_clean_output, dim=1),
            reduction='batchmean'
        )
        loss_re = loss_kl_global + loss_re_cls + 0.2 * loss_re_kl

        loss_intra = F.mse_loss(clean_seq_emb, adv_seq_emb) + F.mse_loss(clean_img_emb, adv_img_emb)
        loss_inter = F.mse_loss(clean_seq_emb, clean_img_emb) + F.mse_loss(adv_seq_emb, adv_img_emb)
        loss_align = loss_intra + loss_inter

        loss = loss_supcon + loss_re + loss_align

        # backward
        optimizer_merge.zero_grad()
        loss.backward()
        optimizer_merge.step()

        _, pred_clean = torch.max(clean_outputs, 1)
        _, pred_adv = torch.max(adv_outputs, 1)
        n_samples += labels.shape[0]
        n_correct_clean += (pred_clean == labels).sum().item()
        n_correct_adv += (pred_adv == labels).sum().item()
        
        losses.append(loss.item())
        epoch_loss += loss.item()

        for i in range(len(labels)):
            label = labels[i]
            p_clean = pred_clean[i].item()
            p_adv = pred_adv[i].item()

            if (label == p_clean):
                n_class_correct_clean[label] += 1
            
            # Adv 类统计
            if (label == p_adv):
                n_class_correct_adv[label] += 1

            n_class_samples[label] += 1
            label_list.append(label)
            pred_list_clean.append(p_clean)
            pred_list_adv.append(p_adv)


    acc_clean = 100. * n_correct_clean / n_samples
    acc_adv = 100. * n_correct_adv / n_samples

    print(f'Clean Accuracy: {acc_clean:.2f}%')
    print(f'Robust (Adv) Accuracy: {acc_adv:.2f}%')

    return acc_clean, acc_adv, epoch_loss


def test(test_loader, seq_model, defense_model, test_mode, do_plot, args=None):

    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(args.num_classes)]
    n_class_samples = [0 for i in range(args.num_classes)]
    label_list = []
    pred_list = []

    defense_model.eval()
    
    for (clean_traj, map_img_sample, map_extra_sample, labels, index) in tqdm(test_loader):

        clean_traj = clean_traj.to(device)
        # adv_traj = adv_traj.to(device)

        map_img_sample = map_img_sample.to(device)
        map_extra_sample = map_extra_sample.to(device)
        labels = labels.to(device)
        index = index.to(device)

        clean_map19 = torch.concat((map_img_sample, map_extra_sample), dim=1)
        
        if test_mode == 'clean':
            clean_seq_emb, clean_img_emb, clean_fusion_emb, outputs, _ = defense_model(clean_traj=clean_traj, adv_traj=None, clean_map19=clean_map19, adv_map19=None, pixel_index=index, compressed_size=args.compressed_size, args=args)

        if test_mode == 'adv':
            adv_traj, pert = attack(clean_traj, seq_model, labels)
            adv_map19 = attack_activate_gaussian_noise(clean_map19, std=args.map_attack_std)
            adv_seq_emb, adv_img_emb, adv_fusion_emb, outputs, _ = defense_model(clean_traj=None, adv_traj=adv_traj, clean_map19=None, adv_map19=adv_map19, pixel_index=index, compressed_size=args.compressed_size, args=args)
                
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            label_list.append(label)
            pred_list.append(pred)

    acc_val = 100.0 * n_correct / n_samples
    acc_list = []
    for i in range(args.num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.4f} %')
        acc_list.append(acc)

    print(f'test acc on {test_mode} = {acc_val:.4f} %')
    print(f'ave test acc on {test_mode}: {(np.array(acc_list).sum() / args.num_classes):.4f} %')

    confmat = ConfusionMatrix(task="multiclass", num_classes=args.num_classes)
    print(confmat(torch.tensor(pred_list), torch.tensor(label_list)))

    return acc_val


def attack(clean_traj, seq_model, labels):
    # Attack in the batch
    features = clean_traj[:, :, 1:]
    # generate mask（same shape as pert），1 presents it could be attacked
    mask = torch.zeros_like(features)

    _, seq_len, feat_dim = features.shape
    num_attack_points = int(seq_len * args.mask_rate)  # random choose mask rate

    # choose the random attack points
    attack_indices = random.sample(range(seq_len), num_attack_points)
    mask[:, attack_indices, :] = 1.0

    adv_seq_traj, pert = models.Pertubation.seq_pgd_attack_modified(seq_model, clean_traj, labels, mask, args)

    return adv_seq_traj, pert


def attack_activate_gaussian_noise(tensor, std=0.01):
    noise = torch.randn_like(tensor) * std
    # 创建一个 mask，只在原图非零的地方应用噪声
    mask = (tensor > 0).float()
    noisy_tensor = tensor + (noise * mask)
    return torch.clamp(noisy_tensor, 0., 1.)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss_v1, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device
        batch_size = features.shape[0]

        features = features.view(-1, features.shape[-1])
        features = F.normalize(features, p=2, dim=1)

        labels = labels.view(-1, 1)
        full_labels = torch.cat([labels, labels], dim=0)

        mask = torch.eq(full_labels, full_labels.T).float().to(device)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask 

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        return loss
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack on CNN_SECA model')

    parser.add_argument("--backbone", type=str, default='Estimator', choices=['SECA', 'Estimator'],
                        help="Backbones.")

    parser.add_argument('--contrastive_loss', action='store_true',
                        help='Normalize lat and lon values.')
    parser.add_argument('--contrasloss_alpha', type=float, default=0.0,
                        help='temperature for loss function')
    
    parser.add_argument("--dataset", type=str, default='geo', choices=['geo', 'mtl'],
                        help="Datasets.")
    parser.add_argument('--traj_length', default=600, type=int,
                        metavar='N', help='the name of datasets')
    # model parameters
    parser.add_argument('--input_img_channel', default=19, type=int, metavar='N',
                        help='kernel size of image')
    parser.add_argument('--input_seq_channel', default=7, type=int, metavar='N',
                        help='kernel size of image')
    parser.add_argument('--crop_size', default=3, type=int, metavar='N',
                        help='crop size for the map images')
    parser.add_argument('--kernel_img_size', default=3, type=int, metavar='N',
                        help='kernel size of image')
    
    parser.add_argument("--supcon_model", type=str, default='Estimator',
                        help="Defense model.")

    parser.add_argument('--seq_only', action='store_true',
                        help='Test on the Seq only')
    parser.add_argument('--mask_load', action='store_true',
                        help='Test on the Seq only')
    parser.add_argument('--mask_save', action='store_true',
                        help='Test on the Seq only')


    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--seq-attack', action='store_true',
                        help='Attack the sequence data of CNN_SECA model.')
    parser.add_argument('--seq-attack-time', default=1, type=int,
                        help='Sequence Attack Iterations. ')
    parser.add_argument('--mask_rate', default=1.0, type=float,
                        help='Mask attack points. ')
    parser.add_argument('--map_attack_std', default=0.1, type=float,
                        help='standard deviation for map attack noise')
    
    parser.add_argument('--no_zero_constraint', default=False, type=bool,
                        help='No zero constraint of CNN_SECA model. ')
    parser.add_argument('--eps_seq', default=1e-1, type=float,
                        help='eps for the attack model for seq. ')
    parser.add_argument('--alpha_seq', default=1e-2, type=float,
                        help='alpha for the attack model for seq. ')
    parser.add_argument('--max_km', default=1e-3, type=float,
                        help='Km threshold of scaling down pert in CNN_SECA model. ')
    parser.add_argument('--wl', default=7, type=int,
                        help='wl smoothing for seq. ')
    parser.add_argument('--order', default=2, type=int,
                        help='order smoothing for seq. ')

    parser.add_argument("--log", type=str, default='Estimator_supcon',
                        help="Where to save logs, checkpoints and debugging images.")

    parser.add_argument('-nc', '--num_classes', default=6, type=int, metavar='N',
                        help='number of classes')
                        
    parser.add_argument('--compressed_size', default=100, type=int, metavar='N',
                        help='compressed size')
    
    parser.add_argument('-ks_img', '--kernel-size-image', default=3, type=int, metavar='N',
                        help='number of classes')

    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    parser.add_argument('--Normalize_latlon', action='store_true',
                        help='Normalize lat and lon values.')

    args = parser.parse_args()
    main(args)
