import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import ConfusionMatrix
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
import random, shutil
import time

import argparse
import models_kdd.utils as utils
import models_kdd.Models_Sequence
import models_kdd.Models_Image
import models_kdd.Models_Fusion
import models_kdd.SupCon_models
from models_kdd.logger import CompleteLogger
from models_kdd.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('walk', 'bike', 'car&taxi', 'bus', 'subway', 'train')

def main(args: argparse.Namespace):

    logger = CompleteLogger(args.log, args.phase)
    print(args)
    print('Method 3, adv emb, as the output')

    cudnn.benchmark = True

    train_loader, test_loader = utils.load_data_Seq_Img(args)

    if args.backbone == 'GPS-Pixel-Align':
        seca_model = models_kdd.Models_Sequence.SECA(input_dims=7, output_dims=128, depth=10).to(device)
        print(f'--> Build Estimator defense model')
        seq_model_wo_softmax = models_kdd.Models_Sequence.SECA_wo_softmax(input_dims=7, output_dims=128, depth=10).to(device)
        map_model_wo_softmax = models_kdd.Models_Image.CNN_map19_wo_softmax(input_channel=args.input_img_channel, ker_size=args.kernel_size_image).to(device)
        align_model = models_kdd.Models_Fusion.Align_Model().to(device)
        defense_model = models_kdd.SupCon_models.GPS_Pixel_Align(seq_model_wo_softmax, map_model_wo_softmax, align_model, args.num_classes).to(device)

    optimizer_merge = torch.optim.Adam(defense_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = -1
    best_clean_test = 0.0
    best_adv_test = 0.0
    best_avg_test = 0.0

    epoch_times = []
    
    # Train the defense model
    print("Training model...")
    for epoch in range(args.epochs):

        start_time = time.perf_counter()
        
        acc_clean, acc_adv, epoch_loss = train(train_loader, seca_model, defense_model, optimizer_merge, args)
        print(f'epoch {epoch+1} / {args.epochs}, loss = {epoch_loss}, acc clean = {acc_clean:.4f}, acc adv = {acc_adv:.4f}')

        # 如果用 CUDA，需要 synchronize，确保 GPU kernel 真正跑完
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Clean Acc: {acc_clean:.4f} | "
            f"Adv Acc: {acc_adv:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        print("--> Testing clean accuracy:")
        acc_clean_test = test(test_loader, seca_model, defense_model, test_mode='clean', do_plot=args.tsne_plot, args=args)

        print("--> Testing adv accuracy:")
        acc_adv_test = test(test_loader, seca_model, defense_model, test_mode='adv', do_plot=args.tsne_plot, args=args)

        avg_test = (acc_clean_test + acc_adv_test) / 2.0

        print(
            f'epoch {epoch+1} test summary: '
            f'clean test = {acc_clean_test:.4f}, '
            f'adv test = {acc_adv_test:.4f}, '
            f'avg test = {avg_test:.4f}'
        )

        if avg_test > best_avg_test:
            best_avg_test = avg_test
            best_clean_test = acc_clean_test
            best_adv_test = acc_adv_test
            best_epoch = epoch + 1

            print(
                f'>>> New best epoch: {best_epoch}, '
                f'best clean test = {best_clean_test:.4f}, '
                f'best adv test = {best_adv_test:.4f}, '
                f'best avg test = {best_avg_test:.4f}'
            )

            torch.save(
            {
                'epoch': best_epoch,
                'defense_model_state_dict': defense_model.state_dict(),
                'optimizer_state_dict': optimizer_merge.state_dict(),
                'best_clean_test': best_clean_test,
                'best_adv_test': best_adv_test,
                'best_avg_test': best_avg_test,
                'args': args,
            },
            logger.get_checkpoint_path('best_defense_baselines_%s.pth') % str(args.baseline_loss_function)
        )

    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times)

    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Average epoch time: {avg_time:.2f}s")

    print("========== Best Test Result ==========")
    print(f'Best epoch: {best_epoch}')
    print(f'Best clean test acc: {best_clean_test:.4f}')
    print(f'Best adv test acc: {best_adv_test:.4f}')
    print(f'Best average test acc: {best_avg_test:.4f}')
    print("======================================")

    logger.close()


def joint_adaptive_pgd_attack(clean_traj, clean_map19, defense_model, labels, args, pixel_index=None):
    traj_eps = args.traj_eps
    traj_alpha = args.traj_alpha

    map_eps = args.map_eps
    map_alpha = args.map_alpha

    steps = args.joint_pgd_steps

    features = clean_traj[:, :, 1:]
    _, seq_len, _ = features.shape

    traj_mask = torch.zeros_like(features)
    num_attack_points = int(seq_len * args.mask_rate)
    attack_indices = random.sample(range(seq_len), num_attack_points)
    traj_mask[:, attack_indices, :] = 1.0

    map_mask = (clean_map19 > 0).float()

    adv_traj = clean_traj.clone().detach()
    adv_map19 = clean_map19.clone().detach()

    for _ in range(steps):
        adv_traj.requires_grad_(True)
        adv_map19.requires_grad_(True)

        # logits = _defense_logits(
        #     defense_model, adv_traj, adv_map19, args, pixel_index=pixel_index
        # )
        # loss = F.cross_entropy(logits, labels)

        loss = 0.0
        for _ in range(args.eot_samples):
            logits = _defense_logits(
                defense_model,
                adv_traj,
                adv_map19,
                args,
                pixel_index=pixel_index,
            )
            loss = loss + F.cross_entropy(logits, labels)

        loss = loss / args.eot_samples

        grad_traj, grad_map = torch.autograd.grad(loss, [adv_traj, adv_map19])

        adv_features = adv_traj.detach()[:, :, 1:]
        clean_features = clean_traj[:, :, 1:]

        adv_features = adv_features + traj_alpha * grad_traj[:, :, 1:].sign() * traj_mask
        adv_features = torch.max(torch.min(adv_features, clean_features + traj_eps), clean_features - traj_eps)

        adv_traj = clean_traj.clone().detach()
        adv_traj[:, :, 1:] = adv_features

        adv_map19 = adv_map19.detach() + map_alpha * grad_map.sign() * map_mask
        adv_map19 = torch.max(
            torch.min(adv_map19, clean_map19 + map_eps),
            clean_map19 - map_eps,
        )
        adv_map19 = torch.clamp(adv_map19, 0.0, 1.0)
        adv_map19 = clean_map19 * (1 - map_mask) + adv_map19 * map_mask

    traj_pert = adv_traj[:, :, 1:] - clean_traj[:, :, 1:]
    map_pert = adv_map19 - clean_map19

    return adv_traj.detach(), adv_map19.detach(), traj_pert.detach(), map_pert.detach()


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

    drift_list = []
    align_list = []

    losses = []
    epoch_loss = 0

    defense_model.train()

    for (clean_traj, map_img_sample, map_extra_sample, labels, class_list, index) in tqdm(train_loader):

        clean_traj = clean_traj.to(device)
        map_img_sample = map_img_sample.to(device)
        map_extra_sample = map_extra_sample.to(device)
        labels = labels.to(device)
        index = index.to(device)
        clean_map19 = torch.concat((map_img_sample, map_extra_sample), dim=1)

        if args.attack_type == "joint_adaptive_pgd":

            adv_traj, adv_map19, traj_pert, map_pert = joint_adaptive_pgd_attack(
                clean_traj, clean_map19, defense_model, labels, args, pixel_index=index
            )


        if args.backbone == 'SECA':
            # outputs = defense_model(clean_traj=clean_traj, adv_traj=adv_traj)
            clean_fusion_emb, clean_outputs = defense_model(clean_traj=clean_traj, adv_traj=None)
            adv_fusion_emb, adv_outputs = defense_model(clean_traj=None, adv_traj=adv_traj)

        if args.backbone == 'Estimator':
            # clean_emb, adv_emb, clean_output, outputs = defense_model(clean_traj=clean_traj, adv_traj=adv_traj, clean_map19=clean_map19, adv_map19=adv_map19)
            clean_seq_emb, clean_img_emb, clean_fusion_emb, clean_outputs = defense_model(clean_traj=clean_traj, adv_traj=None, clean_map19=clean_map19, adv_map19=None, args=args)
            adv_seq_emb, adv_img_emb, adv_fusion_emb, adv_outputs = defense_model(clean_traj=None, adv_traj=adv_traj, clean_map19=None, adv_map19=adv_map19, args=args)

        if args.backbone == 'GPS-Pixel-Align':
            # clean_emb, adv_emb, clean_output, outputs = defense_model(clean_traj=clean_traj, adv_traj=adv_traj, clean_map19=clean_map19, adv_map19=adv_map19)
            clean_seq_emb, clean_img_emb, clean_fusion_emb, clean_outputs, re_clean_output = defense_model(clean_traj=clean_traj, adv_traj=None, clean_map19=clean_map19, adv_map19=None, pixel_index=index, compressed_size=args.compressed_size, args=args)
            adv_seq_emb, adv_img_emb, adv_fusion_emb, adv_outputs, re_adv_output = defense_model(clean_traj=None, adv_traj=adv_traj, clean_map19=None, adv_map19=adv_map19, pixel_index=index, compressed_size=args.compressed_size, args=args)


        if args.backbone == 'GPS-Pixel-Align' and args.loss_function == 'KL+SupCon+Alignment-v1':
            # ===== CE loss =====
            loss_clean_ce = F.cross_entropy(clean_outputs, labels)
            loss_adv_ce = F.cross_entropy(adv_outputs, labels)

            # ===== DCL / SupCon loss =====
            criterion_supcon = SupConLoss_v1(temperature=0.1)
            features = torch.stack([clean_fusion_emb, adv_fusion_emb], dim=0)
            loss_dcl = criterion_supcon(features, labels)

            # ===== AD / KL loss =====
            loss_ad = F.kl_div(
                F.log_softmax(adv_outputs, dim=1),
                F.softmax(clean_outputs.detach(), dim=1),
                reduction='batchmean'
            )

            # ===== CMEA / modality alignment =====
            loss_intra = (
                F.mse_loss(clean_seq_emb, adv_seq_emb) +
                F.mse_loss(clean_img_emb, adv_img_emb)
            )
            loss_inter = (
                F.mse_loss(clean_seq_emb, clean_img_emb) +
                F.mse_loss(adv_seq_emb, adv_img_emb)
            )
            loss_cmea = loss_intra + loss_inter

            # ===== SCI / compressed data loss =====
            loss_sci_cls = F.cross_entropy(re_clean_output, labels)
            loss_sci_kl = F.kl_div(
                F.log_softmax(re_adv_output, dim=1),
                F.softmax(re_clean_output.detach(), dim=1),
                reduction='batchmean'
            )

            # ===== global alignment data loss =====
            loss_global_align = F.mse_loss(clean_outputs, adv_outputs)

            # 总损失 (AD+DCL+CMEA+SCI)
            # ===== final loss =====
            loss = (
                args.lambda_sentry_clean_ce * loss_clean_ce +
                args.lambda_sentry_adv_ce * loss_adv_ce +
                args.lambda_sentry_dcl * loss_dcl +
                args.lambda_sentry_ad * loss_ad +
                args.lambda_sentry_cmea * loss_cmea +
                args.lambda_sentry_sci_cls * loss_sci_cls +
                args.lambda_sentry_sci_kl * loss_sci_kl + 
                args.lambda_sentry_global_align * loss_global_align
            )

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

    print("Fusion Drift:", np.mean(drift_list))
    print("Adv Alignment:", np.mean(align_list))

    return acc_clean, acc_adv, epoch_loss


def test(test_loader, seq_model, defense_model, test_mode, do_plot, args=None):

    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(args.num_classes)]
    n_class_samples = [0 for i in range(args.num_classes)]
    label_list = []
    pred_list = []

    clean_seq_features_list = []
    clean_img_features_list = []
    clean_fusion_features_list = []
    clean_output_list = []
    adv_seq_features_list = []
    adv_img_features_list = []
    adv_fusion_features_list = []
    adv_output_list = []
    all_labels = []

    drift_list = []
    align_list = []

    defense_model.eval()
    
    for (clean_traj, map_img_sample, map_extra_sample, labels, class_list, index) in tqdm(test_loader):

        clean_traj = clean_traj.to(device)
        # adv_traj = adv_traj.to(device)

        map_img_sample = map_img_sample.to(device)
        map_extra_sample = map_extra_sample.to(device)
        labels = labels.to(device)
        index = index.to(device)

        clean_map19 = torch.concat((map_img_sample, map_extra_sample), dim=1)

        all_labels.append(labels.detach().cpu().numpy())
        
        if test_mode == 'clean':

            if args.backbone == 'SECA':
                _, outputs = defense_model(clean_traj=clean_traj, adv_traj=None)

            if args.backbone == 'Estimator':
                clean_seq_emb, clean_img_emb, clean_fusion_emb, outputs = defense_model(clean_traj=clean_traj, adv_traj=None, clean_map19=clean_map19, adv_map19=None, args=args)

                clean_seq_features_list.append(clean_seq_emb.detach().cpu().numpy())
                clean_img_features_list.append(clean_img_emb.detach().cpu().numpy())
                clean_fusion_features_list.append(clean_fusion_emb.detach().cpu().numpy())
                clean_output_list.append(outputs.detach().cpu().numpy())

            if args.backbone == 'GPS-Pixel-Align':
                clean_seq_emb, clean_img_emb, clean_fusion_emb, outputs, _ = defense_model(clean_traj=clean_traj, adv_traj=None, clean_map19=clean_map19, adv_map19=None, pixel_index=index, compressed_size=args.compressed_size, args=args)

                clean_seq_features_list.append(clean_seq_emb.detach().cpu().numpy())
                clean_img_features_list.append(clean_img_emb.detach().cpu().numpy())
                clean_fusion_features_list.append(clean_fusion_emb.detach().cpu().numpy())
                clean_output_list.append(outputs.detach().cpu().numpy())

        if test_mode == 'adv':

            adv_traj, adv_map19, traj_pert, map_pert = joint_adaptive_pgd_attack(clean_traj, clean_map19, defense_model, labels, args, pixel_index=index)

            if args.backbone == 'GPS-Pixel-Align':
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
    # with open('result.txt', 'a') as f:
    #     f.write(f"{acc_list}\n{acc_val:.4f}\n-----------------------------------")
    print(f'test acc on {test_mode} = {acc_val:.4f} %')
    print(f'ave test acc on {test_mode}: {(np.array(acc_list).sum() / args.num_classes):.4f} %')

    print("Fusion Drift:", np.mean(drift_list))
    print("Adv Alignment:", np.mean(align_list))

    confmat = ConfusionMatrix(task="multiclass", num_classes=args.num_classes)
    print(confmat(torch.tensor(pred_list), torch.tensor(label_list)))

    return acc_val


class SupConLoss_v1(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.07):
        super(SupConLoss_v1, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device
        batch_size = features.shape[0]

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

    parser.add_argument("--backbone", type=str, default='GPS-Pixel-Align', choices=['Estimator', 'GPS-Pixel-Align'],
                        help="Backbones.")
    parser.add_argument("--loss_function", type=str, default='None', choices=['KL+SupCon+Alignment-v1'],
                        help="Loss function.")

    parser.add_argument('--lambda_sentry_clean_ce', default=1.0, type=float)
    parser.add_argument('--lambda_sentry_adv_ce', default=1.0, type=float)
    parser.add_argument('--lambda_sentry_dcl', default=1.0, type=float)
    parser.add_argument('--lambda_sentry_ad', default=1.0, type=float)
    parser.add_argument('--lambda_sentry_cmea', default=1.0, type=float)
    parser.add_argument('--lambda_sentry_sci_cls', default=1.0, type=float)
    parser.add_argument('--lambda_sentry_sci_kl', default=0.2, type=float)
    parser.add_argument('--lambda_sentry_global_align', default=1.0, type=float)

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

    parser.add_argument("--tsne_plot", action="store_true")

    parser.add_argument('--seq-attack', action='store_true',
                        help='Attack the sequence data of CNN_SECA model.')
    parser.add_argument('--seq_attack_time', default=5, type=int,
                        help='Sequence Attack Iterations. ')
    parser.add_argument('--mask_rate', default=1.0, type=float,
                        help='Mask attack points. ')
    parser.add_argument('--map_attack_std', default=0.3, type=float,
                        help='standard deviation for map attack noise')
    
    parser.add_argument('--no_zero_constraint', default=False, type=bool,
                        help='No zero constraint of CNN_SECA model. ')
    parser.add_argument('--eps_seq', default=1e-1, type=float,
                        help='eps for the attack model for seq. ')
    parser.add_argument('--alpha_seq', default=1e-2, type=float,
                        help='alpha for the attack model for seq. ')
    parser.add_argument('--max_km', default=1e-3, type=float,
                        help='Km threshold of scaling down pert in CNN_SECA model.')
    parser.add_argument('--wl', default=7, type=int,
                        help='wl smoothing for seq. ')
    parser.add_argument('--order', default=2, type=int,
                        help='order smoothing for seq. ')

    parser.add_argument("--log", type=str, default='SENTRY',
                        help="Where to save logs, checkpoints and debugging images.")

    parser.add_argument('-nc', '--num_classes', default=6, type=int, metavar='N',
                        help='number of classes')
                        
    parser.add_argument('--compressed_size', default=300, type=int, metavar='N',
                        help='compressed size')
    parser.add_argument("--sampling_mode", type=str, default='random_keypoints', choices=["random_keypoints", "uniform_sampling", "learnable_attention", "fixed_random_projection", "seq_attn_img_attn"],
                        help="Sampling Mode.")
    
    parser.add_argument('-ks_img', '--kernel-size-image', default=3, type=int, metavar='N',
                        help='number of classes')

    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    parser.add_argument('--Normalize_latlon', action='store_true',
                        help='Normalize lat and lon values.')

    parser.add_argument('--traj_eps', default=1e-1, type=float,
                        help='eps for the attack model for seq. ')
    parser.add_argument('--traj_alpha', default=1e-2, type=float,
                        help='alpha for the attack model for seq. ')
    parser.add_argument('--map_eps', default=3e-2, type=float,
                        help='eps for the attack model for seq. ')
    parser.add_argument('--map_alpha', default=5e-3, type=float,
                        help='alpha for the attack model for seq. ')

    parser.add_argument('--joint_pgd_steps', default=5, type=int,
                        help='Joint PGD Steps. ')

    parser.add_argument('--attack_type', default='joint_adaptive_pgd', type=str,
                        choices=['image_pgd',
                            'seq_pgd',
                            'joint_adaptive_pgd'])

    parser.add_argument("--eps_map", default=3e-2, type=float)
    parser.add_argument("--alpha_map", default=5e-3, type=float)
    parser.add_argument("--pgd_steps", default=5, type=int)
    parser.add_argument("--eot_samples", default=1, type=int)

    args = parser.parse_args()
    main(args)
