import collections

from test import validate_intent, validate_traj
import torch
import numpy as np
import os
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import pdb
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_intent(start_epoch, model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer):
    pos_weight = torch.tensor(args.intent_positive_weight).to(device) # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight).to(device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(device),
        'BCELoss': torch.nn.BCELoss().to(device),
        'CELoss': torch.nn.CrossEntropyLoss(),
    }
    epoch_loss = {'loss_intent': [], 'loss_traj': []}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_intent_epoch(epoch, model, optimizer, criterions, epoch_loss, train_loader, args, recorder, writer)
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                  f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(args.checkpoint_path, f"epoch_{epoch}_checkpoint.pth"))

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            validate_intent(epoch, model, val_loader, args, recorder, writer)


def train_intent_epoch(epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        
        optimizer.zero_grad()
        intent_logit = model(data)
        gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor).unsqueeze(1)
        loss_edl = edl_loss(torch.digamma, gt_intent_prob, intent_logit.unsqueeze(1), epoch, 5, device)

        gt_disagreement = data['disagree_score'][:, args.observe_length]
        gt_consensus = (1 - gt_disagreement).to(device)
        gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
        #loss_intent = criterions['BCEWithLogitsLoss'](intent_logit , gt_intent)
        loss_intent = sigmoid_focal_loss(intent_logit, gt_intent, alpha=0.4, gamma=2, reduction='mean')

        loss_intent = torch.mean(torch.mul(gt_consensus, loss_intent))
        
        loss = loss_intent + loss_edl*0.01
        loss.backward()
        optimizer.step()

        # Record results
        batch_losses['loss'].append(loss.item())
        batch_losses['loss_intent'].append(loss_intent.item())
        batch_losses['loss_edl'].append(loss_edl.item())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                  f"loss_intent = {np.mean(batch_losses['loss_intent']): .4f} - "
                  f"loss_edl = {np.mean(batch_losses['loss_edl']): .4f}")
        intent_prob = torch.sigmoid(intent_logit)
        recorder.train_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                           gt_intent_prob.detach().cpu().numpy(),
                                           intent_prob.detach().cpu().numpy(),
                                           loss.item(), loss_intent.item())

    epoch_loss['loss_intent'].append(np.mean(batch_losses['loss_intent']))

    recorder.train_intent_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f'LearningRate', optimizer.param_groups[-1]['lr'], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f'Losses/{key}', np.mean(val), epoch)

    return epoch_loss


def edl_loss(func, y, logit, epoch_num, annealing_step=5, device=None):

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y = y.to(device)
    logit = logit.to(device)
    
    # Convert the logit to evidence format
    evidence_pos = F.relu(logit)
    evidence_neg = F.relu(-logit)
    alpha_pos = evidence_pos + 1
    alpha_neg = evidence_neg + 1

    alpha = torch.cat([alpha_neg, alpha_pos], dim=1)
    
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device)
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = kl_divergence(kl_alpha, 2, device=device)  # num_classes is still 2 for binary classification
    
    return torch.mean(A + annealing_coef * kl_div)

def kl_divergence(alpha, num_classes, device=None):

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device) * (1.0 / num_classes)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def train_traj(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer):
    pos_weight = torch.tensor(args.intent_positive_weight).to(device) # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight).to(device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(device),
        'BCELoss': torch.nn.BCELoss().to(device),
        'CELoss': torch.nn.CrossEntropyLoss().to(device),
        'L1Loss': torch.nn.L1Loss().to(device),
    }
    epoch_loss = {'loss_intent': [], 'loss_traj': []}

    for epoch in range(1, args.epochs + 1):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_traj_epoch(epoch, model, optimizer, criterions, epoch_loss, train_loader, args, recorder, writer)
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                  f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}, "
                  f"loss_traj = {np.mean(epoch_loss['loss_traj']): .4f}")

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            validate_traj(epoch, model, val_loader, args, recorder, writer)

        torch.save(model.state_dict(), args.checkpoint_path + f'/latest.pth')


def train_traj_epoch(epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        optimizer.zero_grad()
        traj_pred = model(data)
        traj_gt = data['bboxes'][:, args.observe_length:, :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        loss_traj = torch.tensor(0.).type(FloatTensor)
        if 'bbox_l1' in args.traj_loss:
            loss_bbox_l1 = torch.mean(criterions['L1Loss'](traj_pred, traj_gt))
            batch_losses['loss_bbox_l1'].append(loss_bbox_l1.item())
            loss_traj += loss_bbox_l1

        loss = args.loss_weights['loss_traj'] * loss_traj
        loss.backward()
        optimizer.step()

        # Record results
        batch_losses['loss'].append(loss.item())
        batch_losses['loss_traj'].append(loss_traj.item())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                  f"loss_traj = {np.mean(batch_losses['loss_traj']): .4f}, ")
        recorder.train_traj_batch_update(itern, data, traj_gt.detach().cpu().numpy(), traj_pred.detach().cpu().numpy(),
                                         loss.item(), loss_traj.item())

    epoch_loss['loss_traj'].append(np.mean(batch_losses['loss_traj']))

    recorder.train_traj_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f'LearningRate', optimizer.param_groups[-1]['lr'], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f'Losses/{key}', np.mean(val), epoch)

    return epoch_loss