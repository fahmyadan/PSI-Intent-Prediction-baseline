import os
import torch
import json
import pdb
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def validate_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    with torch.no_grad():
        for itern, data in enumerate(dataloader):
            intent_logit = model.forward(data)
            intent_prob = torch.sigmoid(intent_logit)

            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)


            recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                    intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

            if itern % args.print_freq == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

        recorder.eval_intent_epoch_calculate(writer)

    return recorder


def test_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    recorder.eval_epoch_reset(epoch, niters)
    with torch.no_grad():
        for itern, data in enumerate(dataloader):
            intent_logit = model.forward(data)
            intent_prob = torch.sigmoid(intent_logit)

            # 1. intent loss
            if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
                gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
                gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)

            recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                    intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

        recorder.eval_intent_epoch_calculate(writer)

    return recorder


def predict_intent(model, dataloader, args, dset='test'):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        print(itern)
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)
        for i in range(len(data['frames'])):
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent
            int_prob = intent_prob[i].item()
            int_pred = round(int_prob) # <0.5 --> 0, >=0.5 --> 1.

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['intent'] = int_pred
            dt[vid][pid][fid]['intent_prob'] = int_prob

    with open(os.path.join(args.checkpoint_path, 'results', f'{dset}_intent_pred.json'), 'w') as f:
        json.dump(dt, f)

def validate_traj(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        traj_pred = model(data)
        traj_gt = data['bboxes'][:, args.observe_length: , :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        # if args.normalize_bbox == 'subtract_first_frame':
        #     traj_pred = traj_pred + data['bboxes'][:, :1, :].type(FloatTensor)
        recorder.eval_traj_batch_update(itern, data, traj_gt.detach().cpu().numpy(), traj_pred.detach().cpu().numpy())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

    recorder.eval_traj_epoch_calculate(writer)

    return recorder


def predict_traj(model, dataloader, args, dset='test'):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        traj_pred = model(data)
        # traj_gt = data['original_bboxes'][:, args.observe_length:, :].type(FloatTensor)
        traj_gt = data['bboxes'][:, args.observe_length:, :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        # print("Prediction: ", traj_pred.shape)

        for i in range(len(data['frames'])): # for each sample in a batch
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['traj'] = traj_pred[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    # print("saving prediction...")
    with open(os.path.join(args.checkpoint_path, 'results', f'{dset}_traj_pred.json'), 'w') as f:
        json.dump(dt, f)



def get_test_traj_gt(model, dataloader, args, dset='test'):
    model.eval()
    gt = {}
    for itern, data in enumerate(dataloader):
        traj_pred = model(data)
        traj_gt = data['bboxes'][:, args.observe_length:, :].type(FloatTensor)
        # traj_gt = data['original_bboxes'][:, args.observe_length:, :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        # print("Prediction: ", traj_pred.shape)

        for i in range(len(data['frames'])): # for each sample in a batch
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in gt:
                gt[vid] = {}
            if pid not in gt[vid]:
                gt[vid][pid] = {}
            if fid not in gt[vid][pid]:
                gt[vid][pid][fid] = {}
            gt[vid][pid][fid]['traj'] = traj_gt[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    with open(os.path.join(f'./test_gt/{dset}_traj_gt.json'), 'w') as f:
        json.dump(gt, f)