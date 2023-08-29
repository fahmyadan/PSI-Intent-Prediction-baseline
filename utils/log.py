import os
import numpy as np
from utils.utils import AverageMeter
from utils.metrics import evaluate_traj, evaluate_intent
import json


class RecordResults():
    def __init__(self, args=None, intent=True, traj=True, driving=False, reason=False, evidential=False,
                 extract_prediction=False):
        self.args = args
        self.save_output = extract_prediction
        self.intent = intent
        self.driving = driving
        self.traj = traj
        self.reason = reason
        self.evidential = evidential

        self.all_train_results = {}
        self.all_eval_results = {}
        self.all_val_results = {}

        # cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        self.result_path = os.path.join(self.args.checkpoint_path, 'results')
        if not os.path.isdir(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)

        self._log_file = os.path.join(self.args.checkpoint_path, 'log.txt')
        open(self._log_file, 'w').close()

        # self.log_args(self.args)

    def log_args(self, args):
        args_file = os.path.join(self.args.checkpoint_path, 'args.txt')
        with open(args_file, 'a') as f:
            json.dump(args.__dict__, f, indent=2)
        ''' 
            parser = ArgumentParser()
            args = parser.parse_args()
            with open('commandline_args.txt', 'r') as f:
            args.__dict__ = json.load(f)
        '''

    def train_epoch_reset(self, epoch, nitern):
        # 1. initialize log info
        # (1.1) loss log list
        self.log_loss_total = AverageMeter()
        self.log_loss_intent = AverageMeter()
        self.log_loss_traj = AverageMeter()
        # (1.2) intent
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        # (1.3) trajectory - args.image_shape
        self.traj_gt = []  # normalized, N x 4, (0, 1) range
        self.traj_ori_gt = []
        self.traj_pred = []  # N x 4 dimension, (0, 1) range
        # (1.4) store all results
        self.train_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

        self.frames_list = []
        self.video_list = []
        self.log_loss_driving_speed = AverageMeter()
        self.log_loss_driving_dir = AverageMeter()
        # (1.2) driving
        self.driving_speed_gt = []
        self.driving_speed_pred = []
        self.driving_dir_gt = []
        self.driving_dir_pred = []

    def train_intent_batch_update(self, itern, data, intent_gt, intent_prob_gt, intent_prob, loss, loss_intent):
        # 3. Update training info
        # (3.1) loss log list
        bs = intent_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_intent.update(loss_intent, bs)
        # (3.2) training data info
        if intent_prob != []:
            # (3.3) intent
            self.intention_gt.extend(intent_gt)  # bs
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)  # bs

            # assert len(self.intention_gt[0]) == 1 #self.args.predict_length, intent only predict 1 result
        else:
            pass

        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path+"/training_info.txt", 'a') as f:
                f.write('Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  Intent Loss: {:.4f} \n'.format(
                    self.epoch, self.args.epochs, itern, self.nitern, self.log_loss_total.avg,
                    self.log_loss_intent.avg))


    def train_intent_epoch_calculate(self, writer=None):
        print('----------- Training results: ------------------------------------ ')
        if self.intention_pred:
            intent_results = evaluate_intent(np.array(self.intention_gt), np.array(self.intention_prob_gt),
                                             np.array(self.intention_pred), self.args)
            self.train_epoch_results['intent_results'] = intent_results

        print('----------------------------------------------------------- ')
        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename='train')

        # write scalar to tensorboard
        if writer:
            for key in ['MSE', 'Acc', 'F1', 'mAcc']:
                val = intent_results[key]
                writer.add_scalar(f'Train/Results/{key}', val, self.epoch)

            for i in range(self.args.intent_num):
                for j in range(self.args.intent_num):
                    val = intent_results['ConfusionMatrix'][i][j]
                    writer.add_scalar(f'ConfusionMatrix/train{i}_{j}', val, self.epoch)
    def train_driving_batch_update(self, itern, data, speed_gt, direction_gt, speed_pred_logit, dir_pred_logit,
                                   loss, loss_driving_speed, loss_driving_dir):
        # 3. Update training info
        # (3.1) loss log list
        bs = speed_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_driving_speed.update(loss_driving_speed, bs)
        self.log_loss_driving_dir.update(loss_driving_dir, bs)
        # (3.2) training data info

        self.driving_speed_gt.extend(speed_gt)  # bs
        self.driving_dir_gt.extend(direction_gt)
        self.driving_speed_pred.extend(np.argmax(speed_pred_logit, axis=-1))  # bs
        self.driving_dir_pred.extend(np.argmax(dir_pred_logit, axis=-1))  # bs


        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", 'a') as f:
                f.write('Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  driving speed Loss: {:.4f} |  driving dir Loss: {:.4f} \n'.format(
                    self.epoch, self.args.epochs, itern, self.nitern, self.log_loss_total.avg,
                    self.log_loss_driving_speed.avg, self.log_loss_driving_dir.avg))


    def train_driving_epoch_calculate(self, writer=None):
        print('----------- Training results: ------------------------------------ ')
        if self.driving:
            driving_results = evaluate_driving(np.array(self.driving_speed_gt), np.array(self.driving_dir_gt),
                                             np.array(self.driving_speed_pred), np.array(self.driving_dir_pred),
                                               self.args)
            self.train_epoch_results['driving_results'] = driving_results
            # {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}


        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename='train')

        # write scalar to tensorboard
        if writer:
            for key in ['speed_Acc', 'speed_mAcc', 'direction_Acc', 'direction_mAcc']: # driving_results.keys(): #
                if key not in driving_results.keys():
                    continue
                val = driving_results[key]
                print("results: ", key, val)
                writer.add_scalar(f'Train/Results/{key}', val, self.epoch)
        print('----------------------------------------------------------- ')


    def eval_epoch_reset(self, epoch, nitern, intent=True, traj=True, args=None):
        # 1. initialize log info
        # (1.2) training data info
        self.frames_list = []
        self.video_list = []
        self.ped_list = []
        # (1.3) intent
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        self.intention_rsn_gt = []
        self.intention_rsn_pred = []
        
        # (1.4) trajectory - args.image_shape
        self.traj_gt = []  # normalized, N x 4, (0, 1) range
        self.traj_ori_gt = [] # original bboxes before normalization. equal to bboxes if no normalization
        self.traj_pred = []  # N x 4 dimension, (0, 1) range

        self.eval_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

        self.log_loss_total = AverageMeter()
        self.log_loss_driving_speed = AverageMeter()
        self.log_loss_driving_dir = AverageMeter()

        self.driving_speed_gt = []
        self.driving_speed_pred = []
        self.driving_dir_gt = []
        self.driving_dir_pred = []

    def eval_intent_batch_update(self, itern, data, intent_gt, intent_prob, intent_prob_gt, intent_rsn_gt=None, intent_rsn_pred=None):
        # 3. Update training info
        # (3.1) loss log list
        bs = intent_gt.shape[0]
        # (3.2) training data info
        self.frames_list.extend(data['frames'].detach().cpu().numpy())  # bs x sq_length(60)
        assert len(self.frames_list[0]) == self.args.observe_length
        self.video_list.extend(data['video_id'])  # bs
        self.ped_list.extend(data['ped_id'])
        # print("save record: video list - ", data['video_id'])

        # (3.3) intent
        if intent_prob != []:
            self.intention_gt.extend(intent_gt)  # bs
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)  # bs
            if intent_rsn_gt is not None:
                self.intention_rsn_gt.extend(intent_rsn_gt)
                self.intention_rsn_pred.extend(intent_rsn_pred)
            # assert len(self.intention_gt[0]) == 1 #self.args.predict_length, intent only predict 1 result
        else:
            pass

    def eval_intent_epoch_calculate(self, writer):
        print('----------- Evaluate results: ------------------------------------ ')

        if self.intention_pred:
            intent_results = evaluate_intent(np.array(self.intention_gt), np.array(self.intention_prob_gt),
                                             np.array(self.intention_pred), self.args)
            self.eval_epoch_results['intent_results'] = intent_results

        print('----------------------finished evalcal------------------------------------- ')
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename='eval')
        print('log info finished')

        # write scalar to tensorboard
        if writer:
            for key in ['MSE', 'Acc', 'F1', 'mAcc']:
                val = intent_results[key]
                writer.add_scalar(f'Eval/Results/{key}', val, self.epoch)

            for i in range(self.args.intent_num):
                for j in range(self.args.intent_num):
                    val = intent_results['ConfusionMatrix'][i][j]
                    writer.add_scalar(f'ConfusionMatrix/eval{i}_{j}', val, self.epoch)

    # def save_results(self, prefix=''):
    #     self.result_path = os.path.join(self.args.checkpoint_path, 'results', f'epoch_{self.epoch}', prefix)
    #     if not os.path.isdir(self.result_path):
    #         os.makedirs(self.result_path)
    #     # 1. train results
    #     np.save(self.result_path + "/train_results.npy", self.all_train_results)
    #     # 2. eval results
    #     np.save(self.result_path + "/eval_results.npy", self.all_eval_results)
    #
    #     # 3. save data
    #     np.save(self.result_path + "/intent_gt.npy", self.intention_gt)
    #     np.save(self.result_path + "/intent_prob_gt.npy", self.intention_prob_gt)
    #     np.save(self.result_path + "/intent_pred.npy", self.intention_pred)
    #     np.save(self.result_path + "/frames_list.npy", self.frames_list)
    #     np.save(self.result_path + "/video_list.npy", self.video_list)
    #     np.save(self.result_path + "/ped_list.npy", self.ped_list)
    #     np.save(self.result_path + "/intent_rsn_gt.npy", self.intention_rsn_gt)
    #     np.save(self.result_path + "/intent_rsn_pred.npy", self.intention_rsn_pred)
    #

    # 3. Update traj training info
    def train_traj_batch_update(self, itern, data, traj_gt, traj_pred, loss, loss_traj):
        # evidence: bs x ts x 4: mu,v,alpha,beta

        # (3.1) loss log list
        bs, ts, dim = traj_gt.shape  # bs x 45 x 4
        self.log_loss_total.update(loss, bs)
        self.log_loss_traj.update(loss_traj, bs)
        # (3.2) training data info
        if traj_pred != []:
            self.traj_gt.extend(traj_gt)  # bs x pred_seq(45) x 4
            self.traj_pred.extend(traj_pred)  # bs x pred_seq(45) x 4
        else:
            pass

        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", 'a') as f:
                f.write(
                    'Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  Intent Loss: {:.4f} |  Traj Loss: {:.4f}\n'.format(
                        self.epoch, self.args.epochs, itern, self.nitern, self.log_loss_total.avg,
                        self.log_loss_intent.avg, self.log_loss_traj.avg))

    def train_traj_epoch_calculate(self, writer=None):
        print('----------- Training results: ------------------------------------ ')
        if self.traj_pred != []:
            traj_results = evaluate_traj(np.array(self.traj_gt), np.array(self.traj_pred), self.args)
            self.train_epoch_results['traj_results'] = traj_results
        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results

        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename='train')
        # write scalar to tensorboard
        if writer:
            for key in ['ADE', 'FDE', 'ARB', 'FRB']: #, 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
                for time in ['0.5', '1.0', '1.5']:
                    val = traj_results[key][time]
                    writer.add_scalar(f'Train/Results/{key}_{time}', val, self.epoch)


    def eval_traj_batch_update(self, itern, data, traj_gt, traj_pred):
        # 3. Update training info
        self.frames_list.extend(data['frames'].detach().cpu().numpy())  # bs x sq_length(60)
        assert len(self.frames_list[0]) == self.args.observe_length
        self.video_list.extend(data['video_id'])  # bs
        self.ped_list.extend(data['ped_id'])
        # (3.1) loss log list
        bs, ts, dim = traj_gt.shape # bs x 45 x 4
        self.traj_ori_gt.extend(data['bboxes'].detach().cpu().numpy())

        if traj_pred != []:
            self.traj_gt.extend(traj_gt)  # bs x pred_seq(45) x 4
            self.traj_pred.extend(traj_pred)  # bs x pred_seq(45) x 4
            assert len(self.traj_gt[0]) == self.args.predict_length
        else:
            pass


    def eval_traj_epoch_calculate(self, writer=None):
        print('----------- Eval results: ------------------------------------ ')
        if self.traj_pred != []:
            traj_results = evaluate_traj(np.array(self.traj_gt), np.array(self.traj_pred), self.args)
            self.eval_epoch_results['traj_results'] = traj_results

        # Update epoch to all results
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        # self.log_msg(msg='Epoch {} \n --------------------------'.format(self.epoch), filename='train_results.txt')
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename='eval')
        # write scalar to tensorboard
        if writer:
            for key in ['ADE', 'FDE', 'ARB', 'FRB']: #, 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
                for time in ['0.5', '1.0', '1.5']:
                    val = traj_results[key][time]
                    writer.add_scalar(f'Eval/Results/{key}_{time}', val, self.epoch)
                    print(f'Epoch {self.epoch}: {key}_{time}', val)
        print('----------------------------------------------------------- ')
    def eval_driving_epoch_calculate(self, writer):
        print('----------- Evaluate results: ------------------------------------ ')
        if self.driving:
            driving_results = evaluate_driving(np.array(self.driving_speed_gt), np.array(self.driving_dir_gt),
                                               np.array(self.driving_speed_pred), np.array(self.driving_dir_pred),
                                               self.args)
            self.eval_epoch_results['driving_results'] = driving_results
            # {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}
            for key in self.eval_epoch_results['driving_results'].keys():
                print(key, self.eval_epoch_results['driving_results'][key])
        # Update epoch to all results
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename='eval')


        # write scalar to tensorboard
        if writer:
            for key in ['speed_Acc', 'speed_mAcc', 'direction_Acc', 'direction_mAcc']:
                if key not in driving_results.keys():
                    continue
                val = driving_results[key]
                print("results: ", key, val)
                writer.add_scalar(f'Eval/Results/{key}', val, self.epoch)
        print('log info finished')
        print('----------------------finished results calculation------------------------------------- ')


    def log_msg(self, msg: str, filename: str = None):
        if not filename:
            filename = os.path.join(self.args.checkpoint_path, 'log.txt')
        else:
            pass
        savet_to_file = filename
        with open(savet_to_file, 'a') as f:
            f.write(str(msg) + '\n')

    def log_info(self, epoch: int, info: dict, filename: str = None):
        if not filename:
            filename = 'log.txt'
        else:
            pass
        for key in info:
            savet_to_file = os.path.join(self.args.checkpoint_path, filename + '_' + key + '.txt')
            self.log_msg(msg='Epoch {} \n --------------------------'.format(epoch), filename=savet_to_file)
            with open(savet_to_file, 'a') as f:
                    if type(info[key]) == str:
                        f.write(info[key] + "\n")
                    elif type(info[key]) == dict:
                        for k in info[key]:
                            f.write(k + ": " + str(info[key][k]) + "\n")
                    else:
                        f.write(str(info[key]) + "\n")
            self.log_msg(msg='.................................................'.format(self.epoch), filename=savet_to_file)

