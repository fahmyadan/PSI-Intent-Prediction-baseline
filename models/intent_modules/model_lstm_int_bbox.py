import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
from ..backbones.backbone_base import Backbone

class LSTMIntBbox(nn.Module):
    def __init__(self, args, model_configs):
        super(LSTMIntBbox, self).__init__()
        self.args = args
        self.model_configs = model_configs
        self.observe_length = self.args.observe_length
        self.predict_length = self.args.predict_length

        # self.backbone = args.backbone
        self.backbone = Backbone('resnet34', train_backbone=True)
        # self.intent_predictor = LSTMInt(self.args, self.model_configs['intent_model_opts'])
        # intent predictor, always output (bs x 1) intention logits
        self.intent_predictor = GRUInt(self.args, self.model_configs['intent_model_opts'])
        self.traj_predictor = None

        self.module_list = self.intent_predictor.module_list
        self.network_list = [self.intent_predictor]
        # self._reset_parameters()
        # self.optimizer = None
        # self.build_optimizer(args)

    def forward(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        # global_imgs = data['images']
        # local_imgs = data['cropped_images']
        dec_input_emb = None # as the additional emb for intent predictor
        # bbox: shape = [bs x observe_length x enc_input_dim]
        assert bbox.shape[1] == self.observe_length

        # 1. backbone feature (to be implemented for images)
        if self.backbone is not None:
            #Image Data shape: 128,15,3,224,224 | N-videos/batch size, frames, channels, height, width
            img_in = torch.flatten(data['cropped_images'], start_dim=0, end_dim=1) # N-v
            # img_in = data['images']
            img_feat_maps = self.backbone(img_in)

        # Feature map concat
        new_feat_maps = FeatureConcatenator()(img_feat_maps.values())
        flat_feat_map  = torch.flatten(new_feat_maps, start_dim=0, end_dim=1)

        # 2. intent prediction
        intent_pred = self.intent_predictor(flat_feat_map, dec_input_emb)
        # output shape: bs x int_pred_len=1 x int_dim=1

        return intent_pred.squeeze()

    def build_optimizer(self, args):
        param_group = []
        learning_rate = args.lr
        if self.backbone is not None:
            for name, param in self.backbone.named_parameters():
                if not self.args.freeze_backbone:
                    param.requres_grad = True
                    param_group += [{'params': param, 'lr': learning_rate * 0.1}]
                else:
                    param.requres_grad = False

        for net in self.network_list:  # [reason, intent, traj, truct] networks
            for module in net.module_list:
                param_group += [{'params': module.parameters(), 'lr': learning_rate}]

        optimizer = torch.optim.Adam(param_group, lr=args.lr, eps=1e-7)

        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

    def lr_scheduler(self, cur_epoch, args, gamma=10, power=0.75):
        decay = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def predict_intent(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        # global_imgs = data['images']
        # local_imgs = data['cropped_images']
        dec_input_emb = None # as the additional emb for intent predictor
        # bbox: shape = [bs x observe_length x enc_input_dim]
        assert bbox.shape[1] == self.observe_length

        # 1. backbone feature (to be implemented)
        if self.backbone is not None:
            pass

        # 2. intent prediction
        intent_pred = self.intent_predictor(bbox, dec_input_emb)
        # bs x int_pred_len=1 x int_dim=1
        return intent_pred.squeeze()

class LSTMInt(nn.Module):
    def __init__(self, args, model_opts):
        super(LSTMInt, self).__init__()

        enc_in_dim = model_opts['enc_in_dim']
        enc_out_dim = model_opts['enc_out_dim']
        # dec_in_emb_dim = model_opts['dec_in_emb_dim']
        # dec_out_dim = model_opts['dec_out_dim']
        output_dim = model_opts['output_dim']
        n_layers = model_opts['n_layers']
        dropout = model_opts['dropout']

        self.args = args

        self.enc_in_dim = enc_in_dim  # input bbox+convlstm_output context vector
        self.enc_out_dim = enc_out_dim
        self.encoder = nn.LSTM(
            input_size=self.enc_in_dim,
            hidden_size=self.enc_out_dim,
            num_layers=n_layers,
            batch_first=True,
            bias=True
        )

        self.output_dim = output_dim  # 2/3: intention; 62 for reason; 1 for trust score; 4 for trajectory.

        self.fc = nn.Sequential(
            nn.Linear(self.enc_out_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, self.output_dim)

        )

        if model_opts['output_activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif model_opts['output_activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.module_list = [self.encoder, self.fc] #, self.fc_emb, self.decoder
        # self._reset_parameters()
        # assert self.enc_out_dim == self.dec_out_dim

    def forward(self, enc_input, dec_input_emb=None):
        enc_output, (enc_hc, enc_nc) = self.encoder(enc_input)
        # because 'batch_first=True'
        # enc_output: bs x ts x (1*hiden_dim)*enc_hidden_dim --- only take the last output, concatenated with dec_input_emb, as input to decoder
        # enc_hc:  (n_layer*n_directions) x bs x enc_hidden_dim
        # enc_nc:  (n_layer*n_directions) x bs x enc_hidden_dim
        enc_last_output = enc_output[:, -1:, :]  # bs x 1 x hidden_dim
        output = self.fc(enc_last_output)
        outputs = output.unsqueeze(1) # bs x 1 --> bs x 1 x 1
        return outputs  # shape: bs x predict_length x output_dim, no activation


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

class GRUInt(nn.Module):
    def __init__(self, args, model_opts):
        super(GRUInt, self).__init__()

        enc_in_dim = model_opts['enc_in_dim']
        enc_out_dim = model_opts['enc_out_dim']
        output_dim = model_opts['output_dim']
        n_layers = model_opts['n_layers']
        dropout = model_opts['dropout']

        self.args = args

        self.enc_in_dim = enc_in_dim  # input bbox+convlstm_output context vector
        self.enc_out_dim = enc_out_dim

        self.temp_encoder = nn.GRU(
            input_size=self.enc_in_dim,
            hidden_size=self.enc_out_dim,
            num_layers=n_layers,
            batch_first=True,
            bias=True
        )

        self.output_dim = output_dim  # 2/3: intention; 62 for reason; 1 for trust score; 4 for trajectory.

        self.fc = nn.Sequential(
            nn.Linear(self.enc_out_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, self.output_dim)

        )

        if model_opts['output_activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif model_opts['output_activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.module_list = [self.temp_encoder, self.fc] #, self.fc_emb, self.decoder
        # self._reset_parameters()
        # assert self.enc_out_dim == self.dec_out_dim

    def forward(self, enc_input, dec_input_emb=None):
        enc_output, enc_hidden_state = self.temp_encoder(enc_input)

        #TODO: Fix the output 

        enc_last_output = enc_output[:, -1:, :]  # bs x 1 x hidden_dim
        output = self.fc(enc_last_output)
        outputs = output.unsqueeze(1) # bs x 1 --> bs x 1 x 1
        return outputs  # shape: bs x predict_length x output_dim, no activation


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


class FeatureConcatenator(nn.Module):
    def __init__(self):
        super(FeatureConcatenator, self).__init__()

    def forward(self, feature_maps):
        # Reshape feature maps to a common spatial dimension
        resized_maps = [nn.functional.adaptive_avg_pool2d(fm, (7, 7)) for fm in feature_maps]

        # Concatenate along the channel dimension
        concatenated_features = torch.cat(resized_maps, dim=1)

        return concatenated_features