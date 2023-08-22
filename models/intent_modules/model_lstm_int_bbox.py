import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_size):
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final layer
        self.pooling = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, num_channels, height, width]
        batch_size, sequence_length, C, H, W = x.size()
        
        # Reshape x to: [batch_size * sequence_length, num_channels, height, width]
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.resnet(x)
        
        # Reshape x back to: [batch_size, sequence_length, ...]
        x = x.view(batch_size, sequence_length, -1)
        
        # Average across the sequence_length dimension
        x = x.mean(dim=1)
        
        # Now x is of shape [batch_size, num_features]
        x = self.pooling(x.unsqueeze(2).unsqueeze(3)).squeeze(-1).squeeze(-1)
        
        return x

class LSTMIntBbox(nn.Module):
    def __init__(self, args, model_configs):
        super(LSTMIntBbox, self).__init__()
        self.args = args
        self.model_configs = model_configs
        self.observe_length = self.args.observe_length
        self.predict_length = self.args.predict_length

        self.backbone = models.resnet50(pretrained=True)
        self.intent_predictor = LSTMInt(self.args, self.model_configs['intent_model_opts'])
        # intent predictor, always output (bs x 1) intention logits

        self.module_list = self.intent_predictor.module_list
        self.network_list = [self.intent_predictor]
        self.image_feature_extractor = ImageFeatureExtractor(output_size=(1, 1))
        
    def forward(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        images = data['cropped_images'].type(FloatTensor)  # Assume 'images' key in data dictionary
        description = data['description'].type(FloatTensor)  # Corrected key in data dictionary
        
        # Extract image features
        image_features = self.image_feature_extractor(images)
        
        combined_features = torch.cat([bbox[:, -1, :], image_features, description.mean(dim=1)], dim=1)
        
        # Pass concatenated features to intent predictor
        intent_pred = self.intent_predictor(combined_features.unsqueeze(1), None)
        
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
        self.activation = nn.Sigmoid()

        self.module_list = [self.encoder, self.fc] #, self.fc_emb, self.decoder

    def forward(self, enc_input, dec_input_emb=None):
        enc_output, (enc_hc, enc_nc) = self.encoder(enc_input)
        enc_last_output = enc_output[:, -1:, :] 
        output = self.fc(enc_last_output)
        outputs = output.unsqueeze(1) 
        return outputs  


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
