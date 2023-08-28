import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights, ResNet50_Weights
import pdb
import math
import os
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights
from torchvision.transforms import functional as F
import torch.nn.functional as tF
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
from models.facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class FeatureTemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureTemporalAttention, self).__init__()
        # Feature attention components
        self.query_f = nn.Linear(feature_dim, feature_dim)
        self.key_f = nn.Linear(feature_dim, feature_dim)
        self.value_f = nn.Linear(feature_dim, feature_dim)
        
        # Temporal attention components
        self.query_t = nn.Linear(feature_dim, feature_dim)
        self.key_t = nn.Linear(feature_dim, feature_dim)
        self.value_t = nn.Linear(feature_dim, feature_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Feature attention
        Q_f = self.query_f(x)
        K_f = self.key_f(x)
        V_f = self.value_f(x)
        attention_scores_f = torch.einsum('bti,bti->bti', Q_f, K_f)
        attention_probs_f = self.softmax(attention_scores_f)
        context_vectors_f = torch.einsum('bti,bti->bti', attention_probs_f, V_f)
        
        # Temporal attention
        Q_t = self.query_t(x)
        K_t = self.key_t(x)
        V_t = self.value_t(x)
        attention_scores_t = torch.einsum('bti,btj->btij', Q_t, K_t)
        attention_probs_t = self.softmax(attention_scores_t)
        context_vectors_t = torch.einsum('btij,btj->bti', attention_probs_t, V_t)
        
        # Combine the context vectors (simple sum in this example)
        combined_context = context_vectors_f + context_vectors_t
        
        return self.dropout(combined_context)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.3)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=15)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 0.1), num_layers)
        self.dropout = nn.Dropout(0.3)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)
        return output

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.reshape(batch_size * sequence_length, C, H, W)
        x = self.resnet(x)
        x = x.reshape(batch_size, sequence_length, -1)
        return x
    
class FacialFeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FacialFeatureExtractor,self).__init__(*args, **kwargs)

        self.incep_resnet = InceptionResnetV1(pretrained='vggface2').eval()
        # self.incep_resnet.last_linear = nn.Identity()

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.incep_resnet(x)
        x = x.view(batch_size, sequence_length, -1)
        return x


import torch.nn as nn

class OpticalFlowFeatureExtractor(nn.Module):
    def __init__(self, d_model=16):
        super(OpticalFlowFeatureExtractor, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # First convolution layer for RGB images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, d_model)

    def forward(self, x):
        batch_size, sequence_length, H, W, C = x.size()
        
        x = x.reshape(batch_size * sequence_length, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(batch_size, sequence_length, -1)
        return x


class CrossingIntentPredictor(nn.Module):
    def __init__(self,):
        super(CrossingIntentPredictor, self).__init__()
        self.observe_length = 15
        image_feature_size = 2048  
        description_feature_size = 768
        bbox_feature_size = 4
        skeleton_feature_size = 32
        d_model = 16
        nhead = 4
        num_layers = 3
        dim_feedforward = 32
        flow_map_feature_size = 16


        self.image_feature_extractor = ImageFeatureExtractor()
        self.image_transformer = TransformerEncoder(image_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.optical_flow_feature_extractor = OpticalFlowFeatureExtractor(flow_map_feature_size)
        self.whole_image_transformer = TransformerEncoder(image_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.description_compression = nn.Sequential(
                                        nn.Linear(description_feature_size, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, d_model)
                                    )      
        self.bbox_skeleton_transformer = TransformerEncoder(flow_map_feature_size + skeleton_feature_size + bbox_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.joint_attention = FeatureTemporalAttention(d_model * 4)
        self.batch_norm = nn.BatchNorm1d(d_model * 4) 
        self.fc = nn.Sequential(
            nn.Linear(d_model * 4, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
        
    def forward(self, data):
        bbox = data['bboxes'][:, :self.observe_length, :].type(FloatTensor)
        images = data['cropped_images'][:, :self.observe_length, :].type(FloatTensor).permute(0, 2, 1, 3, 4)
        skeleton = data['skeleton'][:, :self.observe_length, :].type(FloatTensor)
        skeleton = normalize_skeleton_based_on_bbox(skeleton, bbox)
        bbox = normalize_bbox(bbox)
        skeleton = skeleton.view(skeleton.shape[0], self.observe_length, -1)
        whole_images = data['images'][:, :self.observe_length, :].type(FloatTensor).permute(0, 2, 1, 3, 4)
        description = data['description'][:, :self.observe_length, :].type(FloatTensor)
        flow_map = data['cropped_flows'][:, :self.observe_length, :].type(FloatTensor)
        image_features = self.image_feature_extractor(images)
        image_features = self.image_transformer(image_features)
        whole_image_features = self.image_feature_extractor(whole_images)
        whole_image_features = self.whole_image_transformer(whole_image_features)
        description_features = self.description_compression(description)
        flow_map = self.optical_flow_feature_extractor(flow_map)
        bbox_skeleton_flow = torch.cat([bbox, skeleton, flow_map], dim=-1)
        bbox_skeleton_flow_features = self.bbox_skeleton_transformer(bbox_skeleton_flow)

        combined_features = torch.cat([image_features, whole_image_features, 
                                    bbox_skeleton_flow_features, description_features], dim=2)
        context_vectors = self.joint_attention(combined_features)
        context_vectors = context_vectors[:,-1,:]

        output = self.fc(self.batch_norm(context_vectors))
        return output.squeeze()

    def build_optimizer(self):
        backbone_lr = 5e-5
        transformer_lr = 5e-4

        backbone_parameters = list(self.image_feature_extractor.resnet.parameters())
        backbone_param_ids = {id(p): True for p in backbone_parameters}
        backbone_param_ids = {id(p): True for p in backbone_parameters}
        
        transformer_parameters = [p for p in self.parameters() if id(p) not in backbone_param_ids]
        
        param_group = [
            {'params': backbone_parameters, 'lr': backbone_lr},
            {'params': transformer_parameters, 'lr': transformer_lr}
        ]
        
        optimizer = torch.optim.AdamW(param_group, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        return optimizer, scheduler

def normalize_bbox(bbox):
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    normalized_x = x / 1280
    normalized_y = y / 720
    normalized_w = w / 1280
    normalized_h = h / 720
    normalized_bbox = torch.stack([normalized_x, normalized_y, normalized_w, normalized_h], dim=-1)
    return normalized_bbox

def normalize_skeleton_based_on_bbox(skeleton, bbox):
    adjusted_x = skeleton[..., 0] + bbox[..., 0].unsqueeze(-1)
    adjusted_y = skeleton[..., 1] + bbox[..., 1].unsqueeze(-1)
    normalized_x = adjusted_x / 1280
    normalized_y = adjusted_y / 720
    normalized_skeleton = torch.stack([normalized_x, normalized_y], dim=-1)
    return normalized_skeleton

def visualize_flow(flow_image):
    # Normalize to [0, 1]
    flow_image_normalized = (flow_image - flow_image.min()) / (flow_image.max() - flow_image.min())
    plt.imshow(flow_image_normalized)
    plt.colorbar()
    plt.title('Flow Image Visualization')
    plt.show()


