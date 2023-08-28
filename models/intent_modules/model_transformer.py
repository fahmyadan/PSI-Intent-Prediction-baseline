import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights, ResNet50_Weights
import pdb
import math
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
from models.facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class JointAttention(nn.Module):
    def __init__(self, feature_dim):
        super(JointAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, time_steps, feature_dim]
        
        Q = self.dropout(self.query(x))
        K = self.dropout(self.key(x))
        V = self.dropout(self.value(x))

        # Attention scores shape: [batch_size, time_steps, time_steps]
        attention_scores = torch.einsum('bti,btj->btij', Q, K) / math.sqrt(x.size(-1))
        
        attention_probs = self.softmax(attention_scores)

        # Context vectors shape: [batch_size, time_steps, feature_dim]
        context_vectors = torch.einsum('btij,btj->bti', attention_probs, V)
        
        return context_vectors


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, num_features=4):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Linear(feature_dim, num_features)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, *features):
        concat_features = torch.cat(features, dim=-1)
        attention_weights = self.softmax(self.attention(concat_features))
        out_features = []
        for i, feature in enumerate(features):
            out_features.append(self.dropout(feature * attention_weights[..., i:i+1]))
        return out_features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=15)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 0.1), num_layers)
        self.dropout = nn.Dropout(0.1)

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
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.resnet(x)
        x = x.view(batch_size, sequence_length, -1)
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


class CrossingIntentPredictor(nn.Module):
    def __init__(self,):
        super(CrossingIntentPredictor, self).__init__()
        image_feature_size = 2048  # From the output of ResNet50
        description_feature_size = 768
        bbox_feature_size = 4
        skeleton_feature_size = 32
        d_model = 128
        nhead = 4
        num_layers = 2
        self.observe_length = 15
        dim_feedforward = 256
        
        #self.image_feature_extractor = ImageFeatureExtractor()
        #self.image_transformer = TransformerEncoder(image_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.whole_image_feature_extractor = ImageFeatureExtractor()
        self.facial_embeddings = FacialFeatureExtractor()
        self.whole_image_transformer = TransformerEncoder(image_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.description_transformer = TransformerEncoder(description_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.bbox_skeleton_transformer = TransformerEncoder(bbox_feature_size + skeleton_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.feature_attention = FeatureAttention(d_model * 3) 
        self.joint_attention = JointAttention(d_model * 3)
        self.batch_norm = nn.BatchNorm1d(5760) 
        self.fc = nn.Linear(5760, 1)
        
    def forward(self, data):
        bbox = data['bboxes'][:, :self.observe_length, :].type(FloatTensor)
        cropped_images =data['cropped_images'][:, :self.observe_length, :].type(FloatTensor)
        #images = data['cropped_images'][:, :self.observe_length, :].type(FloatTensor)
        skeleton = data['skeleton'][:, :self.observe_length, :].type(FloatTensor)
        skeleton = normalize_skeleton_based_on_bbox(skeleton, bbox)
        bbox = normalize_bbox(bbox)
        skeleton = skeleton.view(skeleton.shape[0], self.observe_length, -1)
        whole_images = data['images'][:, :self.observe_length, :].type(FloatTensor)
        description = data['description'][:, :self.observe_length, :].type(FloatTensor)
        #image_features = self.image_feature_extractor(images)
        #image_features = self.image_transformer(image_features)

        whole_image_features = self.whole_image_feature_extractor(whole_images)
        facial_embeddings =  self.facial_embeddings(cropped_images)
        whole_image_features = self.whole_image_transformer(whole_image_features)
        description_features = self.description_transformer(description)
        bbox_skeleton = torch.cat([bbox, skeleton], dim=-1)
        bbox_skeleton_features = self.bbox_skeleton_transformer(bbox_skeleton)
        # whole_image_features, bbox_skeleton_features, description_features = self.feature_attention(
        #     whole_image_features, bbox_skeleton_features, description_features
        # )
        # combined_features = torch.cat([whole_image_features[:, -1, :], 
        #                                bbox_skeleton_features[:, -1, :],
        #                                description_features[:, -1, :]], dim=1)
        # output = self.fc(self.batch_norm(combined_features))
        combined_features = torch.cat([whole_image_features, bbox_skeleton_features, description_features], dim=2)
        context_vectors = self.joint_attention(combined_features)
        context_vectors = context_vectors.view(context_vectors.size(0), -1)
        output = self.fc(self.batch_norm(context_vectors))
        return output.squeeze()

    def build_optimizer(self):
        backbone_lr = 1e-5
        transformer_lr = 2e-4
        
        # Extract parameters of the ResNet101 backbone
        backbone_parameters = list(self.whole_image_feature_extractor.resnet.parameters())
        backbone_param_ids = {id(p): True for p in backbone_parameters}
        
        # Extract parameters of the other transformers excluding the backbone
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
