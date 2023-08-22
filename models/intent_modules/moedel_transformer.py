import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights
import pdb
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, num_features=4):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Linear(feature_dim, num_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, *features):
        concat_features = torch.cat(features, dim=-1)
        attention_weights = self.softmax(self.attention(concat_features))
        out_features = []
        for i, feature in enumerate(features):
            out_features.append(feature * attention_weights[..., i:i+1])
        return out_features

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 15, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_layers)
        
    def forward(self, src):
        src = self.embedding(src)
        src = src + self.positional_encoding
        output = self.transformer_encoder(src)
        return output

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.resnet(x)
        x = x.view(batch_size, sequence_length, -1)
        return x

class CrossingIntentPredictor(nn.Module):
    def __init__(self,):
        super(CrossingIntentPredictor, self).__init__()
        image_feature_size = 2048  # From the output of ResNet50
        description_feature_size = 768
        bbox_feature_size = 4
        skeleton_feature_size = 32
        d_model = 512
        nhead = 8
        num_layers = 6
        self.observe_length = 15
        dim_feedforward = 2048
        
        self.image_feature_extractor = ImageFeatureExtractor()
        self.image_transformer = TransformerEncoder(image_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.whole_image_feature_extractor = ImageFeatureExtractor()
        self.whole_image_transformer = TransformerEncoder(image_feature_size, d_model, nhead, num_layers, dim_feedforward)
        self.image_feature_bn = nn.BatchNorm1d(image_feature_size)
        self.whole_image_feature_bn = nn.BatchNorm1d(image_feature_size)
        #self.description_transformer = TransformerEncoder(description_feature_size, d_model, nhead, num_layers, dim_feedforward)
        #self.bbox_transformer = TransformerEncoder(bbox_feature_size, d_model, nhead, 2, dim_feedforward)
        #self.skeleton_transformer = TransformerEncoder(skeleton_feature_size, d_model, nhead, 3, dim_feedforward)
        self.bbox_skeleton_transformer = TransformerEncoder(bbox_feature_size + skeleton_feature_size, d_model, nhead, 3, dim_feedforward)
        self.feature_attention = FeatureAttention(d_model * 3) 
        self.batch_norm = nn.BatchNorm1d(d_model * 3) 
        self.fc = nn.Linear(d_model * 3, 1)
        self.sigmoid = nn.Sigmoid()
        xavier_initialize(self)
        
    def forward(self, data):
        bbox = data['bboxes'][:, :self.observe_length, :].type(FloatTensor)
        images = data['cropped_images'][:, :self.observe_length, :].type(FloatTensor)
        skeleton = data['skeleton'][:, :self.observe_length, :].type(FloatTensor)
        skeleton = normalize_skeleton_based_on_bbox(skeleton, bbox)
        bbox = normalize_bbox(bbox)
        skeleton = skeleton.view(skeleton.shape[0], self.observe_length, -1)
        whole_images = data['images'][:, :self.observe_length, :].type(FloatTensor)
        #description = data['description'][:, :self.observe_length, :].type(FloatTensor)
        image_features = self.image_feature_extractor(images)
        N, T, C = image_features.size()
        image_features = image_features.view(N*T, C)
        image_features = self.image_feature_bn(image_features)
        image_features = image_features.view(N, T, C)
        image_features = self.image_transformer(image_features)

        # skeleton_features = self.skeleton_transformer(skeleton)
        whole_image_features = self.whole_image_feature_extractor(whole_images)
        N, T, C = whole_image_features.size()
        whole_image_features = whole_image_features.view(N*T, C)
        whole_image_features = self.whole_image_feature_bn(whole_image_features)
        whole_image_features = whole_image_features.view(N, T, C)
        whole_image_features = self.whole_image_transformer(whole_image_features)
        #description_features = self.description_transformer(description)
        #bbox_features = self.bbox_transformer(bbox)
        bbox_skeleton = torch.cat([bbox, skeleton], dim=-1)
        bbox_skeleton_features = self.bbox_skeleton_transformer(bbox_skeleton)
        image_features, whole_image_features, bbox_skeleton_features = self.feature_attention(
            image_features, whole_image_features, bbox_skeleton_features
        )
        combined_features = torch.cat([image_features[:, -1, :], 
                                       whole_image_features[:, -1, :], 
                                       bbox_skeleton_features[:, -1, :]], dim=1)
        
        final = self.fc(self.batch_norm(combined_features))
        output = self.sigmoid(final)
        return output.squeeze(-1)

    def build_optimizer(self):
        learning_rate = 0.0001
        param_group = []
        param_group += [{'params': self.parameters(), 'lr': learning_rate}]
        optimizer = torch.optim.Adam(param_group, lr=learning_rate, eps=1e-7)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

def xavier_initialize(model):
    """Initialize the weights of the model using Xavier initialization."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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
