import numpy as np
from utils.utils import device
from torch import nn
import torch.nn.functional as F
import torch

class Combiner(nn.Module):
    def __init__(self, clip_feature_dim, projection_dim, hidden_dim):
        super(Combiner, self).__init__()
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, target_imgfeats, text_feats):
        text_projfeats = self.dropout1(F.relu(self.text_projection_layer(text_feats)))
        img_projfeats = self.dropout2(F.relu(self.image_projection_layer(target_imgfeats)))

        raw_combfeats = torch.cat((text_projfeats, img_projfeats), dim=-1)
        combined_feats = self.dropout3(F.relu(self.combiner_layer(raw_combfeats)))
        dynamic_scalar = self.dynamic_scalar(raw_combfeats)
        output = self.output_layer(combined_feats) + \
            dynamic_scalar * text_feats + (1 - dynamic_scalar) * target_imgfeats
        return output
        # return F.normalize(output, dim=-1)
    
    def get_loss(self, refer_feats, combiner_feats, images_in_batch):
        logits = 100 * F.normalize(refer_feats, dim=-1) @ F.normalize(combiner_feats, dim=-1).T
        ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
        return self.crossentropy_criterion(logits, ground_truth)