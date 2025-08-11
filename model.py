# coding=gb2312

import torch
import torch.nn as nn
from efficientnet_pytorch import Efficient
from HigherModels import Attention, MHeadAttention, MeanPooling

class DeformableCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableCNNBlock, self).__init__()
       
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size, stride, padding)
        self.dcnv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.dcnv(x, offset)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SelfNet(nn.Module):
    def __init__(self, label_dim=2, backbone='efficientnet-b0', pretrain=True, head_num=4, model_type=1):
        super(SelfNet, self).__init__()
        
        if not pretrain:
            print(f'{backbone} Models are trained from scratch（No ImageNet pre-training）。')
            self.effnet = EfficientNet.from_name(backbone, in_channels=3)
        else:
            print(f'Use ImageNet pretrained {backbone} model。')
            self.effnet = EfficientNet.from_pretrained(backbone, in_channels=3)
        
    
        last_block = list(self.effnet._blocks)[-1]
        in_channels = last_block._project_conv.out_channels
        

        if backbone == 'efficientnet-b0':
            in_channels = 1280
        




 
        kernel = (1, 3)
        padding = (0, 0) 
        stride = (1, 1)

       
        print(f"Use conv type of  {model_type}.")
        if model_type == 1:
            print("Use Standard Conv")
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        elif model_type == 2:
            print("Use dilated conv (D=1)")
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding, dilation=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        elif model_type == 3:
            print("Use depthwise separable conv")
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding, groups=in_channels),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)
            )
        elif model_type == 4:
            print("Use depthwise conv")
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        elif model_type == 5:
            print("Use pointwise conv(with AdaptivePool)")
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((3, 1))
            )
        elif model_type == 6:
            print("Use deformable conv")
            self.conv_block = DeformableCNNBlock(in_channels, in_channels, kernel_size=kernel, padding=padding)
        else:
            raise ValueError(f"Unsupported model types: {model_type}")





      


        

        if head_num > 1:
            print(f'Model use {head_num} attention heads')
            self.attention = MHeadAttention(
                in_channels,
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid',
                head_num=head_num
            )
        elif head_num == 1:
            print('The model uses a single-head attention mechanism')
            self.attention = Attention(
                in_channels,
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid'
            )
        elif head_num == 0:
            print('Model uses average pooling (no attention mechanism)')
            self.attention = MeanPooling(
                in_channels,
                label_dim,
                att_activation=None,
                cla_activation='sigmoid'
            )
        else:
            raise ValueError('Number of attention heads must be integer >= 0, 0 = average pooling, 1 = single-head attention, >1 = multi-head attention')
        
     
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, label_dim)

    def extract_features(self, x):
   
        x = self.effnet.extract_features(x)  
        return x

    def forward(self, x):


        x = self.extract_features(x) 


        x = self.conv_block(x)



 







        
        x = self.avgpool(x) 
        x = torch.flatten(x, 1) 
        x = self.fc(x)  
        return x

    def freeze_feature_extractor(self):
        """Freeze all layers of the feature extractor (EfficientNet)."""
        for param in self.effnet.parameters():
            param.requires_grad = False
        print(" feature extractor is frozen.")

    def unfreeze_all(self):
        """Unfreeze all layers of the model."""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfreeze all the layers.")