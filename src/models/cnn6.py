import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock5x5(nn.Module):  # for CNN6
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=stride,
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        return x
    
class _CNN6(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, in_channel=1):
        super(_CNN6, self).__init__()

        self.do_dropout = do_dropout
        self.conv_block1 = ConvBlock5x5(in_channels=in_channel, out_channels=64, stride=(1, 1))
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128, stride=(1, 1))
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256, stride=(1, 1))
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512, stride=(1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block2(x)
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block3(x)
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block4(x)
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3)  # mean over time dim
        (x1, _) = torch.max(x, dim=2)  # max over freq dim
        x2 = torch.mean(x, dim=2)  # mean over freq dim (after mean over time)
        x = x1 + x2

        return x
    
class CNN6(nn.Module):
    def __init__(self, num_classes, do_dropout, from_scratch, path_to_weight, in_channel, embed_dim=512):
        super(CNN6, self).__init__()
        
        self.cnn6 = _CNN6(num_classes=num_classes, do_dropout=do_dropout, in_channel=in_channel)
        self.num_features = self.embed_dim = embed_dim
        self.final_feat_dim = self.num_features
        # load state dict only for PANN weights
        # if loading own weights, load them from outside
        if not from_scratch:
            print("weights loading...")
            weights = torch.load(path_to_weight)['model']
            state_dict = {k: v for k, v in weights.items() if k in self.cnn6.state_dict().keys()}
            self.cnn6.load_state_dict(state_dict, strict=False)
            print("weights loaded!")

    def forward(self, x):
        return self.cnn6(x)