import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # (input-7+6)//2+1=向上取整 3*224*224-->64*112*112
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (input-3)//2 ceil_mode=True向上取整 64*56*56

        self.conv2 = BasicConv2d(64, 64,  kernel_size=1, stride=1)
        # 64 * 56 * 56
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1)
        # 64 * 56 * 56 -> 192 * 54 * 54
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 192 * 54 * 54 -> 192 * 27 * 27
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # 192 * 27 * 27 -> 256 * 27 * 27
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 256 * 27 * 27 -> 480 * 13 * 13
        self.max_pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # 480 * 13 * 13 -> 512 * 13 * 13
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 512 * 13 * 13 -> 512 * 13 * 13
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # 512 * 13 * 13 -> 512 * 13 * 13
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # 512 * 13 * 13 -> 512 * 13 * 13
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 512 * 13 * 13 -> 832 * 13 * 13
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        # 832 * 13 * 13 -> 832 * 6 * 6
        self.max_pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # 832 * 6 * 6 - > 832 * 6 * 6
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        # 832 * 6 * 6 - > 1024 * 6  * 6
        # 1024 * 6  * 6 - > 1024 * 1 * 1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.max_pool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.max_pool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.max_pool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.max_pool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Inception(nn.Module):
    def __init__(self, in_channel, channel_1x1, channel_3x3_first, channel_3x3, channel_5x5_first, channel_5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channel, channel_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, channel_3x3_first, kernel_size=1, stride=1),
            BasicConv2d(channel_3x3_first, channel_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, channel_5x5_first, kernel_size=1, stride=1),
            BasicConv2d(channel_5x5_first, channel_5x5, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channel, pool_proj, kernel_size=5, stride=1, padding=2)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

net = GoogLeNet()
print(net)
x = torch.rand((5, 3, 224, 224))
print(net(x).shape)