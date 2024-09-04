# U-Net

## U-Net 代码

```python
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x1p = F.max_pool2d(x1, 2)
        
        x2 = F.relu(self.conv2(x1p))
        x2p = F.max_pool2d(x2, 2)
        
        x3 = F.relu(self.conv3(x2p))
        x3p = F.max_pool2d(x3, 2)
        
        x4 = F.relu(self.conv4(x3p))
        
        # Decoder
        x5 = F.relu(self.upconv1(x4))
        x5 = torch.cat([x5, x3], dim=1)
        x5 = F.relu(self.conv5(x5))
        
        x6 = F.relu(self.upconv2(x5))
        x6 = torch.cat([x6, x2], dim=1)
        x6 = F.relu(self.conv6(x6))
        
        x7 = F.relu(self.upconv3(x6))
        x7 = torch.cat([x7, x1], dim=1)
        x7 = F.relu(self.conv7(x7))
        
        out = torch.sigmoid(self.conv8(x7))
        return out

model = SimpleUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
```

U-Net网络由一个收缩路径（contracting path）和一个扩展路径（expansive path）组成，使其具有U形结构。收缩路径是一张典型的卷积网络，包括卷积的重复应用，每个卷积之后都有一个线性整流函数单元（ReLU）和一个最大汇集作业（max pooling operation）。在收缩过程中，空间与特征信息一减一增。扩张路径通过连续的上卷积和与来自收缩路径的高分辨率特征相连接来组合特征与空间信息。[4]