# VGG（Visual Geometry Group）

VGG是由牛津大学的Visual Geometry Group开发的，首次出现在2014年的论文“Very Deep Convolutional Networks for Large-Scale Image Recognition”中。VGG的主要贡献在于展示了通过增加网络的深度（即层数）可以显著提升系统的性能。

## VGG的特点

+ 统一的卷积核大小：VGG网络使用了多个相同大小（3x3）的卷积核，这是与之前的架构（如AlexNet）不同的地方，后者使用了不同大小的卷积核。
+ 更深的网络结构：VGG提供了多个版本，通常包括VGG16和VGG19，数字代表网络中含有权重层的总数。VGG16包含13个卷积层和3个全连接层，VGG19则包含16个卷积层和3个全连接层。
+ 池化层：在连续的几个卷积层后使用最大池化层来逐渐减少空间尺寸。
