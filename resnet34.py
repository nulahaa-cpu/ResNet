"""ResNet-34 구현 (PyTorch, from scratch).

논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
구조: BasicBlock 기반, 레이어 구성 [3, 4, 6, 3]
"""

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    """패딩 1을 적용한 3x3 합성곱 (공간 크기 유지)."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """다운샘플링용 1x1 합성곱."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1,
        stride=stride, bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet-18/34에서 쓰이는 기본 잔차 블록.

    구조: conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+ shortcut) -> ReLU
    """

    expansion = 1  # 출력 채널 확장 배수 (BasicBlock은 1)

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 차원이 다를 때 shortcut 보정
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 잔차 연결
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """일반화된 ResNet 본체 (구성에 따라 ResNet-18/34 등 생성 가능)."""

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Stem: 7x7 conv (stride 2) -> BN -> ReLU -> maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4개의 stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        # 공간 크기나 채널 수가 바뀌면 shortcut에 1x1 conv 필요
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet34(num_classes=1000):
    """ResNet-34 모델 생성."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == "__main__":
    model = resnet34(num_classes=1000)

    # 파라미터 수 확인 (약 21.8M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-34 파라미터 수: {n_params:,}")

    # 순전파 테스트
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"입력: {tuple(x.shape)} -> 출력: {tuple(y.shape)}")
