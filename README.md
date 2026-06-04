# ResNet (PyTorch)

논문 *"Deep Residual Learning for Image Recognition"* (He et al., 2015)의 ResNet을 PyTorch로 처음부터 구현한 코드입니다.

## 모델

| 모델 | 블록 | 구성 | 파라미터 수 |
|------|------|------|-------------|
| ResNet-34 (`resnet34.py`) | BasicBlock (expansion 1) | `[3, 4, 6, 3]` | 21,797,672 |
| ResNet-50 (`resnet50.py`) | Bottleneck (expansion 4) | `[3, 4, 6, 3]` | 25,557,032 |

두 모델 모두 torchvision 공식 구현과 파라미터 수가 일치합니다.

## 공통 구조

- Stem: 7x7 conv(stride 2) → BN → ReLU → maxpool
- 4 stages: 채널 64 → 128 → 256 → 512 (Bottleneck은 출력 ×4)
- 분류기: AdaptiveAvgPool → FC
- Kaiming 초기화

## 사용법

```python
from resnet34 import resnet34
from resnet50 import resnet50

model34 = resnet34(num_classes=1000)
model50 = resnet50(num_classes=1000)
```

## 요구사항

- Python 3.8+
- PyTorch

```bash
pip install torch
python resnet34.py   # 파라미터 수 + 순전파 형태 확인
python resnet50.py
```
