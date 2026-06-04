"""ResNet (ResNet-34 / ResNet-50 ...) — Keras Functional API 구현.

VGG 실습처럼 '블록을 만드는 함수'를 정의하고,
ResNet 버전별 차이를 configuration(설정)으로 받아
하나의 build_resnet() 함수로 여러 버전을 모두 생성한다.

- Functional API 사용  -> model.summary() 가 모든 레이어를 펼쳐서 출력
- Subclass / Sequential 미사용
"""

import tensorflow as tf
from tensorflow.keras import Input, Model, layers


# ---------------------------------------------------------------------------
# 기본 빌딩 블록을 만드는 함수들 (VGG 실습처럼 함수로 분리)
# ---------------------------------------------------------------------------
def conv_bn(x, filters, kernel_size, strides=1, activation=True, name=None):
    """Conv2D -> BatchNorm -> (ReLU) 를 묶은 기본 단위.

    논문대로 He(MSRA) 초기화 사용, BN을 쓰므로 conv bias는 생략.
    """
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same",
        use_bias=False, kernel_initializer="he_normal",
        name=None if name is None else name + "_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    if activation:
        x = layers.Activation("relu", name=None if name is None else name + "_relu")(x)
    return x


def basic_block(x, filters, strides=1, use_skip=True, name=None):
    """ResNet-18/34용 BasicBlock: 3x3 -> 3x3, 출력 채널 = filters.

    use_skip=True  -> ResNet 블록 (skip connection + residual add)
    use_skip=False -> Plain 블록 (skip 없이 conv만 쌓음)
    """
    in_channels = x.shape[-1]

    y = conv_bn(x, filters, 3, strides=strides, name=None if name is None else name + "_1")
    y = conv_bn(y, filters, 3, strides=1, activation=False,
                name=None if name is None else name + "_2")

    if not use_skip:
        # PlainNet: 잔차 연결 없이 마지막 활성화만 적용
        return layers.Activation("relu", name=None if name is None else name + "_out")(y)

    shortcut = x
    if strides != 1 or in_channels != filters:
        shortcut = conv_bn(x, filters, 1, strides=strides, activation=False,
                           name=None if name is None else name + "_sc")

    y = layers.Add(name=None if name is None else name + "_add")([y, shortcut])
    y = layers.Activation("relu", name=None if name is None else name + "_out")(y)
    return y


def bottleneck_block(x, filters, strides=1, use_skip=True, name=None):
    """ResNet-50/101/152용 Bottleneck: 1x1 -> 3x3 -> 1x1, 출력 채널 = filters*4.

    논문 원본(2015)대로 다운샘플 stride는 '첫 1x1 conv'에 적용한다.
    use_skip=False 이면 skip connection 없는 Plain 블록이 된다.
    """
    expansion = 4
    out_channels = filters * expansion
    in_channels = x.shape[-1]

    y = conv_bn(x, filters, 1, strides=strides, name=None if name is None else name + "_1")
    y = conv_bn(y, filters, 3, strides=1, name=None if name is None else name + "_2")
    y = conv_bn(y, out_channels, 1, strides=1, activation=False,
                name=None if name is None else name + "_3")

    if not use_skip:
        # PlainNet: 잔차 연결 없이 마지막 활성화만 적용
        return layers.Activation("relu", name=None if name is None else name + "_out")(y)

    shortcut = x
    if strides != 1 or in_channels != out_channels:
        shortcut = conv_bn(x, out_channels, 1, strides=strides, activation=False,
                           name=None if name is None else name + "_sc")

    y = layers.Add(name=None if name is None else name + "_add")([y, shortcut])
    y = layers.Activation("relu", name=None if name is None else name + "_out")(y)
    return y


# 설정(config)에서 문자열로 블록을 고를 수 있게 매핑
BLOCK_FNS = {
    "basic": basic_block,
    "bottleneck": bottleneck_block,
}


# ---------------------------------------------------------------------------
# 하나의 생성 함수로 모든 ResNet 버전 생성
# ---------------------------------------------------------------------------
def _build_network(
    input_shape=(224, 224, 3),
    num_classes=1000,
    block_type="basic",        # "basic"(34) | "bottleneck"(50/101/152)
    num_blocks=(3, 4, 6, 3),   # stage별 블록 개수
    filters=(64, 128, 256, 512),
    use_skip=True,             # True: ResNet / False: PlainNet
    name="resnet",
):
    """ResNet/PlainNet 공통 빌더.

    use_skip 만 다를 뿐 레이어 구성은 완전히 동일하므로,
    ResNet vs PlainNet ablation을 공정하게 비교할 수 있다.
    Functional API로 그래프를 평탄하게 구성한다.
    """
    block_fn = BLOCK_FNS[block_type]

    inputs = Input(shape=input_shape)

    # Stem: 7x7 conv(stride 2) -> BN -> ReLU -> 3x3 maxpool(stride 2)
    x = conv_bn(inputs, 64, 7, strides=2, name="stem")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # 4개 stage. stage 0은 다운샘플 없음(stem이 이미 줄임), 이후는 첫 블록에서 stride 2.
    for stage_idx, (f, n) in enumerate(zip(filters, num_blocks)):
        for block_idx in range(n):
            strides = 2 if (block_idx == 0 and stage_idx > 0) else 1
            x = block_fn(x, f, strides=strides, use_skip=use_skip,
                         name=f"stage{stage_idx + 1}_block{block_idx + 1}")

    # 분류기
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name=name)


def build_resnet(
    input_shape=(224, 224, 3),
    num_classes=1000,
    block_type="basic",
    num_blocks=(3, 4, 6, 3),
    filters=(64, 128, 256, 512),
    name="resnet",
):
    """skip connection이 있는 ResNet 생성 (configuration으로 버전 선택)."""
    return _build_network(input_shape, num_classes, block_type, num_blocks,
                          filters, use_skip=True, name=name)


def build_plainnet(
    input_shape=(224, 224, 3),
    num_classes=1000,
    block_type="basic",
    num_blocks=(3, 4, 6, 3),
    filters=(64, 128, 256, 512),
    name="plainnet",
):
    """ResNet과 레이어 구성은 같지만 skip connection이 없는 PlainNet 생성."""
    return _build_network(input_shape, num_classes, block_type, num_blocks,
                          filters, use_skip=False, name=name)


# ---------------------------------------------------------------------------
# 자주 쓰는 버전 단축 함수 (모두 같은 빌더로 생성)
# ---------------------------------------------------------------------------
def resnet18(input_shape=(224, 224, 3), num_classes=1000):
    return build_resnet(input_shape, num_classes, "basic", (2, 2, 2, 2), name="resnet18")


def resnet34(input_shape=(224, 224, 3), num_classes=1000):
    return build_resnet(input_shape, num_classes, "basic", (3, 4, 6, 3), name="resnet34")


def resnet50(input_shape=(224, 224, 3), num_classes=1000):
    return build_resnet(input_shape, num_classes, "bottleneck", (3, 4, 6, 3), name="resnet50")


def resnet101(input_shape=(224, 224, 3), num_classes=1000):
    return build_resnet(input_shape, num_classes, "bottleneck", (3, 4, 23, 3), name="resnet101")


def resnet152(input_shape=(224, 224, 3), num_classes=1000):
    return build_resnet(input_shape, num_classes, "bottleneck", (3, 8, 36, 3), name="resnet152")


def plainnet34(input_shape=(224, 224, 3), num_classes=1000):
    return build_plainnet(input_shape, num_classes, "basic", (3, 4, 6, 3), name="plainnet34")


def plainnet50(input_shape=(224, 224, 3), num_classes=1000):
    return build_plainnet(input_shape, num_classes, "bottleneck", (3, 4, 6, 3), name="plainnet50")


if __name__ == "__main__":
    for build, label in [(resnet34, "ResNet-34"), (plainnet34, "Plain-34"),
                         (resnet50, "ResNet-50"), (plainnet50, "Plain-50")]:
        m = build()
        n = int(sum(tf.size(w) for w in m.trainable_weights))
        print(f"{label:10s} trainable params: {n:,}")
