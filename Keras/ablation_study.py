"""Ablation Study: ResNet vs PlainNet (skip connection 유무 비교).

- 데이터셋: Oxford-IIIT Pet (tensorflow_datasets, 224x224, 37 클래스)
  * torchvision.datasets.OxfordIIITPet 과 동일한 데이터셋의 Keras(tfds) 버전
- 같은 레이어 구성(ResNet-34 vs Plain-34 또는 50 vs 50)으로 학습 후
  검증 정확도 / 검증 손실 곡선을 matplotlib로 비교한다.
- 학습은 '확인용'으로 적은 epoch만 돌린다 (논문 재현이 아니라 경향 확인).

GPU 환경(예: Google Colab)에서 실행 권장.
    pip install tensorflow tensorflow_datasets matplotlib
    python ablation_study.py
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from build_resnet import resnet34, plainnet34, resnet50, plainnet50

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10              # 확인용. 시간이 더 있으면 늘리세요.
NUM_CLASSES = 37        # Oxford-IIIT Pet 품종 수
DEPTH = 34              # 34 또는 50 — 비교할 깊이 선택
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# 데이터 로딩 & 전처리
# ---------------------------------------------------------------------------
def preprocess(image, label, training):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    if training:
        image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_datasets():
    (ds_train, ds_test), info = tfds.load(
        "oxford_iiit_pet",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    ds_train = (
        ds_train.map(lambda x, y: preprocess(x, y, True), num_parallel_calls=AUTOTUNE)
        .shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    )
    ds_test = (
        ds_test.map(lambda x, y: preprocess(x, y, False), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE).prefetch(AUTOTUNE)
    )
    return ds_train, ds_test, info


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def train_one(build_fn, name, ds_train, ds_test):
    print(f"\n{'=' * 50}\n{name} 학습 시작\n{'=' * 50}")
    model = build_fn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS, verbose=1)
    return history


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def plot_comparison(hist_res, hist_plain, depth, out_path="ablation_resnet_vs_plain.png"):
    epochs = range(1, len(hist_res.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 검증 정확도
    axes[0].plot(epochs, hist_res.history["val_accuracy"], "b-o", label=f"ResNet-{depth}")
    axes[0].plot(epochs, hist_plain.history["val_accuracy"], "r-s", label=f"Plain-{depth}")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 검증 손실
    axes[1].plot(epochs, hist_res.history["val_loss"], "b-o", label=f"ResNet-{depth}")
    axes[1].plot(epochs, hist_plain.history["val_loss"], "r-s", label=f"Plain-{depth}")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n그래프 저장: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
def main():
    ds_train, ds_test, _ = make_datasets()

    if DEPTH == 34:
        res_fn, plain_fn = resnet34, plainnet34
    else:
        res_fn, plain_fn = resnet50, plainnet50

    hist_res = train_one(res_fn, f"ResNet-{DEPTH}", ds_train, ds_test)
    hist_plain = train_one(plain_fn, f"Plain-{DEPTH}", ds_train, ds_test)

    plot_comparison(hist_res, hist_plain, DEPTH)

    print("\n=== 최종 검증 정확도 ===")
    print(f"ResNet-{DEPTH}: {hist_res.history['val_accuracy'][-1]:.4f}")
    print(f"Plain-{DEPTH} : {hist_plain.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
