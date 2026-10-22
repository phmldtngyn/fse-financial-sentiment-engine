"""Test toàn bộ pipeline 3-class trước khi upload Kaggle."""
import torch
from src.data.dataset import FSEDataset3Class
from src.models.fse_model import FSEModel
from src.training.losses import FSELoss3Class


def test_dataset():
    print("=" * 60)
    print("TEST 1: Dataset load")
    print("=" * 60)
    ds = FSEDataset3Class("data/processed/train_3class.parquet")
    print(f"✅ Loaded: {len(ds):,} samples")
    item = ds[0]
    print(f"   Keys: {list(item.keys())}")
    print(f"   Label: {item['label'].item()}")
    print(f"   input_ids shape: {item['input_ids'].shape}")


def test_model_forward():
    print("\n" + "=" * 60)
    print("TEST 2: Model forward_batch")
    print("=" * 60)
    model = FSEModel()
    print(f"✅ Model loaded")
    print(f"   Head: {model.sentiment_head}")

    ids = torch.randint(0, 1000, (4, 256))
    mask = torch.ones(4, 256)
    logits = model.forward_batch(ids, mask)
    print(f"✅ forward_batch: output shape {logits.shape}")  # (4, 3)

    assert logits.shape == (4, 3), f"Expected (4,3), got {logits.shape}"


def test_loss_backward():
    print("\n" + "=" * 60)
    print("TEST 3: Loss + backward")
    print("=" * 60)
    model = FSEModel()
    loss_fn = FSELoss3Class()

    ids = torch.randint(0, 1000, (4, 256))
    mask = torch.ones(4, 256)
    labels = torch.tensor([0, 1, 2, 1])

    logits = model.forward_batch(ids, mask)
    loss, parts = loss_fn(logits, labels)
    print(f"✅ Loss: {loss.item():.4f} | parts: {parts}")

    loss.backward()
    print(f"✅ Backward OK — gradients computed")


def test_forward_inference():
    print("\n" + "=" * 60)
    print("TEST 4: Forward inference (bài báo ngắn)")
    print("=" * 60)
    model = FSEModel()
    model.eval()

    v, logits = model("Cổ phiếu VNM tăng trần")
    print(f"✅ v_article shape: {v.shape}")   # (768,)
    print(f"✅ logits shape: {logits.shape}")  # (3,)

    # Empty text
    v2, logits2 = model("")
    print(f"✅ Empty text: v={v2}, logits={logits2}")


if __name__ == "__main__":
    test_dataset()
    test_model_forward()
    test_loss_backward()
    test_forward_inference()
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)