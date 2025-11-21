# ============================================================
# Inference Benchmark 
# ============================================================

import os
import time
import platform
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm 

# ============================================================
# 1. CONFIGURATION
# ============================================================

# === Dataset config ===
DATA_DIR = "dataset" 
NUM_CLASSES = 5
BATCH_SIZE = 32

# === Split config ===
TEST_RATIO = 0.15
VAL_RATIO = 0.15
SEED = 42

# === Model config ===
# Contoh:
#   - DeiT-Small : "deit_small_patch16_224"
#   - Swin-Tiny  : "swin_tiny_patch4_window7_224"
#   - EVA-02     : "eva02_small_patch14_224"
MODEL_DEIT_SMALL = "deit_small_patch16_224"
MODEL_SWIN_TINY = "swin_tiny_patch4_window7_224"
MODEL_EVA_SMALL = "eva02_small_patch14_224"

CKPT_DEIT_1 = "best_model/training1_DEiT/best_model.pth"
CKPT_DEIT_2 = "best_model/training2_DEiT/best_model.pth"

CKPT_SWIN_1 = "best_model/training1_Swin/best_model.pth"
CKPT_SWIN_2 = "best_model/training2_Swin/best_model.pth"

CKPT_EVA_1 = "best_model/training1_EVA-02/best_model.pth"
CKPT_EVA_2 = "best_model/training2_EVA-02/best_model.pth"

TIMM_MODEL_NAME = MODEL_EVA_SMALL  # ganti sesuai model
CHECKPOINT_PATH = CKPT_EVA_2   # path ke .pth hasil training

# === Inference measurement config ===
NUM_WARMUP_BATCHES = 3   # warm-up (tidak dihitung)
MIN_IMAGES_FOR_STATS = 100  # target minimal gambar untuk statistik (kalau test set < 100, pakai jumlah yang ada)

# ============================================================
# 2. DEVICE & HARDWARE INFO
# ============================================================

def get_device_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024 ** 3)
        hw_str = f"GPU: {gpu_name} ({total_vram_gb:.2f} GB VRAM)"
    else:
        device = torch.device("cpu")
        cpu_name = platform.processor() or "Unknown CPU"
        hw_str = f"CPU: {cpu_name}"
    return device, hw_str

device, hardware_str = get_device_info()
print(f"Using device: {device}")
print(f"Hardware    : {hardware_str}")

# ============================================================
# 3. DATASET & TEST LOADER
# ============================================================

def build_test_loader():
    # Base dataset tanpa transform untuk ambil length & indices
    base_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names = base_dataset.classes
    print("Classes:", class_names)
    print("Total images:", len(base_dataset))

    # Transform untuk evaluasi (VAL/TEST)
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # Split index train / val / test
    num_samples = len(base_dataset)
    indices = list(range(num_samples))

    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)

    num_test = int(TEST_RATIO * num_samples)
    num_val = int(VAL_RATIO * num_samples)
    num_train = num_samples - num_test - num_val

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    print(f"Train samples (ignored here): {len(train_indices)}")
    print(f"Val samples   (ignored here): {len(val_indices)}")
    print(f"Test samples                : {len(test_indices)}")

    # Hanya pakai TEST SET untuk inference benchmark
    test_dataset = Subset(
        datasets.ImageFolder(root=DATA_DIR, transform=eval_transform),
        test_indices
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return test_loader, class_names

# ============================================================
# 4. MODEL LOADING
# ============================================================

def load_model(num_classes: int, timm_model_name: str, checkpoint_path: str, device):
    # Build model architecture
    model = timm.create_model(
        timm_model_name,
        pretrained=False,          # pretrained tidak penting, karena load checkpoint hasil training sebelumnya
        num_classes=num_classes
    )

    # Load trained weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model: {timm_model_name}")
    print(f"Loaded weights from: {checkpoint_path}")
    return model

# ============================================================
# 5. INFERENCE TIME MEASUREMENT
# ============================================================

def extract_run_name_from_checkpoint(ckpt_path: str):
    """
    Input:  'best_model/training1_DEiT/best_model.pth'
    Output: 'training1_DEiT'
    """
    parts = ckpt_path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("training") and "_" in p:
            return p
    return "unknown_run"

def measure_inference_time(model, loader, device,
                           checkpoint_path,
                           timm_model_name,
                           hardware_str,
                           num_warmup_batches=2,
                           min_images_for_stats=100):
    """
    Mengukur waktu inferensi + menyimpan hasilnya ke file txt
    """

    # --- Extract RUN NAME dari checkpoint ---
    run_name = extract_run_name_from_checkpoint(checkpoint_path)
    save_path = f"inference/inference_{run_name}.txt"

    model.eval()

    # ----------------- WARM-UP -----------------
    with torch.no_grad():
        warmup = 0
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            _ = model(inputs)
            warmup += 1
            if warmup >= num_warmup_batches:
                break

    # ----------------- INFERENCE MEASUREMENT -----------------
    per_image_times = []
    total_images = 0

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            bs = inputs.size(0)

            t0 = time.time()
            _ = model(inputs)
            t1 = time.time()

            dt = t1 - t0
            per_img = dt / bs

            per_image_times.extend([per_img] * bs)
            total_images += bs

    # convert to numpy
    per_image_times = np.array(per_image_times)

    # ----------------- STATS -----------------
    avg_sec = per_image_times.mean()
    std_sec = per_image_times.std()
    total_sec = per_image_times.sum()

    avg_ms = avg_sec * 1000
    std_ms = std_sec * 1000
    total_ms = total_sec * 1000

    throughput = total_images / total_sec

    # ----------------- PRINT RESULT -----------------
    print("\n===== INFERENCE BENCHMARK RESULT =====")
    print(f"Hardware          : {hardware_str}")
    print(f"Model             : {timm_model_name}")
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"Run Name          : {run_name}")
    print(f"Total test images : {total_images}")
    print(f"Total time        : {total_ms:.2f} ms")
    print(f"Avg per image     : {avg_ms:.4f} ms")
    print(f"Std dev per image : {std_ms:.4f} ms")
    print(f"Throughput        : {throughput:.2f} img/s")
    print("======================================\n")

    # ----------------- SAVE TO TXT -----------------
    lines = [
        "===== INFERENCE BENCHMARK RESULT =====",
        f"Run name           : {run_name}",
        f"Model name         : {timm_model_name}",
        f"Checkpoint         : {checkpoint_path}",
        f"Hardware           : {hardware_str}",
        "",
        f"Total images       : {total_images}",
        f"Total time (ms)    : {total_ms:.2f}",
        f"Avg time (ms/img)  : {avg_ms:.4f}",
        f"Std dev (ms/img)   : {std_ms:.4f}",
        f"Throughput (img/s) : {throughput:.2f}",
        "======================================"
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f">> Saved inference result to {save_path}")

    return {
        "run_name": run_name,
        "total_ms": total_ms,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "throughput": throughput
    }


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    # Build test loader
    test_loader, class_names = build_test_loader()

    # Load trained model
    model = load_model(
        num_classes=NUM_CLASSES,
        timm_model_name=TIMM_MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        device=device
    )

    # Measure inference time
    stats = measure_inference_time(
        model=model,
        loader=test_loader,
        device=device,
        checkpoint_path=CHECKPOINT_PATH,
        timm_model_name=TIMM_MODEL_NAME,
        hardware_str=hardware_str,
        num_warmup_batches=NUM_WARMUP_BATCHES,
        min_images_for_stats=MIN_IMAGES_FOR_STATS
    )

