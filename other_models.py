# compare_models.py (с графиками)
import pandas as pd
import numpy as np
import re
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from pathlib import Path
from copy import deepcopy

# -----------------------------
# 1) Повторяемость
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 2) Загрузка данных
# -----------------------------
DATA_FILE = "all_devices_cleaned.csv"
df = pd.read_csv(DATA_FILE)
print(f"Загружено строк: {len(df)}")

target_col = "price"
drop_cols = ["name", target_col, "device_type_label"]

device_cols = ["is_phone", "is_tablet", "is_laptop"]
for c in device_cols:
    if c not in df.columns:
        df[c] = 0

def infer_device_type(row):
    vals = [row["is_phone"], row["is_tablet"], row["is_laptop"]]
    labels = ["phone", "tablet", "laptop"]
    return labels[int(np.argmax(vals))]

df["device_type_label"] = df.apply(infer_device_type, axis=1)

# -----------------------------
# 3) Feature engineering (как в train_price.py)
# -----------------------------
eps = 1e-6
df["ram_per_screen"] = df["ram_gb"] / (df["screen_size_inch"] + eps)
df["storage_per_ram"] = df["storage_gb"] / (df["ram_gb"] + eps)
df["battery_per_inch"] = df["battery_mah"] / (df["screen_size_inch"] + eps)
df["reviews_per_rating"] = df["total_reviews"] / (df["total_ratings"] + 1.0)
df["engagement_score"] = np.log1p(df["total_ratings"]) * (df["rating"] + eps)
df["camera_x_ram"] = df["camera_megapixels"] * df["ram_gb"]
df["battery_x_ram"] = df["battery_mah"] * df["ram_gb"]

def safe_lower(s):
    if pd.isna(s):
        return ""
    return str(s).lower()

df["name_l"] = df["name"].apply(safe_lower)
if "description" in df.columns:
    df["desc_l"] = df["description"].apply(safe_lower)
else:
    df["desc_l"] = ""
df["text_l"] = df["name_l"] + " " + df["desc_l"]

def extract_first_token_brand(name):
    token = re.sub(r"[^a-z0-9]+", " ", name).strip().split(" ")
    return token[0] if token and token[0] else "unknown"

df["brand_token"] = df["name_l"].apply(extract_first_token_brand)

model_keywords = [
    "iphone", "galaxy", "redmi", "poco", "vivobook", "ideapad", "omen",
    "pavilion", "macbook", "infinix", "realme", "ipad", "tab", "thinkpad",
    "rog", "surface", "moto", "oneplus", "xiaomi"
]
for kw in model_keywords:
    df[f"kw_{kw}"] = df["text_l"].str.contains(rf"\b{re.escape(kw)}\b", regex=True).astype(np.float32)

pattern_features = {
    "has_5g": r"\b5g\b",
    "has_4g": r"\b4g\b|\blte\b",
    "has_wifi": r"\bwi[\-\s]?fi\b|\bwifi\b",
    "has_nvidia": r"\bnvidia\b|\bgeforce\b|\brtx\b|\bgtx\b",
    "has_rtx": r"\brtx\b",
    "has_gtx": r"\bgtx\b",
    "has_amd": r"\bamd\b|\bryzen\b|\bradeon\b",
    "has_intel": r"\bintel\b|\bcore i[3579]\b",
    "has_snapdragon": r"\bsnapdragon\b",
    "has_mediatek": r"\bmediatek\b|\bhelio\b|\bdimensity\b",
    "has_a_bionic": r"\ba\d{2}\s*bionic\b|\bbionic\b",
    "has_oled": r"\boled\b|\bamoled\b|\bsuper amoled\b",
    "has_ips": r"\bips\b",
    "has_stylus_text": r"\bstylus\b|\bpencil\b",
    "has_touchscreen_text": r"\btouch\b|\btouchscreen\b",
    "has_ssd_text": r"\bssd\b|\bnvme\b",
    "has_hdd_text": r"\bhdd\b",
}
for fname, pat in pattern_features.items():
    df[fname] = df["text_l"].str.contains(pat, regex=True).astype(np.float32)

def extract_gen(text):
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s*gen\b", text)
    return float(m.group(1)) if m else 0.0

def extract_cpu_tier(text):
    intel = re.search(r"\bcore\s*i([3579])\b", text)
    if intel:
        return float(intel.group(1))
    ryzen = re.search(r"\bryzen\s*([3579])\b", text)
    if ryzen:
        return float(ryzen.group(1))
    if re.search(r"\ba\d{2}\s*bionic\b", text):
        return 8.0
    return 0.0

def extract_gpu_tier(text):
    if re.search(r"\brtx\s*40", text):
        return 9.0
    if re.search(r"\brtx\s*30", text):
        return 8.0
    if re.search(r"\brtx\s*20", text):
        return 7.0
    if re.search(r"\bgtx\s*16", text):
        return 6.0
    if re.search(r"\bgtx\s*10", text):
        return 5.0
    if re.search(r"\bradeon\b", text):
        return 4.0
    return 0.0

df["cpu_generation"] = df["text_l"].apply(extract_gen).astype(np.float32)
df["cpu_tier"] = df["text_l"].apply(extract_cpu_tier).astype(np.float32)
df["gpu_tier"] = df["text_l"].apply(extract_gpu_tier).astype(np.float32)

df["name_len"] = df["name_l"].str.len().astype(np.float32)
df["desc_len"] = df["desc_l"].str.len().astype(np.float32)
df["name_word_count"] = df["name_l"].str.split().str.len().astype(np.float32)
df["desc_word_count"] = df["desc_l"].str.split().str.len().astype(np.float32)

df["cpu_tier_x_ram"] = df["cpu_tier"] * df["ram_gb"]
df["gpu_tier_x_ram"] = df["gpu_tier"] * df["ram_gb"]
df["cpu_tier_x_storage"] = df["cpu_tier"] * df["storage_gb"]

exclude_cols = {
    "name", target_col, "device_type_label", "name_l", "desc_l", "text_l",
    "brand_token", "cpu_family", "gpu_family"
}
feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

# -----------------------------
# 4) Train/val split и target encoding
# -----------------------------
df = df.dropna(subset=feature_cols + [target_col]).copy()
y_raw = df[target_col].values.astype(np.float32)
y = np.log1p(y_raw).astype(np.float32)

price_bins = pd.qcut(y_raw, q=6, labels=False, duplicates="drop").astype(str)
stratify_key = df["device_type_label"].astype(str) + "_" + price_bins
strat_counts = stratify_key.value_counts()
stratify_safe = stratify_key.where(stratify_key.map(strat_counts) >= 2, df["device_type_label"].astype(str))
safe_counts = stratify_safe.value_counts()
stratify_safe = stratify_safe.where(stratify_safe.map(safe_counts) >= 2, "__other__")
final_counts = stratify_safe.value_counts()
use_stratify = final_counts.min() >= 2

if use_stratify:
    df_train, df_val = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=stratify_safe.values
    )
else:
    print("Предупреждение: stratify отключен")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=SEED)

y_train = np.log1p(df_train[target_col].values.astype(np.float32))
y_val = np.log1p(df_val[target_col].values.astype(np.float32))
y_raw_train = df_train[target_col].values.astype(np.float32)
y_raw_val = df_val[target_col].values.astype(np.float32)

strat_train = (
    df_train["device_type_label"].astype(str)
    + "_"
    + pd.qcut(df_train[target_col], q=6, labels=False, duplicates="drop").astype(str)
).values

def add_target_encoding(train_df, val_df, col, target, min_count=10, alpha=25.0):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
    smooth = smooth.where(stats["count"] >= min_count, global_mean)
    train_enc = train_df[col].map(smooth).fillna(global_mean)
    val_enc = val_df[col].map(smooth).fillna(global_mean)
    return train_enc.astype(np.float32), val_enc.astype(np.float32), \
           {str(k): float(v) for k, v in smooth.to_dict().items()}, float(global_mean)

def cpu_family(text):
    if re.search(r"\bcore\s*i[3579]\b|\bintel\b", text):
        return "intel"
    if re.search(r"\bryzen\b|\bamd\b", text):
        return "amd"
    if re.search(r"\bsnapdragon\b", text):
        return "snapdragon"
    if re.search(r"\bmediatek\b|\bhelio\b|\bdimensity\b", text):
        return "mediatek"
    if re.search(r"\bbionic\b", text):
        return "apple_bionic"
    return "other"

def gpu_family(text):
    if re.search(r"\brtx\b", text):
        return "rtx"
    if re.search(r"\bgtx\b", text):
        return "gtx"
    if re.search(r"\bradeon\b", text):
        return "radeon"
    if re.search(r"\bintel iris\b|\bintel hd\b|\buhd\b", text):
        return "intel_igpu"
    return "other"

df_train["cpu_family"] = df_train["text_l"].apply(cpu_family)
df_val["cpu_family"] = df_val["text_l"].apply(cpu_family)
df_train["gpu_family"] = df_train["text_l"].apply(gpu_family)
df_val["gpu_family"] = df_val["text_l"].apply(gpu_family)

enc_cols = ["brand_token", "cpu_family", "gpu_family", "device_type_label"]
for col in enc_cols:
    tr_enc, va_enc, _, _ = add_target_encoding(df_train, df_val, col, target_col)
    df_train[f"te_{col}"] = tr_enc
    df_val[f"te_{col}"] = va_enc

feature_cols = [c for c in df_train.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])]
X_train = df_train[feature_cols].values.astype(np.float32)
X_val = df_val[feature_cols].values.astype(np.float32)
print(f"Финальное количество признаков: {len(feature_cols)}")

# -----------------------------
# 5) Масштабирование и DataLoader
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

batch_size = 96
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

group_counts = pd.Series(strat_train).value_counts()
sample_weights = np.array([1.0 / group_counts[g] for g in strat_train], dtype=np.float32)
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(sample_weights),
    replacement=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 6) Архитектуры моделей
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class PricePredictor(nn.Module):  # Твоя модель
    def __init__(self, input_dim):
        super().__init__()
        hidden = 150
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.res1 = ResidualBlock(hidden, dropout=0.15)
        self.res2 = ResidualBlock(hidden, dropout=0.10)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x).squeeze(1)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class MLP_GELU_NoRes(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 240),
            nn.GELU(),
            nn.Linear(240, 160),
            nn.GELU(),
            nn.Linear(160, 120),
            nn.GELU(),
            nn.Linear(120, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# -----------------------------
# 7) Метрики
# -----------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def mape_floor(y_true, y_pred, floor=5000.0):
    denom = np.maximum(np.abs(y_true), floor)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

# -----------------------------
# 8) Обучение одной модели (с записью истории)
# -----------------------------
num_epochs = 240
patience = 28

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_model(model_class, seed):
    set_all_seeds(seed)
    model = model_class(X_train.shape[1]).to(device)
    criterion = nn.HuberLoss(delta=0.45)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=7e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = -1
    best_metrics = {}
    best_preds_rub = None

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_mape_floor": []}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(xb)

        train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0.0
        all_preds_log, all_true_log = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item() * len(xb)
                all_preds_log.extend(preds.cpu().numpy())
                all_true_log.extend(yb.cpu().numpy())

        val_loss = total_val_loss / len(val_dataset)
        scheduler.step(val_loss)

        preds_rub = np.expm1(np.array(all_preds_log))
        preds_rub = np.clip(preds_rub, a_min=0.0, a_max=None)
        true_rub = np.expm1(np.array(all_true_log))

        epoch_mae = mae(true_rub, preds_rub)
        epoch_mape_floor = mape_floor(true_rub, preds_rub, floor=5000.0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(epoch_mae)
        history["val_mape_floor"].append(epoch_mape_floor)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_preds_rub = preds_rub.copy()
            best_metrics = {
                "mae": epoch_mae,
                "mape": mape(true_rub, preds_rub),
                "mape_floor": epoch_mape_floor,
                "smape": smape(true_rub, preds_rub),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "best_preds_rub": best_preds_rub,
        "history": history,
    }

# -----------------------------
# 9) Сравнение архитектур (ансамбль из 3 сидов) + графики
# -----------------------------
architectures = {
    "PricePredictor (Res+GELU)": PricePredictor,
    "SimpleMLP (ReLU)": SimpleMLP,
    "MLP GELU NoRes": MLP_GELU_NoRes,
}

ensemble_seeds = [42, 52, 62]
results_summary = {}
histories = {}  # Сохраним историю лучшей одиночной модели для графиков

for name, model_cls in architectures.items():
    print(f"\n===== Обучение архитектуры: {name} =====")
    models = []
    for s in ensemble_seeds:
        res = train_one_model(model_cls, s)
        models.append(res)
    # Ансамбль
    ensemble_preds = np.median(np.vstack([m["best_preds_rub"] for m in models]), axis=0)
    ensemble_metrics = {
        "MAE": mae(y_raw_val, ensemble_preds),
        "MAPE": mape(y_raw_val, ensemble_preds),
        "MAPE@5k": mape_floor(y_raw_val, ensemble_preds, floor=5000.0),
        "SMAPE": smape(y_raw_val, ensemble_preds),
    }
    # Лучшая одиночная
    best = min(models, key=lambda x: x["best_val_loss"])
    single_metrics = best["best_metrics"]
    histories[name] = best["history"]  # для графиков
    results_summary[name] = {
        "Single": single_metrics,
        "Ensemble": ensemble_metrics,
    }
    print(f"  Лучшая одиночная: MAE={single_metrics['mae']:.0f} руб, MAPE@5k={single_metrics['mape_floor']:.1f}%")
    print(f"  Ансамбль: MAE={ensemble_metrics['MAE']:.0f} руб, MAPE@5k={ensemble_metrics['MAPE@5k']:.1f}%")

# -----------------------------
# 10) Итоговая таблица
# -----------------------------
print("\n" + "="*80)
print("Сравнение архитектур (ансамбль из трёх моделей)")
print("="*80)
print(f"{'Модель':<30} {'MAE, руб':>10} {'MAPE@5k, %':>12} {'SMAPE, %':>10}")
for name, metrics in results_summary.items():
    m = metrics["Ensemble"]
    print(f"{name:<30} {m['MAE']:>10.0f} {m['MAPE@5k']:>12.1f} {m['SMAPE']:>10.1f}")

print("\nЛучшая одиночная модель (по Huber Loss):")
print(f"{'Модель':<30} {'MAE, руб':>10} {'MAPE@5k, %':>12} {'SMAPE, %':>10}")
for name, metrics in results_summary.items():
    m = metrics["Single"]
    print(f"{name:<30} {m['mae']:>10.0f} {m['mape_floor']:>12.1f} {m['smape']:>10.1f}")

# -----------------------------
# 11) Графики
# -----------------------------
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

colors = {
    "PricePredictor (Res+GELU)": "#7dd3fc",
    "SimpleMLP (ReLU)": "#f9a8d4",
    "MLP GELU NoRes": "#86efac",
}

for idx, (name, hist) in enumerate(histories.items()):
    ax_loss = axes[idx, 0]
    ax_metric = axes[idx, 1]

    # Loss
    ax_loss.plot(hist["train_loss"], label="Train Loss", color=colors[name], linewidth=2.0, alpha=0.8)
    ax_loss.plot(hist["val_loss"], label="Val Loss", color=colors[name], linestyle="--", linewidth=2.0)
    ax_loss.set_title(f"{name} – Loss", color="white")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Huber Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.2)

    # MAE и MAPE@5k
    ax_metric.plot(hist["val_mae"], label="Val MAE", color="#ffd166", linewidth=2.0)
    ax_metric2 = ax_metric.twinx()
    ax_metric2.plot(hist["val_mape_floor"], label="Val MAPE@5k", color="#ef476f", linewidth=2.0, linestyle=":")
    ax_metric.set_title(f"{name} – Validation metrics", color="white")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("MAE, руб", color="#ffd166")
    ax_metric2.set_ylabel("MAPE@5k, %", color="#ef476f")
    ax_metric.grid(alpha=0.2)
    ax_metric2.grid(False)

plt.tight_layout()
plt.show()