# train_price.py (с MLP_GELU_NoRes)
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
print(f"Файл данных: {DATA_FILE}")

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

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Количество признаков: {len(feature_cols)}")

df = df.dropna(subset=feature_cols + [target_col]).copy()
print(f"После удаления NaN: {len(df)}")

# -----------------------------
# Дополнительные фичи
# -----------------------------
eps = 1e-6
df["ram_per_screen"] = df["ram_gb"] / (df["screen_size_inch"] + eps)
df["storage_per_ram"] = df["storage_gb"] / (df["ram_gb"] + eps)
df["battery_per_inch"] = df["battery_mah"] / (df["screen_size_inch"] + eps)
df["reviews_per_rating"] = df["total_reviews"] / (df["total_ratings"] + 1.0)
df["engagement_score"] = np.log1p(df["total_ratings"].clip(lower=0)) * (df["rating"] + eps)
df["camera_x_ram"] = df["camera_megapixels"] * df["ram_gb"]
df["battery_x_ram"] = df["battery_mah"] * df["ram_gb"]

# -----------------------------
# NLP/Regex фичи
# -----------------------------
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
    if intel: return float(intel.group(1))
    ryzen = re.search(r"\bryzen\s*([3579])\b", text)
    if ryzen: return float(ryzen.group(1))
    if re.search(r"\ba\d{2}\s*bionic\b", text): return 8.0
    return 0.0

def extract_gpu_tier(text):
    if re.search(r"\brtx\s*40", text): return 9.0
    if re.search(r"\brtx\s*30", text): return 8.0
    if re.search(r"\brtx\s*20", text): return 7.0
    if re.search(r"\bgtx\s*16", text): return 6.0
    if re.search(r"\bgtx\s*10", text): return 5.0
    if re.search(r"\bradeon\b", text): return 4.0
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

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Количество признаков после фич-инжиниринга: {len(feature_cols)}")

# -----------------------------
# 3) X, y и лог-цель
# -----------------------------
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
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=SEED, stratify=stratify_safe.values)
else:
    print("Предупреждение: stratify отключен из-за редких групп.")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=SEED)

y_train = np.log1p(df_train[target_col].values.astype(np.float32))
y_val = np.log1p(df_val[target_col].values.astype(np.float32))
y_raw_train = df_train[target_col].values.astype(np.float32)
y_raw_val = df_val[target_col].values.astype(np.float32)
strat_train = (
    df_train["device_type_label"].astype(str) + "_" +
    pd.qcut(df_train[target_col], q=6, labels=False, duplicates="drop").astype(str)
).values

print(f"Train: {len(df_train)}, Val: {len(df_val)}")

# -----------------------------
# Target encoding
# -----------------------------
def add_target_encoding(train_df, val_df, col, target, min_count=10, alpha=20.0):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
    smooth = smooth.where(stats["count"] >= min_count, global_mean)
    train_enc = train_df[col].map(smooth).fillna(global_mean)
    val_enc = val_df[col].map(smooth).fillna(global_mean)
    return train_enc.astype(np.float32), val_enc.astype(np.float32), \
           {str(k): float(v) for k, v in smooth.to_dict().items()}, float(global_mean)

def cpu_family(text):
    if re.search(r"\bcore\s*i[3579]\b|\bintel\b", text): return "intel"
    if re.search(r"\bryzen\b|\bamd\b", text): return "amd"
    if re.search(r"\bsnapdragon\b", text): return "snapdragon"
    if re.search(r"\bmediatek\b|\bhelio\b|\bdimensity\b", text): return "mediatek"
    if re.search(r"\bbionic\b", text): return "apple_bionic"
    return "other"

def gpu_family(text):
    if re.search(r"\brtx\b", text): return "rtx"
    if re.search(r"\bgtx\b", text): return "gtx"
    if re.search(r"\bradeon\b", text): return "radeon"
    if re.search(r"\bintel iris\b|\bintel hd\b|\buhd\b", text): return "intel_igpu"
    return "other"

df_train["cpu_family"] = df_train["text_l"].apply(cpu_family)
df_val["cpu_family"] = df_val["text_l"].apply(cpu_family)
df_train["gpu_family"] = df_train["text_l"].apply(gpu_family)
df_val["gpu_family"] = df_val["text_l"].apply(gpu_family)

enc_cols = ["brand_token", "cpu_family", "gpu_family", "device_type_label"]
te_artifacts = {}
for col in enc_cols:
    tr_enc, va_enc, te_map, te_global = add_target_encoding(df_train, df_val, col, target_col, min_count=10, alpha=25.0)
    df_train[f"te_{col}"] = tr_enc
    df_val[f"te_{col}"] = va_enc
    te_artifacts[col] = {"map": te_map, "global_mean": te_global}

exclude_cols = {
    "name", target_col, "device_type_label", "name_l", "desc_l", "text_l",
    "brand_token", "cpu_family", "gpu_family"
}
feature_cols = [c for c in df_train.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])]

X_train = df_train[feature_cols].values.astype(np.float32)
X_val = df_val[feature_cols].values.astype(np.float32)
print(f"Финальное количество признаков: {len(feature_cols)}")

# -----------------------------
# 4) Масштабирование
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

# -----------------------------
# 5) Модель (MLP GELU NoRes)
# -----------------------------
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 6) Метрики
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
# 7) Обучение + early stopping + ансамбль
# -----------------------------
num_epochs = 240
patience = 28
ensemble_seeds = [42, 52, 62]

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_model(seed):
    set_all_seeds(seed)
    model = MLP_GELU_NoRes(X_train.shape[1]).to(device)
    criterion = nn.HuberLoss(delta=0.45)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=7e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = -1
    best_metrics = {}
    best_preds_rub = None

    train_losses, val_losses, val_maes, val_mapes = [], [], [], []

    print(f"\n=== Обучение модели seed={seed} ===")
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
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        preds_rub = np.expm1(np.array(all_preds_log))
        preds_rub = np.clip(preds_rub, a_min=0.0, a_max=None)
        true_rub = np.expm1(np.array(all_true_log))

        epoch_mae = mae(true_rub, preds_rub)
        epoch_mape = mape(true_rub, preds_rub)
        epoch_mape_floor = mape_floor(true_rub, preds_rub, floor=5000.0)
        epoch_smape = smape(true_rub, preds_rub)

        val_maes.append(epoch_mae)
        val_mapes.append(epoch_mape)

        print(
            f"[seed={seed}] Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"MAE: {epoch_mae:.2f} | MAPE@5k: {epoch_mape_floor:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_preds_rub = preds_rub.copy()
            best_metrics = {
                "mae": epoch_mae, "mape": epoch_mape,
                "mape_floor": epoch_mape_floor, "smape": epoch_smape,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[seed={seed}] Early stopping на эпохе {epoch+1}")
                break

    model.load_state_dict(best_state)
    return {
        "seed": seed, "model": model, "best_state": best_state,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "best_metrics": best_metrics, "best_preds_rub": best_preds_rub,
        "history": {
            "train_losses": train_losses, "val_losses": val_losses,
            "val_maes": val_maes, "val_mapes": val_mapes,
        }
    }

results = []
for seed in ensemble_seeds:
    results.append(train_one_model(seed))

best_single = min(results, key=lambda r: r["best_val_loss"])
print(f"\nЛучшая одиночная модель: seed={best_single['seed']}, "
      f"epoch={best_single['best_epoch']}, val_loss={best_single['best_val_loss']:.4f}")

# Ансамбль
ensemble_pred_matrix = np.vstack([r["best_preds_rub"] for r in results])
ensemble_preds_rub = np.median(ensemble_pred_matrix, axis=0)
ensemble_true_rub = y_raw_val

ensemble_metrics = {
    "mae": mae(ensemble_true_rub, ensemble_preds_rub),
    "mape": mape(ensemble_true_rub, ensemble_preds_rub),
    "mape_floor": mape_floor(ensemble_true_rub, ensemble_preds_rub, floor=5000.0),
    "smape": smape(ensemble_true_rub, ensemble_preds_rub),
}

# Сохранение
Path("artifacts").mkdir(exist_ok=True)
torch.save(best_single["best_state"], "artifacts/best_price_model.pt")
np.save("artifacts/ensemble_val_preds.npy", ensemble_preds_rub)
np.save("artifacts/ensemble_val_true.npy", ensemble_true_rub)
print("Лучшая одиночная модель сохранена: artifacts/best_price_model.pt")

with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

feature_medians = df_train[feature_cols].median(numeric_only=True).to_dict()
feature_medians = {str(k): float(v) for k, v in feature_medians.items()}
preprocess_meta = {
    "data_file": DATA_FILE,
    "feature_cols": feature_cols,
    "feature_medians": feature_medians,
    "te_artifacts": te_artifacts,
}
with open("artifacts/preprocess_meta.json", "w", encoding="utf-8") as f:
    json.dump(preprocess_meta, f, ensure_ascii=False, indent=2)
print("Препроцессинг сохранен: artifacts/scaler.pkl, artifacts/preprocess_meta.json")

# -----------------------------
# 8) Графики
# -----------------------------
plt.style.use("dark_background")
fig = plt.figure(figsize=(13, 4.8), facecolor="#000000")
palette = {"train": "#7dd3fc", "val": "#f9a8d4", "mae": "#86efac", "mape": "#fcd34d"}

hist = best_single["history"]

# --- График 1: Loss ---
ax1 = plt.subplot(1, 3, 1)
ax1.set_facecolor("#0b0b0b")
plt.plot(hist["train_losses"], label="Train Loss", color=palette["train"], linewidth=2.0)
plt.plot(hist["val_losses"], label="Val Loss", color=palette["val"], linewidth=2.0)
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.title("Loss")
plt.legend()
ax1.grid(alpha=0.15, color="#ffffff")
# Опционально: если loss тоже улетает в начале, можно ограничить
# ax1.set_ylim(0, max(max(hist["train_losses"][5:]), max(hist["val_losses"][5:])) * 1.2)

# --- График 2: MAE ---
ax2 = plt.subplot(1, 3, 2)
ax2.set_facecolor("#0b0b0b")
plt.plot(hist["val_maes"], label="Single Val MAE", color=palette["mae"], linewidth=2.0)
plt.axhline(ensemble_metrics["mae"], color="#22d3ee", linestyle="--", label="Ensemble MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (руб)")
plt.title("Средняя абсолютная ошибка")
plt.legend()
ax2.grid(alpha=0.15, color="#ffffff")
ax2.set_ylim(0, 400000)   # <-- ограничение, чтобы не было гигантских начальных значений

# --- График 3: MAPE ---
ax3 = plt.subplot(1, 3, 3)
ax3.set_facecolor("#0b0b0b")
plt.plot(hist["val_mapes"], label="Single Val MAPE", color=palette["mape"], linewidth=2.0)
plt.axhline(ensemble_metrics["mape"], color="#a78bfa", linestyle="--", label="Ensemble MAPE")
plt.xlabel("Epoch")
plt.ylabel("MAPE (%)")
plt.title("Средняя процентная ошибка")
plt.legend()
ax3.grid(alpha=0.15, color="#ffffff")
ax3.set_ylim(0, 100)      # <-- MAPE в процентах, ограничиваем 100%

plt.tight_layout()
plt.show()

# -----------------------------
# 9) Примеры прогнозов (каждый раз разные)
# -----------------------------
print("\n" + "=" * 90)
print("Примеры прогнозов ансамбля (реальная цена vs предсказание)")
print("=" * 90)

# Вместо фиксированного сида используем текущее время или просто np.random
rng = np.random.default_rng()   # каждый раз новые индексы
n_samples = min(10, len(y_raw_val))
sample_idx = rng.choice(len(y_raw_val), size=n_samples, replace=False)

for i in sample_idx:
    real_price = y_raw_val[i]
    pred_price = ensemble_preds_rub[i]
    try:
        name = df_val.iloc[i]["name"]
    except:
        name = "—"
    print(f"Устройство: {str(name)[:60]:<60} | "
          f"Реальная: {real_price:>10.0f} ₽ | "
          f"Прогноз: {pred_price:>10.0f} ₽ | "
          f"Ошибка: {abs(real_price - pred_price):>10.0f} ₽")

          
          
          
print("Мин. прогноз:", ensemble_preds_rub.min())
print("Макс. прогноз:", ensemble_preds_rub.max())
print("Среднее прогнозов:", ensemble_preds_rub.mean())
print("Стд прогнозов:", ensemble_preds_rub.std())