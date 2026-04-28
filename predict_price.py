import argparse
import json
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ARTIFACTS_DIR = "artifacts"

WEIGHTS = {
    "phone": {"ram_gb": 1.30, "storage_gb": 1.10, "battery_mah": 1.15, "screen_size_inch": 0.95},
    "tablet": {"ram_gb": 1.10, "storage_gb": 1.05, "battery_mah": 1.05, "screen_size_inch": 1.00},
    "laptop": {"ram_gb": 0.95, "storage_gb": 1.00, "battery_mah": 0.90, "screen_size_inch": 1.10},
}

MODEL_KEYWORDS = [
    "iphone", "galaxy", "redmi", "poco", "vivobook", "ideapad", "omen",
    "pavilion", "macbook", "infinix", "realme", "ipad", "tab", "thinkpad",
    "rog", "surface", "moto", "oneplus", "xiaomi",
]

PATTERN_FEATURES = {
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


class PricePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 128
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


def build_feature_row(args, meta):
    name = (args.name or "").strip()
    description = (args.description or "").strip()
    device_type = args.device_type.lower().strip()
    text_l = (name + " " + description).lower()
    name_l = name.lower()
    desc_l = description.lower()

    row = {
        "ram_gb": float(args.ram_gb),
        "storage_gb": float(args.storage_gb),
        "screen_size_inch": float(args.screen_size_inch),
        "battery_mah": float(args.battery_mah),
        "camera_megapixels": float(args.camera_megapixels),
        "rating": float(args.rating),
        "total_ratings": float(args.total_ratings),
        "total_reviews": float(args.total_reviews),
    }

    row["offer_price_num"] = 0.0
    row["original_price_num"] = 0.0
    row["discount_ratio"] = 0.0
    row["ram_mb"] = row["ram_gb"] * 1024.0
    row["storage_tb"] = row["storage_gb"] / 1024.0
    row["screen_size_cm"] = row["screen_size_inch"] * 2.54
    row["refresh_hz"] = float(args.refresh_hz)
    row["name_len"] = float(len(name_l))
    row["desc_len"] = float(len(desc_l))
    row["text_len"] = row["name_len"] + row["desc_len"]

    eps = 1e-6
    row["ram_per_screen"] = row["ram_gb"] / (row["screen_size_inch"] + eps)
    row["storage_per_ram"] = row["storage_gb"] / (row["ram_gb"] + eps)
    row["battery_per_inch"] = row["battery_mah"] / (row["screen_size_inch"] + eps)
    row["reviews_per_rating"] = row["total_reviews"] / (row["total_ratings"] + 1.0)
    row["engagement_score"] = np.log1p(max(0.0, row["total_ratings"])) * (row["rating"] + eps)
    row["camera_x_ram"] = row["camera_megapixels"] * row["ram_gb"]
    row["battery_x_ram"] = row["battery_mah"] * row["ram_gb"]

    row["cpu_generation"] = extract_gen(text_l)
    row["cpu_tier"] = extract_cpu_tier(text_l)
    row["gpu_tier"] = extract_gpu_tier(text_l)
    row["cpu_tier_x_ram"] = row["cpu_tier"] * row["ram_gb"]
    row["gpu_tier_x_ram"] = row["gpu_tier"] * row["ram_gb"]
    row["cpu_tier_x_storage"] = row["cpu_tier"] * row["storage_gb"]

    for kw in MODEL_KEYWORDS:
        row[f"kw_{kw}"] = 1.0 if re.search(rf"\b{re.escape(kw)}\b", text_l) else 0.0
    for fname, pat in PATTERN_FEATURES.items():
        row[fname] = 1.0 if re.search(pat, text_l) else 0.0

    row["is_phone"] = 1.0 if device_type == "phone" else 0.0
    row["is_tablet"] = 1.0 if device_type == "tablet" else 0.0
    row["is_laptop"] = 1.0 if device_type == "laptop" else 0.0

    for feature in ["ram_gb", "storage_gb", "battery_mah", "screen_size_inch"]:
        coef = WEIGHTS.get(device_type, {}).get(feature, 1.0)
        row[f"{feature}_weighted"] = row[feature] * coef
        row[f"{feature}__x__is_phone"] = row[feature] * row["is_phone"]
        row[f"{feature}__x__is_tablet"] = row[feature] * row["is_tablet"]
        row[f"{feature}__x__is_laptop"] = row[feature] * row["is_laptop"]

    row["brand_token"] = re.sub(r"[^a-z0-9]+", " ", name_l).strip().split(" ")[0] if name_l else "unknown"
    te = meta.get("te_artifacts", {})
    cpu_f = cpu_family(text_l)
    gpu_f = gpu_family(text_l)
    device_lbl = device_type
    te_inputs = {
        "brand_token": row["brand_token"],
        "cpu_family": cpu_f,
        "gpu_family": gpu_f,
        "device_type_label": device_lbl,
    }
    for col, key in te_inputs.items():
        col_meta = te.get(col, {})
        mapping = col_meta.get("map", {})
        global_mean = float(col_meta.get("global_mean", 0.0))
        row[f"te_{col}"] = float(mapping.get(str(key), global_mean))

    return row


def predict_from_payload(payload: dict, artifacts_dir: str = ARTIFACTS_DIR) -> float:
    class ArgsObj:
        pass

    args = ArgsObj()
    args.name = str(payload.get("name", ""))
    args.description = str(payload.get("description", ""))
    args.device_type = str(payload.get("device_type", "phone")).lower()
    args.ram_gb = float(payload.get("ram_gb", 8.0))
    args.storage_gb = float(payload.get("storage_gb", 128.0))
    args.screen_size_inch = float(payload.get("screen_size_inch", 6.5))
    args.battery_mah = float(payload.get("battery_mah", 5000.0))
    args.camera_megapixels = float(payload.get("camera_megapixels", 50.0))
    args.rating = float(payload.get("rating", 4.3))
    args.total_ratings = float(payload.get("total_ratings", 1000.0))
    args.total_reviews = float(payload.get("total_reviews", 100.0))
    args.refresh_hz = float(payload.get("refresh_hz", 60.0))

    with open(f"{artifacts_dir}/preprocess_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    feature_cols = meta["feature_cols"]
    medians = meta["feature_medians"]
    row = build_feature_row(args, meta)

    x = {c: float(medians.get(c, 0.0)) for c in feature_cols}
    for c, v in row.items():
        if c in x:
            x[c] = float(v)

    X_df = pd.DataFrame([x], columns=feature_cols)
    X_scaled = scaler.transform(X_df.values.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PricePredictor(input_dim=len(feature_cols)).to(device)
    state = torch.load(f"{artifacts_dir}/best_price_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        preds_log = model(torch.tensor(X_scaled, dtype=torch.float32, device=device)).cpu().numpy()

    pred_price = float(np.expm1(preds_log[0]))
    return max(0.0, pred_price)


def main():
    parser = argparse.ArgumentParser(description="Predict electronics price in RUB")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--device_type", type=str, choices=["phone", "tablet", "laptop"], required=True)
    parser.add_argument("--ram_gb", type=float, default=8.0)
    parser.add_argument("--storage_gb", type=float, default=128.0)
    parser.add_argument("--screen_size_inch", type=float, default=6.5)
    parser.add_argument("--battery_mah", type=float, default=5000.0)
    parser.add_argument("--camera_megapixels", type=float, default=50.0)
    parser.add_argument("--rating", type=float, default=4.3)
    parser.add_argument("--total_ratings", type=float, default=1000.0)
    parser.add_argument("--total_reviews", type=float, default=100.0)
    parser.add_argument("--refresh_hz", type=float, default=60.0)
    args = parser.parse_args()

    pred_price = predict_from_payload(
        payload={
            "name": args.name,
            "description": args.description,
            "device_type": args.device_type,
            "ram_gb": args.ram_gb,
            "storage_gb": args.storage_gb,
            "screen_size_inch": args.screen_size_inch,
            "battery_mah": args.battery_mah,
            "camera_megapixels": args.camera_megapixels,
            "rating": args.rating,
            "total_ratings": args.total_ratings,
            "total_reviews": args.total_reviews,
            "refresh_hz": args.refresh_hz,
        }
    )
    print(f"Predicted price (RUB): {pred_price:.2f}")


if __name__ == "__main__":
    main()
