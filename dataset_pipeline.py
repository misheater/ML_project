import re
import numpy as np
import pandas as pd

LAPTOP_FILE = "laptop_dataset.csv"
PHONE_FILE = "mobile_dataset.csv"
TABLET_FILE = "tablet_dataset.csv"
OUT_FILE = "all_devices_cleaned.csv"

PRICE_COLUMN = "offer_price"

WEIGHTS = {
    "phone": {"ram_gb": 1.30, "storage_gb": 1.10, "battery_mah": 1.15, "screen_size_inch": 0.95},
    "tablet": {"ram_gb": 1.10, "storage_gb": 1.05, "battery_mah": 1.05, "screen_size_inch": 1.00},
    "laptop": {"ram_gb": 0.95, "storage_gb": 1.00, "battery_mah": 0.90, "screen_size_inch": 1.10},
}


def extract_float(text, pattern):
    if pd.isna(text):
        return np.nan
    m = re.search(pattern, str(text), flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def normalize_name(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["description"].astype(str)
    n = df["name"].astype(str)
    text = (n + " " + d).str.lower()

    df["ram_gb"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*GB\s*RAM"))
    df["ram_mb"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*MB\s*RAM"))
    df["ram_gb"] = df["ram_gb"].fillna(df["ram_mb"] / 1024.0)

    df["storage_gb"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*GB\s*ROM"))
    df["storage_tb"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*TB"))
    df["storage_gb"] = df["storage_gb"].fillna(df["storage_tb"] * 1024.0)

    df["screen_size_inch"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in)\b"))
    df["screen_size_cm"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*cm"))
    df["screen_size_inch"] = df["screen_size_inch"].fillna(df["screen_size_cm"] / 2.54)

    df["battery_mah"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*mAh"))
    df["camera_megapixels"] = d.apply(lambda x: extract_float(x, r"(\d+(?:\.\d+)?)\s*MP"))

    df["cpu_generation"] = text.apply(lambda x: extract_float(x, r"\b(\d{1,2})(?:st|nd|rd|th)\s*gen\b") or 0.0)
    df["cpu_tier"] = text.apply(cpu_tier).astype(np.float32)
    df["gpu_tier"] = text.apply(gpu_tier).astype(np.float32)
    df["refresh_hz"] = text.apply(lambda x: extract_float(x, r"\b(\d{2,3})\s*hz\b"))

    # Полезные бинарные фичи
    df["has_5g"] = text.str.contains(r"\b5g\b", regex=True).astype(np.float32)
    df["has_stylus"] = text.str.contains(r"\bstylus\b|\bpencil\b", regex=True).astype(np.float32)
    df["has_oled"] = text.str.contains(r"\boled\b|\bamoled\b", regex=True).astype(np.float32)
    df["has_ssd"] = text.str.contains(r"\bssd\b|\bnvme\b", regex=True).astype(np.float32)
    df["has_hdd"] = text.str.contains(r"\bhdd\b", regex=True).astype(np.float32)
    df["has_touch"] = text.str.contains(r"\btouch\b", regex=True).astype(np.float32)
    df["has_wifi_only"] = text.str.contains(r"wi[\-\s]?fi only", regex=True).astype(np.float32)

    # Базовая модельная линейка устройства
    df["model_family"] = text.apply(model_family)

    return df


def cpu_tier(text: str) -> float:
    intel = re.search(r"\bcore\s*i([3579])\b", text)
    if intel:
        return float(intel.group(1))
    ryzen = re.search(r"\bryzen\s*([3579])\b", text)
    if ryzen:
        return float(ryzen.group(1))
    if re.search(r"\bsnapdragon\b", text):
        return 5.5
    if re.search(r"\bmediatek\b|\bhelio\b|\bdimensity\b", text):
        return 4.5
    if re.search(r"\bbionic\b", text):
        return 8.0
    return 0.0


def gpu_tier(text: str) -> float:
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


def model_family(text: str) -> str:
    keys = [
        "iphone", "ipad", "galaxy", "redmi", "poco", "vivobook", "ideapad",
        "omen", "pavilion", "aspire", "macbook", "rog", "infinix", "realme"
    ]
    for k in keys:
        if re.search(rf"\b{re.escape(k)}\b", text):
            return k
    return "other"


def load_one(path: str, device_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["device_type"] = device_type

    df["offer_price_num"] = pd.to_numeric(df.get("offer_price"), errors="coerce")
    df["original_price_num"] = pd.to_numeric(df.get("original_price"), errors="coerce")

    raw_price = df[PRICE_COLUMN].astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", np.nan)
    df["price"] = pd.to_numeric(raw_price, errors="coerce")
    df["discount_ratio"] = 1.0 - (df["offer_price_num"] / (df["original_price_num"] + 1e-8))
    df["discount_ratio"] = df["discount_ratio"].clip(lower=0.0, upper=0.95)

    for c in ["total_ratings", "total_reviews", "rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "description" not in df.columns:
        df["description"] = ""

    df = parse_features(df)

    df["name_norm"] = df["name"].apply(normalize_name)
    df["brand_token"] = df["name_norm"].str.split().str[0].fillna("unknown")
    df["name_len"] = df["name"].astype(str).str.len().astype(np.float32)
    df["desc_len"] = df["description"].astype(str).str.len().astype(np.float32)
    df["text_len"] = df["name_len"] + df["desc_len"]

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df["ram_per_screen"] = df["ram_gb"] / (df["screen_size_inch"] + eps)
    df["storage_per_ram"] = df["storage_gb"] / (df["ram_gb"] + eps)
    df["battery_per_inch"] = df["battery_mah"] / (df["screen_size_inch"] + eps)
    df["reviews_per_rating"] = df["total_reviews"] / (df["total_ratings"] + 1.0)
    df["engagement_score"] = np.log1p(df["total_ratings"].clip(lower=0)) * (df["rating"] + eps)
    df["camera_x_ram"] = df["camera_megapixels"] * df["ram_gb"]
    df["battery_x_ram"] = df["battery_mah"] * df["ram_gb"]
    df["cpu_tier_x_ram"] = df["cpu_tier"] * df["ram_gb"]
    df["gpu_tier_x_ram"] = df["gpu_tier"] * df["ram_gb"]
    df["refresh_x_gpu"] = df["refresh_hz"] * (df["gpu_tier"] + 1.0)
    df["is_gaming_like"] = (
        ((df["gpu_tier"] >= 6) | (df["refresh_hz"] >= 120) | (df["has_rtx_like"] > 0))
        .astype(np.float32)
    )

    for feature in ["ram_gb", "storage_gb", "battery_mah", "screen_size_inch"]:
        df[f"{feature}_weighted"] = df.apply(
            lambda r: r[feature] * WEIGHTS.get(r["device_type"], {}).get(feature, 1.0),
            axis=1,
        )

    return df


def add_hardware_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Тут делаем флаг через уже рассчитанные значения, чтобы не дублировать regex в других местах
    df["has_rtx_like"] = (df["gpu_tier"] >= 7).astype(np.float32)
    return df


def fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)
    return df


def trim_group_outliers(df: pd.DataFrame, min_group_size: int = 25) -> pd.DataFrame:
    # Мягко режем выбросы цены внутри (device_type, brand_token)
    out_parts = []
    grouped = df.groupby(["device_type", "brand_token"], dropna=False)
    for _, g in grouped:
        if len(g) < min_group_size:
            out_parts.append(g)
            continue
        q1 = g["price"].quantile(0.10)
        q9 = g["price"].quantile(0.90)
        iqr = q9 - q1
        lo = max(0.0, q1 - 1.5 * iqr)
        hi = q9 + 1.5 * iqr
        out_parts.append(g[(g["price"] >= lo) & (g["price"] <= hi)])
    return pd.concat(out_parts, ignore_index=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # Сначала удаляем точные дубли строк (кроме идентификаторов/ссылок/даты)
    subset_cols = [c for c in df.columns if c not in ["u_id", "item_link", "created_at"]]
    df = df.drop_duplicates(subset=subset_cols, keep="first").copy()

    # Затем "почти дубли": одинаковый нормализованный name + устройство + очень близкая цена
    df = df.sort_values(["device_type", "name_norm", "price"]).copy()
    df["_dedupe_group_key"] = df["device_type"].astype(str) + "||" + df["name_norm"].astype(str)
    df["_dedupe_price_bucket"] = (df["price"] / 500.0).round().astype("Int64")
    df = df.drop_duplicates(subset=["_dedupe_group_key", "_dedupe_price_bucket"], keep="first")
    df = df.drop(columns=["_dedupe_group_key", "_dedupe_price_bucket"])
    return df


def main():
    df_phone = load_one(PHONE_FILE, "phone")
    df_tablet = load_one(TABLET_FILE, "tablet")
    df_laptop = load_one(LAPTOP_FILE, "laptop")

    df_all = pd.concat([df_phone, df_tablet, df_laptop], ignore_index=True)

    before = len(df_all)
    df_all = df_all[df_all["price"].notna()].copy()
    df_all = dedupe(df_all)
    df_all = add_hardware_flags(df_all)
    df_all = trim_group_outliers(df_all, min_group_size=25)
    after = len(df_all)

    df_all = add_engineered_features(df_all)
    df_all = pd.get_dummies(df_all, columns=["device_type"], prefix="is")
    for feature in ["ram_gb", "storage_gb", "battery_mah", "screen_size_inch"]:
        for dev_col in ["is_phone", "is_tablet", "is_laptop"]:
            if dev_col in df_all.columns:
                df_all[f"{feature}__x__{dev_col}"] = df_all[feature] * df_all[dev_col]

    df_all = fill_numeric(df_all)

    # Оставляем полезные текстовые поля для NLP в train скрипте
    keep_text = ["name", "description", "brand_token", "name_norm"]
    numeric_and_flags = [c for c in df_all.columns if c not in ["u_id", "item_link", "created_at"]]
    cols = []
    for c in keep_text + numeric_and_flags:
        if c in df_all.columns and c not in cols:
            cols.append(c)
    df_all = df_all[cols]

    df_all.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")
    print(f"Rows before cleanup: {before}, after cleanup: {after}")
    print(df_all.head(3))
    print(df_all.shape)


if __name__ == "__main__":
    main()
