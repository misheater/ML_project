"""
sort_devices_pretty.py (исправлено – device_type восстанавливается из one-hot)
Интерактивная сортировка устройств с читаемым выводом.
"""

import pandas as pd
import numpy as np
import re

# ───────────────────── Загрузка и подготовка ─────────────────────
def load_combined():
    original = pd.read_csv("all_devices_cleaned.csv")
    predictions = pd.read_csv("all_predictions.csv")

    # Восстанавливаем device_type
    if "device_type" not in original.columns:
        if {"is_phone", "is_tablet", "is_laptop"}.issubset(original.columns):
            original["device_type"] = "laptop"
            original.loc[original["is_phone"] == 1, "device_type"] = "phone"
            original.loc[original["is_tablet"] == 1, "device_type"] = "tablet"
        else:
            original["device_type"] = "unknown"

    if len(original) != len(predictions):
        print("⚠️ Число строк разное, объединяем по порядку (лучше пересоздать predictions).")
    combined = pd.concat(
        [original, predictions.drop(columns=["name", "device_type"], errors="ignore")],
        axis=1
    )

    # ✅ Новая экономия: предсказанная цена минус реальная (если > 0)
    combined["saved_amount"] = (combined["predicted_price"] - combined["price"]).clip(lower=0)

    return combined




def infer_os(name: str) -> str:
    s = str(name).lower()
    if "windows" in s:
        return "Windows"
    if "mac os" in s or "macbook" in s or "ipad" in s or "iphone" in s:
        return "macOS/iOS"
    if "chrome" in s:
        return "Chrome OS"
    return "Android"

def cpu_short_name(row):
    family = row.get("cpu_family", "")
    if isinstance(family, float) and np.isnan(family):
        family = ""
    family = str(family).lower()
    if "intel" in family:
        gen = int(row.get("cpu_generation", 0))
        tier = int(row.get("cpu_tier", 0))
        if gen:
            return f"Intel Core i{tier} {gen}th Gen"
        return "Intel"
    elif "amd" in family:
        return "AMD Ryzen"
    elif "snapdragon" in family:
        return "Snapdragon"
    elif "mediatek" in family:
        return "MediaTek"
    elif "apple_bionic" in family:
        return "Apple Bionic"
    return "?"

def format_row(row):
    name = str(row.get("name", ""))
    short_name = " ".join(name.split()[:5])

    cpu = cpu_short_name(row)

    storage = f"{int(row.get('storage_gb', 0))} GB"
    if row.get("has_ssd_text", 0) == 1 or row.get("has_ssd", 0) == 1:
        storage += " SSD"
    elif row.get("has_hdd_text", 0) == 1 or row.get("has_hdd", 0) == 1:
        storage += " HDD"

    screen = f"{row.get('screen_size_inch', 0):.1f}\""

    battery = f"{int(row.get('battery_mah', 0))} mAh" if row.get("battery_mah", 0) > 0 else "—"

    discount_pct = row.get("discount_ratio", 0) * 100

    return {
        "Устройство": short_name,
        "Тип": row.get("device_type", ""),
        "Цена, ₽": int(round(row.get("price", 0))),
        "Скидка": f"{discount_pct:.1f}%",
        "Экономия, ₽": int(round(row.get("saved_amount", 0))),
        "Прогноз, ₽": int(round(row.get("predicted_price", 0))),
        "Ошибка, ₽": int(round(abs(row.get("absolute_error", 0)))),
        "Ошибка %": f"{abs(row.get('relative_error_percent', 0)):.1f}",
        "Процессор": cpu,
        "ОС": infer_os(name),
        "RAM": f"{int(row.get('ram_gb', 0))} GB",
        "Накопитель": storage,
        "Экран": screen,
        "Батарея": battery,
    }

# ───────────────────── Интерактивное меню ─────────────────────
def main():
    df = load_combined()

    # Добавляем ОС, если её нет
    if "os" not in df.columns:
        df["os"] = df["name"].apply(infer_os)

    print("\n===== ИНТЕРАКТИВНАЯ СОРТИРОВКА УСТРОЙСТВ =====\n")

    # 1. Тип устройства
    print("Тип устройства:")
    print("  1 - Телефоны")
    print("  2 - Планшеты")
    print("  3 - Ноутбуки")
    print("  4 - Все")
    choice = input("Ваш выбор (1-4): ").strip()
    device_map = {"1": "phone", "2": "tablet", "3": "laptop", "4": "all"}
    dev_type = device_map.get(choice, "all")
    if dev_type != "all":
        df = df[df["device_type"] == dev_type]
        if df.empty:
            print(f"Нет устройств типа '{dev_type}'")
            return

    # 2. Сортировка
    print("\nСортировать по:")
    sort_options = [
        ("discount_ratio", "скидке (наибольшая скидка сверху)"),
        ("price", "реальной цене (от дешёвых)"),
        ("predicted_price", "предсказанной цене (от дешёвых)"),
        ("absolute_error", "абсолютной ошибке прогноза (от меньшей)"),
        ("relative_error_percent", "относительной ошибке % (от меньшей)"),
        ("ram_gb", "объёму RAM"),
        ("storage_gb", "объёму памяти"),
        ("rating", "рейтингу"),
        ("total_reviews", "количеству отзывов"),
        ("saved_amount", "сэкономленной сумме (от большей к меньшей)"),
    ]
    for idx, (col, desc) in enumerate(sort_options, start=1):
        print(f"  {idx} - {desc}")
    print("  0 - свой вариант (ввести название колонки)")

    choice = input("Ваш выбор (0-9): ").strip()
    if choice == "0":
        sort_col = input("Введите точное имя колонки: ").strip()
    else:
        try:
            sort_col = sort_options[int(choice)-1][0]
        except (IndexError, ValueError):
            print("Неверный выбор, сортируем по скидке.")
            sort_col = "discount_ratio"

    if sort_col not in df.columns:
        print(f"Колонка '{sort_col}' не найдена. Доступные: {list(df.columns)}")
        return

    if sort_col in ["discount_ratio", "rating", "total_reviews"]:
        default_asc = False
    else:
        default_asc = True
    asc = input(f"Сортировать по возрастанию? (y/n, по умолчанию {'y' if default_asc else 'n'}): ").strip().lower()
    if asc in ("y", "n"):
        ascending = asc == "y"
    else:
        ascending = default_asc

    df_sorted = df.sort_values(by=sort_col, ascending=ascending)

    # 3. Сколько строк
    top_input = input("Сколько строк показать? (Enter = 20): ").strip()
    top = int(top_input) if top_input.isdigit() else 20

    # Преобразуем в читаемый вид
    top_df = df_sorted.head(top)
    pretty_rows = [format_row(row) for _, row in top_df.iterrows()]
    pretty_df = pd.DataFrame(pretty_rows)

    default_cols = [
        "Устройство", "Тип", "Цена, ₽", "Скидка", "Экономия, ₽",  "Прогноз, ₽",
        "Ошибка, ₽", "Ошибка %", "Процессор", "ОС", "RAM",
        "Накопитель", "Экран", "Батарея"
    ]
    show_cols = [c for c in default_cols if c in pretty_df.columns]

    print(f"\nСортировка по '{sort_col}' {'▲' if ascending else '▼'}, устройство: {dev_type}")
    print(pretty_df[show_cols].to_string(index=False))

    # 4. Сохранение
    save = input("\nСохранить в CSV? (y/n): ").strip().lower()
    if save == "y":
        fname = input("Имя файла (Enter = sorted_devices.csv): ").strip()
        if not fname:
            fname = "sorted_devices.csv"
        pretty_df[show_cols].to_csv(fname, index=False)
        print(f"Сохранено в {fname}")

if __name__ == "__main__":
    main()