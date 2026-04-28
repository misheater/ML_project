import os
import json
from typing import Dict

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from predict_price import predict_from_payload


HELP_TEXT = (
    "Я бот для прогноза цены электроники.\n\n"
    "Команды:\n"
    "/wizard - пошаговый мастер (рекомендуется)\n"
    "/predict <JSON>\n\n"
    "Пример:\n"
    '/predict {"name":"Samsung Galaxy S21 FE 8GB 256GB","description":"8 GB RAM 256 GB ROM Snapdragon 888 120Hz AMOLED 5G","device_type":"phone","ram_gb":8,"storage_gb":256,"screen_size_inch":6.4,"battery_mah":4500,"camera_megapixels":64,"rating":4.5,"total_ratings":8000,"total_reviews":1200,"refresh_hz":120}\n\n'
    "Обязательные поля: name, device_type.\n"
    "device_type: phone | tablet | laptop"
)

(
    W_DEVICE_TYPE,
    W_NAME,
    W_DESCRIPTION,
    W_RAM,
    W_STORAGE,
    W_SCREEN,
    W_BATTERY,
    W_CAMERA,
    W_RATING,
    W_TOTAL_RATINGS,
    W_TOTAL_REVIEWS,
    W_REFRESH,
) = range(12)


def normalize_payload(payload: Dict) -> Dict:
    defaults = {
        "description": "",
        "device_type": "phone",
        "ram_gb": 8.0,
        "storage_gb": 128.0,
        "screen_size_inch": 6.5,
        "battery_mah": 5000.0,
        "camera_megapixels": 50.0,
        "rating": 4.3,
        "total_ratings": 1000.0,
        "total_reviews": 100.0,
        "refresh_hz": 60.0,
    }
    out = defaults.copy()
    out.update(payload)
    out["name"] = str(out.get("name", "")).strip()
    out["device_type"] = str(out.get("device_type", "phone")).lower().strip()
    return out


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот запущен.\n\n" + HELP_TEXT)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _ = predict_from_payload({"name": "healthcheck", "device_type": "phone"})
        await update.message.reply_text("Статус: OK. Модель и артефакты загружены.")
    except Exception as e:
        await update.message.reply_text(f"Статус: ошибка загрузки модели: {e}")


async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Нужен JSON после /predict.\n\n" + HELP_TEXT)
        return

    try:
        payload = json.loads(raw)
        payload = normalize_payload(payload)
        if not payload["name"]:
            await update.message.reply_text("Поле 'name' обязательно.")
            return
        if payload["device_type"] not in {"phone", "tablet", "laptop"}:
            await update.message.reply_text("device_type должен быть phone/tablet/laptop.")
            return

        price = predict_from_payload(payload)
        await update.message.reply_text(
            f"Прогноз цены: {price:,.2f} RUB".replace(",", " ")
        )
    except json.JSONDecodeError:
        await update.message.reply_text("Невалидный JSON.\n\n" + HELP_TEXT)
    except Exception as e:
        await update.message.reply_text(f"Ошибка прогноза: {e}")


async def wizard_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["payload"] = {}
    await update.message.reply_text("Мастер прогноза.\nВведите тип устройства: phone / tablet / laptop")
    return W_DEVICE_TYPE


async def wizard_device(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = update.message.text.strip().lower()
    if t not in {"phone", "tablet", "laptop"}:
        await update.message.reply_text("Неверный тип. Введите: phone / tablet / laptop")
        return W_DEVICE_TYPE
    context.user_data["payload"]["device_type"] = t
    await update.message.reply_text("Введите название устройства (name):")
    return W_NAME


async def wizard_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["payload"]["name"] = update.message.text.strip()
    await update.message.reply_text("Введите description (или '-' если нет):")
    return W_DESCRIPTION


async def wizard_description(update: Update, context: ContextTypes.DEFAULT_TYPE):
    desc = update.message.text.strip()
    context.user_data["payload"]["description"] = "" if desc == "-" else desc
    await update.message.reply_text("RAM (GB), например 8:")
    return W_RAM


async def _read_float(update: Update, err_text: str):
    try:
        return float(update.message.text.strip()), None
    except ValueError:
        await update.message.reply_text(err_text)
        return None, True


async def wizard_ram(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число для RAM, например 8")
    if bad:
        return W_RAM
    context.user_data["payload"]["ram_gb"] = v
    await update.message.reply_text("Storage (GB), например 128:")
    return W_STORAGE


async def wizard_storage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число для Storage, например 128")
    if bad:
        return W_STORAGE
    context.user_data["payload"]["storage_gb"] = v
    await update.message.reply_text("Screen size (inch), например 6.5:")
    return W_SCREEN


async def wizard_screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число для экрана, например 6.5")
    if bad:
        return W_SCREEN
    context.user_data["payload"]["screen_size_inch"] = v
    await update.message.reply_text("Battery (mAh), например 5000:")
    return W_BATTERY


async def wizard_battery(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число для батареи, например 5000")
    if bad:
        return W_BATTERY
    context.user_data["payload"]["battery_mah"] = v
    await update.message.reply_text("Camera (MP), например 50:")
    return W_CAMERA


async def wizard_camera(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число для камеры, например 50")
    if bad:
        return W_CAMERA
    context.user_data["payload"]["camera_megapixels"] = v
    await update.message.reply_text("Рейтинг (например 4.4):")
    return W_RATING


async def wizard_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите рейтинг числом, например 4.4")
    if bad:
        return W_RATING
    context.user_data["payload"]["rating"] = v
    await update.message.reply_text("Количество оценок (total_ratings), например 5000:")
    return W_TOTAL_RATINGS


async def wizard_total_ratings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число total_ratings, например 5000")
    if bad:
        return W_TOTAL_RATINGS
    context.user_data["payload"]["total_ratings"] = v
    await update.message.reply_text("Количество отзывов (total_reviews), например 700:")
    return W_TOTAL_REVIEWS


async def wizard_total_reviews(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число total_reviews, например 700")
    if bad:
        return W_TOTAL_REVIEWS
    context.user_data["payload"]["total_reviews"] = v
    await update.message.reply_text("Refresh rate (Hz), например 60 или 120:")
    return W_REFRESH


async def wizard_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v, bad = await _read_float(update, "Введите число refresh_hz, например 60")
    if bad:
        return W_REFRESH
    context.user_data["payload"]["refresh_hz"] = v

    payload = normalize_payload(context.user_data["payload"])
    try:
        price = predict_from_payload(payload)
        await update.message.reply_text(
            f"Готово.\nПрогноз цены: {price:,.2f} RUB".replace(",", " ")
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка прогноза: {e}")
    return ConversationHandler.END


async def wizard_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Мастер отменён.")
    return ConversationHandler.END


def main():
    token ="8687836720:AAEXJHnLPHywKhTk5y3AI-ptCltyG0xWq5g"
    if not token.strip():
        raise RuntimeError("Задайте TELEGRAM_BOT_TOKEN в переменных окружения.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))

    wizard_handler = ConversationHandler(
        entry_points=[CommandHandler("wizard", wizard_start)],
        states={
            W_DEVICE_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_device)],
            W_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_name)],
            W_DESCRIPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_description)],
            W_RAM: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_ram)],
            W_STORAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_storage)],
            W_SCREEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_screen)],
            W_BATTERY: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_battery)],
            W_CAMERA: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_camera)],
            W_RATING: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_rating)],
            W_TOTAL_RATINGS: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_total_ratings)],
            W_TOTAL_REVIEWS: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_total_reviews)],
            W_REFRESH: [MessageHandler(filters.TEXT & ~filters.COMMAND, wizard_refresh)],
        },
        fallbacks=[CommandHandler("cancel", wizard_cancel)],
    )
    app.add_handler(wizard_handler)
    app.run_polling()


if __name__ == "__main__":
    main()
