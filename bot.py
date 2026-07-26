import os
import logging
import tempfile
from datetime import timedelta

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")  # tiny/base/small/medium

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def format_timestamp(seconds: float) -> str:
    total_ms = int(max(0, seconds) * 1000)
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a video, audio, or voice file and I'll generate an .srt subtitle file for it."
    )


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    file_obj = message.video or message.audio or message.voice or message.document
    if file_obj is None:
        await message.reply_text("Please send a video, audio, or voice file.")
        return

    status_msg = await message.reply_text("Downloading file...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input")
        tg_file = await context.bot.get_file(file_obj.file_id)
        await tg_file.download_to_drive(input_path)

        await status_msg.edit_text("Transcribing... this can take a while for longer files.")

        try:
            segments, info = model.transcribe(input_path, beam_size=5)
            segments = list(segments)
        except Exception as e:
            logger.exception("Transcription failed")
            await status_msg.edit_text(f"Sorry, transcription failed: {e}")
            return

        srt_content = segments_to_srt(segments)
        srt_path = os.path.join(tmp_dir, "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        await status_msg.edit_text("Done! Sending your subtitle file...")
        with open(srt_path, "rb") as f:
            await message.reply_document(document=f, filename="subtitles.srt")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL,
            handle_media,
        )
    )
    app.run_polling()


if __name__ == "__main__":
    main()
