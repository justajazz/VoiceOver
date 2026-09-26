#!/usr/bin/env python3
"""
Озвучка сценария через Gemini 3.8 Flash TTS.

Запуск:
  py voice.py dialogue.txt              # весь диалог одним файлом (до 2 спикеров, встроенные голоса)
  py voice.py dialogue.txt --lines      # каждая реплика отдельным файлом + общий склеенный файл
  py voice.py timed.txt                 # сценарий с таймкодами -> дорожка timeline.wav под видео
  py voice.py timed.txt --fit           # то же + подгонка реплик, которые не влезают в своё окно

Формат сценария:
  @Имя: style                  <- заголовок блока; style после двоеточия можно не писать
  @00:12.5 Имя: style          <- то же с таймкодом начала реплики
  текст реплики с <тегами>
  # строки с решёткой — комментарии

Таймкоды: мм:сс, мм:сс.доли, чч:мм:сс, чч:мм:сс:кадры (для кадров укажи --fps).
"""

import argparse
import base64
import io
import os
import re
import shutil
import subprocess
import sys
import wave
from array import array
from datetime import datetime
from pathlib import Path

# ============ НАСТРОЙКИ ============
MODEL = "gemini-3.8-flash-tts"   # или "gemini-3.8-flash-lite-tts" (дешевле и быстрее)

# Имя спикера из сценария -> голос.
# Встроенный голос (Puck, Charon, Kore, Fola...) или ID своего голоса (voice_...).
VOICES = {
    "Alex": "Puck",
    "Max": "Charon",
}

OUTPUT_DIR = "output"
# ==================================

HEADER_RE = re.compile(
    r"^@\s*(?:(?P<tc>\d+(?::\d{1,2}){1,3}(?:[.,]\d+)?)\s+)?(?P<name>[^:]+?)\s*(?::\s*(?P<style>.*))?$"
)
BACKCHANNEL_RE = re.compile(r"\s*\|[^|]*\|\s*")


# ---------- сценарий ----------

def parse_timecode(tc, fps):
    """'12:05.5' / '01:02:03' / '01:00:05:12' (с кадрами) -> секунды."""
    parts = tc.replace(",", ".").split(":")
    if len(parts) == 4:
        h, m, s, f = parts
        return int(h) * 3600 + int(m) * 60 + int(s) + float(f) / fps
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    m, s = parts
    return int(m) * 60 + float(s)


def parse_script(path, fps):
    """Разбирает сценарий на блоки: [{'speaker', 'style', 'text', 'start'}]."""
    blocks, current = [], None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = HEADER_RE.match(line) if line.startswith("@") else None
        if match:
            tc = match.group("tc")
            current = {
                "speaker": match.group("name").strip(),
                "style": (match.group("style") or "").strip(),
                "text": [],
                "start": parse_timecode(tc, fps) if tc else None,
                "tc": tc,
            }
            blocks.append(current)
        elif current is None:
            sys.exit("Ошибка: сценарий должен начинаться со строки вида '@Имя: style'.")
        else:
            current["text"].append(line)
    for block in blocks:
        block["text"] = " ".join(block["text"])
    return [b for b in blocks if b["text"]]


# ---------- API ----------

def make_part(block, include_speaker):
    """Текст блока + speech_metadata (speaker/style)."""
    meta = {"type": "speech_metadata"}
    if include_speaker:
        meta["speaker"] = block["speaker"]
    if block["style"]:
        meta["style"] = block["style"]
    part = {"type": "text", "text": block["text"]}
    if len(meta) > 1:
        part["annotations"] = [meta]
    return part


def synthesize(client, parts, speech_config):
    """Один запрос к TTS, возвращает байты WAV."""
    interaction = client.interactions.create(
        model=MODEL,
        input=[{"type": "user_input", "content": parts}],
        response_format={"type": "audio"},
        generation_config={"speech_config": speech_config},
    )
    if not interaction.output_audio or not interaction.output_audio.data:
        raise RuntimeError("модель не вернула аудио")
    return base64.b64decode(interaction.output_audio.data)


def synth_line(client, block):
    """Одна реплика одним голосом (бэкченнелы |...| вырезаются)."""
    clean = dict(block)
    clean["text"] = BACKCHANNEL_RE.sub(" ", block["text"]).strip()
    return synthesize(client, [make_part(clean, include_speaker=False)],
                      [{"voice": VOICES[block["speaker"]]}])


# ---------- WAV ----------

def read_wav(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getparams(), w.readframes(w.getnframes())


def duration(data):
    params, _ = read_wav(data)
    return params.nframes / params.framerate


def speed_up(data, factor, tmp_dir):
    """Ускоряет аудио через ffmpeg без изменения высоты голоса."""
    src, dst = tmp_dir / "_in.wav", tmp_dir / "_out.wav"
    src.write_bytes(data)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-filter:a", f"atempo={factor:.3f}", str(dst)], check=True)
    result = dst.read_bytes()
    src.unlink()
    dst.unlink()
    return result


def concat_wavs(paths, out_path, gap):
    """Склеивает WAV-файлы подряд с паузой gap секунд."""
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    silence = b"\x00" * int(params.framerate * gap) * params.sampwidth * params.nchannels
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for i, path in enumerate(paths):
            with wave.open(str(path), "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))
            if i < len(paths) - 1:
                out.writeframes(silence)


def build_timeline(clips, out_path):
    """Раскладывает клипы по их таймкодам в одну дорожку от 00:00. Наложения смешиваются."""
    params, timeline, last_end = None, bytearray(), 0
    for start, data in clips:
        p, frames = read_wav(data)
        params = params or p
        bpf = p.sampwidth * p.nchannels
        offset = int(round(start * p.framerate)) * bpf
        end = offset + len(frames)
        if len(timeline) < end:
            timeline.extend(b"\x00" * (end - len(timeline)))
        if offset >= last_end:
            timeline[offset:end] = frames
        else:
            overlap = min(last_end - offset, len(frames))
            a = array("h", bytes(timeline[offset:offset + overlap]))
            b = array("h", frames[:overlap])
            mixed = array("h", (max(-32768, min(32767, x + y)) for x, y in zip(a, b)))
            timeline[offset:offset + overlap] = mixed.tobytes()
            timeline[offset + overlap:end] = frames[overlap:]
        last_end = max(last_end, end)
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        out.writeframes(bytes(timeline))


def fmt_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


# ---------- режимы ----------

def run_dialogue(client, blocks, out_dir):
    """Весь диалог одним запросом: естественные переходы и бэкченнелы |...|."""
    speakers = list(dict.fromkeys(b["speaker"] for b in blocks))
    if len(speakers) > 2:
        sys.exit(f"В режиме диалога максимум 2 спикера, найдено: {speakers}. Используй --lines.")
    if any(VOICES[s].startswith("voice_") for s in speakers):
        sys.exit("Свои голоса (voice_...) в одном диалоге не поддерживаются. Используй --lines.")
    if len(speakers) == 1:
        print("Один спикер — переключаюсь в режим --lines.")
        return run_lines(client, blocks, out_dir, gap=0.3)

    speech_config = {
        "mode": "conversational",
        "speakers": [{"speaker": s, "voice": VOICES[s]} for s in speakers],
    }
    parts = [make_part(b, include_speaker=True) for b in blocks]
    print(f"Генерирую диалог: {len(blocks)} реплик...")
    path = out_dir / "dialogue.wav"
    path.write_bytes(synthesize(client, parts, speech_config))
    print(f"Готово: {path}")


def run_lines(client, blocks, out_dir, gap):
    """Каждая реплика отдельным запросом и файлом, потом склейка в full.wav."""
    files = []
    for i, block in enumerate(blocks, 1):
        print(f"  [{i:02d}] {block['speaker']}: {block['text'][:60]}...")
        try:
            audio = synth_line(client, block)
        except Exception as err:
            print(f"  [{i:02d}] ОШИБКА: {err} — пропускаю")
            continue
        path = out_dir / f"{i:02d}_{block['speaker']}.wav"
        path.write_bytes(audio)
        files.append(path)
    if files:
        concat_wavs(files, out_dir / "full.wav", gap)
        print(f"Готово: {len(files)} файлов + full.wav в {out_dir}")


def run_timeline(client, blocks, out_dir, fit, max_speed, retries):
    """Реплики по таймкодам -> timeline.wav + отчёт, где что не влезло."""
    ffmpeg = shutil.which("ffmpeg")
    if fit and not ffmpeg:
        print("ffmpeg не найден: ускорения не будет, только перегенерация в более быстром темпе.")

    clips, report = [], []
    for i, block in enumerate(blocks):
        start = block["start"]
        window = blocks[i + 1]["start"] - start if i + 1 < len(blocks) else None
        label = f"[{i + 1:02d}] {fmt_time(start)} {block['speaker']}"
        print(f"  {label}: {block['text'][:50]}...")
        try:
            audio = synth_line(client, block)
        except Exception as err:
            print(f"  {label} ОШИБКА: {err} — пропускаю")
            report.append(f"{label:<28} ОШИБКА: {err}")
            continue
        dur, note = duration(audio), ""

        if fit and window and dur > window:
            faster = dict(block)
            faster["style"] = f"{block['style']}, speaking rapidly" if block["style"] else "speaking rapidly"
            for _ in range(retries):
                try:
                    candidate = synth_line(client, faster)
                except Exception:
                    continue
                if duration(candidate) < dur:
                    audio, dur, note = candidate, duration(candidate), "перегенерировано быстрее"
                if dur <= window:
                    break
            if dur > window and ffmpeg:
                factor = min(dur / window, max_speed)
                audio = speed_up(audio, factor, out_dir)
                dur, note = duration(audio), f"ускорено x{factor:.2f}"

        if window is None or dur <= window:
            status = "OK"
        else:
            status = f"НАЛЕЗАЕТ на следующую на {dur - window:.2f} с"
        window_txt = f"{window:.2f}" if window else "  —  "
        report.append(f"{label:<28} длина {dur:5.2f} с | окно {window_txt} с | {status} {note}".rstrip())

        name = f"{i + 1:02d}_{fmt_time(start).replace(':', 'm')}s_{block['speaker']}.wav"
        (out_dir / name).write_bytes(audio)
        clips.append((start, audio))

    if clips:
        build_timeline(clips, out_dir / "timeline.wav")
    report_text = "\n".join(report)
    (out_dir / "report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nГотово: timeline.wav (начинается с 00:00) и отдельные реплики в {out_dir}")


# ---------- запуск ----------

def main():
    parser = argparse.ArgumentParser(description="Озвучка сценария через Gemini TTS")
    parser.add_argument("script", help="файл сценария, например dialogue.txt")
    parser.add_argument("--lines", action="store_true", help="каждая реплика отдельным файлом")
    parser.add_argument("--gap", type=float, default=0.3, help="пауза между репликами в full.wav, сек")
    parser.add_argument("--fit", action="store_true", help="подгонять реплики, которые не влезают в окно")
    parser.add_argument("--max-speed", type=float, default=1.2, help="максимальное ускорение ffmpeg (по умолчанию 1.2)")
    parser.add_argument("--retries", type=int, default=1, help="сколько раз перегенерировать быстрее (по умолчанию 1)")
    parser.add_argument("--fps", type=float, default=25, help="кадров в секунду для таймкодов с кадрами (по умолчанию 25)")
    parser.add_argument("--start", default=None, help="таймкод начала монтажки, например 01:00:00:00 в DaVinci")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit('Не найден GEMINI_API_KEY. Задай ключ: $env:GEMINI_API_KEY="твой_ключ"')

    blocks = parse_script(args.script, args.fps)
    if not blocks:
        sys.exit("В сценарии нет реплик.")
    unknown = sorted({b["speaker"] for b in blocks} - VOICES.keys())
    if unknown:
        sys.exit(f"Нет голоса для спикеров {unknown}. Добавь их в VOICES в начале файла.")

    timed = [b["start"] is not None for b in blocks]
    if any(timed) and not all(timed):
        missing = [b["speaker"] + ": " + b["text"][:30] for b in blocks if b["start"] is None]
        sys.exit(f"Таймкоды должны быть у всех реплик. Нет у: {missing}")
    if all(timed):
        offset = parse_timecode(args.start, args.fps) if args.start else 0
        for b in blocks:
            b["start"] -= offset
            if b["start"] < 0:
                sys.exit(f"Таймкод {b['tc']} раньше начала монтажки {args.start}.")
        blocks.sort(key=lambda b: b["start"])

    out_dir = Path(OUTPUT_DIR) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "script.txt").write_text(Path(args.script).read_text(encoding="utf-8"), encoding="utf-8")

    from google import genai
    client = genai.Client(api_key=api_key)
    if all(timed):
        run_timeline(client, blocks, out_dir, args.fit, args.max_speed, args.retries)
    elif args.lines:
        run_lines(client, blocks, out_dir, args.gap)
    else:
        run_dialogue(client, blocks, out_dir)


if __name__ == "__main__":
    main()
