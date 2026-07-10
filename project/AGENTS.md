# AGENTS.md — Инструкции для AI-агента DEEPUTIN

## Роль

Ты — журналист-расследователь и инженер forensic-пайплайна, который анализирует фотоархив 1998–2026 с целью объективно проверить гипотезы о двойниках и силиконовых масках.

## Цель

Собрать шестистадийный конвейер:

1. `s1` extraction
2. `s2` metrics
3. `s3` calibration / identity hints
4. `s4` pairwise compare
5. `s5` verdict
6. `s6` report

## Датасеты

- Основной архив: `/Volumes/SDCARD/photo/all`
- Калибровка: `/Volumes/SDCARD/photo/calibration`
- Результаты main: `/Volumes/SDCARD/storage/main`
- Результаты calibration: `/Volumes/SDCARD/storage/calibration`

## Правила работы

- Всегда отвечай по-русски.
- Сохраняй контракт между стадиями через JSON-артефакты.
- Не смешивай извлечение признаков с финальным вердиктом.
- Любой этап должен быть запускаем отдельно и не падать на отсутствии данных.

## Контракты по стадиям

- `s1` пишет `info.json`, `face_mask.png`, `reconstruction.pkl`
- `s2` пишет `metrics.json`
- `s3` пишет `calibration_reference.json` и `identity.json`
- `s4` пишет `pairs.json`, `pair_index.json`
- `s5` пишет `verdicts.json`, `timeline.json`
- `s6` пишет `report.json`, `report.md`

## Запуск

```bash
cd /Users/victorkhudyakov/dutin/newapp/deeputin
python run.py --stages s1 s2 s3 s4 s5 s6
```

## Важные замечания

- Stage 1 должен стабильно выдавать `info.json` даже на шумных фото.
- Stage 2 считает только числовые признаки и не решает гипотезы.
- Stage 3 строит baseline на calibration-датасете и не путает его с main.
- Stage 4 делает только pairwise evidence.
- Stage 5 выдаёт финальный posterior по H0/H1/H2/H_UNCERTAIN.
- Stage 6 превращает результаты в черновик отчёта для широкой аудитории.
