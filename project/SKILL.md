# SKILL.md — Рабочий процесс DEEPUTIN

## Назначение

Этот пакет собирает forensic-пайплайн для фотоархива:

- извлечение фото-артефактов;
- вычисление геометрических и текстурных метрик;
- калибровка на заведомо оригинальном датасете;
- попарное сравнение фото;
- байесовский вердикт;
- текстовый отчёт.

## Последовательность стадий

### 1. Extraction

- определить дату по имени файла;
- оценить позу;
- построить face mask;
- сохранить placeholder reconstruction;
- записать `info.json`.

### 2. Metrics

- вычислить геометрию по реконструкции;
- вычислить текстуру по `face_mask.png`;
- сохранить `metrics.json`.

### 3. Calibration

- собрать статистику по calibration-архиву;
- построить baseline по bucket-ам;
- сохранить `calibration_reference.json`;
- пометить main-фото предварительными гипотезами.

### 4. Pairwise compare

- сравнить соседние фото в пределах bucket-а;
- вычислить расстояния по геометрии и текстуре;
- собрать anomaly flags;
- сохранить `pairs.json` и `pair_index.json`.

### 5. Verdict

- объединить evidence;
- применить priors;
- получить posterior;
- сохранить `verdicts.json` и `timeline.json`.

### 6. Report

- агрегировать verdicts;
- собрать тезисы;
- сохранить `report.json` и `report.md`.

## Запуск

```bash
cd /Users/victorkhudyakov/dutin/newapp/deeputin
python run.py
```

## Дефолтные пути

- `main`: `/Volumes/SDCARD/photo/all`
- `calibration`: `/Volumes/SDCARD/photo/calibration`
- `output/main`: `/Volumes/SDCARD/storage/main`
- `output/calibration`: `/Volumes/SDCARD/storage/calibration`

## Принципы

- Один фото-объект проходит через все стадии без поломки общего пайплайна.
- Если данных мало, стадия должна писать пустой, но валидный результат.
- Все ответы и комментарии в проекте пишутся по-русски.
