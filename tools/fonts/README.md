# Шрифты учебника: Golos Text и JetBrains Mono

Файлы для оформления 3.0 (фаза 6, задача F1): Golos Text — основной текст и заголовки, JetBrains Mono — код и формулы. `tools/build_reader.py` (F2) встраивает их в `START_HERE.html` как `data:`-URL с `font-display:swap`, по одному `@font-face` на файл с его `unicode-range` из [`fonts.json`](fonts.json), и перед этим сверяет размер и SHA-256 каждого файла с `fonts.json` — несовпадение останавливает сборку. Учебник шрифты из сети не загружает; символы вне подмножеств и случай, когда шрифт не загрузился, закрывает запасной стек `Segoe UI` / `Consolas` из токенов `--sans` и `--mono` в `tools/reader.css`.

| Поле | Значение |
| --- | --- |
| Пакеты | `@fontsource/golos-text` 5.3.0 и `@fontsource/jetbrains-mono` 5.3.0 (npm, выпуск 19.07.2026); внутри — Google Fonts: Golos Text v7, JetBrains Mono v24 (обновлены 11.09.2025) |
| Начертания | Golos Text 400, 500, 700; JetBrains Mono 400, 600; только прямое (`normal`) |
| Подмножества | `latin` и `cyrillic` — готовые файлы пакета, своё подмножество не собиралось (fonttools не нужен) |
| Источник | `https://cdn.jsdelivr.net/npm/@fontsource/<пакет>@5.3.0/files/<файл>`, точные адреса — в `fonts.json` |
| Проверка | 09.11.2026: размер и SHA-256 каждого файла совпали со списком файлов npm-пакета на jsDelivr (`data.jsdelivr.com`), у всех woff2 сигнатура `wOF2`. В Microsoft Edge каждый файл загружен через `FontFace`: 67 символов кириллицы (с «Ё», «ё», «№») и 106 символов латиницы, цифр и знаков (с «», —, –, …, ·, ×) нарисованы самим шрифтом, пропусков нет; контрольные `∂Σ√→Әқ` вне подмножеств определились как отсутствующие |
| Размер | 10 файлов woff2, 111 256 байт (108,6 КБ) при лимите 350 КБ; в `data:`-URL (base64) это 148 341 байт (≈ 145 КБ) |
| Лицензия | SIL Open Font License 1.1: [OFL-golos-text.txt](OFL-golos-text.txt), [OFL-jetbrains-mono.txt](OFL-jetbrains-mono.txt) — файлы `LICENSE` из пакетов байт в байт (с CRLF) |

| Семейство | Начертание | Подмножество | Файл | Байт | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| Golos Text | 400 | latin | `golos-text-latin-400-normal.woff2` | 11 992 | `c8246f55dbddb17ceab94b8de2dbcf326dc189f3bc5a8c13d7b14ad05d7d1d81` |
| Golos Text | 400 | cyrillic | `golos-text-cyrillic-400-normal.woff2` | 6 960 | `15a5cbadfc1aa7a541651600b757bf7b67bce4e145b3239de072db5427889605` |
| Golos Text | 500 | latin | `golos-text-latin-500-normal.woff2` | 12 032 | `779e46588bc6f5f18918dcdf87d609a8053e1c925abd9e94d2b46b0254f36b24` |
| Golos Text | 500 | cyrillic | `golos-text-cyrillic-500-normal.woff2` | 7 044 | `22337b726df83521c7f6b2709e4717116eb53106c6140b45d751f36910191ecc` |
| Golos Text | 700 | latin | `golos-text-latin-700-normal.woff2` | 12 380 | `1ac7742b82900b38b483fb65691e8fb786ba16f6c5cae677aa63a128286ce853` |
| Golos Text | 700 | cyrillic | `golos-text-cyrillic-700-normal.woff2` | 7 108 | `42db867ddf4b8b025cb2681a796c8c34633e14f77077c5c5ad950383e357a6f5` |
| JetBrains Mono | 400 | latin | `jetbrains-mono-latin-400-normal.woff2` | 21 168 | `14425ba9c695763c1547f48a206b7aa60350a33ae23de09f0407877f3fcd89eb` |
| JetBrains Mono | 400 | cyrillic | `jetbrains-mono-cyrillic-400-normal.woff2` | 5 328 | `522f5827308102a9c0ea7860b5a11306043610ecf03d150ba32056a06e8c56e2` |
| JetBrains Mono | 600 | latin | `jetbrains-mono-latin-600-normal.woff2` | 21 860 | `400c6bfda18d5d14acad1c15d6dcb9f8e13c015e7286317e0b9a482539bef147` |
| JetBrains Mono | 600 | cyrillic | `jetbrains-mono-cyrillic-600-normal.woff2` | 5 384 | `b450f960a0d29846370fc050b6183727d2075a2bcb1c5595c99374bf083b5a83` |

Лицензии: `OFL-golos-text.txt` — 4 392 байт, SHA-256 `54dcae300688bccf111e43c818a6ee2261d10fc8d7eb5b9c9aa2a1e9b03189c9`; `OFL-jetbrains-mono.txt` — 4 524 байт, SHA-256 `403581b69dac5cff4079205e01c6b467e56af449ecbd7247693ddb1baafa005b`.

## Что шрифты не покрывают

В Markdown курса и решений (177 файлов) 1 821 вхождение 52 символов лежат вне `latin` и `cyrillic`. Чаще всего: `→` (409), `≈` (270), греческие буквы (`η`, `λ`, `μ`, `β`, `α`, `Σ`, `ε`, `π`, `σ`…), индексы и степени (`ᵢ`, `₁`, `₂`, `ᵀ`, `⁻`), `ŷ`, `√`, `∂`, `≤`, `≥`, `≠`, `∞`. Браузер рисует их запасным шрифтом по одному символу (Segoe UI в тексте, Consolas в коде), так что текст остаётся читаемым. В Golos Text греческого нет вообще. У JetBrains Mono есть подмножество `greek` (около 4 КБ на начертание), но в F2 оно не добавлено: в коде и формулах курса всего 3 греческие буквы (`Σ` дважды, `α`), ещё 396 — в обычном тексте, где их рисует Segoe UI. Казахские буквы (`cyrillic-ext`) не включены: курс на русском.

Обновлять файлы — только вместе с этой таблицей и `fonts.json`; тест `tests/test_fonts.py` сверяет размеры и SHA-256. В git файлы хранятся байт в байт (`tools/fonts/** -text` в `.gitattributes`).
