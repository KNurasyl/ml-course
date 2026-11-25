### Фаза 6. Оформление 3.0: тёмная тема и шрифты (после выпуска 1.2.0, ~8 дней)

Решения автора 09.11.2026: тема A «Графит» по умолчанию, светлая остаётся переключателем и для печати; шрифты Golos Text + JetBrains Mono (OFL), встроенные в HTML; размер и ритм текста без изменений (17px / 1,7, колонка ~68 знаков). Макеты: `quality/design-v3/` (копии артефактов «Тёмная тема учебника», «Выбор темы», «Выбор шрифтов»). Markdown уроков не меняется: вся фаза живёт в `tools/`, `course/visuals/` и генераторах рисунков. `check_no_loss.py` обязан проходить без изменений счётчиков.

#### 6.0. Токены темы (вставить в начало `tools/reader.css`; литералы цветов вне `:root` запрещены)

```css
:root{color-scheme:dark;
  --bg:#15171c;--surface:#1b1e25;--paper:#20242c;--line:#2e333d;--line-strong:#3b4150;
  --ink:#e7e5de;--muted:#a3a9b4;
  --accent:#86aaf0;--accent-soft:#1f2a44;      /* ссылки, активный пункт, ID урока, формулы */
  --act:#58c4c6;--act-soft:#12302f;            /* опыты, ответы, ползунки, ключевые слова кода */
  --hint:#e6b566;--hint-soft:#3a2d12;          /* подсказки, фокус, выделение поиска, строки кода */
  --ok:#86d19c;--ok-soft:#173021;--warn:#f0a071;--warn-soft:#3a2416;--error:#f08a8a;  /* только статусы */
  --mark-bg:#4a3c14;--mark-fg:#ffe7a6;--code-bg:#111318;--focus:#e6b566;
  --sans:'Golos Text','Segoe UI',system-ui,sans-serif;--mono:'JetBrains Mono','Cascadia Code',Consolas,ui-monospace,monospace;
  --sidebar:272px}
:root[data-theme="light"],@media print{:root{color-scheme:light;
  --bg:#f6f5f1;--surface:#eeede8;--paper:#ffffff;--line:#d9d6cd;--line-strong:#c2beb3;
  --ink:#1d2430;--muted:#5a6270;--accent:#1f55b0;--accent-soft:#e4ecfb;--act:#0b7478;--act-soft:#dff3f2;
  --hint:#9a6410;--hint-soft:#fbefd6;--ok:#2a7a4b;--ok-soft:#e1f2e6;--warn:#a0541c;--warn-soft:#f9e6d8;--error:#b23b3b;
  --mark-bg:#ffe3a1;--mark-fg:#1d2430;--code-bg:#f0f2f6;--focus:#a65c12}}
```

Контрасты тёмной темы к `--bg` (WCAG 2.1): ink 14,2 · muted 7,6 · accent 7,7 · act 8,7 · hint 9,5 · ok 9,9 · warn 8,5 · error 7,4. Все ≥ 7:1 (AAA). Правило семантики: один блок — один акцент; `--ok/--warn/--error` только для статусов журнала и критических ошибок, никогда для декора. Тени не используются, глубина — яркостью поверхностей (bg → surface → paper; код темнее bg).

| ID | Задача | Файлы | Критерий приёмки | Дни |
| --- | --- | --- | --- | --- |
| D1 | Токены темы: заменить все 51 литерал в `reader.css` на переменные из 6.0; `data-theme` на `<html>`, переключатель «Тема» в шапке рядом с «Печать», выбор в `localStorage` (try/catch); мета `color-scheme` → `dark light`; при первом открытии — тёмная | `tools/reader.css`, `tools/reader.js`, `tools/build_reader.py` | `grep -E '#[0-9a-f]{3,6}' reader.css` даёт совпадения только внутри `:root`-блоков; `check_reader_design.cjs` расширен этой проверкой; скриншоты `quality/design-v3/screenshots/` 1440 и 390 в обеих темах | 1 |
| F1 | Шрифты: скачать woff2 Golos Text 400/500/700 и JetBrains Mono 400/600, подмножество latin + cyrillic, положить в `tools/fonts/` вместе с файлами лицензий OFL; записать точные размеры в `tools/fonts/README.md` | `tools/fonts/` | Файлы в git, суммарно ≤ 350 КБ, лицензии рядом | 0,5 |
| F2 | Встроить `@font-face` с `src:url(data:font/woff2;base64,…)` и `font-display:swap` на этапе сборки; запасной стек Segoe UI / Consolas сохраняется | `tools/build_reader.py` | `START_HERE.html` без внешних запросов; `document.fonts.check('17px "Golos Text"')` истинно в браузерном тесте; рост HTML ≤ 350 КБ | 0,5 |
| D2 | Опыты на токены: `visual-style.css` (21 литерал) и `module-experiments.js` (5) читают переменные; iframe получает тему от родителя (`data-theme` пробрасывается в `srcdoc` при сборке и обновляется `postMessage` при переключении) | `course/visuals/visual-style.css`, `module-experiments.js`, `visual-frame.js`, `reader.js` | 26 опытов открываются в обеих темах без белых рамок; тест в `tests/test_reader_v2.py` на отсутствие литералов | 1 |
| D5 | Mermaid: `theme:'base'` + `themeVariables` из токенов (primaryColor = paper, lineColor = line-strong, textColor = ink, primaryBorderColor = accent); перерисовка при переключении темы | `tools/reader.js` | 33 схемы читаемы в обеих темах, скриншоты трёх любых | 0,5 |
| D3 | Генератор адаптивных SVG: константы `INK/MUTED/BLUE/TEAL/AMBER/LINE/SOFT/OP/WARN` → `var(--ink)` и т. д. (имена по 6.0), белый `<rect>` убрать; SVG встраиваются inline (иначе переменные не работают); перегенерировать 208 файлов | `tools/build_responsive_figures.py`, `course/assets/responsive/`, `build_reader.py` | `check_no_loss`: рисунков не меньше; `metadata/responsive-figures.json` обновлён; визуальная выборка 10 рисунков в обеих темах | 1,5 |
| D4 | Matplotlib-рисунки (34 PNG, 84 SVG): генераторы получают параметр темы и пишут `*-dark.*` рядом с существующими (старые файлы = светлый вариант, не удаляются); `build_reader` вставляет `<picture>` с `<source media="(prefers-color-scheme: dark)">` и подменой при ручном переключении | `tools/build_foundation_figures.py`, `build_quantitative_figures.py`, `build_visuals.py`, `build_module_visuals.py`, `build_product_visuals.py`, `course/assets/` | Каждому рисунку соответствует dark-вариант (тест по списку); палитра matplotlib совпадает с 6.0; старые файлы не изменены (хеши) | 2 |
| D6 | Подсветка Python в `<pre>` на этапе сборки: 4 класса (`k` ключевые слова → `--act`, `s` строки → `--hint`, `n` числа → `--accent`, `c` комментарии → `--muted`), без библиотек, только для блоков с языком python | `tools/build_reader.py` | 142 примера из уроков отрисованы; `assert` в примерах не изменён; HTML валиден | 1 |
| D7 | Печать: `@media print` всегда светлая, без боковой панели и переключателя, рисунки светлого варианта, ссылки с URL в скобках | `tools/reader.css` | PDF одного урока из браузера читаем | 0,5 |
| F4 | Приёмка оформления: `check_reader_design.cjs` проверяет отсутствие внешних запросов, загрузку шрифтов, 0 литералов вне `:root`, контраст пар текст/фон ≥ 4,5 по токенам; скриншоты обеих тем в `quality/design-v3/screenshots/`; отчёт `quality/design-v3/report.json` | `tools/check_reader_design.cjs`, `quality/design-v3/` | Отчёт зелёный, ссылка из `RELEASE_STATUS.md` | 0,5 |

Порядок: D1 → F1 → F2 (учебник уже тёмный, день 2) → D2 → D5 (день 3) → D3 (дни 4–5) → D4 (дни 6–7) → D6 → D7 → F4 (день 8). Учёба начинается после дня 3, не дожидаясь D3–D7.

Чего не делать в этой фазе: инвертировать рисунки `filter:invert()`; загружать шрифты из сети; переносить старые акценты `#245bba` и `#087e82` на тёмный фон (2,8:1 и 3,6:1); чистый `#000` фон и `#fff` текст; тени и градиенты; менять размер шрифта или ширину колонки; трогать Markdown уроков.

После фазы 6 — фаза 7 «Квест-режим» (макет `quality/design-v3/quest-mode.html`), задачи будут оформлены отдельно.
