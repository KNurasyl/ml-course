# Архив подготовки курса 0.1.0

`authoring-0.1.0.zip` — побайтовая копия бывшей рабочей папки `.authoring` на 04.11.2026: 254 файла, 22,0 МБ до сжатия. Внутри генераторы версии 0.1.0, черновые данные страниц, браузерные проверки `browser-qa` вместе с установленным Playwright (`node_modules`), листы контроля рисунков и превью. Перечень всех файлов с размерами и SHA-256, а также SHA-256 самого архива — в [authoring-0.1.0.files.json](authoring-0.1.0.files.json).

**Генераторы архива не запускать.** Они перезаписывают текущие самостоятельные материалы устаревшими версиями 0.1.0. Актуальные уроки, ключи и учебник собираются инструментами из `tools/` по [README курса](../README.md).

Пути вида `.authoring/...` в записях происхождения (например, `metadata/production-foundations.json`) указывают на файлы внутри архива с тем же относительным путём.

Посмотреть список файлов без распаковки:

```powershell
.\.venv\Scripts\python.exe -m zipfile -l archive\authoring-0.1.0.zip
```

Распаковать для чтения — только за пределами папки курса:

```powershell
.\.venv\Scripts\python.exe -m zipfile -e archive\authoring-0.1.0.zip $env:TEMP\authoring-0.1.0
```

Проверить целостность: результат должен совпасть с `archive_sha256` из перечня.

```powershell
(Get-FileHash archive\authoring-0.1.0.zip -Algorithm SHA256).Hash
```
