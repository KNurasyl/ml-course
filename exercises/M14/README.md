# M14: самостоятельный компонент

Версия 1.1.0. Разбери raw JSON. Разрешён только read_document и ровно один строковый document_id из server allowed_ids. Любой лишний ключ, неверный тип, чужой ID, массив или неверный JSON должен вызвать ValueError до чтения. allowed_ids не берётся из raw.

Скопируй starter.py в submissions/M14/my_solution.py, реализуй функцию и сохрани объяснение решения. Не открывай готовый solution до первой попытки.

```powershell
.\.venv\Scripts\python.exe exercises\M14\check.py submissions\M14\my_solution.py
```

Проверка возвращает behavior_passed; самостоятельность и понимание она не определяет. При ошибке сократи свой пример и объясни нарушенный контракт. Стартовый файл намеренно не проходит проверку.

После попытки: [полное решение](../../solutions/exercises/M14/solution.py). Проверка ключа и чувствительности к намеренному дефекту:

```powershell
.\.venv\Scripts\python.exe exercises\M14\check.py solutions\exercises\M14\solution.py --self-test
```

Новый пример после просмотра решения приведён в соответствующем новом уроке и ключе модуля. Нельзя считать повторный запуск этого же ключа самостоятельным результатом.
