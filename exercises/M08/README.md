# M08: causal attention с padding

Версия 1.1.0. Реализуй causal_attention(q,k,v,valid_keys) в копии starter.py внутри submissions/M08. Входы NumPy: Q/K [B,T,D], V [B,T,Dv], valid_keys [B,T] bool. Выход — (O,A), формы [B,T,Dv] и [B,T,T]. Запрещённые keys получают нулевой вес. Каждая строка доступных весов суммируется в1. Query без доступного key отклоняется ValueError.

```powershell
.\.venv\Scripts\python.exe exercises\M08\check.py submissions\M08\solution.py
```

Checker проверяет численный эффект future/padding, а не только shapes. Он не требует считать padding-query полезным выходом: контракт задаёт key-mask, downstream loss отдельно игнорирует padding-target.

B: переставь key/value вместе только среди доступных прошлых позиций одной query — взвешенная сумма сохранится при той же доступности. C через неделю: убери mask и покажи, какой тест обнаружил утечку будущего. Эталон и расчёты — solutions/exercises/M08/solution.py и ключ M08. Критерий завершения: код проходит, а ты объясняешь обе оси матрицы и причину каждого нулевого веса.
