# M10: метрики поиска и отказа

Версия 1.1.0. Реализуй retrieval_scores(ranking,relevant,k,abstained) в копии starter.py. ranking — уникальные ID документов по убыванию score. relevant — ID правильных документов. k — положительный int, bool не допускается. Повтор ID в ranking является ошибкой входного контракта: схлопни chunks до вызова.

Для answerable верни recall по первым k, MRR по первому правильному документу во всём ранге и correct_abstention=None. Для relevant=[] верни recall=None,mrr=None и фактическую правильность отказа. Не включай None в средний recall.

```powershell
.\.venv\Scripts\python.exe exercises\M10\check.py submissions\M10\solution.py
```

B: ranking=[C,A,B], relevant=[A,B],k=1 — recall0, MRR1/2. C через неделю: два answerable recall1 и0, один правильный unanswerable — средний recall0.5 по двум, refusals1/1 отдельно. Полный ключ — solutions/exercises/M10/solution.py. После проверки повтори на своём русском корпусе и покажи различие document recall и достаточности конкретного chunk.
