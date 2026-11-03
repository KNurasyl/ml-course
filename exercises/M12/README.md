# M12: собрать assistant-only labels

Версия 1.1.0. Реализуй training_example(prefix,answer,max_length,pad_id=0). prefix уже включает system/user и маркер начала assistant; answer включает целевой EOS. Верни input_ids,attention_mask,labels одинаковой длины max_length. Padding справа; labels prefix/padding равны−100; assistant labels равны его IDs без ручного сдвига.

Пустой prefix, пустой answer и переполнение длины отклоняются ValueError. Контракт допускает answer из одного EOS как обучение немедленной остановке; подходящий смысл такого примера нужно решить отдельно при подготовке данных.

```powershell
.\.venv\Scripts\python.exe exercises\M12\check.py submissions\M12\solution.py
```

B: prefix=[9,8,7],answer=[2],max_length=4. C через неделю: вручную сдвинутые labels переданы модели, которая тоже сдвигает targets; покажи неправильные пары на четырёх ID. Эталон — solutions/exercises/M12/solution.py, самостоятельный SFT-вариант с тремя сравниваемыми состояниями — solutions/exercises/M12/sft_variant.py. Открывай его после своей попытки по M12-L07.
