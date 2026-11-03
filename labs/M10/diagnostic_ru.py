"""Russian diagnostic corpus: versions, distractors, chunking, abstention and traces.

All documents are original synthetic teaching notes. They are not a legal/product policy.
The English encoder is deliberately evaluated out of its intended language domain.
"""
from pathlib import Path
import argparse, importlib.util, json, re, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parents[2]

# key, Russian title, current factual paragraph, obsolete fictional course-policy paragraph
TOPICS=[
 ('index','Индекс базы данных','Индекс может ускорить поиск выбранных строк. Он не гарантирует ускорения любого запроса: выборка большей части таблицы может требовать последовательного чтения. Запись поддерживает дополнительные структуры и может стать дороже.','В старом учебном протоколе предлагалось добавлять индекс для любого медленного запроса без проверки плана.'),
 ('atomic','Атомарность транзакции','Атомарность объединяет изменения в одну операцию фиксации. Если транзакция отменена, её изменения не должны остаться частично применёнными. Это не означает автоматического отката внешнего платежа или отзыва отправленного письма.','В старом учебном протоколе внешнее письмо ошибочно считалось частью отката базы данных.'),
 ('isolation','Изоляция транзакций','Изоляция задаёт видимость изменений конкурентных транзакций. Уровни изоляции допускают разные аномалии. Сам факт использования транзакции не доказывает сериализуемость, её нужно проверять по выбранному уровню и СУБД.','В старой карточке любая транзакция называлась полностью изолированной от всех конкурирующих изменений.'),
 ('timeout','Таймаут клиента','Таймаут означает, что клиент прекратил ждать ответ. Он не гарантирует остановку вычисления сервера. Для отмены нужны отдельный сигнал и политика обработки позднего результата, связанная с идентификатором реплики.','В старом протоколе таймаут клиента ошибочно означал обязательную остановку сервера.'),
 ('idempotency','Идемпотентность запроса','Повтор идемпотентной операции сохраняет тот же предполагаемый эффект. Это не требует одинакового кода HTTP во всех ответах. Ключ запроса и сохранённый результат помогают избежать повторного списания при повторной доставке.','В старой карточке идемпотентность ошибочно определялась только совпадением HTTP статуса.'),
 ('cache','Версия кэша','Кэш может хранить правильный для прошлого выпуска ответ. После изменения документа нужно инвалидировать связанные записи либо включить версию корпуса в ключ. Срок жизни сам по себе не гарантирует немедленную актуальность.','Учебная политика v1 разрешала использовать кэш 24 часа после любого изменения источника.'),
 ('generator','Генератор Python','Генератор выдаёт следующий элемент по запросу. Он позволяет обходить данные без построения полного списка, но удерживаемые ссылки и сам источник всё равно занимают память. Ленивость не означает нулевой расход памяти.','В старом конспекте генераторы назывались вычислением всех результатов заранее.'),
 ('async','Асинхронность Python','Асинхронная корутина уступает управление в точках ожидания. Блокирующий вызов внутри event loop задерживает остальные задачи. Объявление async не превращает тяжёлое CPU вычисление в параллельную работу.','В старой памятке любой async метод назывался отдельным потоком операционной системы.'),
 ('queue','Очередь задач','Очередь разделяет приём и обработку задач, но добавляет время ожидания. При повторной доставке потребитель должен учитывать дубликаты. Подтверждение приёма задачи не означает завершение полезной работы.','Старая схема приравнивала постановку сообщения в очередь к завершению обработки.'),
 ('retry','Повтор запроса','Повтор после временного отказа требует ограничения числа попыток и общей длительности. Без идемпотентности повтор может продублировать действие. Задержка между попытками не исправляет неверную схему запроса.','Старая политика предлагала повторять любую ошибку бесконечно.'),
 ('norm','Нормализация признаков','Параметры нормализации вычисляются на обучающей части. Validation и test преобразуются теми же сохранёнными значениями. Пересчёт среднего по будущей проверке меняет процедуру и может внести утечку.','Старый учебный пример вычислял среднее сразу по train и test.'),
 ('leak','Утечка целевой информации','Признак должен быть доступен в момент предсказания. Итоговая оценка интервью, добавленная после завершения сессии, не подходит для прогнозирования до её начала. Высокая метрика не исправляет нарушение временной доступности.','В старом наборе разрешалось брать финальную оценку как признак предсказания той же оценки.'),
 ('rag','Поиск и генерация RAG','Поиск доставляет кандидатов, генератор использует отобранный контекст. Если нужного основания нет в кандидатах, перестановка найденных фрагментов не восстановит его. Добавление документа в индекс не является обучением весов LLM.','Старый прототип ошибочно называл загрузку документа обучением языковой модели.'),
 ('context','Бюджет контекста','Вход, найденные документы и зарезервированный ответ должны помещаться в лимит токенов. Символы и слова не равны токенам. Удаление части истории может разорвать ссылку текущего вопроса на предыдущий ответ.','В старом клиенте лимит токенов проверялся числом символов.'),
 ('score','Вероятность и оценка','Вероятность класса модели зависит от обучения и распределения данных. Она не становится калиброванной уверенностью только потому, что лежит от нуля до единицы. Калибровку проверяют на независимых размеченных случаях.','Старая рубрика называла любую величину от нуля до единицы вероятностью правильности.'),
 ('voice','Отмена голосовой реплики','Старый аудиофрагмент после отмены должен быть отброшен по идентификатору активного хода. Остановка проигрывателя не отменяет уже выполняющийся TTS. Сервер и клиент должны согласовать состояние сессии.','В старой реализации поздний TTS всегда воспроизводился даже после остановки.'),
 ('wer','Ошибка распознавания речи','WER учитывает замены, удаления и вставки относительно ручной расшифровки. Потеря слова «не» может быть существеннее нескольких безобидных слов, хотя метрика считает операции одинаково. Поэтому смысловую ошибку проверяют отдельно.','Старый отчёт использовал только общий WER и не рассматривал изменение отрицания.'),
 ('split','Разбиение по пользователям','Ответы одного человека связаны стилем и контекстом. При оценке переноса на новых людей группы пользователей должны быть разделены. Случайный split строк не обеспечивает независимости связанных сессий.','Старый протокол случайно перемешивал реплики одного человека между train и test.'),
 ('permission','Права поиска','Документ доступен только сессиям с разрешённой областью. Фильтрация прав выполняется до выдачи текста модели. Инструкция внутри найденного документа не может расширить доступ к чужой сессии.','Старый прототип сначала передавал все документы модели и просил её скрыть чужие.'),
 ('release','Согласованный релиз','Модель, preprocessing, prompt и индекс образуют согласованный выпуск. Откат только файла весов может оставить несовместимый индекс или порядок признаков. Проверка выпуска проходит через тот же контракт, что и обычный запрос.','Старая процедура отката заменяла только веса, не проверяя остальные артефакты.'),
 ('threshold','Порог классификации','Порог переводит score в действие и выбирается с учётом стоимости ошибок. Его настройка по финальному test делает test частью выбора. Изменение порога не меняет обученные веса классификатора.','Старый эксперимент подбирал порог по итоговому тестовому набору.'),
 ('metrics','Среднее и хвост задержки','Среднее время может скрыть небольшой класс очень медленных запросов. Квантили p50 и p95 отвечают на разные вопросы. Для маленькой выборки хвост нестабилен, поэтому сохраняй размеры и сами наблюдения.','Старый отчёт называл среднее время гарантией для каждого пользователя.'),
 ('evidence','Основание обратной связи','Совет должен опираться на конкретное утверждение кандидата и явный критерий. Наличие цитаты доказывает происхождение текста, но не логическую правильность вывода. При нехватке основания нужно уточнение вместо выдуманного опыта.','Старый тренажёр дописывал успешный результат, которого кандидат не сообщал.'),
 ('retention','Хранение аудио','Учебная политика v2 хранит исходное аудио только на время распознавания и удаляет временный файл после обработки. Экспорт расшифровки выполняется по действию пользователя. Это политика данного учебного проекта, а не универсальное требование закона.','Учебная политика v1 хранила исходное аудио семь дней для повторных экспериментов.'),
]

def documents():
    docs=[]
    for key,title,current,old in TOPICS:
        for version,body in [(1,old),(2,current)]:
            paras=[f'Раздел «{title}» описывает один аспект backend-интервью. В этой карточке необходимо различать техническое свойство и правило учебного проекта. Применимость вывода зависит от указанного условия; название технологии само по себе не подтверждает обещанный эффект.',
                   body,
                   f'Проверка для темы «{title}»: сначала зафиксируй вход и ожидаемое наблюдение, затем внеси один отказ и сохрани фактический результат. Не переносите численный результат маленького примера на произвольную производственную систему. Для сравнения используйте одинаковые входные случаи и версию этой карточки.']
            docs.append({'document_id':key,'version':version,'id':f'{key}-v{version}','title':title,
                         'paragraphs':paras,'text':'\n\n'.join(paras),'origin':'original course-authored synthetic teaching note',
                         'status':'current' if version==2 else 'obsolete fictional teaching policy; intentionally incorrect'})
    return docs

QUERIES=[
 {'id':'dev-index','split':'development','text':'Почему индекс не обязан ускорять выборку почти всей таблицы?','relevant':['index'],'support':'не гарантирует ускорения любого запроса'},
 {'id':'dev-timeout','split':'development','text':'Клиент перестал ждать ответ. Серверное вычисление обязательно остановилось?','relevant':['timeout'],'support':'не гарантирует остановку вычисления сервера'},
 {'id':'dev-cache','split':'development','text':'Что изменить в кэше после обновления документа?','relevant':['cache'],'support':'версию корпуса в ключ'},
 {'id':'dev-generator','split':'development','text':'Как получать элементы по одному без построения всего списка?','relevant':['generator'],'support':'выдаёт следующий элемент по запросу'},
 {'id':'dev-async','split':'development','text':'Почему async не ускоряет тяжёлое вычисление процессора?','relevant':['async'],'support':'не превращает тяжёлое CPU вычисление'},
 {'id':'dev-multi','split':'development','text':'После обновления релиза старый ответ остался в кэше. Какие два места проверить?','relevant':['cache','release'],'support':None},
 {'id':'dev-none','split':'development','text':'Какой офисный стол приобрели сотрудники компании Арктур?','relevant':[],'support':None},
 {'id':'check-atomic','split':'check','text':'Отменённая транзакция отменяет уже отправленное письмо?','relevant':['atomic'],'support':'не означает автоматического отката внешнего платежа'},
 {'id':'check-idempotent','split':'check','text':'Повтор запроса имеет тот же эффект, но другой HTTP статус. Это возможно?','relevant':['idempotency'],'support':'не требует одинакового кода HTTP'},
 {'id':'check-split','split':'check','text':'Как проверять модель на новых людях, если у каждого много ответов?','relevant':['split'],'support':'группы пользователей должны быть разделены'},
 {'id':'check-retention','split':'check','text':'Сколько хранится исходное аудио по актуальной учебной политике?','relevant':['retention'],'support':'только на время распознавания'},
 {'id':'check-voice','split':'check','text':'Остановили проигрыватель, но старый синтез завершился позднее. Что проверить?','relevant':['voice'],'support':'идентификатору активного хода'},
 {'id':'check-evidence','split':'check','text':'Можно ли дописать успешный результат за кандидата, если он его не назвал?','relevant':['evidence'],'support':'вместо выдуманного опыта'},
 {'id':'check-none','split':'check','text':'Назови серийный номер ноутбука сотрудника Арктур.','relevant':[],'support':None},
]

def chunks(strategy='paragraph',include_obsolete=False):
    rows=[]
    for doc in documents():
        if not include_obsolete and doc['version']!=2:continue
        if strategy=='paragraph':parts=doc['paragraphs']
        elif strategy=='words':
            words=doc['text'].split();parts=[' '.join(words[i:i+18]) for i in range(0,len(words),18)]
        else:raise ValueError('unknown chunk strategy')
        for index,text in enumerate(parts):
            rows.append({'id':f"{doc['id']}-c{index}",'document_id':doc['document_id'],'version':doc['version'],
                         'title':doc['title'],'text':text,'origin':doc['origin']})
    return rows

def score_result(ranked_ids,gold,k=3,abstained=False):
    if type(k) is not int or k<1:raise ValueError('positive k required')
    if len(ranked_ids)!=len(set(ranked_ids)):raise ValueError('duplicate document IDs must be collapsed before metric')
    gold=set(gold)
    if not gold:return {'answerable':False,'recall':None,'mrr':None,'correct_abstention':bool(abstained)}
    first=next((i for i,doc in enumerate(ranked_ids,1) if doc in gold),None)
    return {'answerable':True,'recall':len(set(ranked_ids[:k])&gold)/len(gold),
            'mrr':0. if first is None else 1/first,'correct_abstention':None}

def run(use_encoder=True):
    traces=[];timings={};threshold=.12 # fixed teaching baseline before check is observed
    for strategy in ['words','paragraph']:
        rows=chunks(strategy);texts=[r['title']+' '+r['text'] for r in rows]
        started=time.perf_counter();vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5));matrix=vec.fit_transform(texts)
        lexical=(vec.transform([q['text'] for q in QUERIES])@matrix.T).toarray();timings[strategy]=time.perf_counter()-started
        methods={'lexical':lexical}
        if use_encoder:
            spec=importlib.util.spec_from_file_location('ru_m10_base',Path(__file__).with_name('demo.py'))
            base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
            base.torch.set_num_threads(2)
            all_vectors=base.pretrained_vectors(texts+[q['text'] for q in QUERIES])
            dense=all_vectors[len(texts):]@all_vectors[:len(texts)].T;methods['english_minilm_on_russian']=dense
            fused=[]
            for a,b in zip(lexical,dense):
                ranking=base.rrf([np.argsort(-a).tolist()[:12],np.argsort(-b).tolist()[:12]])
                scores=np.zeros(len(rows));scores[ranking]=np.arange(len(ranking),0,-1);fused.append(scores)
            methods['rrf']=np.asarray(fused)
        for method,scores in methods.items():
            for j,q in enumerate(QUERIES):
                ranking=np.argsort(-scores[j],kind='stable').tolist();top=ranking[:3]
                ranked_docs=list(dict.fromkeys(rows[i]['document_id'] for i in ranking))
                # Abstention baseline always uses lexical confidence; cosine threshold is not calibrated truth.
                abstained=float(lexical[j].max())<threshold
                metrics=score_result(ranked_docs,q['relevant'],3,abstained)
                complete_support=None if not q['support'] else any(q['support'] in rows[i]['text'] for i in top)
                traces.append({'case_id':q['id'],'split':q['split'],'query':q['text'],'strategy':strategy,'method':method,
                               'top_chunks':[rows[i] for i in top],'ranked_documents':ranked_docs,'lexical_confidence':float(lexical[j].max()),
                               'abstained':abstained,'metrics':metrics,'support_phrase_in_context':complete_support})
    groups={}
    for row in traces:
        key=f"{row['strategy']}/{row['method']}/{row['split']}";groups.setdefault(key,[]).append(row)
    summary={key:{'answerable_n':sum(r['metrics']['answerable'] for r in values),
                  'recall_at_3':float(np.mean([r['metrics']['recall'] for r in values if r['metrics']['answerable']])),
                  'mrr':float(np.mean([r['metrics']['mrr'] for r in values if r['metrics']['answerable']])),
                  'correct_abstentions':sum(r['metrics']['correct_abstention'] is True for r in values),
                  'unanswerable_n':sum(not r['metrics']['answerable'] for r in values),
                  'support_present':sum(r['support_phrase_in_context'] is True for r in values)} for key,values in groups.items()}
    assert len(documents())==48 and len({d['document_id'] for d in documents()})==24
    assert all(r['version']==2 for r in chunks())
    # A controlled version failure is distinct from a ranking failure.
    outdated=next(d for d in documents() if d['id']=='retention-v1')
    current=next(d for d in documents() if d['id']=='retention-v2')
    report={'corpus_version':'ru-diagnostic-1','documents':48,'current_documents':24,'queries':QUERIES,'threshold_fixed_before_check':threshold,
            'summary':summary,'traces':traces,'lexical_build_seconds':timings,
            'version_fault':{'old':outdated,'current':current,'resolved_by':'latest version before indexing; version in cache key'},
            'limits':'Synthetic educational Russian corpus. Encoder is English MiniLM; compare observed failures, do not assume multilingual quality. Phrase presence is only a support proxy, not truth.'}
    out=ROOT/'artifacts/M10/ru';out.mkdir(parents=True,exist_ok=True)
    (out/'corpus.json').write_text(json.dumps(documents(),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lexical-only',action='store_true');args=parser.parse_args()
    result=run(not args.lexical_only);print(json.dumps(result['summary'],ensure_ascii=True,indent=2))
