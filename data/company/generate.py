"""Synthetic data of one fictional company for the mini-projects MP1-MP6 (REFACTORING_PLAN_2.0, R07).

«Лампа и Чайник» is an invented online store of home appliances in Kazakhstan. Every name, number and text here is
generated from fixed templates and one seed; no real company, person or customer record is used. The domain
lampa-chainik.test is reserved (RFC 2606) and leads nowhere.

    python data/company/generate.py            write everything into data/company/
    python data/company/generate.py --check    regenerate in memory and compare with the files and manifest.json

Output:
  kb/KB-xxx.md              knowledge base for the support operator: 40 documents with id, version and date
  products.csv              60 products: sku, name, category, price, warranty
  customers.csv             800 customers: id, city, segment, registration date (no names, no contacts)
  orders.csv, order_items.csv   3000 orders in 2026 with statuses, payment and delivery
  tickets.jsonl             600 support tickets with a category and extracted fields (labels for MP5)
  operator_requests.jsonl   20 operator requests with the expected JSON fields (MP1)
  schema.sql                PostgreSQL schema of the tables (MP4)
  manifest.json             sha256 of every file; the generator is deterministic
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261118
COMPANY = "Лампа и Чайник"
DOMAIN = "lampa-chainik.test"
CITIES = ["Алматы", "Астана", "Шымкент", "Караганда", "Актобе", "Павлодар", "Усть-Каменогорск", "Атырау"]
CATEGORIES = {  # category: (title, price range in tenge, warranty months, product names)
    "fridge": ("Холодильники", (180_000, 650_000), 24, ["Холодильник Норд", "Холодильник Айсберг", "Холодильник Тундра"]),
    "washer": ("Стиральные машины", (150_000, 420_000), 24, ["Стиральная машина Волна", "Стиральная машина Бриз"]),
    "vacuum": ("Пылесосы", (35_000, 210_000), 12, ["Пылесос Вихрь", "Робот-пылесос Крот"]),
    "laptop": ("Ноутбуки", (240_000, 900_000), 12, ["Ноутбук Сапсан", "Ноутбук Беркут"]),
    "phone": ("Смартфоны", (90_000, 650_000), 12, ["Смартфон Тулпар", "Смартфон Самрук"]),
    "tv": ("Телевизоры", (120_000, 780_000), 24, ["Телевизор Экран", "Телевизор Панорама"]),
    "kitchen": ("Мелкая кухонная техника", (9_000, 95_000), 12, ["Чайник Пар", "Блендер Смерч", "Мультиварка Казан"]),
    "aircon": ("Кондиционеры", (190_000, 520_000), 36, ["Кондиционер Прохлада", "Кондиционер Сквозняк"]),
}
STATUSES = [("delivered", 0.72), ("shipped", 0.07), ("processing", 0.05), ("cancelled", 0.09), ("returned", 0.07)]
PAYMENTS = [("card", 0.55), ("installment", 0.25), ("cash_on_delivery", 0.15), ("invoice", 0.05)]
TICKET_CATEGORIES = ["delivery_delay", "return_request", "warranty_repair", "payment_issue", "installation",
                     "product_question", "order_change", "complaint_other"]


def pick(rng, weighted):
    return rng.choices([v for v, _ in weighted], weights=[w for _, w in weighted])[0]


# ---------------------------------------------------------------- facts shared by the KB, tickets and checks

def facts(rng):
    delivery = {}
    for i, city in enumerate(CITIES):
        delivery[city] = {"days": (1, 2) if i < 2 else (2, 4) if i < 5 else (3, 6),
                          "price": [0, 1_500, 2_500, 3_000, 3_500, 3_500, 4_000, 4_500][i],
                          "free_from": [30_000, 30_000, 50_000, 50_000, 70_000, 70_000, 90_000, 90_000][i],
                          "pickup_points": rng.randint(2, 9) if i < 2 else rng.randint(1, 4),
                          "service_center": f"{city}, ул. Сервисная, {rng.randint(3, 140)}"}
    return {"delivery": delivery, "return_days": 14, "return_days_defective": 30, "refund_card_days": (3, 10),
            "refund_cash_days": 3, "installment_months": [3, 6, 12], "installment_min": 60_000,
            "bonus_rate": 0.02, "bonus_max_share": 0.3, "bonus_ttl_days": 365, "support_hours": "с 9:00 до 21:00 без выходных",
            "support_phone": "+7 700 000 00 00 (номер вымышленный)", "support_email": f"help@{DOMAIN}",
            "install_price": {"aircon": 35_000, "washer": 9_000, "builtin": 15_000}, "install_days": 3,
            "cancel_before": "статуса «Передан в доставку»", "address_change_hours": 24}


def kb_documents(f) -> list[dict]:
    docs = []

    def add(title, topic, body):
        docs.append({"id": f"KB-{len(docs) + 1:03}", "title": title, "topic": topic, "body": body.strip()})

    for city, d in f["delivery"].items():
        free = "бесплатно для всех заказов" if d["price"] == 0 else f"{d['price']:,} ₸; бесплатно от {d['free_from']:,} ₸".replace(",", " ")
        add(f"Доставка: {city}", "delivery", f"""
Курьерская доставка по городу {city} занимает {d['days'][0]}–{d['days'][1]} рабочих дня после подтверждения заказа.
Стоимость доставки: {free}. Крупная техника (холодильники, стиральные машины, кондиционеры) поднимается на этаж бесплатно при наличии грузового лифта; без лифта — 1 000 ₸ за этаж.
Пунктов самовывоза в городе: {d['pickup_points']}. Заказ хранится в пункте 5 дней, затем возвращается на склад и отменяется.
Курьер звонит за час до приезда. Если клиента нет дома, повторная доставка в тот же город стоит столько же, сколько первая.""")
    add("Возврат товара надлежащего качества", "returns", f"""
Товар надлежащего качества можно вернуть в течение {f['return_days']} дней со дня получения, если сохранены упаковка, комплектность и товарный вид.
Не возвращаются: смартфоны и ноутбуки после активации, товары личной гигиены (электробритвы, фены), технически сложные товары по перечню закона.
Возврат оформляется в личном кабинете или через поддержку: укажите номер заказа и причину. Курьер забирает товар бесплатно, если заказ доставлялся курьером.""")
    add("Возврат и обмен товара с недостатком", "returns", f"""
Если в товаре обнаружен заводской недостаток, его можно вернуть или обменять в течение {f['return_days_defective']} дней без экспертизы.
После {f['return_days_defective']} дней и до конца гарантии товар принимается в сервисный центр на диагностику; решение о ремонте, обмене или возврате принимается по её итогам в течение 20 дней.
Повреждения, полученные при доставке, фиксируются в акте при курьере. Без акта претензия о повреждении при доставке рассматривается до 10 дней.""")
    add("Сроки возврата денег", "returns", f"""
Деньги за возвращённый товар возвращаются тем же способом, которым был оплачен заказ.
На банковскую карту — от {f['refund_card_days'][0]} до {f['refund_card_days'][1]} рабочих дней после того, как товар принят на складе; срок зачисления зависит от банка.
При оплате наличными курьеру — переводом на карту клиента в течение {f['refund_cash_days']} рабочих дней после заявки с реквизитами.
Для заказов в рассрочку банк-партнёр закрывает рассрочку; уже внесённые платежи возвращаются на карту.""")
    add("Отмена заказа", "orders", f"""
Заказ можно отменить бесплатно до {f['cancel_before']}. После передачи в доставку отказаться можно при получении; доставка в этом случае не возвращается, если она была платной.
Предоплата за отменённый заказ возвращается по правилам из документа «Сроки возврата денег».""")
    add("Изменение адреса и состава заказа", "orders", f"""
Адрес доставки можно изменить не позднее чем за {f['address_change_hours']} часа до интервала доставки, если новый адрес в том же городе.
Перенос доставки в другой город оформляется как отмена и новый заказ.
Добавить товар в оформленный заказ нельзя; убрать позицию можно до сборки заказа на складе.""")
    add("Статусы заказа", "orders", """
«Оформлен» — заказ принят, ожидает подтверждения. «Подтверждён» — товар зарезервирован. «Собран» — заказ упакован на складе.
«Передан в доставку» — заказ у курьерской службы. «Доставлен» — клиент получил заказ. «Отменён» и «Возвращён» — финальные статусы.
Если статус не меняется больше 3 рабочих дней, поддержка проверяет заказ вручную.""")
    for code, (title, _, months, names) in CATEGORIES.items():
        extra = {"phone": "Аккумулятор — гарантия 6 месяцев.", "laptop": "Аккумулятор — гарантия 6 месяцев; битые пиксели — по стандарту производителя.",
                 "aircon": "Гарантия действует только при установке нашими мастерами или сертифицированной организацией.",
                 "fridge": "Компрессор — гарантия 5 лет.", "washer": "Двигатель — гарантия 5 лет."}.get(code, "")
        add(f"Гарантия: {title.lower()}", "warranty", f"""
На товары категории «{title}» ({', '.join(names)}) действует гарантия {months} месяцев с даты получения. {extra}
Гарантия не распространяется на механические повреждения, следы жидкости и ремонт вне авторизованного сервиса.
Для гарантийного обращения нужен номер заказа или чек. Диагностика бесплатна; срок ремонта — до 30 дней.""")
    add("Оплата картой онлайн", "payment", """
Оплата картой проходит на защищённой странице банка. Списание происходит сразу; при отмене заказа деньги возвращаются по правилам возврата.
Если деньги списались, а заказ остался в статусе «Оформлен», подождите 15 минут: платёж подтверждается автоматически. Если статус не изменился — напишите в поддержку, приложив время платежа и последние 4 цифры карты. Полный номер карты не присылайте никогда.""")
    add("Рассрочка", "payment", f"""
Рассрочка без переплаты на {', '.join(map(str, f['installment_months']))} месяцев оформляется через банк-партнёр для заказов от {f['installment_min']:,} ₸.
Решение банка приходит в течение 15 минут. При отказе банка заказ можно оплатить другим способом без повторного оформления.
Досрочное погашение — без комиссии, в приложении банка.""".replace(",", " "))
    add("Оплата наличными курьеру", "payment", """
Оплата наличными доступна для заказов до 300 000 ₸ при курьерской доставке. Курьер выдаёт электронный чек на телефон или e-mail.
Для заказов дороже 300 000 ₸ нужна предоплата картой или рассрочка.""")
    add("Оплата для юридических лиц", "payment", f"""
Юридические лица оплачивают заказ по счёту. Счёт действует 3 банковских дня; заказ резервируется на это время.
Закрывающие документы (счёт-фактура, накладная) отправляются на e-mail в течение 5 рабочих дней после доставки. Запрос документов — b2b@{DOMAIN}.""")
    add("Установка кондиционеров", "installation", f"""
Стандартная установка сплит-системы — {f['install_price']['aircon']:,} ₸: до 3 метров трассы, монтаж на стену из кирпича или бетона.
Дополнительный метр трассы — 6 000 ₸. Мастер приезжает в течение {f['install_days']} рабочих дней после доставки. Установка на высоте выше второго этажа без балкона требует альпиниста и считается отдельно.""".replace(",", " "))
    add("Подключение стиральных машин", "installation", f"""
Подключение стиральной машины к готовым коммуникациям — {f['install_price']['washer']:,} ₸. Мастер снимает транспортировочные болты и проверяет слив.
Если коммуникаций нет, мастер не прокладывает трубы: нужен сантехник.""".replace(",", " "))
    add("Установка встраиваемой техники", "installation", f"""
Установка встраиваемого холодильника или посудомоечной машины в готовую нишу — {f['install_price']['builtin']:,} ₸.
Подгонка мебели не входит в услугу. Размеры ниши проверяйте по инструкции до заказа: несоответствие ниши — не основание для бесплатного возврата установленной техники.""".replace(",", " "))
    add("Бонусная программа", "loyalty", f"""
За каждую покупку начисляется {int(f['bonus_rate'] * 100)} % суммы бонусами после получения заказа. 1 бонус = 1 ₸.
Бонусами можно оплатить до {int(f['bonus_max_share'] * 100)} % стоимости заказа. Бонусы действуют {f['bonus_ttl_days']} дней с даты начисления.
При возврате товара начисленные за него бонусы списываются.""")
    add("Подарочные сертификаты", "loyalty", """
Электронный сертификат номиналом от 10 000 до 200 000 ₸ действует 1 год. Сертификат можно использовать частично; остаток сохраняется.
Сертификат не обменивается на деньги. Утерянный сертификат восстанавливается по номеру заказа, в котором он был куплен.""")
    for city, d in list(f["delivery"].items())[:4]:
        add(f"Сервисный центр: {city}", "service", f"""
Авторизованный сервисный центр в городе {city}: {d['service_center']}. Приём техники — по будням с 10:00 до 19:00.
Крупную технику (холодильники, стиральные машины) мастер диагностирует на дому; выезд по городу бесплатный в период гарантии.""")
    add("Сервис в городах без сервисного центра", "service", """
В городах Актобе, Павлодар, Усть-Каменогорск и Атырау технику принимают пункты самовывоза и отправляют в сервисный центр Караганды или Астаны.
Доставка в сервис и обратно бесплатная в период гарантии; общий срок с учётом пересылки — до 45 дней.""")
    add("Контакты и часы работы поддержки", "support", f"""
Поддержка работает {f['support_hours']}. Телефон {f['support_phone']}, почта {f['support_email']}, чат на сайте {DOMAIN}.
Ответ в чате — в течение 10 минут в рабочее время, на почту — в течение 1 рабочего дня.""")
    add("Персональные данные", "support", """
Мы храним только данные, нужные для доставки и гарантийного обслуживания. Оператор поддержки никогда не просит полный номер карты, CVV-код или код из SMS.
Удалить аккаунт можно в личном кабинете; данные заказов хранятся 5 лет по требованию закона.""")
    add("Жалобы и эскалация", "support", """
Если ответ поддержки не решил вопрос, попросите передать обращение старшему специалисту. Ответ старшего специалиста — в течение 2 рабочих дней.
Письменная претензия рассматривается в течение 10 дней.""")
    add("Подбор холодильника", "product", """
Для семьи из двух человек обычно хватает 250–300 литров, для четырёх — 350–450 литров. Система No Frost не требует ручной разморозки.
Класс энергопотребления A++ потребляет примерно на 20 % меньше, чем A+. Перед покупкой измерьте дверные проёмы: глубина холодильника с открытой дверью больше на 50–60 см.""")
    add("Подбор кондиционера", "product", """
Для комнаты до 20 м² подходит модель 7 BTU (2,1 кВт), до 30 м² — 9–12 BTU. Инверторная модель тише и экономичнее при постоянной работе.
Кондиционер с функцией обогрева работает на тепло до −15 °C; ниже — только модели с зимним комплектом.""")
    add("Ноутбуки для работы и учёбы", "product", """
Для офисных задач хватает 8 ГБ оперативной памяти; для разработки и работы с графикой рекомендуется от 16 ГБ.
Ноутбуки Беркут поставляются с предустановленной системой; ноутбуки Сапсан — без операционной системы, установка не входит в гарантию.""")
    return docs


def kb_markdown(doc: dict, updated: str) -> str:
    return (f"---\nid: {doc['id']}\ntitle: {doc['title']}\ntopic: {doc['topic']}\nversion: 1\nupdated: {updated}\n"
            f"company: {COMPANY} (вымышленная)\n---\n\n# {doc['title']}\n\n{doc['body']}\n")


# ---------------------------------------------------------------- tables

def products(rng) -> list[dict]:
    rows = []
    for code, (title, (low, high), months, names) in CATEGORIES.items():
        for i in range(60 // len(CATEGORIES) + (1 if code in ("fridge", "phone", "kitchen", "tv") else 0)):
            name = names[i % len(names)]
            rows.append({"sku": f"{code.upper()[:3]}-{100 + i}", "name": f"{name} {rng.choice(['S', 'M', 'L', 'Pro', 'Lite', 'X'])}{rng.randint(1, 9)}",
                         "category": code, "category_title": title, "price_kzt": round(rng.randint(low, high), -3),
                         "warranty_months": months})
    return rows[:60]


def customers(rng) -> list[dict]:
    start = date(2023, 1, 1)
    return [{"customer_id": f"C{n:05}", "city": rng.choices(CITIES, weights=[30, 22, 12, 9, 7, 7, 7, 6])[0],
             "segment": "b2b" if rng.random() < 0.06 else "b2c", "registered_on": (start + timedelta(days=rng.randint(0, 1360))).isoformat()}
            for n in range(1, 801)]


def orders(rng, products_rows, customer_rows, f):
    order_rows, item_rows = [], []
    start = date(2026, 1, 1)
    for n in range(1, 3001):
        customer = rng.choice(customer_rows)
        created = start + timedelta(days=rng.randint(0, 303))
        lines = rng.sample(products_rows, rng.choices([1, 2, 3], weights=[70, 22, 8])[0])
        subtotal = 0
        for product in lines:
            qty = 1 if product["category"] != "kitchen" else rng.choice([1, 1, 2])
            item_rows.append({"order_id": f"LC-{n:06}", "sku": product["sku"], "qty": qty, "price_kzt": product["price_kzt"]})
            subtotal += qty * product["price_kzt"]
        d = f["delivery"][customer["city"]]
        delivery_type = "pickup" if rng.random() < 0.3 else "courier"
        delivery_price = 0 if delivery_type == "pickup" or subtotal >= d["free_from"] else d["price"]
        payment = "invoice" if customer["segment"] == "b2b" else pick(rng, PAYMENTS[:3])
        if payment == "installment" and subtotal < f["installment_min"]:
            payment = "card"
        if payment == "cash_on_delivery" and (subtotal > 300_000 or delivery_type == "pickup"):
            payment = "card"
        status = pick(rng, STATUSES) if created < date(2026, 10, 20) else rng.choice(["processing", "shipped", "delivered"])
        days = rng.randint(*d["days"]) + (rng.randint(2, 6) if rng.random() < 0.08 else 0)
        delivered = (created + timedelta(days=days)).isoformat() if status in ("delivered", "returned") else ""
        order_rows.append({"order_id": f"LC-{n:06}", "customer_id": customer["customer_id"], "city": customer["city"],
                           "created_at": created.isoformat(), "status": status, "payment_method": payment,
                           "delivery_type": delivery_type, "delivery_price_kzt": delivery_price, "items_total_kzt": subtotal,
                           "delivered_at": delivered, "promised_days_max": d["days"][1]})
    return order_rows, item_rows


TICKET_TEMPLATES = {
    "delivery_delay": ["Заказ {order} до сих пор не доставили, обещали за {days} дня. Где он?",
                       "Курьер не приехал, заказ {order}, город {city}. Сколько ещё ждать?",
                       "Здравствуйте! Статус {order} третий день «Передан в доставку». Когда привезут?"],
    "return_request": ["Хочу вернуть «{product}» из заказа {order}, не подошёл. Как оформить?",
                       "Можно вернуть «{product}»? Упаковку сохранил. Заказ {order}.",
                       "Передумал насчёт товара «{product}», заказ {order}, получил 5 дней назад."],
    "warranty_repair": ["«{product}» перестал включаться, куплен в марте, заказ {order}. Это гарантия?",
                        "Сломался товар «{product}» через полгода. Куда нести в городе {city}?",
                        "По гарантии: «{product}» шумит и течёт. Номер заказа {order}."],
    "payment_issue": ["Деньги списались, а заказ {order} висит в статусе «Оформлен».",
                      "Банк отказал в рассрочке на «{product}», что делать с заказом {order}?",
                      "Дважды списали оплату за заказ {order}! Верните деньги."],
    "installation": ["Сколько стоит установка «{product}» в городе {city}? Заказ {order}.",
                     "Когда приедет мастер устанавливать «{product}»? Заказ {order} уже доставлен.",
                     "Мастер не может подключить «{product}», говорит, нет коммуникаций. Заказ {order}."],
    "product_question": ["Подойдёт ли «{product}» для комнаты 25 м²?",
                         "Чем «{product}» отличается от модели подороже? Хочу заказать с доставкой в город {city}.",
                         "Есть ли у модели «{product}» гарантия на компрессор или аккумулятор?"],
    "order_change": ["Можно поменять адрес доставки для заказа {order}? Новый адрес тоже в городе {city}.",
                     "Хочу убрать «{product}» из заказа {order}, остальное оставить.",
                     "Отмените, пожалуйста, заказ {order}, нашёл дешевле."],
    "complaint_other": ["Курьер был груб и бросил коробку с «{product}». Заказ {order}.",
                        "Третий раз пишу в поддержку по заказу {order}, никто не отвечает!",
                        "Почему у вас нет пункта выдачи рядом со мной в городе {city}?"],
}
# Templates that only make sense for some products: a phrase of the template -> allowed product categories.
TEMPLATE_PRODUCTS = {"установка": {"aircon", "washer", "fridge"}, "устанавливать": {"aircon", "washer", "fridge"},
                     "подключить": {"aircon", "washer", "fridge"}, "комнаты 25": {"aircon"},
                     "компрессор или аккумулятор": {"fridge", "phone", "laptop"}, "шумит и течёт": {"fridge", "washer", "aircon"},
                     "рассрочке": {"fridge", "washer", "laptop", "phone", "tv", "aircon"}}
URGENCY = {"delivery_delay": "high", "payment_issue": "high", "complaint_other": "medium", "warranty_repair": "medium",
           "order_change": "high", "return_request": "medium", "installation": "medium", "product_question": "low"}


def tickets(rng, order_rows, item_rows, product_rows, count=600, start_id=1):
    by_sku = {p["sku"]: p for p in product_rows}
    items = {}
    for item in item_rows:
        items.setdefault(item["order_id"], []).append(item["sku"])
    rows = []
    for n in range(start_id, start_id + count):
        category = TICKET_CATEGORIES[(n * 7 + rng.randint(0, 7)) % len(TICKET_CATEGORIES)]
        template = rng.choice(TICKET_TEMPLATES[category])
        allowed = next((cats for marker, cats in TEMPLATE_PRODUCTS.items() if marker in template), None)
        order = rng.choice([o for o in order_rows if allowed is None
                            or any(by_sku[sku]["category"] in allowed for sku in items[o["order_id"]])])
        product = by_sku[rng.choice([sku for sku in items[order["order_id"]]
                                     if allowed is None or by_sku[sku]["category"] in allowed])]
        mentions_order = "{order}" in template
        text = template.format(order=order["order_id"], product=product["name"], city=order["city"],
                               days=order["promised_days_max"])
        rows.append({"ticket_id": f"T{n:05}", "created_at": (date.fromisoformat(order["created_at"]) + timedelta(days=rng.randint(1, 20))).isoformat(),
                     "channel": rng.choice(["chat", "email", "phone_transcript"]), "text": text,
                     "label": {"category": category, "order_id": order["order_id"] if mentions_order else None,
                               "sku": product["sku"] if "{product}" in template else None,
                               "city": order["city"] if "{city}" in template else None, "urgency": URGENCY[category]}})
    return rows


# ---------------------------------------------------------------- output

def to_csv(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


SCHEMA = """-- PostgreSQL schema of the fictional company «Лампа и Чайник» (data/company, R07). Load with:
--   psql -d company -f data/company/schema.sql
--   \\copy products FROM 'data/company/products.csv' CSV HEADER   (the same for customers, orders, order_items)
CREATE TABLE products (sku text PRIMARY KEY, name text NOT NULL, category text NOT NULL, category_title text NOT NULL,
                       price_kzt integer NOT NULL CHECK (price_kzt > 0), warranty_months integer NOT NULL);
CREATE TABLE customers (customer_id text PRIMARY KEY, city text NOT NULL, segment text NOT NULL CHECK (segment IN ('b2c', 'b2b')),
                        registered_on date NOT NULL);
CREATE TABLE orders (order_id text PRIMARY KEY, customer_id text NOT NULL REFERENCES customers, city text NOT NULL,
                     created_at date NOT NULL, status text NOT NULL, payment_method text NOT NULL, delivery_type text NOT NULL,
                     delivery_price_kzt integer NOT NULL, items_total_kzt integer NOT NULL, delivered_at date,
                     promised_days_max integer NOT NULL);
CREATE TABLE order_items (order_id text NOT NULL REFERENCES orders, sku text NOT NULL REFERENCES products,
                          qty integer NOT NULL CHECK (qty > 0), price_kzt integer NOT NULL);
"""


def build() -> dict[str, str]:
    rng = random.Random(SEED)
    f = facts(rng)
    files = {}
    for i, doc in enumerate(kb_documents(f)):
        files[f"kb/{doc['id']}.md"] = kb_markdown(doc, (date(2026, 9, 1) + timedelta(days=i % 30)).isoformat())
    product_rows = products(rng)
    customer_rows = customers(rng)
    order_rows, item_rows = orders(rng, product_rows, customer_rows, f)
    files["products.csv"] = to_csv(product_rows)
    files["customers.csv"] = to_csv(customer_rows)
    files["orders.csv"] = to_csv(order_rows)
    files["order_items.csv"] = to_csv(item_rows)
    files["tickets.jsonl"] = to_jsonl(tickets(rng, order_rows, item_rows, product_rows))
    requests = tickets(random.Random(SEED + 1), order_rows, item_rows, product_rows, count=20, start_id=90001)
    files["operator_requests.jsonl"] = to_jsonl([{"request_id": r["ticket_id"].replace("T", "R"), "text": r["text"], "expected": r["label"]}
                                                for r in requests])
    files["facts.json"] = json.dumps(f, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    files["schema.sql"] = SCHEMA
    files["manifest.json"] = json.dumps({"company": f"{COMPANY} (вымышленная)", "seed": SEED, "generator": "data/company/generate.py",
                                         "files": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sorted(files.items())}},
                                        ensure_ascii=False, indent=2) + "\n"
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv)
    files = build()
    stale = [name for name, text in files.items()
             if not (args.out / name).is_file() or (args.out / name).read_text(encoding="utf-8") != text]
    if args.check:
        if stale:
            raise SystemExit(f"data/company differs from the generator: {', '.join(stale[:5])}")
        print(json.dumps({"files": len(files), "status": "current"}))
        return 0
    for name, text in files.items():
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"files": len(files), "written": len(stale)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
