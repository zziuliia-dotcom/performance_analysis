"""
Анализ эффективности работы интернет-магазина
Итоговая проектная работа: интерактивный дашборд на Streamlit

Структура:
  1. Загрузка данных из трёх CSV-файлов (@st.cache_data)
  2. Диагностика качества: типы, пропуски, дубликаты, аномалии, выбросы
  3. Очистка и приведение типов
  4. Объединение таблиц и расчёт агрегированных показателей
  5. Дашборд: сырые данные, ключевые показатели, визуализация, выводы

Запуск:  streamlit run app_v1.py
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Анализ эффективности интернет-магазина",
    page_icon=":bar_chart:",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Раздел 1. Настройки и справочники
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILES = {
    "orders": "orders.csv",
    "users": "users.csv",
    "items": "items.csv",
}
TABLE_TITLES = {
    "orders": "Заказы (orders)",
    "users": "Пользователи (users)",
    "items": "Товары (items)",
}

# Колонки, которые обязаны быть в каждой таблице
REQUIRED_COLUMNS = {
    "orders": ["order_id", "user_id", "item_id", "order_date", "quantity", "price_per_unit"],
    "users": ["user_id", "user_name", "registration_date", "city", "user_segment"],
    "items": ["item_id", "item_name", "category", "supplier", "base_price"],
}

# Столбцы с датами: приводим к datetime
DATE_COLUMNS = {
    "orders": ["order_date"],
    "users": ["registration_date"],
    "items": [],
}

# Столбцы с числами: приводим к float
NUMERIC_COLUMNS = {
    "orders": ["quantity", "price_per_unit"],
    "users": [],
    "items": ["base_price"],
}

# Столбцы, которые не должны содержать пропусков: такие строки удаляем
CRITICAL_COLUMNS = {
    "orders": ["order_id", "user_id", "item_id", "order_date", "quantity", "price_per_unit"],
    "users": ["user_id"],
    "items": ["item_id"],
}

# Столбцы, где пропуск заполняем значением-заглушкой
TEXT_FILLERS = {
    "user_name": "Неизвестно",
    "city": "Неизвестно",
    "user_segment": "Unknown",
    "item_name": "Товар не указан",
    "category": "Unknown",
    "supplier": "Unknown",
}

PRIMARY_KEYS = {"orders": "order_id", "users": "user_id", "items": "item_id"}

ENCODINGS = ["utf-8", "utf-8-sig", "cp1251", "latin1"]

# Цвета категорий и сегментов, чтобы графики были узнаваемыми
PALETTE = [
    "#2F6F9F", "#E4572E", "#17A398", "#F2A541", "#7B5EA7",
    "#3FA34D", "#C44536", "#4C72B0", "#8C8C8C", "#B5651D",
]
SEGMENT_COLORS = {"vip": "#E4572E", "regular": "#2F6F9F", "new": "#17A398", "Unknown": "#8C8C8C"}
SEGMENT_RU = {"vip": "VIP", "regular": "Постоянные", "new": "Новые", "Unknown": "Не определён"}

WEEKDAYS_RU = [
    "Понедельник", "Вторник", "Среда",
    "Четверг", "Пятница", "Суббота", "Воскресенье",
]
WEEKEND = ["Суббота", "Воскресенье"]

# Даты в формате ISO разбираем как есть, остальные — с dayfirst=True
ISO_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$")

# Уровни проблем: error — критично, warn — влияет на расчёты, info — для сведения
LEVEL_TITLES = {"error": "Ошибки", "warn": "Предупреждения", "info": "Замечания"}


def plural_ru(number, one, few, many):
    """Выбирает форму существительного: 1 заказ, 2 заказа, 5 заказов."""
    remainder = abs(int(number)) % 100
    if 11 <= remainder <= 14:
        return many
    if remainder % 10 == 1:
        return one
    if 2 <= remainder % 10 <= 4:
        return few
    return many


def orders_word(number):
    return f"{int(number)} {plural_ru(number, 'заказ', 'заказа', 'заказов')}"


def money(value, digits=2):
    """Формирует строку вида «97 541.94 ₽» с неразрывными пробелами."""
    number = float(value)
    if not np.isfinite(number):
        return "—"
    formatted = f"{number:,.{digits}f}".replace(",", "\u00a0").replace(",", ".")
    return f"{formatted}\u00a0₽"


def percent(value, digits=1):
    """«54.6 %» из доли 0.546."""
    number = float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number * 100:.{digits}f}\u00a0%"


def number(value, digits=0):
    """«1 234» с неразрывными пробелами."""
    number_value = float(value)
    if not np.isfinite(number_value):
        return "—"
    return f"{number_value:,.{digits}f}".replace(",", "\u00a0")


def date_caption(series):
    """«01.02.2024 — 10.02.2024» по колонке с датами."""
    clean = series.dropna()
    if clean.empty:
        return "нет данных"
    return f"{clean.min():%d.%m.%Y} — {clean.max():%d.%m.%Y}"


# ----------------------------------------------------------------------------
# Раздел 2. Загрузка данных
# ----------------------------------------------------------------------------


def read_csv_any_encoding(path):
    """
    Читает CSV, перебирая кодировки.

    Русские выгрузки из Excel часто сохраняются в cp1251, поэтому utf-8 — не единственный вариант.
    """
    last_error = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Не удалось определить кодировку файла {path}: {last_error}")


@st.cache_data(show_spinner="Загрузка данных из CSV-файлов...")
def load_data(orders_stamp, users_stamp, items_stamp):
    """
    Загружает три CSV-файла и возвращает словарь {таблица: DataFrame}.

    Время изменения файлов передаётся аргументами: Streamlit не считает их частью
    данных, но пересчитывает кэш при их изменении. Так правка CSV подхватывается
    без перезапуска приложения.
    """
    del orders_stamp, users_stamp, items_stamp  # значения нужны только как ключ кэша
    frames = {}
    for key, filename in DATA_FILES.items():
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            st.error(
                f"Не найден файл **{filename}**. "
                f"Поместите `orders.csv`, `users.csv` и `items.csv` в одну папку с `app.py`."
            )
            st.stop()
        try:
            frames[key] = read_csv_any_encoding(path)
        except Exception as error:  # noqa: BLE001 — в интерфейсе показываем текст ошибки
            st.error(f"Не удалось прочитать `{filename}`: {error}")
            st.stop()
    return frames


def data_stamps():
    """Время изменения CSV-файлов — ключ кэша Streamlit."""
    return tuple(
        os.path.getmtime(os.path.join(BASE_DIR, name)) if os.path.exists(os.path.join(BASE_DIR, name)) else 0
        for name in DATA_FILES.values()
    )


# ----------------------------------------------------------------------------
# Раздел 3. Диагностика качества данных
# ----------------------------------------------------------------------------


def issue(level, title, detail):
    """Создаёт запись о проблеме."""
    return {"level": level, "title": title, "detail": detail}


def diagnose_frame(frame, key):
    """Проверяет одну таблицу: схему, типы, пропуски, дубликаты."""
    issues = []
    title = TABLE_TITLES[key]

    # 1. Первичный осмотр: размер датасета
    if frame.empty:
        issues.append(issue("error", "Пустая таблица", f"{title} не содержит ни одной строки."))
        return issues

    missing_columns = [column for column in REQUIRED_COLUMNS[key] if column not in frame.columns]
    if missing_columns:
        issues.append(issue(
            "error", "Отсутствуют обязательные столбцы",
            f"{title}: нет столбцов {', '.join(missing_columns)}. Дальнейший анализ невозможен.",
        ))
        return issues

    issues.append(issue(
        "info", "Размер таблицы",
        f"{title}: {number(len(frame))} строк × {frame.shape[1]} столбцов.",
    ))

    # 2. Пропуски
    nulls = frame[REQUIRED_COLUMNS[key]].isna().sum()
    for column, count in nulls[nulls > 0].items():
        share = count / len(frame)
        if column in CRITICAL_COLUMNS[key]:
            level, action = "error", "строки будут удалены на этапе очистки"
        else:
            level, action = "warn", "пропуск будет заполнен значением-заглушкой"
        issues.append(issue(
            level, f"Пропуски в `{column}`",
            f"{number(count)} из {number(len(frame))} ({percent(share)}). {action.capitalize()}.",
        ))
    if int(nulls.sum()) == 0:
        issues.append(issue("info", "Пропусков нет", f"{title}: все обязательные столбцы заполнены."))

    # 3. Типы данных: object там, где нужен datetime или число
    for column in DATE_COLUMNS[key]:
        if column in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame[column]):
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            broken = int(parsed.isna().sum() - frame[column].isna().sum())
            if broken:
                issues.append(issue(
                    "error", f"Некорректные даты в `{column}`",
                    f"{title}: {number(broken)} значений не распознаются как дата.",
                ))
            else:
                issues.append(issue(
                    "warn", f"Дата хранится как текст в `{column}`",
                    f"{title}: столбец имеет тип object, будет приведён к datetime.",
                ))

    for column in NUMERIC_COLUMNS[key]:
        if column not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        text = frame[column].astype("object").where(frame[column].notna(), None)
        is_number = text.map(lambda value: isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool))
        bad_text = int((~is_number & text.notna()).sum())
        if bad_text:
            issues.append(issue(
                "error", f"Нечисловые значения в `{column}`",
                f"{title}: {number(bad_text)} значений нельзя привести к числу.",
            ))
        else:
            issues.append(issue(
                "warn", f"Число хранится как текст в `{column}`",
                f"{title}: столбец имеет тип {frame[column].dtype}, будет приведён к float.",
            ))

    # 4. Дубликаты
    full_duplicates = int(frame.duplicated().sum())
    if full_duplicates:
        issues.append(issue(
            "warn", "Полные дубликаты строк",
            f"{title}: {number(full_duplicates)} полных копий строк будут удалены.",
        ))
    else:
        issues.append(issue("info", "Полных дубликатов нет", f"{title}: повторов нет."))

    primary_key = PRIMARY_KEYS[key]
    if primary_key in frame.columns:
        duplicated_keys = frame[primary_key].duplicated().sum()
        if duplicated_keys:
            issues.append(issue(
                "error", f"Повторы в `{primary_key}`",
                f"{title}: идентификатор должен быть уникальным, но повторяется {number(duplicated_keys)} раз.",
            ))
        else:
            issues.append(issue(
                "info", f"`{primary_key}` уникален",
                f"{title}: все {number(frame[primary_key].nunique())} идентификаторов уникальны.",
            ))

    # 5. Неизвестные значения в справочниках
    if key == "users" and "user_segment" in frame.columns:
        segments = sorted(frame["user_segment"].dropna().astype(str).str.lower().unique())
        unexpected = [segment for segment in segments if segment not in SEGMENT_RU]
        if unexpected:
            issues.append(issue(
                "warn", "Неизвестные сегменты",
                f"{title}: значения {', '.join(unexpected)} не входят в справочник (vip, regular, new).",
            ))
        else:
            issues.append(issue(
                "info", "Сегменты согласованы", f"{title}: найдены только допустимые значения.",
            ))
    return issues


def diagnose_referential(orders, users, items):
    """Проверяет целостность связей между таблицами."""
    issues = []
    order_users = set(orders["user_id"].dropna())
    order_items = set(orders["item_id"].dropna())
    orphan_users = order_users - set(users["user_id"].dropna())
    orphan_items = order_items - set(items["item_id"].dropna())
    if orphan_users:
        issues.append(issue(
            "error", "Заказы без пользователя",
            f"{len(orphan_users)} значений user_id из orders.csv отсутствуют в users.csv.",
        ))
    else:
        issues.append(issue("info", "Связи заказов и пользователей в порядке", "Все user_id найдены в users.csv."))
    if orphan_items:
        issues.append(issue(
            "error", "Заказы без товара",
            f"{len(orphan_items)} значений item_id из orders.csv отсутствуют в items.csv.",
        ))
    else:
        issues.append(issue("info", "Связи заказов и товаров в порядке", "Все item_id найдены в items.csv."))
    return issues


def diagnose_orders_logic(orders):
    """Проверяет бизнес-логику заказов: положительность цены и количества."""
    issues = []
    quantity = pd.to_numeric(orders["quantity"], errors="coerce")
    price = pd.to_numeric(orders["price_per_unit"], errors="coerce")
    bad_quantity = int((quantity <= 0).sum())
    bad_price = int((price <= 0).sum())
    if bad_quantity:
        issues.append(issue("error", "Неположительное количество",
                            f"{number(bad_quantity)} заказов с quantity <= 0 будут удалены."))
    if bad_price:
        issues.append(issue("error", "Неположительная цена",
                            f"{number(bad_price)} заказов с price_per_unit <= 0 будут удалены."))
    if not bad_quantity and not bad_price:
        issues.append(issue("info", "Количество и цены положительные", "Аномалий в заказах не найдено."))
    return issues


def diagnose_dates(orders, users):
    """Проверяет диапазон дат и логику «регистрация раньше первого заказа»."""
    issues = []
    order_dates = pd.to_datetime(orders["order_date"], errors="coerce", format="mixed")
    if order_dates.notna().any():
        span_days = int((order_dates.max() - order_dates.min()).days) + 1
        issues.append(issue(
            "info", "Период наблюдения",
            f"Заказы с {date_caption(order_dates)}, период {number(span_days)} дн.",
        ))
        far_future = int((order_dates > pd.Timestamp.today().normalize()).sum())
        if far_future:
            issues.append(issue("warn", "Даты в будущем",
                                f"{number(far_future)} заказов датированы позже сегодняшнего дня."))

    registration = users.set_index("user_id")["registration_date"]
    registration = pd.to_datetime(registration, errors="coerce", format="mixed")
    pairs = pd.DataFrame({
        "user_id": orders["user_id"],
        "order_date": order_dates,
    }).dropna(subset=["user_id", "order_date"])
    if not pairs.empty:
        pairs = pairs.join(registration, on="user_id")
        wrong_order = int((pairs["order_date"] < pairs["registration_date"]).sum())
        if wrong_order:
            issues.append(issue(
                "warn", "Заказ раньше регистрации",
                f"{number(wrong_order)} заказов оформлены раньше даты регистрации покупателя.",
            ))
    return issues


def diagnose_outliers(orders):
    """Считает выбросы чека методом IQR (Q1 − 1.5·IQR, Q3 + 1.5·IQR)."""
    quantity = pd.to_numeric(orders["quantity"], errors="coerce")
    price = pd.to_numeric(orders["price_per_unit"], errors="coerce")
    amount = quantity * price
    clean = amount.dropna()
    if len(clean) < 4:
        return []
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    # Нижняя граница ограничивается нулём: отрицательный чек невозможен
    lower = max(q1 - 1.5 * iqr, 0.0)
    upper = q3 + 1.5 * iqr
    outliers = int(((clean < lower) | (clean > upper)).sum())
    if outliers:
        return [issue(
            "warn", "Выбросы по сумме заказа (IQR)",
            f"{orders_word(outliers)} вне границ [{money(lower)}, {money(upper)}]. "
            "Обрезка выбросов доступна в боковой панели.",
        )]
    return [issue(
        "info", "Суммы заказов однородны",
        f"Все чеки лежат в границах [{money(lower)}, {money(upper)}] — грубых выбросов нет.",
    )]


def diagnose_pricing(orders, items):
    """Сравнивает фактическую цену продажи с базовой ценой каталога."""
    issues = []
    pair = orders.merge(items[["item_id", "item_name", "base_price"]], on="item_id", how="left")
    price = pd.to_numeric(pair["price_per_unit"], errors="coerce")
    base = pd.to_numeric(pair["base_price"], errors="coerce")
    valid = price.notna() & base.notna() & (base > 0)
    if not valid.any():
        return issues
    discount = (1 - price[valid] / base[valid]) * 100
    if (discount > 0).any():
        issues.append(issue(
            "info", "Цена продажи ниже базовой",
            f"Фактическая цена ниже base_price в {number(int((discount > 0).sum()))} из "
            f"{number(int(valid.sum()))} заказов: скидка от {discount.min():.1f}\u00a0% "
            f"до {discount.max():.1f}\u00a0%, в среднем {discount.mean():.1f}\u00a0%. "
            "base_price трактуем как прайс до скидки.",
        ))
    return issues


def diagnose_raw(orders_raw, users_raw, items_raw):
    """Полная диагностика трёх CSV «как есть», до приведения типов."""
    issues = []
    for key, frame in (("orders", orders_raw), ("users", users_raw), ("items", items_raw)):
        issues.extend(diagnose_frame(frame, key))

    # Дальнейшие проверки возможны, только если все ключевые столбцы на месте
    complete = all(
        all(column in frame.columns for column in CRITICAL_COLUMNS[key])
        for key, frame in (("orders", orders_raw), ("users", users_raw), ("items", items_raw))
    )
    if not complete:
        return issues

    issues.extend(diagnose_orders_logic(orders_raw))
    issues.extend(diagnose_dates(orders_raw, users_raw))
    issues.extend(diagnose_outliers(orders_raw))
    if all(column in items_raw.columns for column in ("item_id", "base_price")):
        issues.extend(diagnose_pricing(orders_raw, items_raw))
    if all(column in users_raw.columns for column in ("user_id",)) and all(
        column in items_raw.columns for column in ("item_id",)
    ):
        issues.extend(diagnose_referential(orders_raw, users_raw, items_raw))
    return issues


def issue_summary(issues):
    """Считает количество проблем по уровням."""
    summary = {"error": 0, "warn": 0, "info": 0}
    for item in issues:
        summary[item["level"]] = summary.get(item["level"], 0) + 1
    return summary


# ----------------------------------------------------------------------------
# Раздел 4. Очистка и приведение типов
# ----------------------------------------------------------------------------


def to_numeric_safe(series):
    """pd.to_numeric с errors='coerce': непонятные значения становятся NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    cleaned = series.astype("object").where(series.notna(), None)
    return pd.to_numeric(
        pd.Series(
            [str(value).replace("\u00a0", "").replace(" ", "").replace(",", ".") for value in cleaned],
            index=series.index,
            dtype="object",
        ),
        errors="coerce",
    )


def to_datetime_safe(series):
    """
    Приводит столбец к datetime.

    Строки формата ISO разбираются как есть, остальные — с dayfirst=True,
    иначе pandas молча меняет местами месяц и день.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    cleaned = series.astype("object").where(series.notna(), None)
    iso_mask = cleaned.map(lambda value: isinstance(value, str) and bool(ISO_DATE.match(value.strip())))
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(cleaned[iso_mask], errors="coerce", format="mixed")
    other = ~iso_mask & cleaned.notna()
    if other.any():
        try:
            result.loc[other] = pd.to_datetime(cleaned[other], errors="coerce", dayfirst=True)
        except (ValueError, TypeError):
            result.loc[other] = pd.to_datetime(cleaned[other], errors="coerce", format="mixed")
    return result


def clean_frame(frame, key):
    """Приводит типы, заполняет пропуски в тексте и отбрасывает критичные строки."""
    result = frame.copy()
    steps = []

    for column in DATE_COLUMNS[key]:
        if column in result.columns:
            result[column] = to_datetime_safe(result[column])

    for column in NUMERIC_COLUMNS[key]:
        if column in result.columns:
            result[column] = to_numeric_safe(result[column])

    for column, filler in TEXT_FILLERS.items():
        if column in result.columns:
            filled = int(result[column].isna().sum())
            result[column] = result[column].fillna(filler)
            if filled:
                steps.append(f"`{column}`: заполнено пропусков — {number(filled)} (значение «{filler}»)")

    before = len(result)
    result = result.drop_duplicates()
    removed = before - len(result)
    if removed:
        steps.append(f"Удалено полных дубликатов строк — {number(removed)}")

    primary_key = PRIMARY_KEYS[key]
    if primary_key in result.columns:
        before = len(result)
        result = result.sort_values(primary_key).drop_duplicates(subset=primary_key, keep="first")
        removed = before - len(result)
        if removed:
            steps.append(f"Удалено повторов по `{primary_key}` — {number(removed)} (оставлена первая запись)")

    present_critical = [column for column in CRITICAL_COLUMNS[key] if column in result.columns]
    before = len(result)
    result = result.dropna(subset=present_critical)
    removed = before - len(result)
    if removed:
        steps.append(f"Удалено строк с пропусками или некорректными значениями — {number(removed)}")

    if key == "orders":
        before = len(result)
        valid = (result["quantity"] > 0) & (result["price_per_unit"] > 0)
        result = result[valid]
        removed = before - len(result)
        if removed:
            steps.append(f"Удалено заказов с quantity или price <= 0 — {number(removed)}")

    if key == "items":
        for column in NUMERIC_COLUMNS[key]:
            before = len(result)
            result = result[result[column] > 0]
            removed = before - len(result)
            if removed:
                steps.append(f"Удалено товаров с `{column}` <= 0 — {number(removed)}")

    result = result.reset_index(drop=True)
    return result, steps


def build_cleaned(frames, clip_outliers=False):
    """
    Очищает все три таблицы и возвращает (данные, отчёт по очистке).

    Решение по пропускам из шаблона подготовки данных:
      * критичные столбцы (идентификаторы, дата, цена, количество) — строка удаляется,
        потому что по ней нельзя достоверно посчитать выручку;
      * текстовые столбцы — заполняются значением-заглушкой («Unknown»), чтобы
        не терять заказы из-за отсутствующего названия товара или города.
    """
    cleaned = {}
    steps = {}
    for key, frame in frames.items():
        cleaned[key], steps[key] = clean_frame(frame, key)

    orders, users, items = cleaned["orders"], cleaned["users"], cleaned["items"]

    # Заказы-«сироты»: без пользователя или без товара считать выручку некорректно
    known_users = set(users["user_id"])
    known_items = set(items["item_id"])
    before = len(orders)
    orders = orders[orders["user_id"].isin(known_users) & orders["item_id"].isin(known_items)]
    orphan_steps = []
    if before - len(orders):
        orphan_steps.append(f"Удалено заказов без пользователя или товара — {number(before - len(orders))}")
    cleaned["orders"] = orders
    steps["orders"] = steps["orders"] + orphan_steps

    # Обрезка выбросов по IQR — только по решению пользователя
    outlier_steps = []
    if clip_outliers and not orders.empty:
        amount = orders["quantity"] * orders["price_per_unit"]
        q1 = float(amount.quantile(0.25))
        q3 = float(amount.quantile(0.75))
        iqr = q3 - q1
        lower = max(q1 - 1.5 * iqr, 0.0)
        upper = q3 + 1.5 * iqr
        clipped = amount.clip(lower=lower, upper=upper)
        changed = int((clipped != amount).sum())
        if changed:
            orders = orders.assign(amount=clipped)
            outlier_steps.append(
                f"Обрезано выбросов по сумме заказа — {number(changed)} "
                f"(границы {money(lower)} — {money(upper)})"
            )
    cleaned["orders"] = orders
    steps["orders"] = steps["orders"] + outlier_steps

    return cleaned, steps


# ----------------------------------------------------------------------------
# Раздел 5. Объединение таблиц и расчёт показателей
# ----------------------------------------------------------------------------


def build_dataframe(cleaned):
    """
    Объединяет заказы с пользователями и товарами и добавляет расчётные столбцы.

    Соединение — left, чтобы сохранить каждый заказ из orders.csv: строки без пары
    в справочниках уже удалены на этапе очистки, но left-join страхует от потери
    данных, если пользователь добавит записи вручную.
    """
    merged = (
        cleaned["orders"]
        .merge(cleaned["users"], on="user_id", how="left")
        .merge(cleaned["items"], on="item_id", how="left")
    )

    if "amount" in merged.columns:
        merged["revenue"] = merged["amount"]
    else:
        merged["revenue"] = merged["quantity"] * merged["price_per_unit"]

    merged["weekday"] = merged["order_date"].dt.dayofweek
    merged["weekday_ru"] = merged["weekday"].map(lambda value: WEEKDAYS_RU[value] if pd.notna(value) else None)
    merged["is_weekend"] = merged["weekday"].isin([5, 6])
    merged["month"] = merged["order_date"].dt.to_period("M").astype("string")
    merged["day"] = merged["order_date"].dt.date

    merged["base_price"] = merged["base_price"].fillna(0.0)
    valid_base = merged["base_price"] > 0
    merged["discount_pct"] = np.where(
        valid_base,
        (1 - merged["price_per_unit"] / merged["base_price"]) * 100,
        np.nan,
    )
    merged["registration_date"] = pd.to_datetime(merged["registration_date"], errors="coerce")
    merged["days_since_registration"] = (merged["order_date"] - merged["registration_date"]).dt.days
    merged["category"] = merged["category"].fillna("Unknown")
    merged["city"] = merged["city"].fillna("Неизвестно")
    merged["user_segment"] = merged["user_segment"].fillna("Unknown")

    return merged.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def build_metrics(frame):
    """Считает ключевые агрегированные показатели."""
    empty_metrics = {
        "orders": 0, "revenue": 0.0, "users": 0, "avg_check": 0.0, "median_check": 0.0,
        "units": 0, "repeat_share": 0.0, "orders_per_user": 0.0,
        "top_category": None, "top_category_share": 0.0, "top_category_top3_share": 0.0,
        "top_item": None, "top_item_revenue": 0.0, "sold_items": 0,
        "avg_discount": np.nan, "weekend_share": 0.0,
    }
    if frame.empty:
        return empty_metrics

    orders = int(frame["order_id"].nunique())
    revenue = float(frame["revenue"].sum())
    users = int(frame["user_id"].nunique())
    per_user = frame.groupby("user_id")["order_id"].nunique()
    repeat_share = float((per_user > 1).mean()) if len(per_user) else 0.0

    by_category = (
        frame.groupby("category", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
    by_item = (
        frame.groupby("item_name", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )

    return {
        "orders": orders,
        "revenue": revenue,
        "users": users,
        "avg_check": revenue / orders if orders else 0.0,
        "median_check": float(frame["revenue"].median()) if orders else 0.0,
        "units": int(frame["quantity"].sum()),
        "repeat_share": repeat_share,
        "orders_per_user": float(orders / users) if users else 0.0,
        "top_category": by_category.iloc[0]["category"] if len(by_category) else None,
        "top_category_share": float(by_category.iloc[0]["revenue"] / revenue) if revenue and len(by_category) else 0.0,
        "top_category_top3_share": float(by_category.head(3)["revenue"].sum() / revenue) if revenue else 0.0,
        "top_item": by_item.iloc[0]["item_name"] if len(by_item) else None,
        "top_item_revenue": float(by_item.iloc[0]["revenue"]) if len(by_item) else 0.0,
        "sold_items": int(by_item.shape[0]),
        "avg_discount": float(frame["discount_pct"].mean(skipna=True)),
        "weekend_share": float(frame.loc[frame["is_weekend"], "order_id"].nunique() / orders) if orders else 0.0,
    }


def filter_frame(frame, date_range, categories, cities, segments, items):
    """Применяет фильтры боковой панели к объединённому DataFrame."""
    result = frame
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        result = result[(result["order_date"] >= start) & (result["order_date"] < end)]
    if categories:
        result = result[result["category"].isin(categories)]
    if cities:
        result = result[result["city"].isin(cities)]
    if segments:
        result = result[result["user_segment"].isin(segments)]
    if items:
        result = result[result["item_name"].isin(items)]
    return result.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Раздел 6. Аналитические выводы
# ----------------------------------------------------------------------------


def build_insights(frame, metrics, quality, catalog_size):
    """Формирует выводы по данным. Каждый пункт опирается на посчитанные числа."""
    insights = []
    if frame.empty:
        return ["Под текущие фильтры не попал ни один заказ — снимите часть фильтров, чтобы увидеть выводы."]

    by_category = (
        frame.groupby("category", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
    by_weekday = frame.groupby("weekday_ru", as_index=False).agg(
        orders=("order_id", "nunique"), revenue=("revenue", "sum")
    )
    by_weekday["sort"] = by_weekday["weekday_ru"].map({name: index for index, name in enumerate(WEEKDAYS_RU)})
    by_weekday = by_weekday.sort_values("sort")
    by_item = (
        frame.groupby("item_name", as_index=False)
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"), orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    by_city = (
        frame.groupby("city", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
    by_segment = frame.groupby("user_segment", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique"), users=("user_id", "nunique")
    )
    by_segment["segment_ru"] = by_segment["user_segment"].map(SEGMENT_RU).fillna(by_segment["user_segment"])

    # 1. Ключевые деньги
    amounts = frame["revenue"].dropna()
    q1 = float(amounts.quantile(0.25))
    q3 = float(amounts.quantile(0.75))
    iqr = q3 - q1
    iqr_lower = max(q1 - 1.5 * iqr, 0.0)
    iqr_upper = q3 + 1.5 * iqr
    iqr_outliers = int(((amounts < iqr_lower) | (amounts > iqr_upper)).sum())
    check_gap = abs(metrics["avg_check"] - metrics["median_check"]) / metrics["avg_check"] if metrics["avg_check"] else 0.0
    insights.append(
        f"**Общая выручка — {money(metrics['revenue'])}** при {orders_word(metrics['orders'])}; "
        f"средний чек {money(metrics['avg_check'])}, медианный {money(metrics['median_check'])}. "
        f"Расхождение среднего и медианного "
        f"{'минимальное — основная масса чеков сосредоточена в одном диапазоне' if check_gap < 0.05 else 'заметное, в базе есть как мелкие, так и крупные заказы'}. "
        f"По критерию IQR за границы [{money(iqr_lower)}, {money(iqr_upper)}] "
        f"выходят {orders_word(iqr_outliers)}. "
        f"Покупателей — {number(metrics['users'])}, в среднем {metrics['orders_per_user']:.1f} заказа на человека."
    )

    # 2. Концентрация по категориям
    leader = by_category.iloc[0]
    insights.append(
        f"Основная выручка приходится на категорию **«{leader['category']}»** — {money(leader['revenue'])} "
        f"({percent(leader['revenue'] / metrics['revenue'])} выручки). "
        f"На три лидирующие категории приходится {percent(metrics['top_category_top3_share'])} выручки, "
        "то есть ассортимент отдаёт непропорционально много выручки и работает неравномерно."
    )

    # 3. Товар-лидер
    top_item = by_item.iloc[0]
    insights.append(
        f"Товар-лидер продаж — **{top_item['item_name']}**: {money(top_item['revenue'])} "
        f"({percent(top_item['revenue'] / metrics['revenue'])} выручки) при {number(top_item['units'])} шт. "
        f"и {orders_word(top_item['orders'])}. "
        f"В продажах участвует {number(metrics['sold_items'])} из {number(catalog_size)} позиций каталога."
    )

    # 4. День недели: различаем пик по количеству и по выручке
    peak_orders = by_weekday.loc[by_weekday["orders"].idxmax()]
    peak_revenue = by_weekday.loc[by_weekday["revenue"].idxmax()]
    weekday_text = (
        f"Пик по количеству заказов — **{peak_orders['weekday_ru']}** ({orders_word(peak_orders['orders'])}), "
        f"по выручке — **{peak_revenue['weekday_ru']}** ({money(peak_revenue['revenue'])}). "
    )
    weekend_orders = by_weekday[by_weekday["weekday_ru"].isin(WEEKEND)]["orders"].sum()
    weekday_orders = by_weekday[~by_weekday["weekday_ru"].isin(WEEKEND)]["orders"].sum()
    if weekend_orders > weekday_orders:
        weekday_text += (
            f"На выходные приходится {orders_word(weekend_orders)} против {orders_word(weekday_orders)} "
            "в будни — гипотеза о пике в выходные подтверждается."
        )
    else:
        weekday_text += (
            f"На выходные приходится {orders_word(weekend_orders)} против {orders_word(weekday_orders)} "
            "в будни, поэтому пик приходится на будние дни, а не на выходные — "
            "гипотеза о выходных не подтверждается."
        )
    insights.append(weekday_text)

    # 5. Повторные покупки
    repeat_rate = metrics["repeat_share"]
    insights.append(
        f"Повторные покупки есть у {percent(repeat_rate)} покупателей "
        f"(совершили больше одного заказа). "
        + (
            "Это хороший показатель удержания: база лояльна, стоит развивать программу лояльности."
            if repeat_rate >= 0.4
            else "База обновляется чаще, чем возвращается, — стоит изучить причины оттока."
        )
    )

    # 6. География
    city_leader = by_city.iloc[0]
    city_share = city_leader["revenue"] / metrics["revenue"]
    insights.append(
        f"География продаж: **«{city_leader['city']}»** приносит {money(city_leader['revenue'])} "
        f"({percent(city_share)} выручки). Всего задействовано {by_city.shape[0]} городов; "
        f"на топ-3 приходится {percent(by_city.head(3)['revenue'].sum() / metrics['revenue'])}."
    )

    # 7. Сегменты
    segment_leader = by_segment.loc[by_segment["revenue"].idxmax()]
    vip = by_segment[by_segment["user_segment"] == "vip"]
    if len(vip):
        vip_row = vip.iloc[0]
        vip_avg = vip_row["revenue"] / vip_row["orders"] if vip_row["orders"] else 0.0
        insights.append(
            f"По выручке лидирует сегмент **{segment_leader['segment_ru']}** — "
            f"{money(segment_leader['revenue'])} ({percent(segment_leader['revenue'] / metrics['revenue'])}). "
            f"При этом VIP-клиенты ({number(vip_row['users'])} чел.) дают средний чек "
            f"{money(vip_avg)} — "
            + (
                "ниже среднего по базе, то есть статус VIP пока не конвертируется в деньги."
                if vip_avg < metrics["avg_check"]
                else "выше среднего, статус VIP работает корректно."
            )
        )
    else:
        insights.append(
            f"По выручке лидирует сегмент **{segment_leader['segment_ru']}** — "
            f"{money(segment_leader['revenue'])} ({percent(segment_leader['revenue'] / metrics['revenue'])})."
        )

    # 8. Ценовая дисциплина
    discount = frame["discount_pct"].dropna()
    if not discount.empty:
        insights.append(
            f"Фактическая цена продажи ниже базовой цены каталога в среднем на "
            f"**{discount.mean():.1f}\u00a0%** (разброс {discount.min():.1f}–{discount.max():.1f}\u00a0%). "
            "Если `base_price` — это прайс до скидки, то средняя скидка очень велика и требует "
            "проверки: возможно, цены в orders.csv и items.csv заданы в разных валютах или масштабах."
        )

    # 9. Товары без продаж
    not_sold = max(catalog_size - metrics["sold_items"], 0)
    if not_sold:
        insights.append(
            f"Часть позиций каталога не продаётся: {number(not_sold)} шт. из {number(catalog_size)}. "
            "Проверьте остатки и цены — наличие товара само по себе выручку не даёт."
        )

    # 10. Качество данных
    summary = issue_summary(quality)
    insights.append(
        f"Качество данных: ошибок — {number(summary['error'])}, предупреждений — "
        f"{number(summary['warn'])}. Подробный разбор — во вкладке «Качество данных»."
    )
    return insights


# ----------------------------------------------------------------------------
# Раздел 7. Графики (matplotlib)
# ----------------------------------------------------------------------------


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Единое оформление осей всех графиков."""
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)
    if title:
        ax.set_title(title, fontsize=12, color="#222222", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#666666")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#666666")


def short_labels(text, limit=32):
    """Обрезает длинные названия товаров, чтобы подписи на графике не наезжали."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def plot_top_items(frame, top_n=10):
    """Горизонтальная столбчатая диаграмма: топ-N товаров по выручке."""
    if frame.empty:
        return None
    data = (
        frame.groupby("item_name", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False).head(top_n).sort_values("revenue")
    )
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.45 * len(data))))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    colors[-1] = "#E4572E"
    ax.barh([short_labels(name) for name in data["item_name"]], data["revenue"], color=colors, height=0.68)
    style_axes(
        ax,
        title=f"Топ-{len(data)} товаров по выручке",
        xlabel="Выручка, ₽",
    )
    ax.grid(axis="y", visible=False)
    for position, value in enumerate(data["revenue"]):
        ax.text(value, position, f"  {number(value)} ₽", va="center", fontsize=8, color="#444444")
    ax.set_xlim(0, float(data["revenue"].max()) * 1.18)
    fig.tight_layout()
    return fig


def plot_category_share(frame):
    """Круговая диаграмма: доля категорий в выручке."""
    if frame.empty:
        return None
    data = (
        frame.groupby("category", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
    data = data[data["revenue"] > 0]
    if data.empty:
        return None
    labels = [f"{name} — {percent(value / data['revenue'].sum(), 1)}" for name, value in zip(data["category"], data["revenue"])]
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(data))]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    # matplotlib возвращает (wedges, texts), если autopct не задан
    wedges, _ = ax.pie(
        data["revenue"],
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.96, 0.5), fontsize=9, frameon=False)
    ax.set_title("Доля категорий в выручке", fontsize=12, color="#222222", pad=14, loc="left")
    ax.text(0, 0.06, money(data["revenue"].sum(), 0), ha="center", va="center", fontsize=13, color="#222222")
    ax.text(0, -0.12, "всего выручки", ha="center", va="center", fontsize=9, color="#777777")
    fig.tight_layout()
    return fig


def plot_weekday_orders(frame):
    """Столбчатая диаграмма: заказы по дням недели."""
    if frame.empty:
        return None
    data = frame.groupby("weekday_ru", as_index=False).agg(
        orders=("order_id", "nunique"), revenue=("revenue", "sum")
    )
    order_map = {name: index for index, name in enumerate(WEEKDAYS_RU)}
    data["sort"] = data["weekday_ru"].map(order_map)
    data = data.sort_values("sort")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#E4572E" if name in WEEKEND else "#2F6F9F" for name in data["weekday_ru"]]
    ax.bar(data["weekday_ru"], data["orders"], color=colors, width=0.6, label="Заказы")
    style_axes(ax, title="Количество заказов по дням недели", ylabel="Заказов, шт.")

    twin = ax.twinx()
    twin.plot(
        data["weekday_ru"], data["revenue"],
        color="#F2A541", marker="o", linewidth=2, markersize=5, label="Выручка",
    )
    twin.set_ylabel("Выручка, ₽", fontsize=9, color="#666666")
    twin.tick_params(colors="#555555", labelsize=9)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color("#CCCCCC")

    ax.tick_params(axis="x", rotation=30)
    for position, value in enumerate(data["orders"]):
        ax.text(position, value, str(int(value)), ha="center", va="bottom", fontsize=8, color="#444444")
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = twin.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right,
              loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_daily_revenue(frame):
    """Линейный график: динамика выручки по дням периода."""
    if frame.empty:
        return None
    data = frame.groupby("day", as_index=False).agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    data = data.sort_values("day")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(data["day"], data["revenue"], color="#2F6F9F", linewidth=2, marker="o", markersize=5)
    ax.fill_between(data["day"], data["revenue"], alpha=0.15, color="#2F6F9F")
    style_axes(ax, title="Динамика выручки по дням", ylabel="Выручка, ₽")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0)
    for _, row in data.iterrows():
        ax.annotate(number(row["revenue"]), (row["day"], row["revenue"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#666666")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_city_revenue(frame, top_n=8):
    """Горизонтальная столбчатая диаграмма: выручка по городам."""
    if frame.empty:
        return None
    data = (
        frame.groupby("city", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False).head(top_n).sort_values("revenue")
    )
    fig, ax = plt.subplots(figsize=(7.5, max(2.8, 0.5 * len(data))))
    colors = ["#17A398" if name != data["city"].iloc[-1] else "#E4572E" for name in data["city"]]
    ax.barh(data["city"], data["revenue"], color=colors, height=0.62)
    style_axes(ax, title="Выручка по городам", xlabel="Выручка, ₽")
    ax.grid(axis="y", visible=False)
    for position, value in enumerate(data["revenue"]):
        ax.text(value, position, f"  {number(value)} ₽", va="center", fontsize=8, color="#444444")
    ax.set_xlim(0, float(data["revenue"].max()) * 1.2)
    fig.tight_layout()
    return fig


def plot_segment_orders(frame):
    """Столбчатая диаграмма: средний чек по сегментам покупателей."""
    if frame.empty:
        return None
    data = (
        frame.groupby("user_segment", as_index=False)
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    )
    data["avg_check"] = np.where(data["orders"] > 0, data["revenue"] / data["orders"], 0.0)
    data["segment_ru"] = data["user_segment"].map(SEGMENT_RU).fillna(data["user_segment"])
    data = data.sort_values("avg_check")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [SEGMENT_COLORS.get(segment, "#8C8C8C") for segment in data["user_segment"]]
    ax.bar(data["segment_ru"], data["avg_check"], color=colors, width=0.55)
    style_axes(ax, title="Средний чек по сегментам", ylabel="Средний чек, ₽")
    for position, value in enumerate(data["avg_check"]):
        ax.text(position, value, f"  {number(value)} ₽", ha="center", va="bottom", fontsize=8, color="#444444")
    ax.set_ylim(0, float(data["avg_check"].max()) * 1.22)
    fig.tight_layout()
    return fig


def show_figure(fig):
    """Показывает график и освобождает фигуру, чтобы не течь память при перерисовке."""
    if fig is None:
        st.info("Нет данных для построения графика по текущим фильтрам.")
        return
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Раздел 8. Сборка интерфейса
# ----------------------------------------------------------------------------

RAW_DISPLAY_COLUMNS = [
    "order_id", "order_date", "user_id", "user_name", "user_segment", "city",
    "item_id", "item_name", "category", "supplier", "quantity",
    "price_per_unit", "base_price", "revenue", "discount_pct",
    "weekday_ru", "is_weekend", "days_since_registration",
]

COLUMN_CONFIG = {
    "order_id": st.column_config.NumberColumn("Заказ", format="%d"),
    "order_date": st.column_config.DatetimeColumn("Дата и время", format="DD.MM.YYYY HH:mm"),
    "user_id": st.column_config.NumberColumn("Пользователь", format="%d"),
    "user_name": st.column_config.TextColumn("Покупатель"),
    "user_segment": st.column_config.TextColumn("Сегмент"),
    "city": st.column_config.TextColumn("Город"),
    "item_id": st.column_config.NumberColumn("Товар (ID)", format="%d"),
    "item_name": st.column_config.TextColumn("Товар"),
    "category": st.column_config.TextColumn("Категория"),
    "supplier": st.column_config.TextColumn("Поставщик"),
    "quantity": st.column_config.NumberColumn("Кол-во", format="%d шт."),
    "price_per_unit": st.column_config.NumberColumn("Цена продажи", format="%.2f ₽"),
    "base_price": st.column_config.NumberColumn("Базовая цена", format="%.2f ₽"),
    "revenue": st.column_config.NumberColumn("Выручка", format="%.2f ₽"),
    "discount_pct": st.column_config.NumberColumn("Скидка, %", format="%.1f"),
    "weekday_ru": st.column_config.TextColumn("День недели"),
    "is_weekend": st.column_config.CheckboxColumn("Выходной"),
    "days_since_registration": st.column_config.NumberColumn("Дней с регистрации", format="%d"),
}


def render_sidebar(frames, cleaned_default):
    """Собирает боковую панель с фильтрами и настройками очистки."""
    sidebar = st.sidebar
    sidebar.title("Фильтры")

    clip_outliers = sidebar.checkbox(
        "Обрезать выбросы по IQR",
        value=False,
        help="Сумма заказа ограничивается границами Q1 − 1.5·IQR и Q3 + 1.5·IQR. "
             "По умолчанию выключено, чтобы выручка отражала фактические продажи.",
    )

    orders = cleaned_default["orders"]
    date_min = orders["order_date"].min().date() if not orders.empty else None
    date_max = orders["order_date"].max().date() if not orders.empty else None

    date_range = None
    if date_min and date_max:
        date_range = sidebar.date_input(
            "Период",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            format="DD.MM.YYYY",
            key="filter_dates",
        )

    def multiselect(label, values, key):
        options = sorted(str(value) for value in values.dropna().unique())
        chosen = sidebar.multiselect(label, options, key=key)
        return chosen if chosen else None

    categories = multiselect("Категории", cleaned_default["items"]["category"], "filter_categories")
    cities = multiselect("Города", cleaned_default["users"]["city"], "filter_cities")
    segments = multiselect("Сегменты", cleaned_default["users"]["user_segment"], "filter_segments")
    items = multiselect("Товары", cleaned_default["items"]["item_name"], "filter_items")

    top_n = sidebar.slider("Товаров в топе", 5, 20, 10, 1, key="top_n")
    if sidebar.button("Сбросить фильтры", width="stretch"):
        for key in ("filter_dates", "filter_categories", "filter_cities", "filter_segments", "filter_items", "top_n"):
            st.session_state.pop(key, None)
        st.rerun()

    sidebar.divider()
    with sidebar.expander("О загруженных файлах"):
        for key, frame in frames.items():
            sidebar.write(f"**{TABLE_TITLES[key]}** — {number(len(frame))} строк")
        sidebar.caption(f"Папка данных: {BASE_DIR}")

    return {
        "date_range": date_range,
        "categories": categories,
        "cities": cities,
        "segments": segments,
        "items": items,
        "top_n": top_n,
        "clip_outliers": clip_outliers,
    }


def render_header(frames, frame, metrics):
    """Шапка приложения с описанием периода и ключевыми числами верхнего уровня."""
    st.title("Анализ эффективности работы интернет-магазина")
    st.caption(
        "Итоговая проектная работа. Дашборд собран на Streamlit, Pandas и Matplotlib "
        "по трём CSV-файлам: orders, users, items."
    )

    period_start = frame["order_date"].min()
    period_end = frame["order_date"].max()
    span = (period_end - period_start).days + 1 if pd.notna(period_start) and pd.notna(period_end) else 0

    columns = st.columns(4)
    columns[0].metric("Заказов", number(metrics["orders"]))
    columns[1].metric("Выручка", money(metrics["revenue"]))
    columns[2].metric("Покупателей", number(metrics["users"]))
    columns[3].metric("Средний чек", money(metrics["avg_check"]))

    st.caption(
        f"Период данных: {date_caption(frame['order_date'])} ({number(span)} дн.) · "
        f"строк после фильтрации: {number(len(frame))} из {number(sum(len(value) for value in frames.values()))} "
        "в исходных файлах"
    )
    st.divider()


def render_raw_tab(frame):
    """Вкладка «Сырые данные»."""
    st.subheader("Сырые данные")
    st.caption(
        f"Итоговая таблица после объединения и очистки: {number(len(frame))} строк. "
        "Фильтры боковой панели применяются и здесь, и во всех остальных вкладках."
    )

    display = frame[[column for column in RAW_DISPLAY_COLUMNS if column in frame.columns]].copy()
    display["order_date"] = pd.to_datetime(display["order_date"])
    display["revenue"] = display["revenue"].round(2)
    display["price_per_unit"] = display["price_per_unit"].round(2)
    display["base_price"] = display["base_price"].round(2)
    display["discount_pct"] = display["discount_pct"].round(1)

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={key: value for key, value in COLUMN_CONFIG.items() if key in display.columns},
        height=460,
    )

    left, right = st.columns([3, 1])
    with left:
        st.download_button(
            "Скачать отфильтрованные данные (CSV)",
            data=display.to_csv(index=False).encode("utf-8-sig"),
            file_name="orders_analit.csv",
            mime="text/csv",
            width="stretch",
        )
    with right:
        st.caption(f"Сумма выручки в таблице: {money(display['revenue'].sum())}")

    with st.expander("Описание столбцов итоговой таблицы"):
        st.markdown(
            """
| Столбец | Источник | Смысл |
|---|---|---|
| `order_id` | orders | Идентификатор заказа |
| `order_date` | orders | Дата и время заказа, приведены к `datetime` |
| `user_id`, `user_name`, `user_segment`, `city` | users | Покупатель, его сегмент и город |
| `item_id`, `item_name`, `category`, `supplier` | items | Товар, категория и поставщик |
| `quantity` | orders | Количество единиц в заказе |
| `price_per_unit` | orders | Фактическая цена продажи за единицу |
| `base_price` | items | Базовая цена каталога (прайс до скидки) |
| `revenue` | расчёт | `quantity × price_per_unit` — выручка по заказу |
| `discount_pct` | расчёт | Скидка относительно базовой цены, % |
| `weekday_ru`, `is_weekend` | расчёт | День недели и признак выходного |
| `days_since_registration` | расчёт | Сколько дней прошло с регистрации покупателя |
"""
        )


def render_metrics_tab(frame, metrics):
    """Вкладка «Ключевые показатели»."""
    st.subheader("Ключевые показатели")
    st.caption("Показатели рассчитаны по данным, отобранным фильтрами боковой панели.")

    top_row = st.columns(4)
    top_row[0].metric("Заказов", number(metrics["orders"]), help="Уникальных order_id в выборке")
    top_row[1].metric("Выручка", money(metrics["revenue"]), help="Сумма quantity × price_per_unit")
    top_row[2].metric("Уникальных покупателей", number(metrics["users"]), help="Уникальных user_id в выборке")
    top_row[3].metric("Средний чек", money(metrics["avg_check"]), help="Выручка / количество заказов")

    bottom_row = st.columns(4)
    bottom_row[0].metric("Медианный чек", money(metrics["median_check"]), help="Медиана суммы заказа")
    bottom_row[1].metric("Продано единиц", number(metrics["units"]), help="Сумма quantity по всем заказам")
    bottom_row[2].metric(
        "Доля повторных покупателей", percent(metrics["repeat_share"]),
        help="Доля покупателей, оформивших больше одного заказа",
    )
    bottom_row[3].metric(
        "Заказов на покупателя", f"{metrics['orders_per_user']:.2f}", help="Заказы / уникальные покупатели",
    )

    st.divider()

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("**Распределение выручки по категориям**")
        by_category = (
            frame.groupby("category", as_index=False)
            .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"), units=("quantity", "sum"))
            .sort_values("revenue", ascending=False)
        )
        if by_category.empty:
            st.info("Нет данных.")
        else:
            total = by_category["revenue"].sum()
            by_category["share"] = by_category["revenue"] / total if total else 0.0
            by_category["avg_check"] = np.where(
                by_category["orders"] > 0, by_category["revenue"] / by_category["orders"], 0.0
            )
            max_share = float(by_category["share"].max()) or 1.0
            st.dataframe(
                by_category.rename(columns={
                    "category": "Категория", "revenue": "Выручка, ₽", "orders": "Заказов",
                    "units": "Единиц", "share": "Доля выручки", "avg_check": "Средний чек, ₽",
                }),
                hide_index=True,
                width="stretch",
                column_config={
                    "Категория": st.column_config.TextColumn("Категория", width="medium"),
                    "Выручка, ₽": st.column_config.NumberColumn("Выручка, ₽", format="%.2f ₽"),
                    "Заказов": st.column_config.NumberColumn("Заказов", format="%d"),
                    "Единиц": st.column_config.NumberColumn("Единиц", format="%d"),
                    "Доля выручки": st.column_config.ProgressColumn(
                        "Доля выручки", format="%.1f %%", min_value=0.0, max_value=max_share,
                    ),
                    "Средний чек, ₽": st.column_config.NumberColumn("Средний чек, ₽", format="%.2f ₽"),
                },
            )
            st.caption("Полоса «Доля выручки» показывает вклад категории относительно лидера.")

    with right:
        st.markdown("**Описательная статистика чеков**")
        if frame.empty:
            st.info("Нет данных.")
        else:
            stats = frame["revenue"].describe()
            stats.index = ["Количество", "Среднее", "Стандартное отклонение", "Минимум",
                           "25-й перцентиль", "Медиана", "75-й перцентиль", "Максимум"]
            stats_frame = pd.DataFrame({
                "Показатель": stats.index,
                "Значение": stats.values,
            })
            st.dataframe(
                stats_frame,
                hide_index=True,
                width="stretch",
                column_config={"Значение": st.column_config.NumberColumn("Значение, ₽", format="%.2f ₽")},
            )
            st.markdown("**Топ-5 покупателей по выручке**")
            top_users = (
                frame.groupby(["user_id", "user_name"], as_index=False)
                .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
                .sort_values("revenue", ascending=False).head(5)
            )
            st.dataframe(
                top_users.rename(columns={
                    "user_id": "ID", "user_name": "Покупатель", "revenue": "Выручка, ₽", "orders": "Заказов",
                }),
                hide_index=True,
                width="stretch",
                column_config={
                    "ID": st.column_config.NumberColumn("ID", format="%d"),
                    "Покупатель": st.column_config.TextColumn("Покупатель"),
                    "Выручка, ₽": st.column_config.NumberColumn("Выручка, ₽", format="%.2f ₽"),
                    "Заказов": st.column_config.NumberColumn("Заказов", format="%d"),
                },
            )


def render_visualization_tab(frame, top_n):
    """Вкладка «Визуализация»."""
    st.subheader("Визуализация")
    st.caption(
        "Графики построены на Matplotlib и учитывают фильтры боковой панели. "
        "Красным выделены выходные дни и лидирующие категории."
    )

    if frame.empty:
        st.warning("Под текущие фильтры не попал ни один заказ. Снимите фильтры, чтобы увидеть графики.")
        return

    show_figure(plot_top_items(frame, top_n))
    st.divider()

    left, right = st.columns(2)
    with left:
        show_figure(plot_category_share(frame))
    with right:
        show_figure(plot_segment_orders(frame))
    st.divider()

    show_figure(plot_weekday_orders(frame))
    st.divider()

    left, right = st.columns(2)
    with left:
        show_figure(plot_daily_revenue(frame))
    with right:
        show_figure(plot_city_revenue(frame))


def render_quality_tab(frames, cleaned, issues, steps):
    """Вкладка «Качество данных»."""
    st.subheader("Качество данных")
    st.caption(
        "Проверки идут в порядке шаблона подготовки данных: схема и типы → пропуски → "
        "дубликаты → аномалии → выбросы → целостность связей."
    )

    summary = issue_summary(issues)
    columns = st.columns(3)
    columns[0].metric("Ошибок", number(summary["error"]), help="Критичные проблемы, влияющие на расчёты")
    columns[1].metric("Предупреждений", number(summary["warn"]), help="Данные считаются, но требуют внимания")
    columns[2].metric("Замечаний", number(summary["info"]), help="Для сведения, расчёты не затронуты")

    st.markdown("**Что было до очистки и что стало после**")
    comparison = pd.DataFrame([
        {
            "Таблица": TABLE_TITLES[key],
            "Строк в файле": number(len(frames[key])),
            "Строк после очистки": number(len(cleaned[key])),
            "Потеряно": number(len(frames[key]) - len(cleaned[key])),
        }
        for key in DATA_FILES
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")

    with st.expander("Что именно было сделано при очистке", expanded=True):
        any_step = False
        for key in DATA_FILES:
            steps_for_table = steps.get(key) or []
            st.markdown(f"**{TABLE_TITLES[key]}**")
            if steps_for_table:
                for text in steps_for_table:
                    st.markdown(f"- {text}")
                    any_step = True
            else:
                st.markdown("- Изменений не потребовалось: типы корректны, пропусков и дубликатов нет.")
        if not any_step:
            st.caption("Все три таблицы прошли проверку без изменений.")

    st.markdown("**Результаты проверок**")
    for level in ("error", "warn", "info"):
        level_issues = [item for item in issues if item["level"] == level]
        if not level_issues:
            continue
        icon = {"error": "🔴", "warn": "🟠", "info": "🔵"}[level]
        with st.expander(f"{icon} {LEVEL_TITLES[level]}: {len(level_issues)}", expanded=(level != "info")):
            for item in level_issues:
                st.markdown(f"**{item['title']}** — {item['detail']}")


def render_insights_tab(frame, metrics, issues, catalog_size):
    """Вкладка «Аналитические выводы»."""
    st.subheader("Аналитические выводы")
    st.caption(
        "Выводы формируются по отфильтрованным данным: текст меняется вместе с фильтрами, "
        "поэтому каждое утверждение можно проверить на графиках."
    )

    insights = build_insights(frame, metrics, issues, catalog_size)
    for index, text in enumerate(insights, start=1):
        st.markdown(f"{index}. {text}")

    st.divider()
    st.markdown("**Рекомендации для руководства**")

    if frame.empty:
        st.info("Снимите фильтры, чтобы получить рекомендации по конкретной выборке.")
        return

    weekday_names = sorted(set(frame["weekday_ru"].dropna()), key=lambda name: WEEKDAYS_RU.index(name))
    repeat_share = metrics["repeat_share"]
    not_sold = max(catalog_size - metrics["sold_items"], 0)
    discount = frame["discount_pct"].dropna()

    recommendations = [
        f"**Держать фокус на категории «{metrics['top_category']}».** Она даёт "
        f"{percent(metrics['top_category_share'])} выручки — на неё стоит направлять маркетинговый бюджет "
        "и контролировать наличие ключевых позиций.",
        f"**Управлять загрузкой по дням недели.** Наиболее нагруженные дни — "
        f"{', '.join(weekday_names)}: в них нужно усилить склад и поддержку, "
        "а в дни спада — проводить акции, чтобы выровнять поток.",
        f"**Развивать повторные покупки.** {percent(repeat_share)} покупателей уже возвращались"
        + (
            " — программа лояльности и напоминания дадут прирост без затрат на привлечение трафика."
            if repeat_share >= 0.4
            else " — но удержание пока слабое, сначала нужно разобраться с причинами оттока."
        ),
    ]
    if not discount.empty and discount.mean() > 10:
        recommendations.append(
            f"**Проверить прайс-лист.** Фактическая цена продажи в среднем на {discount.mean():.1f}\u00a0% "
            "ниже базовой цены каталога. Если это не запланированная акция, то либо обновляется "
            "справочник товаров, либо теряется маржа."
        )
    if not_sold:
        recommendations.append(
            f"**Пересмотреть неликвид.** {number(not_sold)} позиций каталога не продались за период — "
            "проверьте остатки, цены и размещение на сайте."
        )
    st.markdown("\n".join(f"{index}. {text}" for index, text in enumerate(recommendations, start=1)))


def render_about_tab(frames, frame):
    """Вкладка «О работе»: методология и состав этапов."""
    st.subheader("О работе")
    st.markdown(
        """
**Предмет анализа.** Эффективность работы интернет-магазина по трём CSV-файлам за период наблюдения.
Дашборд отвечает на вопросы: сколько заработано, кто покупает, что продаётся, когда и где,
и насколько качественные исходные данные.

**Этапы пайплайна в коде `app_v1.py`:**

| Этап | Что делает | Функция |
|---|---|---|
| Загрузка | Читает три CSV с перебором кодировок, кэширует результат | `load_data` |
| Диагностика | Схема, типы, пропуски, дубликаты, аномалии, выбросы IQR, целостность связей | `diagnose_raw` |
| Очистка | Приводит типы, заполняет текстовые пропуски, удаляет критичные строки | `clean_frame`, `build_cleaned` |
| Объединение | `orders` → `users` → `items` через `merge` по `user_id` и `item_id` | `build_dataframe` |
| Показатели | Считает KPI и строит разрезы для визуализации | `build_metrics` |
| Выводы | Формирует текстовые заключения по текущей выборке | `build_insights` |

**Решения по очистке данных:**

* строки с пропусками в идентификаторах, дате, цене или количестве **удаляются** — без них
  выручка посчиталась бы искажённо;
* пропуски в текстовых полях (название товара, город, поставщик) **заполняются** значением
  «Unknown» / «Неизвестно», чтобы сохранить заказ в аналитике;
* выбросы по сумме заказа **не трогаются по умолчанию** и обрезаются только по явному
  согласию пользователя (чекбокс в боковой панели);
* `base_price` трактуется как прайс до скидки, а `price_per_unit` — как фактическая цена продажи.

**Интерактивность.** Фильтры по периоду, категориям, городам, сегментам и товарам применяются
ко всем вкладкам одновременно; пересчёт метриков, графиков и выводов происходит мгновенно.
"""
    )
    st.caption(
        f"Исходные файлы: {', '.join(DATA_FILES.values())} · "
        f"строк в файлах: {number(sum(len(value) for value in frames.values()))} · "
        f"строк после очистки и фильтрации: {number(len(frame))}"
    )


def main():
    """Собирает приложение: данные → очистка → фильтры → вкладки."""
    frames_raw = load_data(*data_stamps())
    issues = diagnose_raw(frames_raw["orders"], frames_raw["users"], frames_raw["items"])

    # Базовый набор данных нужен сайдбару до того, как известно значение чекбокса
    cleaned_base, _ = build_cleaned(frames_raw, clip_outliers=False)
    sidebar_state = render_sidebar(frames_raw, cleaned_base)

    cleaned, steps = build_cleaned(frames_raw, clip_outliers=sidebar_state["clip_outliers"])
    frame = build_dataframe(cleaned)
    frame = filter_frame(
        frame,
        sidebar_state["date_range"],
        sidebar_state["categories"],
        sidebar_state["cities"],
        sidebar_state["segments"],
        sidebar_state["items"],
    )
    metrics = build_metrics(frame)
    catalog_size = int(cleaned["items"]["item_id"].nunique())

    render_header(frames_raw, frame, metrics)

    tabs = st.tabs([
        "Сырые данные", "Ключевые показатели", "Визуализация",
        "Качество данных", "Аналитические выводы", "О работе",
    ])
    with tabs[0]:
        render_raw_tab(frame)
    with tabs[1]:
        render_metrics_tab(frame, metrics)
    with tabs[2]:
        render_visualization_tab(frame, sidebar_state["top_n"])
    with tabs[3]:
        render_quality_tab(frames_raw, cleaned, issues, steps)
    with tabs[4]:
        render_insights_tab(frame, metrics, issues, catalog_size)
    with tabs[5]:
        render_about_tab(frames_raw, frame)


if __name__ == "__main__":
    main()