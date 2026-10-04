"""
Анализ эффективности работы интернет-магазина
Итоговая проектная работа: интерактивный дашборд на Streamlit

Структура:
  1. Контракт чтения файлов: разделитель, кодировка, типы ключей, форматы дат
  2. Загрузка данных из трёх CSV-файлов (@st.cache_data)
  3. Диагностика качества: типы, пропуски, дубликаты, аномалии, выбросы
  4. Очистка и приведение типов
  5. Объединение таблиц (validate="m:1") и расчёт агрегированных показателей
  6. Дашборд: сырые данные, ключевые показатели, визуализация, выводы

Запуск:  streamlit run app.py
"""

import os

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

# ----------------------------------------------------------------------------
# КОНТРАКТ ЧТЕНИЯ ФАЙЛОВ
#
# Единственное место, где описано, как устроена выгрузка. Всё остальное —
# диагностика, очистка, расчёты — берёт правила отсюда, а не угадывает их.
#
# Принцип «объявляй, а не угадывай» выбран не из эстетики. Перебор кодировок
# выглядит безобидно, но cp1251 и latin1 декодируют ЛЮБЫЕ байты без ошибки,
# поэтому битый utf-8 тихо превращается в мусор, и в отчёт попадают данные,
# которых в источнике нет. Ни разбор дат, ни разбор разделителей, ни выбор
# кодировки не догадываются: неизвестное значение считается битым и попадает
# в диагностику. Поменялась выгрузка — правится этот словарь.
# ----------------------------------------------------------------------------
FILE_SPECS = {
    "orders": {
        "file": "orders.csv",
        "sep": ",",
        "encoding": "utf-8",
        # Типы ключей объявляются при чтении (по шаблону Merge_по_ключам).
        # Nullable Int64 принимает и 10, и "10", и пустое значение.
        "dtypes": {"order_id": "Int64", "user_id": "Int64", "item_id": "Int64"},
        "dates": {"order_date": "%Y-%m-%d %H:%M:%S"},
    },
    "users": {
        "file": "users.csv",
        "sep": ",",
        "encoding": "utf-8",
        "dtypes": {"user_id": "Int64"},
        "dates": {"registration_date": "%Y-%m-%d"},
    },
    "items": {
        "file": "items.csv",
        "sep": ",",
        "encoding": "utf-8",
        "dtypes": {"item_id": "Int64"},
        "dates": {},
    },
}

# BOM допускаем всегда: это не смена кодировки, а метка файла из Excel
ENCODING_FALLBACK = {"utf-8": "utf-8-sig"}

# КОНТРАКТ ФОРМАТОВ ЧИСЕЛ — честно, включая то, что НЕ поддерживается.
# Поддерживается: 2999.98, "2 999,98" (пробелы и запятая как десятичный знак),
# а также символ валюты на конце или в начале: "2 999,98 ₽".
# Не поддерживается: "1.234,56" — точка как разделитель тысяч.
NUMERIC_CONTRACTS = {
    ("orders", "price_per_unit"): "десятичная точка: 2999.98 или 2 999,98",
    ("orders", "quantity"): "целое число: 2",
    ("items", "base_price"): "десятичная точка: 2999.98 или 2 999,98",
}
UNSUPPORTED_NUMERIC_FORMAT = "`1.234,56` (точка как разделитель тысяч)"

# Символы, которые можно молча убрать: валюта не часть числа, а мусор вокруг
# значения. Точка сюда НЕ входит намеренно — см. to_numeric_safe.
CURRENCY_SYMBOLS = ("₽", "руб.", "руб", "RUB", "rub", "€", "EUR", "$", "USD")

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
DATE_COLUMNS = {key: list(spec["dates"]) for key, spec in FILE_SPECS.items()}

# Доли нераспознанных значений, выше которых считаем, что источник сменил формат
DATE_BREAK_RATIO = 0.05
NUMERIC_BREAK_RATIO = 0.05

# Столбцы с числами: приводим к float
NUMERIC_COLUMNS = {
    "orders": ["quantity", "price_per_unit"],
    "users": [],
    "items": ["base_price"],
}

# Числа, которые нужны для расчётов, но не обязательны для самого товара.
# Их приводим к типу, но НЕ удаляем строку при пропуске: иначе товар выпал бы
# из справочника вместе со всеми его заказами.
OPTIONAL_NUMERIC_COLUMNS = {
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

# Ключи соединения: эти столбцы должны иметь одинаковый тип во всех таблицах,
# где они встречаются. Именно по ним идут merge.
JOIN_KEYS = {
    "orders": ["user_id", "item_id"],
    "users": ["user_id"],
    "items": ["item_id"],
}

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

# Уровни проблем: error — критично, warn — влияет на расчёты, info — для сведения
LEVEL_TITLES = {"error": "Ошибки", "warn": "Предупреждения", "info": "Замечания"}

# Пороги достаточности данных. Пока выборка меньше, закономерности не выдаются за выводы:
# описательные показатели остаются, а утверждения о закономерностях помечаются как гипотезы.
ANALYTICS_MIN_DAYS = 90
ANALYTICS_MIN_ORDERS = 500
WEEKDAY_MIN_REPEATS = 4


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


def delta_word(value, base):
    """Отклонение от среднего по базе со знаком: «+37 %», «-24 %»."""
    if not base:
        return "—"
    change = (float(value) - float(base)) / float(base)
    return f"{'+' if change >= 0 else '-'}{abs(change) * 100:.0f} %"


def counted_after(value, one, many):
    """Счётчик в родительном падеже: «из 1 позиции», «из 4 позиций», «из 11 позиций».

    После предлога «из» число всегда стоит в родительном падеже, поэтому
    формы 2–4 совпадают с множественным числом, а не с «позициями».
    """
    tail = int(value) % 100
    if 11 <= tail <= 14:
        return f"{int(value)} {many}"
    if tail % 10 == 1:
        return f"{int(value)} {one}"
    return f"{int(value)} {many}"


def concentration_text(group_count, top_share, noun_forms):
    """Сравнение доли топ-3 с равномерным распределением.

    При трёх и менее группах сравнение вырождается в тождество «100 % против
    100 %», и вывод о концентрации противоречил бы самим числам. В этом
    случае возвращается только сам факт: групп столько-то, доля столько-то.
    """
    groups = int(group_count)
    top = float(top_share)
    uniform = min(3, groups) / groups if groups else 0.0
    nominative = f"{number(groups)} {plural_ru(groups, *noun_forms)}"
    genitive = counted_after(groups, noun_forms[0], noun_forms[2])
    if groups <= 3:
        return (
            f"В выборке {nominative} — сравнение с равномерным распределением "
            "здесь ничего не добавляет."
        )
    ratio = top / uniform if uniform else 1.0
    if ratio >= 1.3:
        verdict = "выручка сконцентрирована"
    elif ratio > 1.0:
        verdict = "выручка распределена с умеренным перекосом в пользу лидеров"
    else:
        verdict = "выручка распределена равномерно"
    return (
        f"На топ-3 из {genitive} приходится {percent(top)} выручки против "
        f"{percent(uniform)} при равномерном распределении — {verdict}."
    )


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


def share(part, total):
    """
    Доля значения в итоге с защитой от нуля.

    При total == 0 возвращает «—», а не nan и не RuntimeWarning:
    на пустой выборке доля не определена, и показывать надо именно это.
    """
    total_value = float(total)
    if not np.isfinite(total_value) or total_value == 0:
        return "—"
    return percent(float(part) / total_value)


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


# Преобразование типов. Объявлены до диагностики, потому что диагностика
# проверяет данные ровно тем же преобразователем, что и очистка.


def strip_currency(text):
    """Убирает символы валюты с краёв значения: они не часть числа."""
    result = text.strip()
    changed = True
    while changed:
        changed = False
        for symbol in CURRENCY_SYMBOLS:
            if result.lower().startswith(symbol.lower()):
                result = result[len(symbol):].strip()
                changed = True
            if result.lower().endswith(symbol.lower()):
                result = result[: -len(symbol)].strip()
                changed = True
    return result


def to_numeric_safe(series):
    """
    Приводит столбец к числу по контракту NUMERIC_CONTRACTS.

    Осознанное ограничение: поддерживается только десятичная точка и запятая как
    десятичный знак (`2999.98` и `2 999,98`). Значение `1.234,56` НЕ разбирается.

    Почему так: чтобы принять точку как разделитель тысяч, пришлось бы решить, что
    значит `1.234` — это 1234 или 1.234. Однозначного ответа нет, значит любая
    догадка рано или поздно посчитает выручку, которой в источнике нет. Поэтому
    значение, которое нельзя разобрать однозначно, превращается в пропуск и
    попадает в диагностику с примером — это громко и сразу исправимо. Умный
    разбор был бы тихим и опасным.

    Символ валюты, наоборот, убирается: это однозначный мусор вокруг числа,
    где ошибки быть не может.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    cleaned = series.astype("object").where(series.notna(), None)
    return pd.to_numeric(
        pd.Series(
            [
                strip_currency(str(value)).replace("\u00a0", "").replace(" ", "").replace(",", ".")
                for value in cleaned
            ],
            index=series.index,
            dtype="object",
        ),
        errors="coerce",
    )


def date_format_of(key, column):
    """Возвращает ожидаемый формат даты для столбца или None, если контракт не задан."""
    return FILE_SPECS.get(key, {}).get("dates", {}).get(column)


def numeric_contract_of(key, column):
    """Человекочитаемое описание ожидаемого формата числа для сообщений."""
    return NUMERIC_CONTRACTS.get((key, column), "десятичная точка")


def count_numeric_breaks(frame, key, column):
    """Сколько значений числа не соответствуют контракту формата."""
    if column not in frame.columns:
        return 0, 0
    present = int(frame[column].notna().sum())
    if not present:
        return 0, 0
    parsed = to_numeric_safe(frame[column])
    broken = int(parsed.isna().sum() - frame[column].isna().sum())
    return broken, present


def to_datetime_strict(series, fmt):
    """
    Приводит столбец к datetime СТРОГО по формату из контракта FILE_SPECS.

    Нераспознанное значение становится NaT, а не «догадкой»: поломка источника
    попадает в диагностику и в счётчик потерь, а не подменяет данные тихо.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    cleaned = series.astype("object").where(series.notna(), None)
    return pd.to_datetime(cleaned, errors="coerce", format=fmt)


def as_datetime(series):
    """Готовый datetime отдаём как есть — разбирать его повторно незачем."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def count_date_breaks(frame, key, column):
    """Сколько значений даты не соответствуют контракту формата."""
    fmt = date_format_of(key, column)
    if column not in frame.columns:
        return 0, 0, None
    present = frame[column].notna().sum()
    if not present:
        return 0, 0, fmt
    parsed = to_datetime_strict(frame[column], fmt) if fmt else frame[column]
    broken = int(parsed.isna().sum() - frame[column].isna().sum())
    return broken, int(present), fmt


# ----------------------------------------------------------------------------
# Раздел 2. Загрузка данных
# ----------------------------------------------------------------------------


class SourceFileError(Exception):
    """Файл не соответствует контракту чтения. Сообщение готово для интерфейса."""

    def __init__(self, message, hint=""):
        super().__init__(message)
        self.message = message
        self.hint = hint


def read_csv_by_spec(path, key):
    """
    Читает CSV строго по контракту FILE_SPECS: указанные разделитель, кодировка и типы.

    Кодировка не перебирается. Если объявленная не подошла — это ошибка источника,
    и её надо показать, а не замаскировать: cp1251 и latin1 принимают любые байты,
    поэтому «успешное» чтение битого файла даёт мусор вместо данных.
    """
    spec = FILE_SPECS[key]
    encoding = spec["encoding"]
    try:
        return pd.read_csv(
            path,
            sep=spec["sep"],
            encoding=encoding,
            dtype=dict(spec["dtypes"]),
            low_memory=False,
        )
    except UnicodeDecodeError as error:
        fallback = ENCODING_FALLBACK.get(encoding)
        if fallback:
            # BOM-метка — не поломка данных, а особенность выгрузки из Excel
            try:
                return pd.read_csv(
                    path,
                    sep=spec["sep"],
                    encoding=fallback,
                    dtype=dict(spec["dtypes"]),
                    low_memory=False,
                )
            except Exception:  # noqa: BLE001 — ниже сообщаем про исходную ошибку
                pass
        raise SourceFileError(
            f"`{spec['file']}` не читается в кодировке {encoding}: {error}",
            hint=(
                f"Если файл сохранён в другой кодировке, поменяйте `encoding` для "
                f"таблицы **{TABLE_TITLES[key]}** в FILE_SPECS. Перебирать кодировки "
                f"автоматически нельзя: cp1251 и latin1 принимают любые байты и "
                f"превращают битый файл в молчаливый мусор."
            ),
        ) from error
    except pd.errors.EmptyDataError as error:
        raise SourceFileError(
            f"`{spec['file']}` пуст: в файле нет ни заголовка, ни данных.",
            hint="Проверьте, что выгрузка не обрезалась при сохранении.",
        ) from error


def diagnose_reading(frame, key):
    """
    Ловит ошибки формата, из-за которых read_csv отработал не так, как ожидалось.

    Главный случай — файл с разделителем «;»: pandas не падает, а собирает всю
    строку в один столбец. Тогда validate_schema сообщила бы «нет всех обязательных
    столбцов», хотя столбцы есть, просто слипшиеся. Поэтому ловим это здесь и
    объясняем настоящую причину.
    """
    if frame.shape[1] > 1:
        return None
    declared = FILE_SPECS[key]["sep"]
    return (
        f"`{FILE_SPECS[key]['file']}` прочитан как один столбец, хотя в нём "
        f"{len(REQUIRED_COLUMNS[key])} обязательных полей. Скорее всего, разделитель "
        f"в файле не `{declared}` — русский Excel по умолчанию сохраняет через `;`. "
        f"Поменяйте `sep` в FILE_SPECS."
    )


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
    for key in FILE_SPECS:
        spec = FILE_SPECS[key]
        path = os.path.join(BASE_DIR, spec["file"])
        if not os.path.exists(path):
            st.error(
                f"Не найден файл **{spec['file']}**. "
                f"Поместите `orders.csv`, `users.csv` и `items.csv` в одну папку с `app.py`."
            )
            st.stop()
        try:
            frames[key] = read_csv_by_spec(path, key)
        except SourceFileError as error:
            st.error(error.message, icon="🚫")
            if error.hint:
                st.info(error.hint)
            st.stop()
        except Exception as error:  # noqa: BLE001 — в интерфейсе показываем текст ошибки
            st.error(f"Не удалось прочитать `{spec['file']}`: {error}", icon="🚫")
            st.stop()
    return frames


def data_stamps():
    """Время изменения CSV-файлов — ключ кэша Streamlit."""
    return tuple(
        os.path.getmtime(os.path.join(BASE_DIR, FILE_SPECS[key]["file"]))
        if os.path.exists(os.path.join(BASE_DIR, FILE_SPECS[key]["file"]))
        else 0
        for key in FILE_SPECS
    )


# ----------------------------------------------------------------------------
# Раздел 3. Диагностика качества данных
# ----------------------------------------------------------------------------


def issue(level, title, detail):
    """Создаёт запись о проблеме."""
    return {"level": level, "title": title, "detail": detail}


def validate_schema(frames):
    """Возвращает по каждой таблице список обязательных столбцов, которых нет."""
    problems = {}
    for key, frame in frames.items():
        missing = [column for column in REQUIRED_COLUMNS[key] if column not in frame.columns]
        if missing:
            problems[key] = missing
    return problems


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

    # 3. Типы данных: проверяем ровно тем же преобразователем, что идёт в очистку.
    # Иначе диагностика и очистка расходятся во мнениях об одном и том же столбце.
    for column in DATE_COLUMNS[key]:
        if column not in frame.columns:
            continue
        broken, present, fmt = count_date_breaks(frame, key, column)
        if broken:
            sample = frame.loc[frame[column].notna()].head(3)[column].tolist()
            issues.append(issue(
                "error", f"Некорректные даты в `{column}`",
                f"{title}: {number(broken)} из {number(present)} значений не соответствуют "
                f"контракту `{fmt}`. Примеры: {', '.join(str(value) for value in sample)}.",
            ))
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue
        elif fmt:
            issues.append(issue(
                "warn", f"Дата хранится как текст в `{column}`",
                f"{title}: столбец имеет тип object, будет приведён к datetime по `{fmt}`.",
            ))

    optional_numeric = set(OPTIONAL_NUMERIC_COLUMNS.get(key, []))
    for column in NUMERIC_COLUMNS[key]:
        if column not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        # Считаем поломки через to_numeric_safe — ровно тот же путь, что и в очистке.
        # Прежняя проверка isinstance ругалась на "1 499,99", хотя это корректное число.
        parsed = to_numeric_safe(frame[column])
        bad_text = int(parsed.isna().sum() - frame[column].isna().sum())
        if bad_text:
            # Показываем именно те значения, которые не разобрались, а не первые
            # попавшиеся: иначе пример не объясняет, что именно считается битым.
            broken_mask = parsed.isna() & frame[column].notna()
            sample = frame.loc[broken_mask, column].head(3).tolist()
            issues.append(issue(
                "error", f"Нечисловые значения в `{column}`",
                f"{title}: {number(bad_text)} значений нельзя привести к числу. "
                f"Ожидается {numeric_contract_of(key, column)}. "
                f"Примеры: {', '.join(str(value) for value in sample)}.",
            ))
        else:
            issues.append(issue(
                "warn", f"Число хранится как текст в `{column}`",
                f"{title}: столбец имеет тип {frame[column].dtype}, будет приведён к float.",
            ))

    # Пропуски в необязательных числах не удаляют строку, а лишь обесценивают расчёт
    for column in optional_numeric:
        if column not in frame.columns:
            continue
        missing = int(frame[column].isna().sum())
        if missing:
            issues.append(issue(
                "warn", f"Пропуски в необязательном `{column}`",
                f"{title}: {number(missing)} значений пропущено. Товар сохранится, "
                f"но производные от этой колонки (например, скидка) посчитать нельзя.",
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
    order_dates = as_datetime(orders["order_date"])
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
    registration = as_datetime(registration)
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
            f"{number(int(valid.sum()))} заказов: отклонение от {discount.min():.1f}\u00a0% "
            f"до {discount.max():.1f}\u00a0%, в среднем {discount.mean():.1f}\u00a0%. "
            "Смысл поля base_price требует сверки с прайс-листом.",
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


def normalize_segments(series):
    """
    Приводит сегмент к каноническому виду: без пробелов и в нижнем регистре.

    `VIP`, `Vip`, ` vip ` и `vip` — это один и тот же сегмент. Без приведения к
    единому виду они попадают в четыре разные группы, и график показывает
    четыре категории там, где покупатель один.
    """
    return (
        series.astype("object")
        .where(series.notna(), None)
        .map(lambda value: str(value).strip().lower() if pd.notna(value) else pd.NA)
    )


def clean_frame(frame, key):
    """Приводит типы, заполняет пропуски в тексте и отбрасывает критичные строки."""
    result = frame.copy()
    steps = []

    for column in DATE_COLUMNS[key]:
        if column not in result.columns:
            continue
        fmt = date_format_of(key, column)
        broken, present, _ = count_date_breaks(result, key, column)
        result[column] = to_datetime_strict(result[column], fmt) if fmt else as_datetime(result[column])
        if broken:
            steps.append(
                f"`{column}`: не распознано по контракту `{fmt}` — {number(broken)} "
                f"из {number(present)}, такие строки отброшены"
            )

    for column in NUMERIC_COLUMNS[key]:
        if column in result.columns:
            broken, present = count_numeric_breaks(result, key, column)
            result[column] = to_numeric_safe(result[column])
            if broken:
                steps.append(
                    f"`{column}`: не соответствует контракту формата "
                    f"({numeric_contract_of(key, column)}) — {number(broken)} "
                    f"из {number(present)}, такие строки отброшены"
                )

    if "user_segment" in result.columns:
        result["user_segment"] = normalize_segments(result["user_segment"])

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
        if "quantity" in result.columns and "price_per_unit" in result.columns:
            before = len(result)
            valid = (result["quantity"] > 0) & (result["price_per_unit"] > 0)
            result = result[valid]
            removed = before - len(result)
            if removed:
                steps.append(f"Удалено заказов с quantity или price <= 0 — {number(removed)}")

    # Необязательные числа (base_price) НЕ являются поводом удалять товар:
    # у товара без базовой цены просто нельзя посчитать скидку, но сам товар
    # и все его заказы остаются в анализе.
    if key == "items":
        for column in OPTIONAL_NUMERIC_COLUMNS[key]:
            if column not in result.columns:
                continue
            unknown = int(result[column].isna().sum())
            if unknown:
                steps.append(
                    f"`{column}`: товаров без базовой цены — {number(unknown)} "
                    f"(товары сохранены, скидка по ним не рассчитана)"
                )

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


def align_key_types(cleaned):
    """
    Приводит ключи соединения к одному типу во всех таблицах.

    Читаем мы их с объявленным dtype, но если файл правили руками, ключ может
    оказаться строкой в одном файле и числом в другом — и pandas падает с
    «You are trying to merge on Int64 and str columns». Идентификатор по сути
    имя, поэтому обе стороны приводим к строке и отсекаем пробелы.

    Выравниваются только ключи соединения. `order_id` не трогаем: он не участвует
    в merge и остаётся числом, чтобы форматирование в таблице работало.
    """
    for frame_key, columns in JOIN_KEYS.items():
        frame = cleaned.get(frame_key)
        if frame is None:
            continue
        for column in columns:
            if column in frame.columns:
                frame[column] = (
                    frame[column].astype("object").map(
                        lambda value: str(value).strip() if pd.notna(value) else pd.NA
                    )
                )
    return cleaned


def build_dataframe(cleaned):
    """
    Объединяет заказы с пользователями и товарами и добавляет расчётные столбцы.

    Соединение — left, чтобы сохранить каждый заказ из orders.csv: строки без пары
    в справочниках уже удалены на этапе очистки, но left-join страхует от потери
    данных, если пользователь добавит записи вручную.

    validate="m:1" ставится по шаблону Merge_по_ключам: он запрещает молча размножить
    строки, если в справочнике вдруг окажутся два товара с одним item_id.
    """
    cleaned = align_key_types(cleaned)
    orders_before = len(cleaned["orders"])
    merged = cleaned["orders"].merge(
        cleaned["users"], on="user_id", how="left", validate="m:1"
    )
    merged = merged.merge(cleaned["items"], on="item_id", how="left", validate="m:1")

    # По шаблону «Проверка корректности результата»: left-join не должен менять
    # число строк. Если изменил — это размножение, а не соединение.
    if len(merged) != orders_before:
        raise ValueError(
            f"left-join изменил число строк: было {orders_before}, стало {len(merged)}. "
            f"Вероятно, в справочнике дубли ключа — но validate='m:1' этого не пропустил, "
            f"значит проблема в типах ключей."
        )

    # Соединение сделано по строкам, чтобы не зависеть от того, как ключ прочитан:
    # `10`, `"10"` и ` 10 ` должны стыковаться. Для показа возвращаем числа там,
    # где это возможно, иначе колонка в таблице отрисовалась бы пустотой.
    for column in ("user_id", "item_id"):
        if column in merged.columns:
            converted = pd.to_numeric(merged[column], errors="coerce")
            if converted.notna().all() and (converted.dropna() % 1 == 0).all():
                merged[column] = converted.astype("Int64")

    if "amount" in merged.columns:
        merged["revenue"] = merged["amount"]
    else:
        merged["revenue"] = merged["quantity"] * merged["price_per_unit"]

    # Календарные колонки строим только если дата заказа на месте. Иначе вместо
    # KeyError получаем пустые значения: дашборд должен пережить нарушенную схему
    # и показать диагностику, а не упасть с трейсбеком.
    if "order_date" in merged.columns:
        merged["weekday"] = merged["order_date"].dt.dayofweek
        merged["month"] = merged["order_date"].dt.to_period("M").astype("string")
        merged["day"] = merged["order_date"].dt.date
    else:
        for column in ("weekday", "month", "day"):
            merged[column] = pd.Series(pd.NA, index=merged.index, dtype="object")
    merged["weekday_ru"] = merged["weekday"].map(lambda value: WEEKDAYS_RU[value] if pd.notna(value) else None)
    merged["is_weekend"] = merged["weekday"].isin([5, 6])

    if "base_price" in merged.columns:
        merged["base_price"] = merged["base_price"].fillna(0.0)
        valid_base = merged["base_price"] > 0
        merged["discount_pct"] = np.where(
            valid_base,
            (1 - merged["price_per_unit"] / merged["base_price"]) * 100,
            np.nan,
        )
    else:
        merged["base_price"] = np.nan
        merged["discount_pct"] = np.nan
    if "registration_date" in merged.columns:
        merged["registration_date"] = pd.to_datetime(merged["registration_date"], errors="coerce")
    if "order_date" in merged.columns and "registration_date" in merged.columns:
        merged["days_since_registration"] = (merged["order_date"] - merged["registration_date"]).dt.days
    else:
        merged["days_since_registration"] = np.nan
    merged["category"] = merged["category"].fillna("Unknown")
    merged["city"] = merged["city"].fillna("Неизвестно")
    merged["user_segment"] = merged["user_segment"].fillna("Unknown")

    if "order_date" in merged.columns:
        merged = merged.sort_values(["order_date", "order_id"])
    else:
        merged = merged.sort_values(["order_id"])
    return merged.reset_index(drop=True)


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
        "sold_items": int(frame["item_id"].nunique()) if orders else 0,
        "avg_discount": float(frame["discount_pct"].mean(skipna=True)),
        "weekend_share": float(frame.loc[frame["is_weekend"], "order_id"].nunique() / orders) if orders else 0.0,
    }


def normalize_date_range(value):
    """
    Приводит значение виджета date_input к паре дат.

    Виджет отдаёт либо кортеж из двух дат, либо одну дату (если пользователь
    выбрал один день), либо пустое значение. Раньше код без проверки звал len()
    по объекту datetime.date и падал на одиночной дате.

    Возвращает (start, end, note), где note — пояснение для пользователя,
    если входное значение пришлось поправить.
    """
    if value is None:
        return None, None, None
    raw = list(value) if isinstance(value, (list, tuple)) else [value]
    dates = []
    for item in raw:
        if item is None or (isinstance(item, float) and np.isnan(item)):
            continue
        try:
            dates.append(pd.Timestamp(item).normalize())
        except (ValueError, TypeError):
            continue
    if not dates:
        return None, None, None
    start, end = min(dates), max(dates)
    note = None
    if start == end:
        note = f"Выбран один день — {start:%d.%m.%Y}, фильтр применится к нему."
    elif len(dates) == 2 and raw[0] is not None and raw[1] is not None:
        first, second = pd.Timestamp(raw[0]).normalize(), pd.Timestamp(raw[1]).normalize()
        if second < first:
            note = (
                f"Начало позже конца ({first:%d.%m.%Y} → {second:%d.%m.%Y}), "
                f"диапазон переупорядочен."
            )
    return start, end, note


def filter_frame(frame, date_range, categories, cities, segments, items):
    """Применяет фильтры боковой панели к объединённому DataFrame."""
    result = frame
    start, end, _ = normalize_date_range(date_range)
    if start is not None and end is not None:
        stop = end + pd.Timedelta(days=1)
        result = result[(result["order_date"] >= start) & (result["order_date"] < stop)]
    if categories:
        result = result[result["category"].isin(categories)]
    if cities:
        result = result[result["city"].isin(cities)]
    if segments:
        result = result[result["user_segment"].isin(segments)]
    if items:
        result = result[result["item_name"].isin(items)]
    return result.reset_index(drop=True)


def catalog_scope(reference, categories, items):
    """Число позиций справочника, попадающих под фильтры боковой панели.

    Категория и товар сужают сам каталог, поэтому знаменатель для вывода о
    неликвиде обязан считаться по той же области, что и продажи: иначе при
    выбранной «Электронике» товары остальных категорий попадали бы в
    «не продавшиеся». Город, сегмент и даты каталог не ограничивают — товар,
    не купленный в Москве или в конкретный день, остаётся позицией каталога,
    и сравнение с полным справочником тут корректно.
    """
    scope = reference
    if categories:
        scope = scope[scope["category"].isin(categories)]
    if items:
        scope = scope[scope["item_name"].isin(items)]
    return int(scope["item_id"].nunique())


def filter_scope_note(categories, cities, segments, items):
    """Строка с активными фильтрами для подписи к числам выводов.

    Период в подпись не входит: он и так назван в каждом выводе. Без
    фильтров возвращается пустая строка, и скобки в тексте не появляются.
    """
    parts = []
    if categories:
        parts.append(f"категории — {', '.join(categories)}")
    if cities:
        parts.append(f"города — {', '.join(cities)}")
    if segments:
        parts.append(f"сегменты — {', '.join(segments)}")
    if items:
        parts.append(f"товары — {', '.join(items)}")
    return f"фильтр: {'; '.join(parts)}" if parts else ""


def catalog_scope_for(reference, categories, cities, segments, items):
    """Знаменатель для неликвида и подпись к нему по состоянию боковой панели.

    Собрано в одну функцию, чтобы `main` не мог передать в выводы полный каталог,
    посчитав область вручную и забрав половину состояния фильтров.
    """
    return (
        catalog_scope(reference, categories, items),
        filter_scope_note(categories, cities, segments, items),
    )


# ----------------------------------------------------------------------------
# Раздел 6. Аналитические выводы
# ----------------------------------------------------------------------------


def evidence_level(frame):
    """Оценивает, позволяет ли объём выборки делать выводы о закономерностях."""
    empty = {"days": 0, "orders": 0, "sufficient": False, "weekday_repeats": 0, "weekday_ok": False}
    if frame.empty or "order_date" not in frame.columns:
        return empty
    dates = pd.to_datetime(frame["order_date"], errors="coerce").dropna()
    if dates.empty:
        return empty
    start, end = dates.min().normalize(), dates.max().normalize()
    days = int((end - start).days) + 1
    span = pd.date_range(start, end, freq="D")
    repeats = int(np.bincount(span.dayofweek, minlength=7).min())
    orders = int(frame["order_id"].nunique())
    return {
        "days": days,
        "orders": orders,
        "sufficient": days >= ANALYTICS_MIN_DAYS and orders >= ANALYTICS_MIN_ORDERS,
        "weekday_repeats": repeats,
        "weekday_ok": repeats >= WEEKDAY_MIN_REPEATS,
    }


def weekday_profile(frame):
    """Заказы и выручка в пересчёте на один день недели с учётом числа дней в окне.

    Возвращает все семь строк, даже если по какому-то дню заказов не было вовсе.
    Это не косметика: без такой строки её календарные дни выпадали из знаменателя,
    и «на выходных 4.0 заказа в день» получалось делением заказов воскресенья на
    число воскресений вместо деления на число всех выходных. Пустой день честно
    даёт ноль, а не исчезает из расчёта.
    """
    dates = pd.to_datetime(frame["order_date"], errors="coerce").dropna() if "order_date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    counts = np.zeros(7, dtype=int)
    if not dates.empty:
        span = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="D")
        counts = np.bincount(span.dayofweek, minlength=7)
    if frame.empty:
        grouped = pd.DataFrame({"weekday_ru": [], "orders": [], "revenue": []})
    else:
        grouped = frame.groupby("weekday_ru", as_index=False).agg(
            orders=("order_id", "nunique"), revenue=("revenue", "sum")
        )
    profile = pd.DataFrame({"weekday_ru": list(WEEKDAYS_RU)}).merge(grouped, on="weekday_ru", how="left")
    profile["orders"] = profile["orders"].fillna(0).astype(int)
    profile["revenue"] = profile["revenue"].fillna(0.0)
    profile["days"] = [int(counts[index]) for index in range(7)]
    safe_days = profile["days"].replace(0, np.nan)
    profile["orders_per_day"] = profile["orders"] / safe_days
    profile["revenue_per_day"] = profile["revenue"] / safe_days
    return profile


def build_insights(frame, metrics, quality, catalog_size, scope_note=""):
    """Формирует выводы по данным. Каждый пункт опирается на посчитанные числа.

    `catalog_size` — размер справочника в области активных фильтров (см.
    `catalog_scope`), `scope_note` — человекочитаемый перечень фильтров для
    подписи к числам, которые без неё выглядели бы как данные по всему
    каталогу.
    """
    insights = []
    if frame.empty:
        return ["Под текущие фильтры не попал ни один заказ — снимите часть фильтров, чтобы увидеть выводы."]
    scope = f" ({scope_note})" if scope_note else ""

    level = evidence_level(frame)
    profile = weekday_profile(frame)

    def add(text, hypothesis=False):
        """Добавляет пункт в выводы; гипотезы помечаются, пока объёма данных недостаточно."""
        prefix = "" if hypothesis is False or level["sufficient"] else "**Гипотеза.** "
        insights.append(prefix + text)

    by_category = (
        frame.groupby("category", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
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
    by_segment["avg_check"] = np.where(
        by_segment["orders"] > 0, by_segment["revenue"] / by_segment["orders"], 0.0
    )

    # 1. Ключевые деньги
    top_amounts = frame["revenue"].dropna().nlargest(3)
    top_orders = int(len(top_amounts))
    top_revenue = float(top_amounts.sum())
    check_gap = (
        (metrics["avg_check"] - metrics["median_check"]) / metrics["avg_check"]
        if metrics["avg_check"] else 0.0
    )
    money_text = (
        f"**Общая выручка — {money(metrics['revenue'])}** при {orders_word(metrics['orders'])}; "
        f"средний чек {money(metrics['avg_check'])}, медианный {money(metrics['median_check'])}. "
    )
    if top_orders >= 2 and metrics["revenue"]:
        tail = (
            "распределение правостороннее, медиана надёжнее среднего"
            if check_gap >= 0 else
            "распределение смещено влево, медиана надёжнее среднего"
        )
        money_text += (
            f"Средний чек на {abs(check_gap) * 100:.1f} % "
            f"{'выше' if check_gap >= 0 else 'ниже'} медианного, при этом {number(top_orders)} самых крупных "
            f"{plural_ru(top_orders, 'заказ', 'заказа', 'заказов')} дают "
            f"{share(top_revenue, metrics['revenue'])} выручки: {tail}. "
        )
    money_text += (
        f"Покупателей — {number(metrics['users'])}, в среднем "
        f"{metrics['orders_per_user']:.1f} заказа на человека."
    )
    add(f"Период наблюдения: {date_caption(frame['order_date'])} ({number(level['days'])} дн.). {money_text}")

    # 2. Концентрация по категориям
    leader = by_category.iloc[0]
    leader_items = int(frame.loc[frame["category"] == leader["category"], "item_id"].nunique())
    leader_share = float(leader["revenue"] / metrics["revenue"]) if metrics["revenue"] else 0.0
    add(
        f"Основная выручка приходится на категорию **«{leader['category']}»** — {money(leader['revenue'])} "
        f"({share(leader['revenue'], metrics['revenue'])} выручки) при {number(leader_items)} из "
        f"{counted_after(metrics['sold_items'], 'позиции', 'позиций')}. "
        + concentration_text(
            len(by_category),
            metrics["top_category_top3_share"],
            ("категория", "категории", "категорий"),
        )
    )
    risk_text = (
        "сбой поставок в этой категории снимет больше половины оборота"
        if leader_share > 0.5 else
        "даже полная остановка этой категории унесёт меньше половины оборота — "
        "зависимость от неё рабочая, а не критическая"
    )
    add(
        f"Зависимость от одного класса товара: на «{leader['category']}» приходится "
        f"{percent(metrics['top_category_share'])} выручки, поэтому {risk_text}. "
        "Маркетинговый бюджет и контроль остатков сюда оправданы, "
        "но остальные категории нужно развивать параллельно.",
        hypothesis=True,
    )

    # 3. Товар-лидер
    top_item = by_item.iloc[0]
    covered = metrics["sold_items"] >= catalog_size if catalog_size else True
    coverage_text = (
        "ассортимент выбранной области покрыт полностью, поэтому рост возможен только за счёт "
        "количества и цены, а не расширения каталога"
        if covered else
        "покрытие выбранной области неполное, часть позиций каталога в продажах не участвует"
    )
    add(
        f"Товар-лидер продаж — **{top_item['item_name']}**: {money(top_item['revenue'])} "
        f"({share(top_item['revenue'], metrics['revenue'])} выручки) при {number(top_item['units'])} шт. "
        f"и {orders_word(top_item['orders'])}. "
        f"В продажах участвует {number(metrics['sold_items'])} из "
        f"{counted_after(catalog_size, 'позиции', 'позиций')} каталога{scope} — {coverage_text}."
    )

    # 4. День недели: сравниваем дни, а не календарные даты
    if not profile.empty and level["weekday_ok"]:
        peak_orders = profile.loc[profile["orders_per_day"].idxmax()]
        peak_revenue = profile.loc[profile["revenue_per_day"].idxmax()]
        weekend = profile[profile["weekday_ru"].isin(WEEKEND)]
        weekdays = profile[~profile["weekday_ru"].isin(WEEKEND)]
        weekend_days = int(weekend["days"].sum())
        weekday_days = int(weekdays["days"].sum())
        weekend_per_day = weekend["orders"].sum() / weekend_days if weekend_days else 0.0
        weekday_per_day = weekdays["orders"].sum() / weekday_days if weekday_days else 0.0
        base_orders = metrics["orders"] / level["days"] if level["days"] else 0.0
        base_revenue = metrics["revenue"] / level["days"] if level["days"] else 0.0
        weekday_text = (
            f"Пик по количеству заказов — **{peak_orders['weekday_ru']}**: "
            f"{peak_orders['orders_per_day']:.1f} заказов в день против {base_orders:.1f} в среднем по окну. "
            f"Пик по выручке — **{peak_revenue['weekday_ru']}**: "
            f"{money(peak_revenue['revenue_per_day'])} в день против {money(base_revenue)} в среднем. "
            f"На выходных {weekend_per_day:.1f} заказов в день против {weekday_per_day:.1f} в будни"
        )
        if abs(weekend_per_day - weekday_per_day) < 0.5 * base_orders:
            weekday_text += (
                " — разница укладывается в случайные колебания, оснований считать выходные "
                "просевающими нет."
            )
        elif weekend_per_day > weekday_per_day:
            weekday_text += " — выходные выглядят сильнее будних, но разница невелика."
        else:
            weekday_text += " — будни выглядят сильнее выходных, но разница невелика."
        add(weekday_text, hypothesis=True)
    else:
        spread = profile["orders_per_day"].dropna() if "orders_per_day" in profile.columns else profile["orders"]
        low = f"{spread.min():.1f}" if len(spread) else "—"
        high = f"{spread.max():.1f}" if len(spread) else "—"
        add(
            f"Режим недели определить нельзя: в окне из {number(level['days'])} дн. минимальное число "
            f"повторов дня недели — {number(level['weekday_repeats'])}, для сравнения нужно минимум "
            f"{number(WEEKDAY_MIN_REPEATS)}. Наблюдаемый разброс {low}–{high} заказов в день лежит "
            "в пределах случайных колебаний, поэтому вывод о пике делать нельзя."
        )

    # 5. Покупки внутри окна
    per_user = frame.groupby("user_id")["order_id"].nunique()
    repeat_users = int((per_user > 1).sum())
    add(
        f"Доля покупателей с повторной покупкой внутри окна — {percent(metrics['repeat_share'])}: "
        f"{number(repeat_users)} из {number(len(per_user))} оформили больше одного заказа за период "
        f"{date_caption(frame['order_date'])}. Это доля покупателей с несколькими заказами, а не "
        "показатель удержания: истории покупок до начала окна в данных нет, поэтому возвращаемость "
        "невозможно отделить от обычной частоты покупок.",
        hypothesis=True,
    )

    # 6. География
    city_leader = by_city.iloc[0]
    add(
        f"География продаж: **«{city_leader['city']}»** приносит {money(city_leader['revenue'])} "
        f"({share(city_leader['revenue'], metrics['revenue'])} выручки). "
        + concentration_text(
            len(by_city),
            float(by_city.head(3)["revenue"].sum() / metrics["revenue"]) if metrics["revenue"] else 0.0,
            ("город", "города", "городов"),
        )
    )

    # 7. Сегменты
    segment_leader = by_segment.loc[by_segment["revenue"].idxmax()]
    by_check = by_segment.sort_values("avg_check", ascending=False)
    checks = ", ".join(
        f"{row['segment_ru']} — {money(row['avg_check'])}" for _, row in by_check.iterrows()
    )
    add(
        f"По выручке лидирует сегмент **{segment_leader['segment_ru']}** — "
        f"{money(segment_leader['revenue'])} "
        f"({share(segment_leader['revenue'], metrics['revenue'])}). "
        f"Средний чек по сегментам: {checks}."
    )
    if len(by_check) >= 2:
        richest = by_check.iloc[0]
        poorest = by_check.iloc[-1]
        add(
            f"Средний чек не растёт вместе с лояльностью: лучший — у сегмента "
            f"«{richest['segment_ru']}» ({money(richest['avg_check'])}, "
            f"{delta_word(richest['avg_check'], metrics['avg_check'])} к среднему по базе), худший — "
            f"у «{poorest['segment_ru']}» ({money(poorest['avg_check'])}, "
            f"{delta_word(poorest['avg_check'], metrics['avg_check'])}). Самые дорогие покупки делают "
            "те, кто ещё не стал постоянным.",
            hypothesis=True,
        )

    # 8. Ценовая дисциплина
    discount = frame["discount_pct"].dropna()
    if not discount.empty:
        ratio = (frame["base_price"] / frame["price_per_unit"])
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        below = int((frame["price_per_unit"] < frame["base_price"]).sum())
        positions = int(frame["item_id"].nunique())
        checks_total = int(len(frame))
        price_text = (
            f"Фактическая цена продажи ниже базовой цены каталога в среднем на "
            f"**{discount.mean():.1f} %** "
            f"(разброс {discount.min():.1f}–{discount.max():.1f} %)."
        )
        if not ratio.empty:
            price_text += (
                f" Отношение `base_price` к цене продажи держится в диапазоне "
                f"{ratio.min():.1f}×–{ratio.max():.1f}× при медиане {ratio.median():.1f}×, "
                f"и цена ниже базовой в {number(below)} из {number(checks_total)} "
                f"{plural_ru(checks_total, 'заказе', 'заказах', 'заказах')} "
                f"({number(positions)} {plural_ru(positions, 'позиции', 'позиции', 'позиций')} каталога)."
            )
        add(price_text)
        add(
            "Равномерной скидочной лестницы такого вида не бывает: `base_price` почти наверняка "
            "задан в другом масштабе, чем фактическая цена продажи. До выяснения природы поля "
            "расчёты маржи и скидки будут неверными — это первоочередная проверка.",
            hypothesis=True,
        )

    # 9. Товары без продаж
    # Короткое окно не позволяет судить о неликвиде: за 10 дней позиция могла просто
    # не попасть в спрос. Молчащий счёт в таком случае вводит в заблуждение сильнее,
    # чем отсутствие вывода, поэтому при недостаточном окне пункт не показывается.
    not_sold = max(catalog_size - metrics["sold_items"], 0)
    if not_sold and level["days"] >= ANALYTICS_MIN_DAYS:
        insights.append(
            f"За выбранный период не продались {number(not_sold)} из "
            f"{counted_after(catalog_size, 'позиции', 'позиций')} каталога{scope}. "
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
# Раздел 9. Сборка интерфейса
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
        _, _, date_note = normalize_date_range(date_range)
        if date_note:
            sidebar.info(date_note)

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

    dataset_level = evidence_level(cleaned_default["orders"])
    if not dataset_level["sufficient"]:
        sidebar.caption(
            f"Данных мало для выводов о закономерностях: {number(dataset_level['days'])} дн., "
            f"{orders_word(dataset_level['orders'])}. Описательные показатели точны, "
            "закономерности помечены как гипотезы."
        )

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
| `base_price` | items | Базовая цена из справочника товаров |
| `revenue` | расчёт | `quantity × price_per_unit` — выручка по заказу |
| `discount_pct` | расчёт | Отличие фактической цены от базовой, % |
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
        if frame.empty:
            st.info("Нет данных.")
        else:
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

    date_breaks = collect_date_breaks(frames)
    if date_breaks:
        st.warning(
            "Источник, похоже, сменил формат даты — часть значений не соответствует контракту",
            icon="⚠️",
        )
        st.dataframe(
            pd.DataFrame(date_breaks),
            hide_index=True,
            width="stretch",
            column_config={
                "Таблица": st.column_config.TextColumn("Таблица"),
                "Столбец": st.column_config.TextColumn("Столбец"),
                "Ожидается": st.column_config.TextColumn("Ожидается"),
                "Не распознано": st.column_config.TextColumn("Не распознано"),
                "Доля": st.column_config.TextColumn("Доля"),
            },
        )
        st.caption(
            "Разбор дат строгий: значение не угадывается, а считается битым. "
            f"Если формат выгрузки изменился намеренно, поправьте `dates` в FILE_SPECS. "
            f"Порог срабатывания — {percent(DATE_BREAK_RATIO, 0)} нераспознанных значений."
        )

    numeric_breaks = collect_numeric_breaks(frames)
    if numeric_breaks:
        st.error("Источник изменил формат чисел", icon="🚫")
        st.caption(
            f"Больше {percent(NUMERIC_BREAK_RATIO, 0)} значений не соответствует объявленному "
            f"формату. Расчёты остановлены: продолжать с молчаливыми пропусками опаснее, "
            f"чем остановиться."
        )
        st.dataframe(
            pd.DataFrame(numeric_breaks),
            hide_index=True,
            width="stretch",
            column_config={
                "Таблица": st.column_config.TextColumn("Таблица"),
                "Столбец": st.column_config.TextColumn("Столбец"),
                "Ожидается": st.column_config.TextColumn("Ожидается"),
                "Не распознано": st.column_config.TextColumn("Не распознано"),
                "Доля": st.column_config.TextColumn("Доля"),
            },
        )
        st.warning(
            f"Формат с точкой как разделителем тысяч ({UNSUPPORTED_NUMERIC_FORMAT}) "
            f"не поддерживается сознательно: `1.234` — это 1234 или 1.234, и выбрать "
            f"безопасно нельзя. Если данные пришли именно в таком виде, приведите их "
            f"к точке как десятичному знаку в источнике."
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
        for key in FILE_SPECS
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")

    with st.expander("Что именно было сделано при очистке", expanded=True):
        any_step = False
        for key in FILE_SPECS:
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


def collect_date_breaks(frames):
    """Собирает сводку по столбцам дат, где доля нераспознанных превысила порог."""
    rows = []
    for key, frame in frames.items():
        for column in DATE_COLUMNS.get(key, []):
            if column not in frame.columns:
                continue
            broken, present, fmt = count_date_breaks(frame, key, column)
            if not present:
                continue
            ratio = broken / present
            if ratio > DATE_BREAK_RATIO:
                rows.append({
                    "Таблица": TABLE_TITLES[key],
                    "Столбец": f"`{column}`",
                    "Ожидается": f"`{fmt}`",
                    "Не распознано": f"{number(broken)} из {number(present)}",
                    "Доля": percent(ratio),
                })
    return rows


def collect_numeric_breaks(frames, keys=None):
    """Собирает сводку по столбцам с числами, где доля нераспознанных превысила порог."""
    wanted = set(keys) if keys is not None else set(NUMERIC_COLUMNS)
    rows = []
    for key in sorted(wanted):
        frame = frames.get(key)
        if frame is None:
            continue
        for column in NUMERIC_COLUMNS.get(key, []):
            broken, present = count_numeric_breaks(frame, key, column)
            if not present:
                continue
            ratio = broken / present
            if ratio > NUMERIC_BREAK_RATIO:
                rows.append({
                    "Таблица": TABLE_TITLES[key],
                    "Столбец": f"`{column}`",
                    "Ожидается": numeric_contract_of(key, column),
                    "Не распознано": f"{number(broken)} из {number(present)}",
                    "Доля": percent(ratio),
                })
    return rows


def render_insights_tab(frame, metrics, issues, catalog_size, scope_note=""):
    """Вкладка «Аналитические выводы»."""
    st.subheader("Аналитические выводы")
    st.caption(
        "Выводы формируются по отфильтрованным данным: текст меняется вместе с фильтрами, "
        "поэтому каждое утверждение можно проверить на графиках."
    )

    level = evidence_level(frame)
    if not level["sufficient"]:
        st.warning(
            f"**Данных недостаточно для выводов о закономерностях.** "
            f"Период {number(level['days'])} дн., {orders_word(level['orders'])}. "
            f"Режим недели определяется при {number(WEEKDAY_MIN_REPEATS)} повторах каждого дня недели, "
            f"устойчивая аналитика спроса — при {number(ANALYTICS_MIN_DAYS)} днях и "
            f"{number(ANALYTICS_MIN_ORDERS)} заказах. Ниже показатели посчитаны точно, "
            "но утверждения о закономерностях помечены как **гипотезы** и требуют проверки "
            "на более длинном периоде."
        )

    insights = build_insights(frame, metrics, issues, catalog_size, scope_note)
    for index, text in enumerate(insights, start=1):
        st.markdown(f"{index}. {text}")

    st.divider()
    st.markdown("**Рекомендации для руководства**")

    if frame.empty:
        st.info("Снимите фильтры, чтобы получить рекомендации по конкретной выборке.")
        return

    profile = weekday_profile(frame)

    def marked(text):
        """Помечает рекомендацию как гипотезу, пока объёма данных недостаточно."""
        return text if level["sufficient"] else "**Гипотеза.** " + text

    recommendations = [
        marked(
            f"**Держать фокус на категории «{metrics['top_category']}».** Она даёт "
            f"{percent(metrics['top_category_share'])} выручки — на неё стоит направлять маркетинговый "
            "бюджет и контролировать наличие ключевых позиций."
        ),
    ]

    if level["weekday_ok"] and not profile.empty:
        ranked = profile.sort_values("revenue_per_day", ascending=False)
        top_days = ranked.head(3)
        weak_day = ranked.iloc[-1]
        base_revenue = metrics["revenue"] / level["days"] if level["days"] else 0.0
        days_text = ", ".join(
            f"{row['weekday_ru']} ({money(row['revenue_per_day'])}/день)" for _, row in top_days.iterrows()
        )
        recommendations.append(
            marked(
                f"**Управлять загрузкой по дням недели.** Наиболее доходные дни — {days_text} "
                f"против {money(base_revenue)} в среднем по окну; наименее доходный — "
                f"{weak_day['weekday_ru']} ({money(weak_day['revenue_per_day'])}/день). "
                "В дни пика стоит усилить склад и поддержку, в дни спада — планировать акции."
            )
        )
    else:
        recommendations.append(
            f"**Не планировать режим недели по этим данным.** Окно в {number(level['days'])} дн. "
            f"даёт меньше {number(WEEKDAY_MIN_REPEATS)} повторов каждого дня недели, поэтому "
            "различия между днями нельзя отличить от случайных колебаний. "
            "Для планирования смен и акций нужно собрать несколько месяцев истории."
        )

    per_user = frame.groupby("user_id")["order_id"].nunique()
    repeat_users = int((per_user > 1).sum())
    recommendations.append(
        marked(
            f"**Измерять возврат, а не оценивать по доле повторных.** "
            f"{number(repeat_users)} из {number(len(per_user))} покупателей сделали больше одного "
            f"заказа внутри окна ({percent(metrics['repeat_share'])}), но удержание по этим данным "
            "не считается — нужна история покупок до начала периода."
        )
    )

    discount = frame["discount_pct"].dropna()
    if not discount.empty and discount.mean() > 10:
        recommendations.append(
            f"**Сверить `base_price` с фактическими ценами — это приоритет.** Фактическая цена ниже "
            f"базовой в среднем на {discount.mean():.1f} %, и это касается всех позиций каталога. "
            "Если это не запланированная акция, то либо обновляется справочник товаров, "
            "либо теряется маржа."
        )
    not_sold = max(catalog_size - metrics["sold_items"], 0)
    # Тот же порог, что и в выводах: на коротком окне неликвид не определяется,
    # поэтому рекомендация «пересмотреть неликвид» была бы действием по шуму.
    if not_sold and level["days"] >= ANALYTICS_MIN_DAYS:
        scope = f" ({scope_note})" if scope_note else ""
        recommendations.append(
            f"**Пересмотреть неликвид.** За выбранный период не продались {number(not_sold)} из "
            f"{number(catalog_size)} позиций каталога{scope} — проверьте остатки, цены и размещение."
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

**Этапы пайплайна в коде `app.py`:**

| Этап | Что делает | Функция |
|---|---|---|
| Загрузка | Читает три CSV строго по контракту `FILE_SPECS` и кэширует результат | `load_data` |
| Проверка схемы | Сверяет состав столбцов с контрактом и останавливает приложение с объяснением | `validate_schema` |
| Диагностика | Схема, типы, пропуски, дубликаты, аномалии, выбросы IQR, целостность связей | `diagnose_raw` |
| Очистка | Приводит типы, заполняет текстовые пропуски, удаляет критичные строки | `clean_frame`, `build_cleaned` |
| Объединение | `orders` → `users` → `items` через `merge` с `validate="m:1"` | `build_dataframe` |
| Показатели | Считает KPI и строит разрезы для визуализации | `build_metrics` |
| Выводы | Формирует текстовые заключения по текущей выборке | `build_insights` |

**Решения по очистке данных:**

* строки с пропусками в идентификаторах, дате, цене или количестве **удаляются** — без них
  выручка посчиталась бы искажённо;
* пропуски в текстовых полях (название товара, город, поставщик) **заполняются** значением
  «Unknown» / «Неизвестно», чтобы сохранить заказ в аналитике;
* выбросы по сумме заказа **не трогаются по умолчанию** и обрезаются только по явному
  согласию пользователя (чекбокс в боковой панели);
* `base_price` — базовая цена из справочника товаров, `price_per_unit` — фактическая цена продажи;
  соотношение этих полей требует сверки, поэтому скидка по ним отдельно не утверждается;
* пропуск или неположительное значение в `base_price` **не удаляет товар**: товар и его заказы
  остаются в анализе, но скидка по ним не рассчитывается.

**Контракт форматов дат.** Формат выгрузки известен и фиксирован, поэтому даты разбираются
строго по заявленному шаблону, а не угадываются:

| Столбец | Формат |
|---|---|
| `orders.order_date` | `%Y-%m-%d %H:%M:%S` |
| `users.registration_date` | `%Y-%m-%d` |

Эвристики вида `dayfirst=True` или `format='mixed'` сознательно не используются. На датах вида
`03/04/2024` они дают 3 апреля или 4 марта в зависимости от версии pandas, и расхождение в месяц
незаметно — а это уже искажённые выводы. Вместо угадывания значение, не совпавшее с контрактом,
считается битым: строка отбрасывается, а на вкладке «Качество данных» появляется предупреждение,
если таких значений больше 5%. Сменился формат выгрузки — правится `dates` в `FILE_SPECS`.

**Контракт чтения файлов.** Для каждого файла объявлены разделитель, кодировка и типы ключей
(`FILE_SPECS`). Кодировка не перебирается, разделитель не угадывается: `cp1251` и `latin1` принимают
любые байты без ошибки, поэтому битый utf-8 при переборе тихо превратился бы в мусор, а файл с `;`
прочитался бы одним столбцом. Неизвестное значение считается битым и попадает в диагностику.

**Контракт форматов чисел и одно осознанное ограничение.** Поддерживаются `2999.98` и `2 999,98`
(пробелы и запятая как десятичный знак). Формат `1.234,56`, где точка — разделитель тысяч,
**не поддерживается сознательно**: `1.234` — это 1234 или 1.234, однозначно выбрать нельзя, а любая
догадка рано или поздно посчитает выручку, которой в источнике нет. Такие значения не угадываются,
а попадают в диагностику с примером; если их больше 5%, расчёты останавливаются с явным сообщением.
Приходят данные с точкой-разделителем тысяч — приводите их к точке как десятичному знаку в источнике.

**Строгость объединения.** Ключи приводятся к общему типу, поэтому `user_id` как строка в одном
файле и как число в другом больше не роняют `merge`. Все соединения выполняются с `validate="m:1"`
(по шаблону `Merge_по_ключам`): размножение строк из-за дублей справочника считается ошибкой,
а не нормой. После соединения проверяется, что число строк не изменилось.

**Интерактивность.** Фильтры по периоду, категориям, городам, сегментам и товарам применяются
ко всем вкладкам одновременно; пересчёт метриков, графиков и выводов происходит мгновенно.
"""
    )
    st.caption(
        f"Исходные файлы: {', '.join(spec['file'] for spec in FILE_SPECS.values())} · "
        f"строк в файлах: {number(sum(len(value) for value in frames.values()))} · "
        f"строк после очистки и фильтрации: {number(len(frame))}"
    )
    st.caption(
        "Версия: app.py — явный контракт чтения файлов, строгая проверка merge, "
        "контракт чисел с остановкой расчётов при поломке формата"
    )


def main():
    """Собирает приложение: данные → очистка → фильтры → вкладки."""
    frames_raw = load_data(*data_stamps())

    # Разделитель ловим первым: если файл с «;» слипся в один столбец,
    # validate_schema сообщила бы «нет всех обязательных столбцов» — то есть
    # указала бы на следствие вместо причины.
    separator_problems = {
        key: message
        for key, frame in frames_raw.items()
        if (message := diagnose_reading(frame, key))
    }
    if separator_problems:
        st.error("Не удалось продолжить анализ: файл прочитан не как таблица", icon="🚫")
        for message in separator_problems.values():
            st.markdown(message)
        st.caption(f"Папка с данными: {BASE_DIR}")
        st.stop()

    # Схема проверяется до всего остального. Без этого шага очистка упала бы
    # с KeyError на первом же отсутствующем столбце, и пользователь увидел бы
    # трейсбек вместо понятного объяснения, чего именно не хватает.
    schema_problems = validate_schema(frames_raw)
    if schema_problems:
        st.error("Не удалось продолжить анализ: в исходных файлах нарушена схема", icon="🚫")
        for key, missing in schema_problems.items():
            st.markdown(
                f"**{TABLE_TITLES[key]}** (`{FILE_SPECS[key]['file']}`) — "
                f"отсутствуют обязательные столбцы: {', '.join(f'`{name}`' for name in missing)}"
            )
        st.info(
            "Ожидается следующий набор столбцов:\n\n"
            + "\n\n".join(
                f"* **{TABLE_TITLES[key]}** — {', '.join(REQUIRED_COLUMNS[key])}"
                for key in FILE_SPECS
            )
        )
        st.caption(f"Папка с данными: {BASE_DIR}")
        st.stop()

    # Массовый сбой формата чисел останавливает расчёты: считать выручку по
    # частично распознанным ценам — это тихо выдумывать данные.
    numeric_problems = collect_numeric_breaks(frames_raw)
    if numeric_problems:
        st.error("Источник изменил формат чисел", icon="🚫")
        st.dataframe(
            pd.DataFrame(numeric_problems),
            hide_index=True,
            width="stretch",
            column_config={
                "Таблица": st.column_config.TextColumn("Таблица"),
                "Столбец": st.column_config.TextColumn("Столбец"),
                "Ожидается": st.column_config.TextColumn("Ожидается"),
                "Не распознано": st.column_config.TextColumn("Не распознано"),
                "Доля": st.column_config.TextColumn("Доля"),
            },
        )
        st.warning(
            f"Больше {percent(NUMERIC_BREAK_RATIO, 0)} значений не соответствует "
            f"объявленному формату. Расчёты остановлены: продолжить с молчаливыми "
            f"пропусками опаснее, чем остановиться."
        )
        st.info(
            f"Формат {UNSUPPORTED_NUMERIC_FORMAT} **не поддерживается сознательно**: "
            f"`1.234` — это 1234 или 1.234, и однозначно выбрать нельзя, поэтому "
            f"значение считается битым, а не угадывается. Ожидается: "
            f"`2999.98` или `2 999,98`. Если данные пришли с точкой как разделителем "
            f"тысяч, приведите их к точке как десятичному знаку в источнике."
        )
        st.caption(f"Папка с данными: {BASE_DIR}")
        st.stop()

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
    catalog_size, scope_note = catalog_scope_for(
        cleaned["items"],
        sidebar_state["categories"],
        sidebar_state["cities"],
        sidebar_state["segments"],
        sidebar_state["items"],
    )

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
        render_insights_tab(frame, metrics, issues, catalog_size, scope_note)
    with tabs[5]:
        render_about_tab(frames_raw, frame)


if __name__ == "__main__":
    main()