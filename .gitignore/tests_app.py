"""
Проверки пайплайна дашборда интернет-магазина

Запуск:
    python tests_app.py                          спросит, какой файл и что прогнать
    python tests_app.py --all                    спросит только файл, прогонит всё
    python tests_app.py -f app.py --all          без диалога: файл и всё
    python tests_app.py -s invariants            только инварианты
    python tests_app.py -f app.py -s fuzz -n 200 фаззинг на 200 итераций
    python tests_app.py -f app.py -s all -v      всё с фактом по каждой проверке

Что здесь и зачем:

Сценарии проверяют «как я ожидаю себя повести» на конкретных поломках: на
копиях реальных данных подставляются заведомо дефектные значения, после чего
прогоняется весь путь «диагностика -> очистка -> объединение -> показатели».
Каждый сценарий заранее знает ожидаемый исход, поэтому расхождение видно сразу.

Инварианты проверяют другое: математические тождества, которые не могут
нарушиться из-за данных. Красный инвариант означает ошибку в коде, а не в выгрузке.

Фаззинг проверяет то, о чём мы не подумали: 500 случайных мутаций копий данных,
после которых требуется ровно два свойства — пайплайн не падает и не выдумывает
отрицательную выручку.

Все проверки работают на копиях и не трогают исходные файлы.
Код возврата: 0 — все проверки прошли, 1 — есть провалы.
"""

import argparse
import importlib.util
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_CANDIDATES = ("app.py", "app_v4.py")


def load_app(filename):
    """Загружает дашборд как модуль, не запуская интерфейс Streamlit."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise SystemExit(f"Файл не найден: {path}")
    spec = importlib.util.spec_from_file_location(f"dash_{filename[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


# Символы дашборда, которые нужны проверкам. Импорт поимённо, а не через префикс
# модуля: так тела сценариев остаются такими же, какими были внутри приложения.
def bind(app):
    names = (
        "BASE_DIR FILE_SPECS NUMERIC_BREAK_RATIO SourceFileError "
        "diagnose_raw diagnose_reading validate_schema read_csv_by_spec "
        "collect_numeric_breaks build_cleaned build_dataframe build_metrics "
        "filter_frame normalize_date_range "
        "money number percent share "
        "plot_top_items plot_category_share plot_weekday_orders "
        "plot_daily_revenue plot_city_revenue plot_segment_orders "
        "load_data data_stamps build_insights evidence_level weekday_profile "
        "catalog_scope filter_scope_note catalog_scope_for plural_ru counted_after "
        "concentration_text WEEKEND "
        "WEEKDAYS_RU ANALYTICS_MIN_DAYS ANALYTICS_MIN_ORDERS WEEKDAY_MIN_REPEATS"
    ).split()
    missing = [name for name in names if not hasattr(app, name)]
    if missing:
        raise SystemExit(
            "В дашборде нет функций, нужных проверкам: " + ", ".join(missing)
            + ". Возможно, файл повреждён или это не тот файл."
        )
    for name in names:
        globals()[name] = getattr(app, name)


# ---------------------------------------------------------------------------
# Помощники
# ---------------------------------------------------------------------------


def clone_frames(frames):
    """Глубокая копия таблиц: сценарий не должен трогать данные дашборда."""
    return {key: frame.copy(deep=True) for key, frame in frames.items()}


def run_pipeline(frames, clip_outliers=False):
    """Полный путь обработки: диагностика, очистка, объединение, показатели."""
    issues = diagnose_raw(frames["orders"], frames["users"], frames["items"])
    cleaned, steps = build_cleaned(frames, clip_outliers=clip_outliers)
    merged = build_dataframe(cleaned)
    metrics = build_metrics(merged)
    return {"issues": issues, "cleaned": cleaned, "steps": steps, "merged": merged, "metrics": metrics}


def issue_titles(issues, level=None, needle=None):
    """Заголовки диагностик, опционально отфильтрованные по уровню и слову."""
    return [
        item["title"] for item in issues
        if (level is None or item["level"] == level)
        and (needle is None or needle in item["title"])
    ]


def steps_mentioning(steps, key, needle):
    return any(needle in line for line in steps.get(key, []))


# ---------------------------------------------------------------------------
# Мутаторы: ломают копию данных заданным образом
# ---------------------------------------------------------------------------


def empty_orders(frames):
    frames["orders"] = frames["orders"].iloc[0:0].copy()
    return frames


def drop_order_date(frames):
    frames["orders"] = frames["orders"].drop(columns=["order_date"])
    return frames


def nan_quantity(frames):
    frames["orders"].loc[frames["orders"].index[0], "quantity"] = np.nan
    return frames


def nan_city(frames):
    frames["users"].loc[frames["users"].index[0], "city"] = np.nan
    return frames


def duplicate_order(frames):
    # Повторяется именно order_id, а не вся строка: полные дубликаты ловятся
    # отдельной проверкой, здесь мы тестируем отбрасывание по ключу.
    twin = frames["orders"].iloc[[0]].copy()
    twin["quantity"] = twin["quantity"] + 1
    frames["orders"] = pd.concat([frames["orders"], twin], ignore_index=True)
    return frames


def orphan_order(frames):
    frames["orders"].loc[frames["orders"].index[0], "user_id"] = 999999
    return frames


def zero_quantity(frames):
    frames["orders"].loc[frames["orders"].index[0], "quantity"] = 0
    return frames


def zero_price(frames):
    frames["orders"].loc[frames["orders"].index[0], "price_per_unit"] = -10
    return frames


def foreign_date_format(frames):
    frames["orders"]["order_date"] = frames["orders"]["order_date"].str.replace("-", "/", regex=False)
    return frames


def partly_broken_dates(frames):
    mask = frames["orders"].index % 4 == 0
    frames["orders"].loc[mask, "order_date"] = frames["orders"].loc[mask, "order_date"].str.replace(
        "-", "/", regex=False
    )
    return frames


def prices_as_text(frames):
    frames["orders"]["price_per_unit"] = frames["orders"]["price_per_unit"].map(
        lambda value: f"{value:,.2f}".replace(",", " ").replace(".", ",")
    )
    return frames


def zero_base_price(frames):
    frames["items"]["base_price"] = 0.0
    return frames


def nan_base_price(frames):
    mask = frames["items"].index % 2 == 0
    frames["items"].loc[mask, "base_price"] = np.nan
    return frames


def unknown_segment(frames):
    frames["users"].loc[frames["users"].index[0], "user_segment"] = "vip_2024"
    return frames


def single_dimension(frames):
    first_user = frames["users"]["user_id"].iloc[0]
    first_item = frames["items"]["item_id"].iloc[0]
    frames["orders"] = frames["orders"][
        (frames["orders"]["user_id"] == first_user) & (frames["orders"]["item_id"] == first_item)
    ]
    frames["users"] = frames["users"][frames["users"]["user_id"] == first_user]
    frames["items"] = frames["items"][frames["items"]["item_id"] == first_item]
    frames["users"]["city"] = frames["users"]["city"].iloc[0]
    frames["items"]["category"] = frames["items"]["category"].iloc[0]
    return frames


def extreme_values(frames):
    frames["orders"].loc[frames["orders"].index[0], "quantity"] = 10 ** 9
    frames["orders"].loc[frames["orders"].index[1], "price_per_unit"] = 10 ** 9
    return frames


def keys_as_strings(frames):
    """Ключи во всех трёх файлах как строки — регресс на str против int64 в merge."""
    for table, column in (("orders", "user_id"), ("users", "user_id"),
                          ("orders", "item_id"), ("items", "item_id")):
        frames[table][column] = frames[table][column].map(lambda value: str(value))
    return frames


def keys_with_spaces(frames):
    """Ключи с пробелами: идентификатор это имя, пробел в нём не имеет смысла."""
    for table, column in (("orders", "user_id"), ("users", "user_id"),
                          ("orders", "item_id"), ("items", "item_id")):
        frames[table][column] = frames[table][column].map(lambda value: f" {value} ")
    return frames


def one_key_as_float(frames):
    """Один ключ-идентификатор прочитан как float — регресс на 1.0 против 1."""
    frames["users"]["user_id"] = frames["users"]["user_id"].astype("float64")
    return frames


def dot_thousands_prices(frames):
    """Цены форматом `1.234,56`, который не поддерживается сознательно."""
    frames["orders"]["price_per_unit"] = frames["orders"]["price_per_unit"].map(
        lambda value: f"{value:,.2f}".replace(",", ".")
    )
    return frames


def currency_symbol_prices(frames):
    frames["orders"]["price_per_unit"] = frames["orders"]["price_per_unit"].map(
        lambda value: f"{value:.2f} руб."
    )
    return frames


def scientific_quantity(frames):
    frames["orders"]["quantity"] = frames["orders"]["quantity"].map(
        lambda value: f"{int(value):e}" if value and value > 0 else value
    )
    return frames


def sentinel_values(frames):
    """Значения-заглушки вместо пропусков: N/A, null, тире и дефис."""
    sentinels = ["N/A", "null", "—", "-"]
    frames["orders"]["price_per_unit"] = [
        sentinels[position % len(sentinels)] for position in range(len(frames["orders"]))
    ]
    return frames


def duplicate_reference_keys(frames):
    """В справочнике товаров два товара с одним item_id."""
    twin = frames["items"].iloc[[0]].copy()
    twin["item_name"] = "Дубль товара"
    frames["items"] = pd.concat([frames["items"], twin], ignore_index=True)
    return frames


def empty_items_reference(frames):
    frames["items"] = frames["items"].iloc[0:0].copy()
    return frames


def mixed_case_segments(frames):
    """Один сегмент записан четырьмя разными способами."""
    users = frames["users"]
    target = str(users["user_segment"].iloc[0])
    variants = [target.upper(), f" {target.capitalize()} ", target.capitalize(), target]
    users.loc[users.index[: len(variants)], "user_segment"] = variants
    return frames


def overpricing(frames):
    """Цена продажи выше базовой: скидка уходит в минус.

    Умножать на константу здесь нельзя: в данных базовая цена выше цены продажи
    до 37 раз, так что даже x2 не даёт наценки. Поэтому цену задаём от базовой
    явно — через сам товар.
    """
    base_by_item = frames["items"].set_index("item_id")["base_price"]
    frames["orders"]["price_per_unit"] = frames["orders"].apply(
        lambda row: float(base_by_item.get(row["item_id"], np.nan)) * 1.5
        if pd.notna(base_by_item.get(row["item_id"], np.nan))
        else row["price_per_unit"],
        axis=1,
    )
    return frames


def extra_and_reordered_columns(frames):
    """Лишний столбец и другой порядок колонок — схема должна устоять."""
    orders = frames["orders"]
    orders["служебный_столбец"] = "ignore me"
    frames["orders"] = orders[list(orders.columns)[::-1]]
    return frames


def read_with_semicolon(frames):
    """Файл с разделителем `;` — русский Excel по умолчанию пишет так."""
    path = os.path.join(BASE_DIR, FILE_SPECS["orders"]["file"])
    frames["orders"] = pd.read_csv(path, sep=";", low_memory=False)
    return frames


def build_figures(result):
    """Все шесть графиков дашборда на одном наборе данных."""
    frame = result["merged"]
    figures = [
        plot_top_items(frame), plot_category_share(frame), plot_weekday_orders(frame),
        plot_daily_revenue(frame), plot_city_revenue(frame), plot_segment_orders(frame),
    ]
    built = sum(1 for figure in figures if figure is not None)
    for figure in figures:
        if figure is not None:
            plt.close(figure)
    return figures, built


# ---------------------------------------------------------------------------
# Сценарии: поломка данных и ожидаемая реакция пайплайна
# ---------------------------------------------------------------------------


def check_clean_data(frames):
    before = len(frames["orders"])
    result = run_pipeline(frames)
    after = len(result["cleaned"]["orders"])
    fact = f"{before} -> {after} строк, выручка {money(result['metrics']['revenue'])}"
    return fact, after == before


def check_missing_column(frames):
    frames = drop_order_date(frames)
    problems = validate_schema(frames)
    reported = "order_date" in problems.get("orders", [])
    try:
        result = run_pipeline(frames)
        built = len(result["cleaned"]["orders"]) >= 0
    except Exception as error:
        return f"падает: {type(error).__name__}: {error}", False
    fact = f"в схеме: {problems.get('orders', 'нет')}"
    return fact, reported and built


def check_empty_orders(frames):
    result = run_pipeline(empty_orders(frames))
    metrics = result["metrics"]
    fact = f"заказов {number(metrics['orders'])}, выручка {money(metrics['revenue'])}"
    return fact, metrics["orders"] == 0 and metrics["revenue"] == 0


def check_single_row(frames):
    frames["orders"] = frames["orders"].iloc[[0]].copy()
    result = run_pipeline(frames)
    metrics = result["metrics"]
    fact = f"заказов {number(metrics['orders'])}, выручка {money(metrics['revenue'])}"
    return fact, metrics["orders"] == 1 and metrics["revenue"] > 0


def check_nan_quantity(frames):
    result = run_pipeline(nan_quantity(frames))
    fact = f"осталось {len(result['cleaned']['orders'])} строк"
    mentioned = steps_mentioning(result["steps"], "orders", "пропусками или некорректными")
    return fact, len(result["cleaned"]["orders"]) == len(frames["orders"]) - 1 and mentioned


def check_nan_city(frames):
    result = run_pipeline(nan_city(frames))
    kept = len(result["cleaned"]["orders"])
    filled = "Неизвестно" in set(result["cleaned"]["users"]["city"])
    fact = f"заказов {kept}, заглушка применена: {'да' if filled else 'нет'}"
    return fact, kept == len(frames["orders"]) and filled


def check_duplicate_order(frames):
    expected = len(frames["orders"])
    result = run_pipeline(duplicate_order(frames))
    left = len(result["cleaned"]["orders"])
    fact = f"{expected + 1} -> {left} строк"
    removed = steps_mentioning(result["steps"], "orders", "повторов по `order_id`")
    return fact, left == expected and removed


def check_orphan_order(frames):
    result = run_pipeline(orphan_order(frames))
    fact = f"осталось {len(result['cleaned']['orders'])} строк"
    removed = steps_mentioning(result["steps"], "orders", "без пользователя или товара")
    return fact, len(result["cleaned"]["orders"]) == len(frames["orders"]) - 1 and removed


def check_zero_quantity(frames):
    result = run_pipeline(zero_quantity(frames))
    fact = f"осталось {len(result['cleaned']['orders'])} строк"
    return fact, len(result["cleaned"]["orders"]) == len(frames["orders"]) - 1


def check_negative_price(frames):
    result = run_pipeline(zero_price(frames))
    fact = f"осталось {len(result['cleaned']['orders'])} строк"
    return fact, len(result["cleaned"]["orders"]) == len(frames["orders"]) - 1


def check_foreign_date_format(frames):
    result = run_pipeline(foreign_date_format(frames))
    issues = result["issues"]
    errors = issue_titles(issues, level="error", needle="order_date")
    fact = f"ошибок о датах: {len(errors)}, заказов осталось {len(result['cleaned']['orders'])}"
    return fact, bool(errors) and result["metrics"]["orders"] == 0


def check_partly_broken_dates(frames):
    frames = partly_broken_dates(frames)
    result = run_pipeline(frames)
    total = len(frames["orders"])
    kept = result["metrics"]["orders"]
    fact = f"распознано {kept} из {total}"
    return fact, kept == round(total * 0.75)


def check_prices_as_text(frames):
    result = run_pipeline(prices_as_text(frames))
    errors = issue_titles(result["issues"], level="error", needle="price_per_unit")
    kept = result["metrics"]["orders"]
    fact = f"ошибок о нечисловых: {len(errors)}, заказов {kept}"
    return fact, not errors and kept == len(frames["orders"])


def check_zero_base_price(frames):
    result = run_pipeline(zero_base_price(frames))
    items_left = len(result["cleaned"]["items"])
    orders_left = result["metrics"]["orders"]
    fact = f"товаров {items_left}, заказов {orders_left}"
    return fact, items_left == len(frames["items"]) and orders_left == len(frames["orders"])


def check_nan_base_price(frames):
    result = run_pipeline(nan_base_price(frames))
    items_left = len(result["cleaned"]["items"])
    orders_left = result["metrics"]["orders"]
    with_discount = int(result["merged"]["discount_pct"].notna().sum())
    fact = f"товаров {items_left}, заказов {orders_left}, со скидкой {with_discount}"
    return fact, items_left == len(frames["items"]) and with_discount < orders_left


def check_unknown_segment(frames):
    result = run_pipeline(unknown_segment(frames))
    notes = issue_titles(result["issues"], level="warn", needle="сегмент")
    errors = issue_titles(result["issues"], level="error", needle="сегмент")
    fact = f"warn: {len(notes)}, error: {len(errors)}"
    return fact, bool(notes) and not errors


def check_single_dimension(frames):
    result = run_pipeline(single_dimension(frames))
    figures, built = build_figures(result)
    fact = f"построено {built} из {len(figures)}"
    return fact, built == len(figures)


def check_extreme_values(frames):
    result = run_pipeline(extreme_values(frames))
    revenue = result["metrics"]["revenue"]
    fact = f"выручка {money(revenue)}"
    return fact, bool(np.isfinite(revenue))


def check_iqr_clipping(frames):
    plain = run_pipeline(frames)["metrics"]["revenue"]
    clipped_result = run_pipeline(frames, clip_outliers=True)
    clipped = clipped_result["metrics"]["revenue"]
    fact = f"{money(plain)} -> {money(clipped)}"
    reported = steps_mentioning(clipped_result["steps"], "orders", "Обрезано выбросов")
    return fact, clipped < plain and reported


def check_zero_revenue(frames):
    result = run_pipeline(empty_orders(frames))
    metrics = result["metrics"]
    fact = f"share(0, 0) = {share(0, metrics['revenue'])}"
    return fact, share(0, metrics["revenue"]) == "—"


def check_single_day_filter(frames):
    result = run_pipeline(frames)
    frame = result["merged"]
    day = frame["order_date"].iloc[0].date()
    filtered = filter_frame(frame, day, None, None, None, None)
    fact = f"{len(frame)} -> {len(filtered)} строк за {day:%d.%m.%Y}"
    return fact, 0 < len(filtered) < len(frame)


def check_reversed_date_range(frames):
    result = run_pipeline(frames)
    frame = result["merged"]
    last = frame["order_date"].max().date()
    first = frame["order_date"].min().date()
    filtered = filter_frame(frame, (last, first), None, None, None, None)
    _, _, note = normalize_date_range((last, first))
    fact = note or "без пояснения"
    return fact, len(filtered) == len(frame) and note is not None


def check_plots_on_empty(frames):
    result = run_pipeline(empty_orders(frames))
    figures, _ = build_figures(result)
    fact = f"пустых фигур: {sum(1 for figure in figures if figure is None)} из {len(figures)}"
    return fact, all(figure is None for figure in figures)


def check_keys_as_strings(frames):
    plain = run_pipeline(frames)["metrics"]["revenue"]
    result = run_pipeline(keys_as_strings(frames))
    revenue = result["metrics"]["revenue"]
    fact = f"выручка {money(plain)} -> {money(revenue)}"
    return fact, result["metrics"]["orders"] > 0 and abs(revenue - plain) < 0.01


def check_keys_with_spaces(frames):
    expected = run_pipeline(frames)["metrics"]["orders"]
    result = run_pipeline(keys_with_spaces(frames))
    fact = f"заказов {result['metrics']['orders']} из {expected}"
    return fact, result["metrics"]["orders"] == expected


def check_one_key_as_float(frames):
    expected = run_pipeline(frames)["metrics"]["orders"]
    result = run_pipeline(one_key_as_float(frames))
    fact = f"заказов {result['metrics']['orders']} из {expected}"
    return fact, result["metrics"]["orders"] == expected


def check_dot_thousands_prices(frames):
    original = frames["orders"]["price_per_unit"]
    # Точка появляется только у значений с разрядом тысяч, поэтому ломается часть,
    # а не все строки: 1.234 — это 1234 или 1.234, и выбрать нельзя.
    should_break = int((original >= 1000).sum())
    frames = dot_thousands_prices(frames)
    result = run_pipeline(frames)
    errors = issue_titles(result["issues"], level="error", needle="price_per_unit")
    drops = steps_mentioning(result["steps"], "orders", "price_per_unit")
    fact = (
        f"ожидалось поломок {should_break}, ошибок о нечисловых: {len(errors)}, "
        f"заказов осталось {result['metrics']['orders']} из {len(original)}"
    )
    # Цена не угадывается: такие заказы отбрасываются с явной ошибкой, а не
    # молча считаются по неверной цене.
    return fact, bool(errors) and bool(drops) and result["metrics"]["orders"] == len(original) - should_break


def check_currency_symbol_prices(frames):
    expected = run_pipeline(frames)["metrics"]["orders"]
    result = run_pipeline(currency_symbol_prices(frames))
    fact = f"заказов {result['metrics']['orders']} из {expected}"
    return fact, result["metrics"]["orders"] == expected


def check_scientific_quantity(frames):
    expected = run_pipeline(frames)["metrics"]["orders"]
    result = run_pipeline(scientific_quantity(frames))
    fact = f"заказов {result['metrics']['orders']} из {expected}"
    return fact, result["metrics"]["orders"] == expected


def check_sentinel_values(frames):
    frames = sentinel_values(frames)
    result = run_pipeline(frames)
    errors = issue_titles(result["issues"], level="error", needle="price_per_unit")
    # N/A и null pandas понимает сам, а «—» и «-» — нет: их надо увидеть и отбросить.
    fact = f"ошибок о нечисловых: {len(errors)}, заказов осталось {result['metrics']['orders']}"
    return fact, bool(errors) and result["metrics"]["orders"] == 0


def check_duplicate_reference_keys(frames):
    expected = run_pipeline(frames)["metrics"]["orders"]
    frames = duplicate_reference_keys(frames)
    cleaned = build_cleaned(frames)[0]
    merged = build_dataframe(cleaned)
    fact = f"товаров после очистки {len(cleaned['items'])}, заказов {len(merged)} из {expected}"
    return fact, len(merged) == expected


def check_empty_items_reference(frames):
    result = run_pipeline(empty_items_reference(frames))
    metrics = result["metrics"]
    fact = f"товаров 0, заказов {number(metrics['orders'])}, выручка {money(metrics['revenue'])}"
    return fact, metrics["orders"] == 0 and metrics["revenue"] == 0.0


def check_mixed_case_segments(frames):
    frames = mixed_case_segments(frames)
    result = run_pipeline(frames)
    segments = set(result["cleaned"]["users"]["user_segment"].dropna())
    fact = f"уникальных сегментов: {len(segments)} — {', '.join(sorted(segments))}"
    return fact, all(segment == segment.strip().lower() for segment in segments)


def check_overpricing(frames):
    result = run_pipeline(overpricing(frames))
    merged = result["merged"]
    minimum = float(merged["discount_pct"].min())
    fact = f"минимальная скидка {percent(minimum)}"
    return fact, bool(np.isfinite(result["metrics"]["revenue"])) and minimum < 0


def check_extra_and_reordered_columns(frames):
    plain = run_pipeline(frames)["metrics"]["revenue"]
    frames = extra_and_reordered_columns(frames)
    problems = validate_schema(frames)
    result = run_pipeline(frames)
    fact = f"нарушений схемы: {len(problems)}, выручка {money(plain)} -> {money(result['metrics']['revenue'])}"
    return fact, not problems and abs(result["metrics"]["revenue"] - plain) < 0.01


def check_semicolon_delimiter(frames):
    frames = read_with_semicolon(frames)
    message = diagnose_reading(frames["orders"], "orders")
    problems = validate_schema(frames)
    fact = (
        f"прочитан как {frames['orders'].shape[1]} столбец, "
        f"в схеме числится отсутствующими {len(problems.get('orders', []))}"
    )
    return fact, bool(message) and ";" in message and bool(problems.get("orders"))


def check_broken_utf8(frames):
    import tempfile

    del frames
    folder = tempfile.mkdtemp()
    try:
        path = os.path.join(folder, "broken.csv")
        with open(path, "wb") as handle:
            handle.write("order_id,user_id\n1,\xff\xfe\xfd1\n2,2\n".encode("latin1"))
        try:
            read_csv_by_spec(path, "orders")
        except SourceFileError as error:
            fact = f"SourceFileError: {error.message[:60]}..."
            return fact, "utf-8" in error.message
        return "прочитан без ошибки", False
    finally:
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        os.rmdir(folder)


def check_empty_csv_file(frames):
    import tempfile

    del frames
    folder = tempfile.mkdtemp()
    try:
        path = os.path.join(folder, "empty.csv")
        open(path, "w", encoding="utf-8").close()
        try:
            read_csv_by_spec(path, "orders")
        except SourceFileError as error:
            return error.message[:70], "пуст" in error.message
        return "прочитан без ошибки", False
    finally:
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        os.rmdir(folder)


def check_contract_reading(frames):
    del frames
    loaded = {}
    for key, spec in FILE_SPECS.items():
        path = os.path.join(BASE_DIR, spec["file"])
        loaded[key] = read_csv_by_spec(path, key)
    dtype_ok = all(
        str(loaded[key][column].dtype) == "Int64"
        for key in FILE_SPECS
        for column in FILE_SPECS[key]["dtypes"]
        if column in loaded[key].columns
    )
    shape_ok = all(loaded[key].shape[1] > 1 for key in FILE_SPECS)
    fact = f"формы {loaded['orders'].shape}, типы ключей Int64: {'да' if dtype_ok else 'нет'}"
    return fact, dtype_ok and shape_ok


def check_mass_numeric_break(frames):
    frames = dot_thousands_prices(frames)
    rows = collect_numeric_breaks(frames, keys={"orders"})
    detected = bool(rows)
    fact = f"столбцов с поломкой формата: {len(rows)}, порог {percent(NUMERIC_BREAK_RATIO, 0)}"
    return fact, detected and len(rows) == 1


def weekday_name_count(text):
    """Сколько разных названий дней недели упомянуто в тексте."""
    return sum(1 for name in WEEKDAYS_RU if name in text)


def stretch_to_long_window(merged, weeks=18):
    """Растягивает реальные данные на длинное окно для проверки порогов достаточности."""
    parts = []
    for shift in range(weeks):
        chunk = merged.copy()
        offset = pd.Timedelta(days=7 * shift)
        chunk["order_date"] = chunk["order_date"] + offset
        chunk["order_id"] = chunk["order_id"].astype(str) + f"_{shift}"
        chunk["user_id"] = chunk["user_id"].astype(str) + f"_{shift}"
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def collect_insights(merged, issues, catalog_size, kinds=False):
    """Выводы приложения для произвольного набора заказов."""
    return build_insights(merged, build_metrics(merged), issues, catalog_size, "", kinds=kinds)


def check_evidence_flags_short_window(frames):
    """Короткое окно должно распознаваться как недостаточное для выводов."""
    result = run_pipeline(clone_frames(frames))
    level = evidence_level(result["merged"])
    not_sufficient = level["sufficient"] is False
    not_weekday = level["weekday_ok"] is False
    fact = (
        f"дней {level['days']}, заказов {level['orders']}, "
        f"повторов дня недели {level['weekday_repeats']}, "
        f"sufficient={level['sufficient']}, weekday_ok={level['weekday_ok']}"
    )
    return fact, not_sufficient and not_weekday and level["orders"] > 0


def check_insights_mark_hypotheses(frames):
    """Пока данных мало, закономерности отделены от фактов."""
    data = run_insights_of(clone_frames(frames))
    marked = sum(1 for flag in data["kinds"] if flag)
    fact = f"выводов {len(data['texts'])}, из них закономерностей {marked}"
    return fact, marked > 0 and marked < len(data["texts"])


def check_insights_have_no_iqr(frames):
    """В выводах не должно остаться утверждений по критерию IQR."""
    result = run_insights_of(clone_frames(frames))
    joined = " ".join(result["texts"]).lower()
    fact = f"выводов {len(result['texts'])}, упоминаний IQR: {joined.count('iqr')}"
    return fact, "iqr" not in joined and "выброс" not in joined


def check_insights_no_all_weekdays(frames):
    """Ни один вывод не перечисляет все семь дней недели как нагруженные."""
    data = run_insights_of(clone_frames(frames))
    catalog_size = int(data["result"]["cleaned"]["items"]["item_id"].nunique())
    stretched = collect_insights(
        stretch_to_long_window(data["result"]["merged"]), data["result"]["issues"], catalog_size
    )
    short_worst = max((weekday_name_count(text) for text in data["texts"]), default=0)
    long_worst = max((weekday_name_count(text) for text in stretched), default=0)
    fact = f"максимум дней недели в выводе: короткое окно {short_worst}, длинное {long_worst}"
    return fact, short_worst < 5 and long_worst < 5


def check_long_window_drops_hypotheses(frames):
    """На длинном окне и большом объёме метки гипотез исчезают."""
    result = run_pipeline(clone_frames(frames))
    stretched = stretch_to_long_window(result["merged"])
    level = evidence_level(stretched)
    catalog_size = int(result["cleaned"]["items"]["item_id"].nunique())
    _, kinds = collect_insights(
        stretch_to_long_window(result["merged"]), result["issues"], catalog_size, kinds=True
    )
    marked = sum(1 for flag in kinds if flag)
    fact = (
        f"дней {level['days']}, заказов {level['orders']}, "
        f"sufficient={level['sufficient']}, закономерностей {marked}"
    )
    return fact, level["sufficient"] is True and level["weekday_ok"] is True and not marked


def run_insights_of(frames):
    """Прогон пайплайна и сборка выводов — общий шаг для сценариев про тексты."""
    result = run_pipeline(frames)
    catalog_size = int(result["cleaned"]["items"]["item_id"].nunique())
    texts, kinds = build_insights(
        result["merged"], result["metrics"], result["issues"], catalog_size, "", kinds=True
    )
    return {
        "result": result,
        "texts": texts,
        "kinds": kinds,
        "level": evidence_level(result["merged"]),
    }


def expected_catalog_size(reference, categories, items):
    """Знаменатель неликвида, посчитанный мимо `catalog_scope`.

    Нужен, чтобы проверка была независимой: если считать ожидаемое значение той же
    функцией, что и код приложения, то поломка в ней пройдёт незамеченной.
    """
    scope = reference
    if categories:
        scope = scope[scope["category"].isin(list(categories))]
    if items:
        scope = scope[scope["item_name"].isin(list(items))]
    return len(set(scope["item_id"].tolist()))


def duplicate_item_name(frames):
    """Две заказываемые позиции справочника делят одно название при разных item_id."""
    items = frames["items"].copy()
    items.loc[items.index[0], "item_name"] = items.loc[items.index[1], "item_name"]
    frames["items"] = items
    return frames


def invariant_weekday_profile_has_all_seven_days(result, frames):
    """Профиль недели всегда содержит семь дней, даже если по какому-то их нет."""
    merged = result["merged"]
    profile = weekday_profile(merged)
    empty_profile = weekday_profile(merged.iloc[0:0])
    level = evidence_level(merged)
    total = int(profile["days"].sum())
    fact = (
        f"строк {len(profile)}, сумма дней {total}, длина окна {level['days']}, "
        f"строк в пустом профиле {len(empty_profile)}"
    )
    ok = len(profile) == 7 and len(empty_profile) == 7 and total == level["days"]

    # День недели без заказов не должен исчезать из профиля вместе с календарными днями.
    stretched = stretch_to_long_window(merged)
    dates = pd.to_datetime(stretched["order_date"])
    holeless = stretched[dates.dt.dayofweek.ne(5)].reset_index(drop=True)
    if not holeless.empty:
        gap_level = evidence_level(holeless)
        gap_profile = weekday_profile(holeless)
        gap_days = int(gap_profile["days"].sum())
        fact += (
            f"; без субботы строк {len(gap_profile)}, сумма дней {gap_days}, "
            f"окно {gap_level['days']}"
        )
        ok = ok and len(gap_profile) == 7 and gap_days == gap_level["days"]
    return ok, fact


def check_weekend_rate_survives_missing_day(frames):
    """Пустой день недели не выбрасывает свои календарные дни из знаменателя."""
    result = run_pipeline(clone_frames(frames))
    stretched = stretch_to_long_window(result["merged"])
    dates = pd.to_datetime(stretched["order_date"])
    without_saturday = stretched[dates.dt.dayofweek.ne(5)].reset_index(drop=True)
    level = evidence_level(without_saturday)
    if not level["weekday_ok"]:
        return "окно не дотянуло до порога, проверка не выполнена", False
    texts = build_insights(
        without_saturday,
        build_metrics(without_saturday),
        result["issues"],
        int(result["cleaned"]["items"]["item_id"].nunique()),
        "",
    )
    span = pd.date_range(
        pd.to_datetime(without_saturday["order_date"]).min().normalize(),
        pd.to_datetime(without_saturday["order_date"]).max().normalize(),
        freq="D",
    )
    weekend_days = int((span.dayofweek >= 5).sum())
    weekend_orders = int(
        without_saturday.loc[
            pd.to_datetime(without_saturday["order_date"]).dt.dayofweek >= 5, "order_id"
        ].nunique()
    )
    expected = weekend_orders / weekend_days if weekend_days else 0.0
    profile = weekday_profile(without_saturday)
    fact = (
        f"строк в профиле {len(profile)}, выходных дней {weekend_days}, "
        f"заказов в выходные {weekend_orders}, верно {expected:.1f}"
    )
    weekdays = profile[~profile["weekday_ru"].isin(WEEKEND)]
    weekday_days = int(weekdays["days"].sum())
    weekday_orders = int(
        without_saturday.loc[
            pd.to_datetime(without_saturday["order_date"]).dt.dayofweek < 5, "order_id"
        ].nunique()
    )
    expected_weekday = weekday_orders / weekday_days if weekday_days else 0.0
    stated = [text for text in texts if "выходных" in text]
    if not stated:
        return fact + ", сравнение в тексте не найдено", False
    fact += f", будни верно {expected_weekday:.1f}"
    ok = f"На выходных {expected:.1f} заказов в день" in stated[0]
    ok = ok and f"в будни" in stated[0]
    return fact, ok


def check_catalog_scope_follows_filters(frames):
    """Знаменатель для неликвида сужается вместе с категориями и товарами."""
    result = run_pipeline(clone_frames(frames))
    reference = result["cleaned"]["items"]
    merged = result["merged"]
    full = catalog_scope(reference, None, None)
    category = str(reference.iloc[0]["category"])
    city = str(merged["city"].iloc[0])
    item = str(reference.iloc[0]["item_name"])
    by_category = catalog_scope(reference, [category], None)
    by_city = catalog_scope(reference, None, None)
    by_item = catalog_scope(reference, None, [item])
    facts = {
        "все": full,
        f"категория {category}": by_category,
        f"город {city}": by_city,
        f"товар {item}": by_item,
    }
    narrows = 1 <= by_category < full and by_item == 1
    return str(facts), narrows and by_city == full


def check_sold_items_counts_item_ids(frames):
    """sold_items считается по item_id: одноимённые позиции не сливаются в одну."""
    plain = run_pipeline(clone_frames(frames))
    baseline = plain["metrics"]["sold_items"]
    result = run_pipeline(duplicate_item_name(clone_frames(frames)))
    merged = result["merged"]
    by_id = int(merged["item_id"].nunique())
    by_name = int(merged["item_name"].nunique())
    fact = (
        f"обычные данные: {baseline}; после слияния названий "
        f"по item_id {by_id}, по item_name {by_name}, в показателях {result['metrics']['sold_items']}"
    )
    return fact, by_id > by_name and result["metrics"]["sold_items"] == by_id


def invariant_unsold_stays_inside_catalog_scope(result, frames):
    """Число неликвида считается по каталогу в области фильтров, а не по полному."""
    data = run_insights_of(clone_frames(frames))
    reference = data["result"]["cleaned"]["items"]
    # Неликвид показывается только на длинном окне, поэтому и проверять его нужно там.
    merged = stretch_to_long_window(data["result"]["merged"])
    category = str(reference.iloc[0]["category"])
    city = str(merged["city"].iloc[0])
    problems = []
    for label, cats, cities in (
        ("без фильтров", None, None),
        (f"категория {category}", [category], None),
        (f"категория {category} + город {city}", [category], [city]),
    ):
        frame = filter_frame(merged, None, cats, cities, None, None)
        size = expected_catalog_size(reference, cats, None)
        note = filter_scope_note(cats, cities, None, None)
        texts = build_insights(frame, build_metrics(frame), data["result"]["issues"], size, note)
        expected = max(size - build_metrics(frame)["sold_items"], 0)
        stated = [text for text in texts if "не продались" in text]
        if expected == 0:
            if stated:
                problems.append(f"{label}: заявлено при нулевом неликвиде")
            continue
        if not stated:
            problems.append(f"{label}: неливид {expected} не упомянут")
        elif f"{expected} из {size}" not in stated[0]:
            problems.append(f"{label}: в тексте нет «{expected} из {size}»")
    fact = f"нарушений: {problems or 'нет'}"
    return not problems, fact


def invariant_unsold_hidden_on_short_window(result, frames):
    """На окне короче порога вывод о неликвиде не показывается вовсе."""
    data = run_insights_of(clone_frames(frames))
    reference = data["result"]["cleaned"]["items"]
    merged = data["result"]["merged"]
    city = str(merged["city"].iloc[0])
    problems = []
    for label, source in (("короткое окно", merged), ("длинное окно", stretch_to_long_window(merged))):
        for suffix, cats, cities in (("", None, None), (" + город", None, [city])):
            frame = filter_frame(source, None, cats, cities, None, None)
            days = evidence_level(frame)["days"]
            size = expected_catalog_size(reference, cats, None)
            note = filter_scope_note(cats, cities, None, None)
            texts = build_insights(frame, build_metrics(frame), data["result"]["issues"], size, note)
            stated = [text for text in texts if "не продались" in text]
            enough = days >= ANALYTICS_MIN_DAYS
            if enough and not stated and size > build_metrics(frame)["sold_items"]:
                problems.append(f"{label}{suffix}: неливид есть, но не упомянут")
            if not enough and stated:
                problems.append(f"{label}{suffix}: неливид показан при {days} дн.")
    fact = f"нарушений: {problems or 'нет'}, порог {ANALYTICS_MIN_DAYS} дн."
    return not problems, fact


def invariant_filter_scope_note_is_explicit(result, frames):
    """Подпись к числам пуста без фильтров и называет каждый активный фильтр."""
    empty = filter_scope_note(None, None, None, None)
    only_city = filter_scope_note(None, ["Москва"], None, None)
    both = filter_scope_note(["Электроника"], ["Москва"], ["vip"], ["Скейтборд"])
    ok = empty == "" and "Москва" in only_city and "категории" not in only_city
    ok = ok and all(word in both for word in ("категории", "города", "сегменты", "товары"))
    fact = f"пусто='{empty}', город='{only_city}', всё='{both}'"
    return ok, fact


def invariant_plural_forms_after_numbers(result, frames):
    """Числительные 2–4 не склоняются как 5: «3 позиции», а не «3 позиций»."""
    cases = {
        1: ("позиции", "заказ"),
        2: ("позиции", "заказа"),
        3: ("позиции", "заказа"),
        4: ("позиции", "заказа"),
        5: ("позиций", "заказов"),
        11: ("позиций", "заказов"),
        21: ("позиции", "заказ"),
    }
    wrong = [
        value for value, (position, order) in cases.items()
        if plural_ru(value, "позиции", "позиции", "позиций") != position
        or plural_ru(value, "заказ", "заказа", "заказов") != order
    ]
    fact = f"неверных форм: {wrong or 'нет'}"
    return not wrong, fact


def check_scope_pair_follows_sidebar_state(frames):
    """Связка знаменателя и подписи считается по всему состоянию фильтров разом."""
    reference = run_pipeline(clone_frames(frames))["cleaned"]["items"]
    category = str(reference.iloc[0]["category"])
    item = str(reference.iloc[1]["item_name"])
    cases = {
        "пусто": (None, None, None, None),
        "категория": ([category], None, None, None),
        "город": (None, ["Москва"], None, None),
        "категория+товар": ([category], None, None, [item]),
    }
    wrong = []
    for label, (cats, cities, segs, its) in cases.items():
        size, note = catalog_scope_for(reference, cats, cities, segs, its)
        if size != expected_catalog_size(reference, cats, its):
            wrong.append(f"{label}: размер {size}")
        if (note == "") != (label == "пусто"):
            wrong.append(f"{label}: подпись '{note}'")
    fact = f"расхождений: {wrong or 'нет'}"
    return fact, not wrong


def check_insights_no_foreign_unsold(frames):
    """Текст о неликвиде не упоминает товары за пределами выбранных категорий."""
    data = run_insights_of(clone_frames(frames))
    reference = data["result"]["cleaned"]["items"]
    merged = stretch_to_long_window(data["result"]["merged"])
    category = str(reference.iloc[0]["category"])
    city = str(merged["city"].iloc[0])
    cats, cities = [category], [city]
    frame = filter_frame(merged, None, cats, cities, None, None)
    scoped = expected_catalog_size(reference, cats, None)
    full = expected_catalog_size(reference, None, None)
    sold = build_metrics(frame)["sold_items"]
    expected = max(scoped - sold, 0)
    note = filter_scope_note(cats, cities, None, None)
    texts = build_insights(frame, build_metrics(frame), data["result"]["issues"], scoped, note)
    stated = [text for text in texts if "не продались" in text]
    fact = (
        f"каталог в области фильтра {scoped}, полный {full}, продано {sold}, "
        f"ожидалось «{expected} из {scoped}», строк {len(stated)}"
    )
    if expected == 0:
        return fact, not stated
    return fact, bool(stated) and f"{expected} из {scoped}" in stated[0]


SCENARIOS = (
    ("Чистые данные не теряются", "все строки на месте, выручка не меняется", check_clean_data),
    ("Отсутствует обязательный столбец", "схема найдена, падения нет", check_missing_column),
    ("Пустая таблица заказов", "приложение работает, выручка 0", check_empty_orders),
    ("Одна строка заказов", "показатели считаются по единственной строке", check_single_row),
    ("Пропуск в quantity", "строка удалена, потеря отмечена в отчёте", check_nan_quantity),
    ("Пропуск в городе", "заказ сохранён, город заполнен заглушкой", check_nan_city),
    ("Дубликат order_id", "повтор удалён, осталась одна запись", check_duplicate_order),
    ("Заказ без пользователя", "сирота удалён на этапе связей", check_orphan_order),
    ("quantity = 0", "заказ отброшен как неположительный", check_zero_quantity),
    ("Отрицательная цена", "заказ отброшен как неположительный", check_negative_price),
    ("Источник сменил формат даты", "контракт нарушен, приложение не падает", check_foreign_date_format),
    ("Часть дат не соответствует контракту", "распознано ровно три четверти", check_partly_broken_dates),
    ("Цены текстом «1 499,99»", "числа распознаны, ложной ошибки нет", check_prices_as_text),
    ("base_price = 0", "товары и заказы сохранены", check_zero_base_price),
    ("base_price = пропуск", "товары сохранены, скидка не считается", check_nan_base_price),
    ("Неизвестный сегмент", "предупреждение, а не ошибка", check_unknown_segment),
    ("Один город, категория и покупатель", "все шесть графиков строятся", check_single_dimension),
    ("Экстремальные значения", "нет исключений, выручка конечна", check_extreme_values),
    ("Обрезка выбросов по IQR", "выручка уменьшается, границы зафиксированы", check_iqr_clipping),
    ("Нулевая выручка", "доля не определена, деления на ноль нет", check_zero_revenue),
    ("Фильтр по одной дате", "день применяется, а не игнорируется", check_single_day_filter),
    ("Перевёрнутый диапазон дат", "диапазон переупорядочен, фильтр применим", check_reversed_date_range),
    ("Графики на пустой выборке", "возвращают пусто, а не исключение", check_plots_on_empty),
    ("Ключи во всех файлах как строки", "merge не падает, выручка не меняется", check_keys_as_strings),
    ("Ключи с пробелами", "обрезка пробелов, стыковка сохраняется", check_keys_with_spaces),
    ("Ключ справочника прочитан как float", "типы выровнены, merge не падает", check_one_key_as_float),
    ("Цены форматом «1.234,56»", "отказ по контракту, а не угадывание", check_dot_thousands_prices),
    ("Цены с символом валюты", "символ отброшен, число распознано", check_currency_symbol_prices),
    ("quantity в научной нотации", "число распознано, строки сохранены", check_scientific_quantity),
    ("Значения-заглушки N/A, null, —, -", "считаются пропуском, а не числом", check_sentinel_values),
    ("Дубль item_id в справочнике товаров", "merge не размножает строки", check_duplicate_reference_keys),
    ("Пустой справочник товаров", "заказы отброшены, приложение живо", check_empty_items_reference),
    ("Регистр и пробелы в сегментах", "четыре записи схлопываются в один сегмент", check_mixed_case_segments),
    ("Цена выше базовой", "скидка отрицательная, расчёт не ломается", check_overpricing),
    ("Лишние и переставленные столбцы", "схема устояла, выручка не меняется", check_extra_and_reordered_columns),
    ("Разделитель «;» вместо запятой", "понятная диагностика, а не «нет столбцов»", check_semicolon_delimiter),
    ("Битый UTF-8", "ошибка чтения вместо молчаливого мусора", check_broken_utf8),
    ("Пустой CSV-файл", "понятное сообщение, не трассировка", check_empty_csv_file),
    ("Чтение по контракту на реальных файлах", "разделитель, кодировка и типы из FILE_SPECS", check_contract_reading),
    ("Массовый сбой формата чисел", "порог 5% ловит поломку источника", check_mass_numeric_break),
    ("Короткое окно данных", "помечается как недостаточное для выводов", check_evidence_flags_short_window),
    ("Метки гипотез", "закономерности помечены, показатели — нет", check_insights_mark_hypotheses),
    ("IQR убран из выводов", "в тексте выводов нет критерия IQR", check_insights_have_no_iqr),
    ("Дни недели не перечисляются все", "нет списка из семи дней как нагруженных", check_insights_no_all_weekdays),
    ("Длинное окно и большой объём", "метки гипотез исчезают сами", check_long_window_drops_hypotheses),
    ("Знаменатель неликвида идёт за фильтром", "категория и товар сужают каталог, город — нет", check_catalog_scope_follows_filters),
    ("sold_items по item_id", "одноимённые позиции не сливаются в одну", check_sold_items_counts_item_ids),
    ("Неливид не выходит за область фильтра", "в тексте только позиции выбранных категорий", check_insights_no_foreign_unsold),
    ("Знаменатель и подпись считаются разом", "размер и текст берутся из одного состояния", check_scope_pair_follows_sidebar_state),
    ("Пустой день недели не ломает расчёт", "выходные считаются по всем выходным дням", check_weekend_rate_survives_missing_day),
)


def run_scenarios(frames, verbose=True):
    """
    Прогоняет все сценарии и возвращает результаты.

    Любое исключение внутри сценария считается провалом, а не причиной
    сломать весь прогон. RuntimeWarning тоже считается провалом: он означает,
    что где-то в расчётах появилось деление на ноль или 0/0.
    """
    rows = []
    for name, expectation, function in SCENARIOS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fact, passed = function(clone_frames(frames))
            runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
            if runtime_warnings:
                passed = False
                fact = f"{fact} + RuntimeWarning: {runtime_warnings[0].message}"
        except Exception as error:
            fact, passed = f"{type(error).__name__}: {error}", False
        rows.append({
            "name": name,
            "expectation": expectation,
            "fact": fact,
            "passed": bool(passed),
        })
        if verbose:
            print(f"  {'OK  ' if passed else 'FAIL'}  {name}")
    return rows


# ---------------------------------------------------------------------------
# Инварианты: свойства, которые должны быть верны при ЛЮБЫХ данных
# ---------------------------------------------------------------------------


def invariant_shares_sum_to_100(result, frames):
    """Сумма долей категорий вместе с остатком даёт ровно 100%."""
    frame = result["merged"]
    by_category = frame.groupby("category")["revenue"].sum()
    total = float(frame["revenue"].sum())
    if not total:
        return True, f"выручка {money(total)}, доли не определены"
    covered = float(by_category.sum())
    rest = 1.0 - covered / total
    return abs(rest) < 1e-9, f"покрыто категориями {percent(covered / total)}, остаток {percent(rest)}"


def invariant_all_categories_filter(result, frames):
    """Фильтр по всем категориям не меняет выборку."""
    frame = result["merged"]
    categories = sorted(frame["category"].dropna().unique())
    filtered = filter_frame(frame, None, categories, None, None, None)
    same = len(filtered) == len(frame) and abs(filtered["revenue"].sum() - frame["revenue"].sum()) < 0.01
    return same, f"категорий {len(categories)}, строк {len(frame)} -> {len(filtered)}"


def invariant_row_order_does_not_matter(result, frames):
    """Перемешивание строк не меняет выручку."""
    frame = result["merged"]
    shuffled = frame.sample(frac=1.0, random_state=42)
    same = abs(float(shuffled["revenue"].sum()) - float(frame["revenue"].sum())) < 0.01
    return same, f"выручка {money(frame['revenue'].sum())} -> {money(shuffled['revenue'].sum())}"


def invariant_group_sum_equals_total(result, frames):
    """Сумма выручки по любым разрезам равна общей выручке."""
    frame = result["merged"]
    total = float(frame["revenue"].sum())
    by_city = float(frame.groupby("city")["revenue"].sum().sum())
    by_item = float(frame.groupby("item_name")["revenue"].sum().sum())
    same = abs(by_city - total) < 0.01 and abs(by_item - total) < 0.01
    return same, f"итого {money(total)}, по городам {money(by_city)}, по товарам {money(by_item)}"


def invariant_join_keeps_row_count(result, frames):
    """left-join не размножает и не теряет строки заказов."""
    orders_count = len(result["cleaned"]["orders"])
    merged_count = len(result["merged"])
    return orders_count == merged_count, f"заказов {orders_count}, после merge {merged_count}"


def invariant_order_id_unique_after_join(result, frames):
    """После объединения заказ остаётся одним и тем же заказом."""
    merged = result["merged"]
    unique = bool(merged["order_id"].is_unique)
    return unique, f"строк {len(merged)}, уникальных order_id {merged['order_id'].nunique()}"


def invariant_iqr_off_keeps_revenue(result, frames):
    """С обрезкой выбросов выключенной выручка не меняется."""
    plain = result["metrics"]["revenue"]
    unclipped = run_pipeline(clone_frames(frames), clip_outliers=False)["metrics"]["revenue"]
    same = abs(plain - unclipped) < 0.01
    return same, f"выручка {money(plain)}, без обрезки {money(unclipped)}"


BANNED_CLAIMS = (
    "дадут прирост",
    "база лояльна",
    "хороший показатель удержания",
    "не подтверждается",
    "масса чеков сосредоточена",
)


def invariant_insights_are_text(result, frames):
    """Каждый вывод — непустая строка, пригодная для показа пользователю."""
    data = run_insights_of(clone_frames(frames))
    texts = data["texts"]
    broken = [text for text in texts if not isinstance(text, str) or not text.strip()]
    fact = f"выводов {len(texts)}, некорректных {len(broken)}"
    return (bool(texts) and not broken), fact


def invariant_no_unsupported_claims(result, frames):
    """В выводах нет утверждений, которые данные не подтверждают."""
    data = run_insights_of(clone_frames(frames))
    joined = " ".join(data["texts"]).lower()
    found = [phrase for phrase in BANNED_CLAIMS if phrase in joined]
    fact = f"запрещённых формулировок: {found or 'нет'}"
    return not found, fact


def invariant_evidence_days_match_span(result, frames):
    """Число дней в окне совпадает с размахом дат заказов."""
    merged = result["merged"]
    level = evidence_level(merged)
    dates = pd.to_datetime(merged["order_date"]).dropna()
    expected = int((dates.max().normalize() - dates.min().normalize()).days) + 1
    fact = f"evidence_level {level['days']}, размах дат {expected}"
    return level["days"] == expected, fact


def invariant_weekday_profile_days_sum(result, frames):
    """Сумма повторов дней недели равна длине окна: каждый день учтён один раз."""
    merged = result["merged"]
    level = evidence_level(merged)
    profile = weekday_profile(merged)
    total = int(profile["days"].sum())
    fact = f"сумма дней по профилю {total}, длина окна {level['days']}"
    return total == level["days"], fact


def invariant_repeat_share_consistent(result, frames):
    """Доля покупателей с повторной покупкой в выводе совпадает с показателем."""
    merged = result["merged"]
    per_user = merged.groupby("user_id")["order_id"].nunique()
    expected = float((per_user > 1).mean()) if len(per_user) else 0.0
    actual = float(result["metrics"]["repeat_share"])
    fact = f"по заказам {percent(expected)}, в показателях {percent(actual)}"
    return abs(expected - actual) < 1e-9, fact


def invariant_no_tautological_concentration(result, frames):
    """Сравнение «топ-3 против равномерного» не превращается в тождество.

    При трёх и менее группах доля топ-3 по определению равна 100 %, и любое
    утверждение о концентрации противоречило бы самим числам.
    """
    merged = stretch_to_long_window(result["merged"])
    problems = []
    for column in ("category", "city"):
        groups = merged.groupby(column)["revenue"].sum()
        for size in (1, 2, 3):
            sample = merged[merged[column].isin(groups.index[:size])]
            if sample.empty:
                continue
            joined = " ".join(
                build_insights(sample, build_metrics(sample), result["issues"], 20, "")
            )
            for phrase in (
                "против 100.0 % при равномерном распределении",
                "выраженно неравномерен",
            ):
                if phrase in joined:
                    problems.append(f"{column} ≤ {size}: {phrase}")
    fact = f"тождественных сравнений {len(problems)}" if problems else "тождественных сравнений нет"
    return not problems, fact


def invariant_counts_agree_after_preposition(result, frames):
    """Счётчик после «из» стоит в родительном падеже.

    «3 из 4 позиций», а не «3 из 4 позиции»: после предлога множественное
    число среднего рода не совпадает с родительным падежом.
    """
    merged = stretch_to_long_window(result["merged"])
    items = sorted(merged["item_id"].unique())
    problems = []
    for size in range(1, 6):
        if size > len(items):
            break
        sample = merged[merged["item_id"].isin(items[:size])]
        if sample.empty:
            continue
        joined = " ".join(
            build_insights(sample, build_metrics(sample), result["issues"], size + 3, "")
        )
        nominative = plural_ru(size, "позиции", "позиции", "позиций")
        genitive = counted_after(size, "позиции", "позиций")
        if f"из {genitive}" not in joined:
            problems.append(f"{size} позиций: нет «из {genitive}»")
        elif nominative != genitive.split()[-1] and f"из {size} {nominative}" in joined:
            problems.append(f"{size}: «из {nominative}» вместо «из {genitive}»")
    fact = f"нарушений {len(problems)}" if problems else "нарушений нет"
    return not problems, fact


def invariant_halves_claim_matches_share(result, frames):
    """Утверждение про половину оборота соответствует доле лидера."""
    merged = stretch_to_long_window(result["merged"])
    totals = merged.groupby("category")["revenue"].sum().sort_values(ascending=False)
    problems = []
    for size in (2, 4, 9):
        if size > len(totals):
            break
        sample = merged[merged["category"].isin(totals.index[:size])]
        if sample.empty:
            continue
        lead_share = float(totals.iloc[0] / totals.sum())
        joined = " ".join(
            build_insights(sample, build_metrics(sample), result["issues"], 20, "")
        )
        claims_more = "снимет больше половины оборота" in joined
        claims_less = "меньше половины оборота" in joined
        if lead_share > 0.5 and not claims_more:
            problems.append(f"{size} категорий, доля {percent(lead_share)}: нет утверждения о половине+")
        if lead_share <= 0.5 and not claims_less:
            problems.append(f"{size} категорий, доля {percent(lead_share)}: нет утверждения о половине-")
    fact = f"несовпадений {len(problems)}" if problems else "несовпадений нет"
    return not problems, fact


LLM_TELLS = (
    "Гипотеза.",
    "почти наверняка",
    "первоочередная проверка",
    "невозможно отделить",
    "лежит в пределах случайных колебаний",
    "выглядят сильнее",
    "Маркетинговый бюджет",
    "стоит проверить",
    "рекомендуется обратить внимание",
    "является ключевым",
    "данные демонстрируют",
    "следует отметить",
    "таким образом",
    "итоговый вывод",
)


def invariant_no_llm_tells(result, frames):
    """В тексте нет служебных меток и шаблонных оборотов.

    Метка «Гипотеза.» на каждом пункте и канцелярские обороты выдают
    машинную генерацию; различие между фактом и закономерностью задаётся
    флагами, а не словами в тексте.
    """
    problems = []
    for label, data in (
        ("окно 10 дней", result["merged"]),
        ("длинное окно", stretch_to_long_window(result["merged"])),
    ):
        texts = build_insights(
            data, build_metrics(data), result["issues"],
            int(result["cleaned"]["items"]["item_id"].nunique()), "",
        )
        joined = " ".join(texts)
        found = [phrase for phrase in LLM_TELLS if phrase.lower() in joined.lower()]
        if found:
            problems.append(f"{label}: {found}")
    fact = f"оборотов найдено {len(problems)}" if problems else "оборотов нет"
    return not problems, fact


def invariant_no_genitive_error_after_prep(result, frames):
    """После «при» стоит предложный падеж: «при 40 заказах», не «при 40 заказов»."""
    problems = []
    for label, data in (
        ("окно 10 дней", result["merged"]),
        ("длинное окно", stretch_to_long_window(result["merged"])),
    ):
        texts = build_insights(
            data, build_metrics(data), result["issues"],
            int(result["cleaned"]["items"]["item_id"].nunique()), "",
        )
        joined = " ".join(texts)
# Проверяем любое предложение с числительным рядом со словом «заказ»:
        # после «при» нужен предложный падеж, после «в N из M» — родительный.
        for match in re.findall(r"(?:при|из) [\d.,  ]+ заказ\w*", joined):
            if not match.endswith(("заказах", "заказов")):
                problems.append(f"{label}: «{match}»")
    fact = f"нарушений {len(problems)}" if problems else "нарушений нет"
    return not problems, fact


def invariant_tail_claim_matches_concentration(result, frames):
    """Вывод о «крупных заказах» соответствует их доле в выручке.

    Доля 1 % при 720 заказах — это отношение в 2.4 раза к равномерному
    ожиданию, но крупных заказов там нет; обратное «ровно» при 18 %
    противоречило бы самим числам.
    """
    problems = []
    for label, data in (
        ("окно 10 дней", result["merged"]),
        ("длинное окно", stretch_to_long_window(result["merged"])),
    ):
        orders = int(len(data))
        top_share = (
            float(data["revenue"].nlargest(3).sum() / data["revenue"].sum())
            if orders and data["revenue"].sum() else 0.0
        )
        texts = build_insights(data, build_metrics(data), result["issues"], 20, "")
        # Текст начинается с заглавной буквы, поэтому сравнение без учёта регистра.
        joined = " ".join(texts).lower()
        if top_share >= 0.15 and "единичных крупных заказах" not in joined:
            problems.append(f"{label}: доля {percent(top_share)}, а вывода о крупных заказах нет")
        if top_share < 0.03 and "крупных отдельных заказов нет" not in joined:
            problems.append(f"{label}: доля {percent(top_share)}, а вывода об отсутствии крупных заказов нет")
        if 0.03 <= top_share < 0.15 and "умеренный" not in joined:
            problems.append(f"{label}: доля {percent(top_share)}, не сработала умеренная формулировка")
    fact = f"несовпадений {len(problems)}" if problems else "несовпадений нет"
    return not problems, fact


INVARIANTS = (
    ("В тексте нет служебных меток и штампов", invariant_no_llm_tells),
    ("После «при» предложный падеж", invariant_no_genitive_error_after_prep),
    ("Вывод о крупных заказах соответствует доле", invariant_tail_claim_matches_concentration),
    ("Фильтр по всем категориям не меняет выборку", invariant_all_categories_filter),
    ("Порядок строк не влияет на выручку", invariant_row_order_does_not_matter),
    ("Сумма по разрезам равна общей выручке", invariant_group_sum_equals_total),
    ("left-join не меняет число строк", invariant_join_keeps_row_count),
    ("order_id уникален после объединения", invariant_order_id_unique_after_join),
    ("Выключенный IQR не меняет выручку", invariant_iqr_off_keeps_revenue),
    ("Выводы — непустые строки", invariant_insights_are_text),
    ("Нет неподтверждённых утверждений", invariant_no_unsupported_claims),
    ("Длительность окна совпадает с размахом дат", invariant_evidence_days_match_span),
    ("Повторы дней недели покрывают окно целиком", invariant_weekday_profile_days_sum),
    ("Профиль недели всегда из семи дней", invariant_weekday_profile_has_all_seven_days),
    ("Доля повторных покупателей согласована", invariant_repeat_share_consistent),
    ("Сравнение топ-3 не превращается в тождество", invariant_no_tautological_concentration),
    ("Счётчик после «из» в родительном падеже", invariant_counts_agree_after_preposition),
    ("Утверждение о половине оборота соответствует доле", invariant_halves_claim_matches_share),
    ("Неливид не выходит за область фильтра", invariant_unsold_stays_inside_catalog_scope),
    ("Неликвид скрыт на коротком окне", invariant_unsold_hidden_on_short_window),
    ("Подпись фильтра явно называет активные", invariant_filter_scope_note_is_explicit),
    ("Числительные 2–4 склоняются верно", invariant_plural_forms_after_numbers),
)


def run_invariants(frames, verbose=True):
    """Проверяет математические тождества на текущих данных."""
    result = run_pipeline(clone_frames(frames))
    rows = []
    for name, function in INVARIANTS:
        try:
            passed, fact = function(result, clone_frames(frames))
        except Exception as error:
            passed, fact = False, f"{type(error).__name__}: {error}"
        rows.append({"name": name, "fact": fact, "passed": bool(passed)})
        if verbose:
            print(f"  {'OK  ' if passed else 'FAIL'}  {name}")
    return rows


# ---------------------------------------------------------------------------
# Фаззинг: случайные мутации копий данных
# ---------------------------------------------------------------------------

FUZZ_ITERATIONS = 500
FUZZ_SEED = 20240

FUZZ_MUTATIONS = (
    "пропуск в ключе",
    "пустая строка",
    "ноль в цене",
    "отрицательная цена",
    "минус в количестве",
    "дата в другом формате",
    "дата в будущем",
    "число текстом",
    "запятая как разделитель",
    "точка как разделитель тысяч",
    "символ валюты",
    "научная нотация",
    "дробный quantity",
    "пропуск в базовой цене",
    "дубль ключа",
    "неизвестный сегмент",
    "неизвестная категория",
    "пустой город",
    "гигантское число",
)


def fuzz_frame(frames, random_state):
    """Одна случайная мутация набора таблиц. Мутирует копию, не оригинал."""
    rng = np.random.default_rng(random_state)
    frames = clone_frames(frames)
    mutation = FUZZ_MUTATIONS[int(rng.integers(0, len(FUZZ_MUTATIONS)))]
    orders = frames["orders"]
    row = int(rng.integers(0, len(orders))) if len(orders) else 0

    if mutation == "пропуск в ключе" and len(orders):
        orders.loc[orders.index[row], "order_id"] = np.nan
    elif mutation == "пустая строка" and len(frames["items"]):
        # Пустая строка в текстовом поле: к id её подставлять нельзя,
        # идентификатор числовой, и pandas правомерно не даст вписать "" в Int64.
        frames["items"].loc[frames["items"].index[0], "item_name"] = ""
    elif mutation == "ноль в цене" and len(orders):
        orders.loc[orders.index[row], "price_per_unit"] = 0
    elif mutation == "отрицательная цена" and len(orders):
        orders.loc[orders.index[row], "price_per_unit"] = -abs(float(orders["price_per_unit"].iloc[0]))
    elif mutation == "минус в количестве" and len(orders):
        orders.loc[orders.index[row], "quantity"] = -3
    elif mutation == "дата в другом формате" and len(orders):
        orders["order_date"] = orders["order_date"].astype(str).str.replace("-", "/", regex=False)
    elif mutation == "дата в будущем" and len(orders):
        orders["order_date"] = "2099-12-31 23:59:59"
    elif mutation == "число текстом" and len(orders):
        orders["price_per_unit"] = orders["price_per_unit"].map(lambda value: f"{value:,.2f}".replace(",", " ").replace(".", ","))
    elif mutation == "запятая как разделитель" and len(orders):
        orders["price_per_unit"] = orders["price_per_unit"].map(lambda value: f"{float(value):.2f}".replace(".", ","))
    elif mutation == "точка как разделитель тысяч" and len(orders):
        orders["price_per_unit"] = orders["price_per_unit"].map(lambda value: f"{float(value):,.2f}".replace(",", "."))
    elif mutation == "символ валюты" and len(orders):
        orders["price_per_unit"] = orders["price_per_unit"].map(lambda value: f"{float(value):.2f} руб.")
    elif mutation == "научная нотация" and len(orders):
        orders["quantity"] = orders["quantity"].map(lambda value: f"{int(value)}e0")
    elif mutation == "дробный quantity" and len(orders):
        orders["quantity"] = orders["quantity"].astype("float64") + 0.5
    elif mutation == "пропуск в базовой цене":
        mask = frames["items"].index[::2]
        frames["items"].loc[mask, "base_price"] = np.nan
    elif mutation == "дубль ключа" and len(orders):
        frames["items"] = pd.concat([frames["items"], frames["items"].iloc[[0]]], ignore_index=True)
    elif mutation == "неизвестный сегмент":
        frames["users"].loc[frames["users"].index[0], "user_segment"] = "vip_2024"
    elif mutation == "неизвестная категория":
        frames["items"].loc[frames["items"].index[0], "category"] = "Новая категория"
    elif mutation == "пустой город":
        frames["users"].loc[frames["users"].index[0], "city"] = ""
    elif mutation == "гигантское число" and len(orders):
        orders.loc[orders.index[row], "quantity"] = 10 ** 12
    return frames, mutation


def run_fuzzing(frames, iterations=500, seed=20240, verbose=True):
    """
    Прогоняет N случайных мутаций и требует двух инвариантов:
    пайплайн не падает, выручка не отрицательна и конечна.

    Возвращает сводку и список упавших мутаций с описанием причины.
    """
    failures = []
    passed = 0
    for iteration in range(iterations):
        mutated, mutation = fuzz_frame(frames, seed + iteration)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = run_pipeline(mutated)
            revenue = float(result["metrics"]["revenue"])
            runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
            if runtime_warnings:
                raise AssertionError(f"RuntimeWarning: {runtime_warnings[0].message}")
            if not np.isfinite(revenue):
                raise AssertionError(f"выручка не конечна: {revenue}")
            if revenue < 0:
                raise AssertionError(f"выручка отрицательна: {revenue}")
            if len(result["merged"]) > len(mutated["orders"]):
                raise AssertionError(
                    f"строк после merge {len(result['merged'])} больше, чем заказов {len(mutated['orders'])}"
                )
            passed += 1
        except Exception as error:
            failures.append({"iteration": iteration, "mutation": mutation, "error": f"{type(error).__name__}: {error}"})
        if verbose and iterations > 50 and (iteration + 1) % 100 == 0:
            print(f"  ... {iteration + 1}/{iterations}, провалов {len(failures)}")
    return {
        "iterations": iterations,
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Выбор что тестировать и вывод результатов
# ---------------------------------------------------------------------------

SUITES = (
    ("scenarios", "Сценарии", "поломки данных и ожидаемая реакция пайплайна"),
    ("invariants", "Инварианты", "тождества, верные при любых данных"),
    ("fuzz", "Фаззинг", "случайные мутации копий данных"),
)


def available_apps():
    return [name for name in APP_CANDIDATES if os.path.exists(os.path.join(BASE_DIR, name))]


def prompt_choice(title, options):
    """Печатает нумерованный список и возвращает выбранный ключ."""
    print(f"\n{title}")
    for position, (key, label, detail) in enumerate(options, start=1):
        print(f"  {position}) {label:<22} {detail}")
    while True:
        raw = input(f"> [1] ").strip()
        if not raw:
            return options[0][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("  Нужно число от 1 до", len(options))


def warn_if_copies_diverge(chosen):
    """Предупредить, если выбранная копия разошлась с рабочей версией."""
    if chosen == "app_v4.py" or "app.py" not in available_apps():
        return
    with open(os.path.join(BASE_DIR, "app.py"), "rb") as handle:
        copy_bytes = handle.read()
    with open(os.path.join(BASE_DIR, "app_v4.py"), "rb") as handle:
        working_bytes = handle.read()
    if copy_bytes == working_bytes:
        print("app.py и app_v4.py идентичны.")
    else:
        print("ВНИМАНИЕ: app.py отличается от app_v4.py — копию надо пересоздать.")


def report(title, rows, show_all):
    passed = sum(1 for row in rows if row["passed"])
    total = len(rows)
    print(f"\n{title}: {passed} / {total}")
    failures = [row for row in rows if not row["passed"]]
    for row in failures:
        print(f"  FAIL  {row['name']}")
        print(f"        ожидалось: {row.get('expectation', '-')}")
        print(f"        получено:  {row['fact']}")
    if not failures and total:
        if show_all:
            for row in rows:
                print(f"        {row['name']}: {row['fact']}")
        else:
            print("  все проверки пройдены")
    return passed, total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Проверки пайплайна дашборда интернет-магазина",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-f", "--file", choices=APP_CANDIDATES, help="какой файл дашборда тестировать")
    parser.add_argument("-s", "--suite", choices=("all",) + tuple(key for key, _, _ in SUITES),
                        help="что прогнать: all, scenarios, invariants, fuzz")
    parser.add_argument("--all", action="store_true",
                        help="прогнать всё: сценарии, инварианты и фаззинг")
    parser.add_argument("-n", "--iterations", type=int, default=FUZZ_ITERATIONS, help="итераций фаззинга")
    parser.add_argument("-v", "--verbose", action="store_true", help="показать факт по каждой проверке")
    return parser.parse_args()


def choose(args):
    """Возвращает (файл, наборы) — из аргументов либо через диалог."""
    present = available_apps()
    if not present:
        raise SystemExit("В папке нет ни app.py, ни app_v4.py.")

    if args.file:
        chosen_file = args.file
    else:
        chosen_file = prompt_choice(
            "Что тестировать?",
            [(name, name, "файл дашборда") for name in present],
        )

    if args.suite:
        suites = tuple(key for key, _, _ in SUITES) if args.suite == "all" else (args.suite,)
    elif args.all:
        suites = tuple(key for key, _, _ in SUITES)
    else:
        picked = prompt_choice(
            "Что прогнать?",
            list(SUITES) + [("all", "Всё", "сценарии, инварианты и фаззинг")],
        )
        suites = tuple(key for key, _, _ in SUITES) if picked == "all" else (picked,)

    return chosen_file, suites


def main():
    args = parse_args()
    chosen_file, suites = choose(args)
    show_all = args.verbose or "scenarios" not in suites

    print(f"=== Проверки пайплайна: {chosen_file} ===")
    warn_if_copies_diverge(chosen_file)

    app, _ = load_app(chosen_file)
    bind(app)
    frames = load_data(*data_stamps())
    print(f"Данные: заказов {len(frames['orders'])}, "
          f"пользователей {len(frames['users'])}, товаров {len(frames['items'])}\n")

    total_passed = 0
    total_checks = 0

    if "scenarios" in suites:
        rows = run_scenarios(frames, verbose=args.verbose)
        passed, total = report("Сценарии", rows, show_all)
        total_passed += passed
        total_checks += total
    if "invariants" in suites:
        rows = run_invariants(frames, verbose=args.verbose)
        total_passed += report("Инварианты", rows, show_all)[0]
        total_checks += len(rows)
    if "fuzz" in suites:
        summary = run_fuzzing(frames, iterations=args.iterations, seed=FUZZ_SEED, verbose=args.verbose)
        passed, total = summary["passed"], summary["iterations"]
        print(f"\nФаззинг: {passed} / {total}")
        for failure in summary["failures"][:10]:
            print(f"  FAIL  итерация {failure['iteration']}, {failure['mutation']}: {failure['error']}")
        total_passed += passed
        total_checks += total

    print(f"\nИТОГО: {total_passed} / {total_checks} проверок, "
          f"провалов: {total_checks - total_passed}")
    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())