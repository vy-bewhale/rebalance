import streamlit as st
import pandas as pd
import numpy as np
import datetime
import core # Импортируем наш модуль с логикой
import plots # Импортируем наш модуль с графиками
from typing import Dict, List

# --- Настройка страницы ---
st.set_page_config(layout="wide", page_title="Бэктестер Ребалансировки")

# --- CSS для уменьшения шрифтов И ДОБАВЛЕНИЯ РАМОК КОНТЕЙНЕРАМ ---
st.markdown("""
<style>
    /* Уменьшаем шрифт для опций и выбранных элементов в multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        font-size: 0.7em !important; /* ЕЩЕ МЕНЬШЕ */
        /* background-color: #eee !important; /* Стандартный цвет */
        /* color: #333 !important; */
        /* border-radius: 0.25rem !important; */
        /* padding: 0.15rem 0.4rem !important; */
    }
    div[data-baseweb="select"] > div {
        font-size: 0.8em !important; /* ЕЩЕ МЕНЬШЕ */
    }
    /* Уменьшаем шрифт для опций в выпадающем списке */
     div[data-baseweb="popover"] li {
        font-size: 0.8em !important; /* ЕЩЕ МЕНЬШЕ */
     }
     /* Уменьшаем шрифт для подписей к весам (уже есть small, сделаем чуть меньше) */
    .stNumberInput small {
         font-size: 0.75em !important;
     }

    /* --- Стили для рамок контейнеров --- */
    /* Выбираем контейнеры верхнего уровня в основной области */
    .main .stBlock > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid #e0e0e0; /* Светло-серая рамка */
        border-radius: 5px;       /* Скругленные углы */
        padding: 1.5rem;            /* Внутренний отступ */
        margin-bottom: 1.5rem;      /* Внешний отступ снизу */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Легкая тень */
    }
</style>
""", unsafe_allow_html=True)

# Уменьшенный заголовок
st.subheader("Интерактивный Бэктестер Ребалансировки Портфеля")

# --- Определения активов, групп и цветов ---

# Расширенный словарь тикеров
COMMON_TICKERS = {
    # Акции США
    'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'IWM': 'Russell 2000 ETF', 'DIA': 'Dow Jones ETF',
    # Акции Развитые страны
    'EFA': 'MSCI EAFE ETF', 'VEA': 'FTSE Developed Markets ETF',
    # Акции Развивающиеся рынки
    'EEM': 'MSCI Emerging Markets ETF', 'VWO': 'FTSE Emerging Markets ETF',
    # Облигации США (Казначейские)
    'IEF': '7-10 Year Treasury Bond ETF', 'TLT': '20+ Year Treasury Bond ETF', 'SHY': '1-3 Year Treasury Bond ETF',
    # Облигации США (Корпоративные/Совокупные)
    'AGG': 'US Aggregate Bond ETF', 'LQD': 'Investment Grade Corporate Bond ETF', 'HYG': 'High Yield Corporate Bond ETF',
    # Облигации Международные
    'BNDX': 'Total Intl Bond ETF (hedged)',
    # Золото
    'GLD': 'Gold ETF',
    # Недвижимость
    'VNQ': 'Real Estate ETF',
    # Сырье
    'DBC': 'Commodity Index Tracking Fund',
    # Криптовалюты
    'BTC-USD': 'Bitcoin USD',
    # Валюты
    'FXE': 'Euro ETF (vs USD)',
    'FXY': 'Japanese Yen ETF (vs USD)',
    'FXB': 'British Pound ETF (vs USD)'
    # Добавить еще, если нужно
}

# Группы активов (сопоставление тикера с названием группы)
ASSET_GROUPS = {
    'SPY': 'Акции США', 'QQQ': 'Акции США', 'IWM': 'Акции США', 'DIA': 'Акции США',
    'EFA': 'Акции Разв.', 'VEA': 'Акции Разв.',
    'EEM': 'Акции Разв.', 'VWO': 'Акции Разв.',
    'IEF': 'Обл. США (Казн.)', 'TLT': 'Обл. США (Казн.)', 'SHY': 'Обл. США (Казн.)',
    'AGG': 'Обл. США (Сумм.)', 'LQD': 'Обл. США (Корп.)', 'HYG': 'Обл. США (ВДО)',
    'BNDX': 'Обл. Межд.',
    'GLD': 'Золото',
    'VNQ': 'Недвиж.',
    'DBC': 'Сырье',
    'BTC-USD': 'Крипто',
    'FXE': 'Валюты', 'FXY': 'Валюты', 'FXB': 'Валюты',
    'Cash': 'Кэш' # Добавляем группу для Кэша
}

# Цвета для групп (можно настроить) - используем палитру Plotly по умолчанию
import plotly.express as px
# Генерируем достаточное количество цветов из стандартной палитры
unique_groups = sorted(list(set(ASSET_GROUPS.values())))
colors = px.colors.qualitative.Plotly # или Set3, Pastell, etc.
GROUP_COLORS = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
GROUP_COLORS['Кэш'] = '#AAAAAA' # Серый для кэша

# --- Callback для обновления весов ---
def adjust_weights():
    # Получаем текущие выбранные НАЗВАНИЯ из состояния мултиселекта
    current_selected_names = st.session_state.selected_assets_multiselect
    # Получаем соответствующие ТИКЕРЫ
    current_tickers = [ticker for ticker, name in COMMON_TICKERS.items() if name in current_selected_names]
    # Список для установки весов
    assets_for_weighting = current_tickers + ['Cash']
    num_assets_plus_cash = len(assets_for_weighting)
    new_default_weight = 100.0 / num_assets_plus_cash if num_assets_plus_cash > 0 else 0

    # Обновляем веса в session_state для текущего выбора
    for asset in assets_for_weighting:
        st.session_state[f"weight_{asset}"] = new_default_weight

    # Опционально: Очистить состояние весов для тикеров, которые были убраны
    # (но это не обязательно, так как виджеты для них не будут отображаться)
    # all_possible_weights = {f"weight_{ticker}" for ticker in COMMON_TICKERS} | {"weight_Cash"}
    # current_weight_keys = {f"weight_{asset}" for asset in assets_for_weighting}
    # for key in all_possible_weights - current_weight_keys:
    #     if key in st.session_state:
    #         del st.session_state[key]

# --- Боковая панель для ввода параметров ---
st.sidebar.header("Параметры Бэктеста")

# Ввод тикеров через multiselect
default_tickers = ['SPY', 'GLD', 'IEF', 'BTC-USD'] # Изменил дефолт
default_names = [COMMON_TICKERS.get(t, t) for t in default_tickers]

# Инициализируем веса по умолчанию в session_state при первом запуске
if 'selected_assets_multiselect' not in st.session_state:
    st.session_state.selected_assets_multiselect = default_names
    adjust_weights() # Вызываем для установки начальных весов

selected_names = st.sidebar.multiselect(
    "Выберите активы",
    options=list(COMMON_TICKERS.values()), # Показываем названия
    key="selected_assets_multiselect", # Ключ для доступа к состоянию и для callback
    on_change=adjust_weights # Назначаем callback
)

# Получаем тикеры из выбранных названий (уже берется из session_state)
# tickers = [ticker for ticker, name in COMMON_TICKERS.items() if name in selected_names]
# Вместо этого, используем то, что сейчас в состоянии (callback уже обновил)
tickers = [ticker for ticker, name in COMMON_TICKERS.items() if name in st.session_state.selected_assets_multiselect]

# Создаем карту тикер -> название для выбранных
selected_ticker_map = {ticker: name for ticker, name in COMMON_TICKERS.items() if name in st.session_state.selected_assets_multiselect}

# Создаем словарь цветов для выбранных тикеров
selected_ticker_colors = {
    ticker: GROUP_COLORS.get(ASSET_GROUPS.get(ticker, 'Другое'), '#333333') # Цвет по умолчанию, если нет группы
    for ticker in tickers
}

# Ввод дат
default_end_date = datetime.date.today()
default_start_date = default_end_date - datetime.timedelta(days=10*365) # Примерно 10 лет назад
start_date = st.sidebar.date_input("Начальная дата", default_start_date)
end_date = st.sidebar.date_input("Конечная дата", default_end_date)

# Выбор частоты ребалансировки
rebalance_freq_options = {'ME': 'Ежемесячно', 'QE': 'Ежеквартально', 'YE': 'Ежегодно'} # Обновлено на новые коды
rebalance_freq_display = st.sidebar.selectbox(
    "Частота ребалансировки (Календарная)", # Добавил уточнение
    options=list(rebalance_freq_options.keys()),
    format_func=lambda x: rebalance_freq_options[x]
)

# Новый ввод: Порог изменения цены для ребалансировки
price_change_threshold = st.sidebar.number_input(
    "Порог ребалансировки по цене (%)",
    min_value=1.0,
    max_value=100.0,
    value=10.0, # Значение по умолчанию 10%
    step=0.5,
    help="Ребалансировка сработает, если цена ЛЮБОГО актива изменится на этот % с момента последней ребалансировки."
)

# Безрисковая ставка
rf_rate_percent = st.sidebar.number_input("Безрисковая ставка (% годовых)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
rf_rate_decimal = rf_rate_percent / 100.0

# Начальный капитал
initial_capital = 100000.0
st.sidebar.markdown(f"**Начальный капитал:** `${initial_capital:,.0f}`")

# Ввод весов
st.sidebar.subheader("Целевые веса (%)")
target_weights_input: Dict[str, float] = {}
if tickers:
    cols = st.sidebar.columns(2)
    col_idx = 0
    assets_for_weighting = tickers + ['Cash']

    for asset in assets_for_weighting: # asset здесь - это тикер (SPY, GLD...) или 'Cash'
        session_key = f"weight_{asset}"
        # Убедимся, что ключ существует в состоянии (это должно делаться в callback)
        if session_key not in st.session_state:
             # Эта логика должна быть в adjust_weights, но на всякий случай оставим fallback
             num_assets_plus_cash = len(tickers) + 1
             fallback_default = 100.0 / num_assets_plus_cash if num_assets_plus_cash > 0 else 0
             st.session_state[session_key] = fallback_default

        with cols[col_idx % len(cols)]:
            st.markdown(f"<small>Вес {asset}</small>", unsafe_allow_html=True)
            # Убираем key и передаем value явно из session_state
            weight = st.number_input(
                label=f"Вес {asset}", # Label нужен для внутренней идентификации, но он скрыт
                label_visibility="collapsed",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state[session_key], # Явно берем значение из состояния
                step=1.0
                # key=session_key # Убрали key, чтобы избежать конфликта
            )
            target_weights_input[asset] = weight / 100.0 # Собираем значение как и раньше
        col_idx += 1

    # Показываем сумму весов и предупреждение
    # Сумму считаем из ТЕКУЩИХ значений виджетов (target_weights_input), а не из session_state
    current_sum_weights = sum(target_weights_input.values()) * 100
    st.sidebar.metric("Текущая сумма весов", f"{current_sum_weights:.1f}%")
    if abs(current_sum_weights - 100.0) > 0.1:
        st.sidebar.warning("Сумма весов должна быть равна 100%!")

# Кнопка запуска
run_button = st.sidebar.button("Запустить Бэктест")

# --- Основная область для вывода результатов ---
st.divider()

if run_button:
    # 1. Валидация ввода (проверяем tickers, даты, и сумму из target_weights_input)
    errors = []
    if not tickers:
        errors.append("Выберите хотя бы один актив.")
    if start_date >= end_date:
        errors.append("Начальная дата должна быть раньше конечной даты.")
    # Используем веса из target_weights_input для финальной проверки суммы
    if tickers and abs(sum(target_weights_input.values()) * 100 - 100.0) > 0.1:
        errors.append("Сумма целевых весов должна быть равна 100%.")

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    # Нормализуем финальные веса из target_weights_input
    total_weight_input = sum(target_weights_input.values())
    if total_weight_input > 0:
        normalized_weights = {k: v / total_weight_input for k, v in target_weights_input.items()}
    else:
         normalized_weights = {} # Если сумма 0, то весов нет

    # 2. Выполнение расчетов
    price_data = None
    backtest_results = None
    drawdown_results = None # Добавляем переменную для данных просадок
    metrics = None

    try:
        with st.spinner('Загрузка исторических данных...'):
            price_data = core.load_price_data(tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            st.error("Не удалось загрузить данные. Проверьте выбранные активы и период.")
            st.stop()

        # Проверка на наличие всех тикеров после загрузки (на всякий случай)
        if not all(ticker in price_data.columns for ticker in tickers):
             st.error(f"Не удалось загрузить данные для всех указанных активов. Проверьте: {tickers}")
             st.stop()

        with st.spinner('Запуск бэктеста...'):
            # Запускаем бэктест
            backtest_output = core.run_backtest(
                price_data=price_data,
                target_weights=normalized_weights,
                rebalance_freq=rebalance_freq_display,
                initial_capital=initial_capital,
                weight_deviation_threshold=price_change_threshold # Передаем новый параметр
            )
            if backtest_output is None:
                st.error("Ошибка при выполнении бэктеста.")
                st.stop()
            # Распаковываем ЧЕТЫРЕ результата
            backtest_results, drawdown_results, rebalance_log, hover_weights_text = backtest_output

        # 2.5 Расчет метрик
        if backtest_results is None or drawdown_results is None:
            st.error("Ошибка: Результаты бэктеста или просадок отсутствуют после выполнения.")
            st.stop()

        try:
            # Вычисляем безрисковую ставку в виде десятичной дроби
            rf_rate_decimal = rf_rate_percent / 100.0

            # Рассчитываем метрики, передавая лог ребалансировок
            with st.spinner('Расчет итоговых метрик...'):
                # Передаем только результаты стоимостей в метрики
                metrics = core.calculate_metrics(backtest_results, rf_rate_decimal, rebalance_log)

        except Exception as e:
            st.error(f"Произошла непредвиденная ошибка при расчете метрик: {e}")
            st.exception(e)
            st.stop()

    except Exception as e:
        st.error(f"Произошла непредвиденная ошибка при загрузке данных: {e}")
        st.exception(e)
        st.stop()

    # 3. Отображение результатов
    # Оборачиваем каждую секцию в контейнер
    with st.container():
        st.subheader("1. Нормализованные цены активов")
        prices_fig = plots.plot_normalized_prices(price_data, selected_ticker_map)
        if prices_fig:
            st.plotly_chart(prices_fig, use_container_width=True)
        else:
            st.warning("Не удалось построить график нормализованных цен.")

    with st.container():
        st.subheader("2. Динамика стоимости портфелей")
        equity_fig = plots.plot_equity_curves(backtest_results, selected_ticker_map, hover_weights_text)
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            st.warning("Не удалось построить график эквити.")

    with st.container():
        st.subheader("3. Итоговые метрики")
        if metrics:
            metrics_df = pd.DataFrame(metrics).T

            # Определяем порядок стратегий
            # Восстанавливаем Комби в main_order
            main_order = ['Ребаланс (Календарь)', 'Ребаланс (Доля %)', 'Ребаланс (Комби)', 'B&H (Целевые веса)']
            individual_bh_order = sorted(
                [name for name in metrics_df.index if name.startswith('B&H (') and name != 'B&H (Целевые веса)'],
                key=lambda x: x.split('(')[1]
            )
            strategy_order = main_order + individual_bh_order
            # Переиндексируем и удаляем строки, где все значения NaN
            metrics_df = metrics_df.reindex([s for s in strategy_order if s in metrics_df.index]).dropna(how='all')

            if not metrics_df.empty:
                metrics_df_display = metrics_df.copy()

                # --- Убираем временный вывод --- 
                # st.write("Отладочные данные для RF:")
                # st.dataframe(metrics_df_display[['End Value', 'Max Drawdown %', 'Max Drawdown Abs $', 'Peak Before MDD', 'Recovery Factor']], use_container_width=True)
                # ---------------------------------

                # Форматирование (основное)
                metrics_df_display['End Value'] = metrics_df['End Value'].map("${:,.0f}".format)
                metrics_df_display['CAGR'] = metrics_df['CAGR'].map("{:.2%}".format)
                metrics_df_display['Max Drawdown %'] = metrics_df['Max Drawdown %'].map("{:.2f}%".format)
                # Убираем отображение новых колонок из финальной таблицы
                metrics_df_display['Volatility'] = metrics_df['Volatility'].map("{:.2%}".format)
                metrics_df_display['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].map("{:.2f}".format)
                metrics_df_display['Sortino Ratio'] = metrics_df['Sortino Ratio'].apply(lambda x: "∞" if np.isinf(x) else f"{x:.2f}")
                metrics_df_display['Recovery Factor'] = metrics_df['Recovery Factor'].apply(lambda x: "∞" if np.isinf(x) else f"{x:.2f}")

                # Убираем колонки Start Value, Abs Drawdown, Peak из финального вывода
                columns_to_drop = ['Start Value', 'Max Drawdown Abs $', 'Peak Before MDD']
                metrics_df_display = metrics_df_display.drop(columns=[col for col in columns_to_drop if col in metrics_df_display.columns])

                # Названия колонок (финальные)
                # Добавляем "Кол-во ребал." в список названий
                final_columns = [
                    "Конечная стоимость", "CAGR (год. дох-ть)",
                    "Макс. просадка", "Волатильность (год.)", "Коэф. Шарпа",
                    "Коэф. Сортино", "Фактор восст.", "Кол-во ребал."
                ]
                if len(metrics_df_display.columns) == len(final_columns):
                     metrics_df_display.columns = final_columns
                else:
                     print(f"Warning: Column count mismatch. Expected {len(final_columns)}, got {len(metrics_df_display.columns)}")
                     # Попытка переименовать только существующие, но может привести к ошибкам
                     # rename_map = dict(zip(metrics_df_display.columns[:len(final_columns)], final_columns))
                     # metrics_df_display = metrics_df_display.rename(columns=rename_map)
                     pass # Оставляем оригинальные имена, если не совпадает

                # Форматирование Rebalance Count
                # Убедимся, что колонка существует перед форматированием
                if 'Rebalance Count' in metrics_df_display.columns:
                     metrics_df_display['Rebalance Count'] = metrics_df['Rebalance Count'].map(lambda x: "{:.0f}".format(x) if not pd.isna(x) else "-") # Отображаем как целое или прочерк

                st.dataframe(metrics_df_display, use_container_width=True)
            else:
                 st.warning("Не удалось рассчитать метрики ни для одной стратегии.")
        else:
            st.warning("Не удалось рассчитать метрики.")

    # Секция 4: Просадки портфелей (новый пункт)
    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    st.subheader("4. Просадки портфелей")
    # Вызываем новую функцию графика, передаем drawdown_results
    drawdown_fig = plots.plot_drawdown_curves(drawdown_results)
    if drawdown_fig:
        st.plotly_chart(drawdown_fig, use_container_width=True)
    else:
        # Сообщение, если график просадок не удалось построить
        st.warning("Не удалось построить график просадок.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Настройте параметры в боковой панели и нажмите 'Запустить Бэктест'.") 