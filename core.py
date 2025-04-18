import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import date
from typing import List, Dict, Optional, Tuple

# --- Константы ---
TRADING_DAYS_PER_YEAR = 252

# --- Функции загрузки данных ---

@st.cache_data # Кэшируем загрузку данных
def load_price_data(tickers: List[str], start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """
    Загружает исторические цены закрытия для указанных тикеров.

    Args:
        tickers (List[str]): Список тикеров для загрузки.
        start_date (date): Начальная дата.
        end_date (date): Конечная дата.

    Returns:
        Optional[pd.DataFrame]: DataFrame с ценами закрытия ('Close') и столбцом 'Cash' = 1.0,
                                или None в случае ошибки загрузки или отсутствия данных.
    """
    if not tickers:
        print("Error: No tickers provided.")
        return None
    try:
        # Загружаем данные, используем auto_adjust=True для получения скорректированных цен
        # (хотя для ETF это менее критично, чем для акций)
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            print(f"Error: No data downloaded for tickers {tickers} in the specified period.")
            return None

        close_prices = data['Close'] # Получаем цены закрытия
        # Обработка случая одного тикера (yfinance возвращает Series)
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=tickers[0])

        # Создаем явную копию, чтобы избежать SettingWithCopyWarning
        prices = close_prices.copy()

        # Проверка на наличие всех тикеров в результате (уже на копии)
        if not all(ticker in prices.columns for ticker in tickers):
            missing = [t for t in tickers if t not in prices.columns]
            print(f"Error: Could not download data for all tickers. Missing: {missing}")
            return None

        # Удаляем строки, где есть NaN хотя бы для одного тикера (без inplace=True)
        prices = prices.dropna()

        if prices.empty:
             print(f"Error: Data became empty after dropping NaN for tickers {tickers}.")
             return None

        # Добавляем столбец 'Cash' (теперь это безопасно)
        prices['Cash'] = 1.0
        return prices

    except Exception as e:
        print(f"Error downloading data for {tickers}: {e}")
        return None

# --- Функции бэктестинга ---

# Функция для расчета просадки
def calculate_drawdown_series(series: pd.Series) -> pd.Series:
    """Рассчитывает временной ряд просадки для заданной серии стоимости."""
    cumulative_max = series.cummax()
    # Избегаем деления на 0, заменяя 0 на NaN
    cumulative_max_safe = cumulative_max.replace(0, np.nan)
    drawdown = (series - cumulative_max_safe) / cumulative_max_safe
    # Заполняем NaN нулями (например, в начале, пока просадки нет)
    drawdown = drawdown.fillna(0)
    return drawdown

@st.cache_data # Кэшируем результаты бэктеста
def run_backtest(price_data: pd.DataFrame, target_weights: Dict[str, float],
                 rebalance_freq: str, initial_capital: float,
                 price_change_threshold: float) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Выполняет бэктестинг стратегий ребалансировки и сравнение с Buy & Hold.

    Args:
        price_data (pd.DataFrame): DataFrame с ценами (включая 'Cash').
        target_weights (Dict[str, float]): Словарь целевых весов (доли, сумма = 1.0).
        rebalance_freq (str): Частота ребалансировки (Календарная: 'ME', 'QE', 'YE').
        initial_capital (float): Начальный капитал.
        price_change_threshold (float): Порог изменения цены в % для запуска ценовой ребалансировки.

    Returns:
        Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
            Кортеж из двух DataFrame:
            1. results: DataFrame со стоимостями портфелей для разных стратегий.
            2. drawdown_results: DataFrame с рядами просадок для основных стратегий.
            Или None в случае ошибки.
    """
    if price_data is None or price_data.empty or not target_weights:
        return None

    assets = [col for col in target_weights.keys() if col != 'Cash']
    if not all(asset in price_data.columns for asset in assets):
         print("Error: Price data missing for some assets defined in target_weights.")
         return None

    # Конвертируем порог из % в доли
    threshold_decimal_upper = 1.0 + price_change_threshold / 100.0
    threshold_decimal_lower = 1.0 - price_change_threshold / 100.0

    # Определяем активы для ребалансировки (из target_weights) и все активы (из price_data)
    assets_to_rebalance = [col for col in target_weights.keys() if col != 'Cash']
    all_price_assets = [col for col in price_data.columns if col != 'Cash'] # Все активы с ценами

    # --- Общая инициализация для всех стратегий ребалансировки ---
    first_date = price_data.index[0]

    # Рассчитываем начальные доли и кэш ОДИН РАЗ
    initial_total_value = initial_capital
    initial_holdings_template = {} # Шаблон долей
    for ticker in assets_to_rebalance:
        target_value = initial_total_value * target_weights.get(ticker, 0.0)
        initial_price = price_data.at[first_date, ticker]
        shares = 0.0
        if not pd.isna(initial_price) and initial_price > 0 and not pd.isna(target_value):
            shares = target_value / initial_price
        else:
            print(f"Warning: Initial price/target issue for {ticker}. Setting shares to 0.")
        initial_holdings_template[ticker] = shares
    # Добавляем нулевые доли для активов, не входящих в target_weights
    for ticker in all_price_assets:
         if ticker not in initial_holdings_template:
              initial_holdings_template[ticker] = 0.0
    initial_cash = initial_total_value * target_weights.get('Cash', 0.0)

    # Рассчитываем начальную стоимость портфеля (для проверки)
    initial_asset_value_check = 0.0
    for ticker in all_price_assets:
         shares = initial_holdings_template.get(ticker, 0.0)
         price = price_data.at[first_date, ticker]
         if not pd.isna(price) and price > 0 and shares > 0:
             initial_asset_value_check += shares * price
    initial_total_value_check = initial_asset_value_check + initial_cash
    if not np.isclose(initial_total_value_check, initial_capital):
        print(f"Warning: Initial calculated value check ({initial_total_value_check:.2f}) differs from capital ({initial_capital:.2f}).")

    # --- 1. Логика КАЛЕНДАРНОЙ ребалансировки --- (как раньше, но вынесена в отдельную часть)
    portfolio_cal = pd.DataFrame(index=price_data.index)
    portfolio_cal['Holdings'] = pd.Series(dtype=object)
    portfolio_cal['Cash'] = np.nan
    portfolio_cal['Total_Value'] = np.nan

    portfolio_cal.at[first_date, 'Holdings'] = initial_holdings_template.copy()
    portfolio_cal.at[first_date, 'Cash'] = initial_cash
    portfolio_cal.at[first_date, 'Total_Value'] = initial_total_value_check # Используем проверенное значение

    freq_map = {'M': 'ME', 'Q': 'QE', 'A': 'YE'}
    actual_freq = freq_map.get(rebalance_freq, rebalance_freq)
    rebalance_dates_cal_raw = pd.date_range(start=first_date, end=portfolio_cal.index[-1], freq=actual_freq)
    rebalance_dates_cal = portfolio_cal.index.intersection(rebalance_dates_cal_raw)

    for i in range(1, len(portfolio_cal.index)):
        current_date = portfolio_cal.index[i]
        prev_date = portfolio_cal.index[i-1]
        # Копируем состояние
        portfolio_cal.at[current_date, 'Holdings'] = portfolio_cal.at[prev_date, 'Holdings'].copy()
        portfolio_cal.at[current_date, 'Cash'] = portfolio_cal.at[prev_date, 'Cash']
        # Пересчитываем стоимость
        current_total_value = 0.0
        holdings_dict = portfolio_cal.at[current_date, 'Holdings']
        for ticker in all_price_assets:
            shares = holdings_dict.get(ticker, 0.0)
            price = price_data.at[current_date, ticker]
            if shares > 0 and not pd.isna(price) and price > 0:
                current_total_value += shares * price
        current_total_value += portfolio_cal.at[current_date, 'Cash']
        portfolio_cal.at[current_date, 'Total_Value'] = current_total_value
        # Ребалансировка?
        if current_date in rebalance_dates_cal:
            # (логика ребалансировки как раньше)
            new_holdings = holdings_dict.copy()
            cash_after_rebalance = 0.0
            if current_total_value > 0:
                for ticker in assets_to_rebalance:
                    target_val = current_total_value * target_weights.get(ticker, 0.0)
                    price = price_data.at[current_date, ticker]
                    shares = 0.0
                    if not pd.isna(price) and price > 0 and target_val > 0:
                        shares = target_val / price
                    new_holdings[ticker] = shares
                cash_after_rebalance = current_total_value * target_weights.get('Cash', 0.0)
            portfolio_cal.at[current_date, 'Holdings'] = new_holdings
            portfolio_cal.at[current_date, 'Cash'] = cash_after_rebalance

    calendar_rebalanced_values = portfolio_cal['Total_Value'].copy().rename('Calendar_Rebalanced_Value')

    # --- 2. Логика ребалансировки ПО ПОРОГУ ИЗМЕНЕНИЯ ЦЕНЫ --- (без изменений)
    portfolio_pb = pd.DataFrame(index=price_data.index)
    portfolio_pb['Holdings'] = pd.Series(dtype=object)
    portfolio_pb['Cash'] = np.nan
    portfolio_pb['Total_Value'] = np.nan
    # Доп. состояние: цены на момент последней ребалансировки
    portfolio_pb['Prices_Last_Rebalance'] = pd.Series(dtype=object)

    # Инициализация на первую дату
    portfolio_pb.at[first_date, 'Holdings'] = initial_holdings_template.copy()
    portfolio_pb.at[first_date, 'Cash'] = initial_cash
    portfolio_pb.at[first_date, 'Total_Value'] = initial_total_value_check
    # Запоминаем цены на первую дату (только для тех, где есть вес)
    initial_prices_last_rebalance = {ticker: price_data.at[first_date, ticker]
                                     for ticker in assets_to_rebalance
                                     if not pd.isna(price_data.at[first_date, ticker])}
    portfolio_pb.at[first_date, 'Prices_Last_Rebalance'] = initial_prices_last_rebalance

    for i in range(1, len(portfolio_pb.index)):
        current_date = portfolio_pb.index[i]
        prev_date = portfolio_pb.index[i-1]
        # Копируем состояние
        portfolio_pb.at[current_date, 'Holdings'] = portfolio_pb.at[prev_date, 'Holdings'].copy()
        portfolio_pb.at[current_date, 'Cash'] = portfolio_pb.at[prev_date, 'Cash']
        portfolio_pb.at[current_date, 'Prices_Last_Rebalance'] = portfolio_pb.at[prev_date, 'Prices_Last_Rebalance'].copy()
        # Пересчитываем стоимость
        current_total_value = 0.0
        holdings_dict = portfolio_pb.at[current_date, 'Holdings']
        for ticker in all_price_assets:
            shares = holdings_dict.get(ticker, 0.0)
            price = price_data.at[current_date, ticker]
            if shares > 0 and not pd.isna(price) and price > 0:
                current_total_value += shares * price
        current_total_value += portfolio_pb.at[current_date, 'Cash']
        portfolio_pb.at[current_date, 'Total_Value'] = current_total_value

        # Проверка условия ребалансировки по цене
        trigger_rebalance = False
        prices_last_rebalance = portfolio_pb.at[current_date, 'Prices_Last_Rebalance']
        for ticker in assets_to_rebalance: # Проверяем только целевые активы
            current_price = price_data.at[current_date, ticker]
            last_rebalance_price = prices_last_rebalance.get(ticker)
            if pd.isna(current_price) or pd.isna(last_rebalance_price) or last_rebalance_price <= 0:
                continue # Пропускаем проверку, если цена некорректна
            # Проверка порогов
            if (current_price > last_rebalance_price * threshold_decimal_upper) or \
               (current_price < last_rebalance_price * threshold_decimal_lower):
                trigger_rebalance = True
                break # Достаточно одного триггера

        # Ребалансировка, если триггер сработал
        if trigger_rebalance:
            new_holdings = holdings_dict.copy()
            cash_after_rebalance = 0.0
            if current_total_value > 0:
                # (логика ребалансировки как раньше)
                for ticker in assets_to_rebalance:
                    target_val = current_total_value * target_weights.get(ticker, 0.0)
                    price = price_data.at[current_date, ticker]
                    shares = 0.0
                    if not pd.isna(price) and price > 0 and target_val > 0:
                        shares = target_val / price
                    new_holdings[ticker] = shares
                cash_after_rebalance = current_total_value * target_weights.get('Cash', 0.0)
                portfolio_pb.at[current_date, 'Holdings'] = new_holdings
                portfolio_pb.at[current_date, 'Cash'] = cash_after_rebalance
                # !!! Обновляем цены последней ребалансировки !!!
                new_prices_last_rebalance = {ticker: price_data.at[current_date, ticker]
                                             for ticker in assets_to_rebalance
                                             if not pd.isna(price_data.at[current_date, ticker])}
                portfolio_pb.at[current_date, 'Prices_Last_Rebalance'] = new_prices_last_rebalance
            else:
                 print(f"Warning: Cannot rebalance by price band on {current_date} due to zero/negative total value.")

    price_band_rebalanced_values = portfolio_pb['Total_Value'].copy().rename('Price_Band_Value')

    # --- 3. Логика КОМБИНИРОВАННОЙ ребалансировки (Календарь ИЛИ Цена % с НЕзависимым сбросом цены) --- (новая версия)
    portfolio_comb = pd.DataFrame(index=price_data.index)
    portfolio_comb['Holdings'] = pd.Series(dtype=object)
    portfolio_comb['Cash'] = np.nan
    portfolio_comb['Total_Value'] = np.nan
    portfolio_comb['Prices_Last_Price_Trigger_Rebalance'] = pd.Series(dtype=object) # Отслеживаем цены для ЦЕНОВОГО триггера

    # Инициализация
    portfolio_comb.at[first_date, 'Holdings'] = initial_holdings_template.copy()
    portfolio_comb.at[first_date, 'Cash'] = initial_cash
    portfolio_comb.at[first_date, 'Total_Value'] = initial_total_value_check
    # Цены, от которых отсчитываем ПРОЦЕНТНОЕ изменение
    initial_prices_last_price_trigger_rebalance = {ticker: price_data.at[first_date, ticker]
                                                  for ticker in assets_to_rebalance
                                                  if not pd.isna(price_data.at[first_date, ticker])}
    portfolio_comb.at[first_date, 'Prices_Last_Price_Trigger_Rebalance'] = initial_prices_last_price_trigger_rebalance

    # Используем те же календарные даты: rebalance_dates_cal

    for i in range(1, len(portfolio_comb.index)):
        current_date = portfolio_comb.index[i]
        prev_date = portfolio_comb.index[i-1]
        # Копируем состояние
        portfolio_comb.at[current_date, 'Holdings'] = portfolio_comb.at[prev_date, 'Holdings'].copy()
        portfolio_comb.at[current_date, 'Cash'] = portfolio_comb.at[prev_date, 'Cash']
        portfolio_comb.at[current_date, 'Prices_Last_Price_Trigger_Rebalance'] = portfolio_comb.at[prev_date, 'Prices_Last_Price_Trigger_Rebalance'].copy()
        # Пересчитываем стоимость
        current_total_value = 0.0
        holdings_dict = portfolio_comb.at[current_date, 'Holdings']
        for ticker in all_price_assets:
            shares = holdings_dict.get(ticker, 0.0)
            price = price_data.at[current_date, ticker]
            if shares > 0 and not pd.isna(price) and price > 0:
                current_total_value += shares * price
        current_total_value += portfolio_comb.at[current_date, 'Cash']
        portfolio_comb.at[current_date, 'Total_Value'] = current_total_value

        # Проверка триггеров
        calendar_trigger = current_date in rebalance_dates_cal
        price_trigger = False
        prices_baseline_for_price_trigger = portfolio_comb.at[current_date, 'Prices_Last_Price_Trigger_Rebalance']
        for ticker in assets_to_rebalance:
            current_price = price_data.at[current_date, ticker]
            last_price_rebalance_price = prices_baseline_for_price_trigger.get(ticker)
            if pd.isna(current_price) or pd.isna(last_price_rebalance_price) or last_price_rebalance_price <= 0:
                continue
            if (current_price > last_price_rebalance_price * threshold_decimal_upper) or \
               (current_price < last_price_rebalance_price * threshold_decimal_lower):
                price_trigger = True
                break

        # Ребалансировка, если ЛЮБОЙ триггер сработал
        if calendar_trigger or price_trigger:
            new_holdings = holdings_dict.copy()
            cash_after_rebalance = 0.0
            if current_total_value > 0:
                # (стандартная логика ребалансировки)
                for ticker in assets_to_rebalance:
                    target_val = current_total_value * target_weights.get(ticker, 0.0)
                    price = price_data.at[current_date, ticker]
                    shares = 0.0
                    if not pd.isna(price) and price > 0 and target_val > 0:
                        shares = target_val / price
                    new_holdings[ticker] = shares
                cash_after_rebalance = current_total_value * target_weights.get('Cash', 0.0)
                portfolio_comb.at[current_date, 'Holdings'] = new_holdings
                portfolio_comb.at[current_date, 'Cash'] = cash_after_rebalance

                # !!! Обновляем цены для ЦЕНОВОГО триггера ТОЛЬКО ЕСЛИ он сработал !!!
                if price_trigger:
                    new_prices_last_price_trigger_rebalance = {ticker: price_data.at[current_date, ticker]
                                                               for ticker in assets_to_rebalance
                                                               if not pd.isna(price_data.at[current_date, ticker])}
                    portfolio_comb.at[current_date, 'Prices_Last_Price_Trigger_Rebalance'] = new_prices_last_price_trigger_rebalance
            # else: # Убрал варнинг, т.к. он был и в других циклах
            #      print(f"Warning: Cannot perform combined rebalance on {current_date} due to zero/negative total value.")

    combined_rebalanced_values = portfolio_comb['Total_Value'].copy().rename('Combined_Value')

    # --- 4. Логика Buy & Hold (на основе целевых весов) --- (без изменений)
    initial_prices = price_data.iloc[0]
    bh_target_assets = [col for col in target_weights.keys() if col != 'Cash']
    initial_investment_per_target_asset = {ticker: initial_capital * target_weights.get(ticker, 0.0) for ticker in bh_target_assets}
    initial_target_shares = {}
    for ticker in bh_target_assets:
         price = initial_prices.get(ticker, np.nan)
         if not pd.isna(price) and price > 0:
             initial_target_shares[ticker] = initial_investment_per_target_asset[ticker] / price
         else:
             initial_target_shares[ticker] = 0.0
    cash_bh_target = initial_capital * target_weights.get('Cash', 0.0)
    bh_target_values = pd.Series(index=price_data.index, dtype=float)
    asset_prices_bh_target = price_data[bh_target_assets]
    portfolio_asset_values_bh_target = asset_prices_bh_target.mul(pd.Series(initial_target_shares), axis=1)
    bh_target_values = portfolio_asset_values_bh_target.sum(axis=1) + cash_bh_target
    bh_target_values.rename('BH_Target_Value', inplace=True)

    # --- 5. Логика Buy & Hold (для каждого актива отдельно) --- (без изменений)
    individual_bh_results = {}
    all_tickers = [col for col in price_data.columns if col != 'Cash']
    for ticker in all_tickers:
        initial_price = initial_prices.get(ticker, np.nan)
        if pd.isna(initial_price) or initial_price <= 0:
            bh_individual_values = pd.Series(np.nan, index=price_data.index, name=f"BH_{ticker}")
        else:
            initial_shares_individual = initial_capital / initial_price
            bh_individual_values = price_data[ticker] * initial_shares_individual
            bh_individual_values.rename(f"BH_{ticker}", inplace=True)
        individual_bh_results[f"BH_{ticker}"] = bh_individual_values

    # Собираем все результаты СТОИМОСТЕЙ
    all_results_list = [calendar_rebalanced_values, price_band_rebalanced_values, combined_rebalanced_values, bh_target_values] + list(individual_bh_results.values())
    results = pd.concat(all_results_list, axis=1)
    results.ffill(inplace=True)

    # --- Расчет рядов ПРОСАДОК для основных стратегий --- (добавляем Combined_Value)
    drawdown_results_dict = {}
    main_strategy_cols = ['Calendar_Rebalanced_Value', 'Price_Band_Value', 'Combined_Value', 'BH_Target_Value']
    for col in main_strategy_cols:
        if col in results.columns:
            series = pd.to_numeric(results[col], errors='coerce').dropna()
            if not series.empty:
                 drawdown_results_dict[col] = calculate_drawdown_series(series)
            else:
                 drawdown_results_dict[col] = pd.Series(np.nan, index=results.index, name=col)
        else:
            drawdown_results_dict[col] = pd.Series(np.nan, index=results.index, name=col)

    drawdown_results = pd.concat(drawdown_results_dict, axis=1)

    # Возвращаем КОРТЕЖ из двух DataFrame
    return results, drawdown_results

# --- Функции расчета метрик ---

def _calculate_cagr(series: pd.Series) -> float:
    """Рассчитывает CAGR для временного ряда стоимости."""
    if series.empty or pd.isna(series.iloc[0]) or series.iloc[0] == 0:
        return np.nan
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if pd.isna(end_value):
        # Если конечного значения нет, попробуем взять предпоследнее non-NA
        valid_series = series.dropna()
        if valid_series.empty:
             return np.nan
        end_value = valid_series.iloc[-1]
        end_date = valid_series.index[-1]
    else:
        end_date = series.index[-1]

    start_date = series.index[0]
    num_years = (end_date - start_date).days / 365.25
    if num_years <= 0:
        return np.nan
    if end_value <= 0 and start_value > 0:
        return -1.0
    if end_value < 0 and start_value < 0:
         return np.nan # Неопределенность
    # Избегаем деления на ноль или отрицательный старт
    if start_value <= 0:
         return np.nan
    ratio = end_value / start_value
    if ratio < 0: # Не может быть отрицательного отношения для CAGR
         return np.nan
    cagr = ratio ** (1 / num_years) - 1
    return cagr

def _calculate_max_drawdown(series: pd.Series) -> (float, float, float):
    """Рассчитывает максимальную просадку (в %), абсолютную просадку и пик перед ней."""
    if series.empty or series.isna().all():
        return np.nan, np.nan, np.nan
    series = series.dropna()
    if series.empty:
         return np.nan, np.nan, np.nan
    cumulative_max = series.cummax()
    # Заменяем 0 на NaN, чтобы избежать деления на ноль, если серия начинается с 0 или имеет 0
    safe_cumulative_max = cumulative_max.replace(0, np.nan)
    drawdown = (series - safe_cumulative_max) / safe_cumulative_max
    drawdown = drawdown.fillna(0) # Заполняем NaN (возникающие из-за safe_cumulative_max) нулями

    if drawdown.empty:
         return 0.0, 0.0, series.iloc[0] if not series.empty else np.nan

    max_drawdown_pct = drawdown.min()
    # Если все просадки 0, то ищем первый индекс
    try:
         idx_min = drawdown.idxmin()
    except ValueError: # Может возникнуть, если все значения одинаковы
         idx_min = series.index[0]

    peak_before_max_drawdown = cumulative_max.loc[idx_min] if idx_min in cumulative_max.index else series.iloc[0]

    if pd.isna(max_drawdown_pct) or pd.isna(peak_before_max_drawdown):
         max_drawdown_abs = 0.0 # Или np.nan? Если pct NaN, то и abs NaN
    else:
         max_drawdown_abs = max_drawdown_pct * peak_before_max_drawdown

    return max_drawdown_pct, max_drawdown_abs, peak_before_max_drawdown


def _calculate_sharpe(series: pd.Series, risk_free_rate_annual: float) -> float:
    """Рассчитывает коэффициент Шарпа (аннуализированный)."""
    if series.empty or series.isna().all() or len(series) < 2:
        return np.nan
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return np.nan
    # Обработка inf/-inf
    if np.isinf(daily_returns).any():
         print("Warning: Inf values detected in daily returns for Sharpe calculation.")
         daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
         if daily_returns.empty:
              return np.nan
    # Проверка на нулевое стандартное отклонение
    std_dev = daily_returns.std()
    if std_dev < 1e-10 or pd.isna(std_dev):
         # Если стандартное отклонение 0, Шарп не определен, если только средняя доходность не равна безрисковой
         mean_return = daily_returns.mean()
         risk_free_daily = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
         if abs(mean_return - risk_free_daily) < 1e-10:
              return 0.0 # Или np.nan? Часто считают 0, если нет избыточной доходности и риска
         else:
              # Бесконечный Шарп? Возвращаем NaN как неопределенность
              return np.nan
    excess_returns = daily_returns - (risk_free_rate_annual / TRADING_DAYS_PER_YEAR)
    sharpe_ratio = np.mean(excess_returns) / std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe_ratio

def _calculate_sortino(series: pd.Series, risk_free_rate_annual: float) -> float:
    """Рассчитывает коэффициент Сортино (аннуализированный)."""
    if series.empty or series.isna().all() or len(series) < 2:
        return np.nan
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return np.nan
    # Обработка inf/-inf
    if np.isinf(daily_returns).any():
         print("Warning: Inf values detected in daily returns for Sortino calculation.")
         daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
         if daily_returns.empty:
              return np.nan

    target_return_daily = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    downside_returns = daily_returns[daily_returns < target_return_daily]

    # Проверяем, есть ли отрицательные отклонения
    if downside_returns.empty:
        # Если нет доходностей ниже целевой, Сортино не определен или бесконечен?
        # Если средняя доходность > целевой, то риск = 0, можно вернуть inf или nan
        # Если средняя доходность <= целевой, но все >= target, можно вернуть 0?
        # Часто возвращают NaN в этом случае.
        return np.nan

    downside_deviation = np.sqrt(np.mean((downside_returns - target_return_daily)**2))

    # Проверка на нулевое отклонение вниз
    if downside_deviation < 1e-10 or pd.isna(downside_deviation):
        # Если риск = 0, Сортино не определен, если только избыточная доходность не 0
        mean_excess_return = np.mean(daily_returns - target_return_daily)
        if abs(mean_excess_return) < 1e-10:
             return 0.0 # Нет изб. доходности, нет риска -> 0
        else:
             # Положительная изб. доходность при нулевом риске -> Бесконечность? Вернем NaN.
             return np.nan

    mean_excess_return = np.mean(daily_returns - target_return_daily)
    sortino_ratio = mean_excess_return / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino_ratio


def _calculate_recovery_factor(series: pd.Series, max_drawdown_abs: float) -> float:
     """Рассчитывает фактор восстановления."""
     if series.empty or series.isna().all():
         return np.nan
     series = series.dropna()
     if series.empty or len(series) < 2:
         return np.nan
     # max_drawdown_abs должен быть отрицательным или 0
     if pd.isna(max_drawdown_abs) or max_drawdown_abs > 1e-10 : # Проверка, что не положительный
         # Если просадки не было (max_drawdown_abs ~ 0) или она не рассчитана, фактор не определен
         return np.nan
     if abs(max_drawdown_abs) < 1e-10: # Если просадка 0
        # Если есть прибыль, фактор восстановления бесконечен? Вернем NaN
        # Если прибыли нет, фактор 0/0? Вернем NaN
        return np.nan

     total_profit = series.iloc[-1] - series.iloc[0]
     if pd.isna(total_profit):
          return np.nan
     # Делим на абсолютную величину просадки
     recovery_factor = total_profit / abs(max_drawdown_abs)
     return recovery_factor


@st.cache_data # Кэшируем расчет метрик
def calculate_metrics(backtest_results: pd.DataFrame, risk_free_rate_annual: float) -> Dict[str, Dict[str, float]]:
    """
    Рассчитывает набор метрик эффективности для результатов бэктеста.

    Args:
        backtest_results (pd.DataFrame): DataFrame с результатами бэктеста для всех стратегий
                                           (колонки 'Rebalanced_Value', 'BH_Target_Value', 'BH_SPY', 'BH_GLD', ...).
        risk_free_rate_annual (float): Годовая безрисковая ставка (десятичная дробь, например, 0.035).

    Returns:
        Dict[str, Dict[str, float]]: Словарь с метриками для всех стратегий.
                                       Ключи метрик: 'Start Value', 'End Value', 'CAGR',
                                       'Max Drawdown %', 'Volatility', 'Sharpe Ratio',
                                       'Sortino Ratio', 'Recovery Factor'.
                                       Значения могут быть np.nan, если расчет невозможен.
    """
    metrics = {}
    strategies = {}

    # Динамически определяем стратегии и их отображаемые имена
    strategy_names = {}
    if 'Calendar_Rebalanced_Value' in backtest_results.columns:
        strategy_names['Calendar_Rebalanced_Value'] = 'Ребаланс (Календарь)'
    if 'Price_Band_Value' in backtest_results.columns:
        strategy_names['Price_Band_Value'] = 'Ребаланс (Цена %)'
    if 'Combined_Value' in backtest_results.columns:
        strategy_names['Combined_Value'] = 'Ребаланс (Комби)'
    if 'BH_Target_Value' in backtest_results.columns:
        strategy_names['BH_Target_Value'] = 'B&H (Целевые веса)'
    # Добавляем индивидуальные B&H
    for col in backtest_results.columns:
        if col.startswith('BH_') and col != 'BH_Target_Value':
            ticker = col.split('_', 1)[1]
            strategy_names[col] = f'B&H ({ticker})'

    metric_keys = ['Start Value', 'End Value', 'CAGR', 'Max Drawdown %', 'Max Drawdown Abs $', 'Peak Before MDD', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Recovery Factor']
    nan_metrics = {key: np.nan for key in metric_keys}

    for col_name, display_name in strategy_names.items():
        if col_name in backtest_results.columns:
            strategies[display_name] = backtest_results[col_name]
        else:
            # Если стратегии нет, создаем пустую серию с NaN чтобы не ломать расчеты
            strategies[display_name] = pd.Series(np.nan, index=backtest_results.index)

    if not strategies:
        print("Error: Backtest results DataFrame does not contain any valid strategy columns.")
        return {name: nan_metrics.copy() for name in strategy_names.values()}

    for name, series in strategies.items():
        valid_series = series.dropna()
        if valid_series.empty or len(valid_series) < 2:
            metrics[name] = nan_metrics.copy()
            if not series.empty:
                 metrics[name]['Start Value'] = series.iloc[0] if not pd.isna(series.iloc[0]) else np.nan
                 metrics[name]['End Value'] = series.iloc[-1] if not pd.isna(series.iloc[-1]) else np.nan
            continue
        start_value = valid_series.iloc[0]
        end_value = valid_series.iloc[-1]
        cagr = _calculate_cagr(valid_series)
        max_drawdown_pct, max_drawdown_abs, peak = _calculate_max_drawdown(valid_series)
        volatility = valid_series.pct_change().std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = _calculate_sharpe(valid_series, risk_free_rate_annual)
        sortino = _calculate_sortino(valid_series, risk_free_rate_annual)
        recovery = _calculate_recovery_factor(valid_series, max_drawdown_abs)

        metrics[name] = {
            'Start Value': start_value if not pd.isna(start_value) else np.nan,
            'End Value': end_value if not pd.isna(end_value) else np.nan,
            'CAGR': cagr,
            'Max Drawdown %': max_drawdown_pct * 100 if not pd.isna(max_drawdown_pct) else np.nan,
            'Max Drawdown Abs $': max_drawdown_abs if not pd.isna(max_drawdown_abs) else np.nan,
            'Peak Before MDD': peak if not pd.isna(peak) else np.nan,
            'Volatility': volatility * 100 if not pd.isna(volatility) else np.nan,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Recovery Factor': recovery
        }

    for display_name in strategy_names.values():
        if display_name not in metrics:
            metrics[display_name] = nan_metrics.copy()

    return metrics 