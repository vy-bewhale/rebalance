import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict
import numpy as np

def plot_normalized_prices(price_data: pd.DataFrame, ticker_map: Dict[str, str]) -> Optional[go.Figure]:
    """
    Строит график нормализованных цен активов.

    Args:
        price_data (pd.DataFrame): DataFrame с ценами закрытия активов (индекс - дата).
        ticker_map (Dict[str, str]): Словарь для отображения тикеров как названий.

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None, если входные данные некорректны.
    """
    if price_data is None or price_data.empty:
        return None

    price_data_assets = price_data.drop(columns=['Cash'], errors='ignore')
    if price_data_assets.empty:
        print("Warning: Price data for normalization is empty.")
        return None

    try:
        if (price_data_assets.iloc[0] == 0).any():
            print("Warning: Zero prices found in the first row. Cannot normalize.")
            return None
        normalized_data = price_data_assets / price_data_assets.iloc[0] * 100
    except Exception as e:
        print(f"Error during price normalization: {e}")
        return None

    fig = go.Figure()

    for ticker in normalized_data.columns:
        display_name = ticker_map.get(ticker, ticker)
        fig.add_trace(go.Scatter(
            x=normalized_data.index,
            y=normalized_data[ticker],
            mode='lines',
            name=display_name,
        ))

    fig.update_layout(
        title='Динамика нормализованных цен активов (база 100 на начало периода)',
        xaxis_title='Дата',
        yaxis_title='Нормализованная цена',
        legend_title='Активы',
        legend=dict(font=dict(size=10)),
        hovermode='x unified'
    )
    return fig

def plot_equity_curves(backtest_results: pd.DataFrame, ticker_map: Dict[str, str]) -> Optional[go.Figure]:
    """
    Строит график кривых эквити для всех рассчитанных стратегий.

    Args:
        backtest_results (pd.DataFrame): DataFrame с результатами бэктеста.
        ticker_map (Dict[str, str]): Словарь для отображения тикеров как названий (для B&H по активам).

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None, если входные данные некорректны.
    """
    if backtest_results is None or backtest_results.empty:
        return None

    strategy_columns = {}
    if 'Calendar_Rebalanced_Value' in backtest_results.columns:
        strategy_columns['Calendar_Rebalanced_Value'] = 'Ребаланс (Календарь)'
    if 'Price_Band_Value' in backtest_results.columns:
        strategy_columns['Price_Band_Value'] = 'Ребаланс (Цена %)'
    if 'Combined_Value' in backtest_results.columns:
        strategy_columns['Combined_Value'] = 'Ребаланс (Комби)'
    if 'BH_Target_Value' in backtest_results.columns:
        strategy_columns['BH_Target_Value'] = 'B&H (Целевые веса)'
    for col in backtest_results.columns:
        if col.startswith('BH_') and col != 'BH_Target_Value':
            ticker = col.split('_', 1)[1]
            display_ticker_name = ticker_map.get(ticker, ticker)
            strategy_columns[col] = f'B&H ({display_ticker_name})'

    fig = go.Figure()

    main_strategies = ['Calendar_Rebalanced_Value', 'Price_Band_Value', 'Combined_Value', 'BH_Target_Value']
    for col_name in main_strategies:
        if col_name in backtest_results.columns:
             display_name = strategy_columns.get(col_name, col_name)
             fig.add_trace(go.Scatter(
                 x=backtest_results.index,
                 y=backtest_results[col_name],
                 mode='lines',
                 name=display_name,
             ))

    for col_name, display_name in strategy_columns.items():
        if col_name not in main_strategies and col_name in backtest_results.columns:
             fig.add_trace(go.Scatter(
                 x=backtest_results.index,
                 y=backtest_results[col_name],
                 mode='lines',
                 name=display_name,
                 line=dict(dash='dot')
             ))

    fig.update_layout(
        title='Динамика стоимости портфелей',
        xaxis_title='Дата',
        yaxis_title='Стоимость портфеля',
        yaxis_type='log',
        legend_title='Стратегия',
        legend=dict(font=dict(size=10)),
        hovermode='x unified'
    )
    return fig

def plot_drawdown_curves(drawdown_results: pd.DataFrame) -> Optional[go.Figure]:
    """Строит график временных рядов просадок для основных стратегий.

    Args:
        drawdown_results (pd.DataFrame): DataFrame с рядами просадок.

    Returns:
        Optional[go.Figure]: Объект фигуры Plotly или None.
    """
    if drawdown_results is None or drawdown_results.empty:
        return None

    fig = go.Figure()

    # Определяем имена для легенды (совпадают с именами в calculate_metrics)
    strategy_names_map = {
        'Calendar_Rebalanced_Value': 'Ребаланс (Календарь)',
        'Price_Band_Value': 'Ребаланс (Цена %)',
        'Combined_Value': 'Ребаланс (Комби)',
        'BH_Target_Value': 'B&H (Целевые веса)'
    }

    for col in drawdown_results.columns:
        if col in strategy_names_map:
            # Умножаем на 100 для отображения в процентах
            drawdown_percent = drawdown_results[col] * 100
            display_name = strategy_names_map[col]
            fig.add_trace(go.Scatter(
                x=drawdown_percent.index,
                y=drawdown_percent,
                mode='lines',
                name=display_name
            ))

    fig.update_layout(
        title='Динамика просадок портфелей',
        xaxis_title='Дата',
        yaxis_title='Просадка (%)',
        legend_title='Стратегия',
        legend=dict(font=dict(size=10)),
        hovermode='x unified',
        yaxis_tickformat='.1f' # Формат оси Y с одним знаком после запятой
    )
    # Устанавливаем диапазон оси Y, чтобы 0 был наверху
    min_drawdown = drawdown_results.min().min() * 100 # Минимальная просадка по всем стратегиям в %
    if not pd.isna(min_drawdown):
         fig.update_yaxes(range=[min_drawdown * 1.1, 5]) # Чуть ниже минимума и до 5%
    else:
         fig.update_yaxes(range=[-50, 5]) # Диапазон по умолчанию, если нет данных

    return fig 