import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit



def plot_candles(
    df,
    trendlines=None,  # Liste de dicts contenant slope, intercept, start_time, end_time
):
    # Candlestick trace
    candle = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles',
        increasing_line_color='green',
        decreasing_line_color='red'
    )

    fig = go.Figure(data=[candle])

    # Ajoute les trendlines si spécifiées
    if trendlines:
        for i, trend in enumerate(trendlines):
            slope = trend.get('slope')
            intercept = trend.get('intercept')
            start_time = trend.get('start_time')
            end_time = trend.get('end_time')
            color = trend.get('color', 'dodgerblue')
            dash = trend.get('dash', 'dash')
            name = trend.get('name', f'Trend {i+1}')

            if start_time not in df.index or end_time not in df.index:
                print(f"⚠️ Timestamps non trouvés dans l'index pour la trendline {i+1}.")
                continue

            start_idx = df.index.get_loc(start_time)
            end_idx = df.index.get_loc(end_time)

            trend_line = np.full(len(df), np.nan)
            X_segment = np.arange(end_idx - start_idx + 1)
            trend_segment = slope * X_segment + intercept
            trend_line[start_idx:end_idx + 1] = trend_segment

            trend_trace = go.Scatter(
                x=df.index,
                y=trend_line,
                mode='lines',
                name=name,
                line=dict(color=color, width=2, dash=dash)
            )

            fig.add_trace(trend_trace)


    # Layout final
    fig.update_layout(
        title="Trendlines",
        xaxis_title='Date',
        yaxis_title='Prix',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=600,
        width=1000
    )

    fig.show()

    return fig


def get_best_fit_slope_intercept(y : np.array):
    # best fit line
    X = np.arange(len(y)) # Independent variable
    coefs = np.polyfit(X, y, 1)

    # Get the coefficients
    slope = coefs[0]
    intercept = coefs[1]

    return slope, intercept



@njit
def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    intercept = -slope * pivot + y[pivot]
    err = 0.0

    for i in range(len(y)):
        line_val = slope * i + intercept
        diff = line_val - y[i]

        if support and diff > 1e-5:
            return -1.0
        elif not support and diff < -1e-5:
            return -1.0

        err += diff * diff

    return err



@njit
def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray):
    slope_unit = (np.max(y) - np.min(y)) / len(y)

    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    
    get_derivative = True
    derivative = 0.0

    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                return (best_slope, -best_slope * pivot + y[pivot])

            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y[pivot])


@njit
def fit_trendlines(y: np.ndarray):
    x = np.arange(len(y)).astype(np.float64)
    
    # Calculer les coefficients de la régression linéaire (moindres carrés)
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    num = 0.0
    den = 0.0
    for i in range(n):
        num += (x[i] - x_mean) * (y[i] - y_mean)
        den += (x[i] - x_mean) ** 2

    slope = num / den
    intercept = y_mean - slope * x_mean

    # Points de la droite
    line_points = slope * x + intercept

    # Trouver les pivots
    max_diff = -np.inf
    min_diff = np.inf
    upper_pivot = 0
    lower_pivot = 0

    for i in range(n):
        diff = y[i] - line_points[i]
        if diff > max_diff:
            max_diff = diff
            upper_pivot = i
        if diff < min_diff:
            min_diff = diff
            lower_pivot = i

    # Optimisation
    support_coefs = optimize_slope(True, lower_pivot, slope, y)
    resist_coefs = optimize_slope(False, upper_pivot, slope, y)

    return (support_coefs, resist_coefs)



@njit
def candles_close_to_trendline_numba(y: np.ndarray, slope: float, intercept: float, median_candle_width: float, support: bool) -> int:
    count = 0
    for i in range(len(y)):
        trend_val = slope * i + intercept
        if support:
            if y[i] - trend_val <= median_candle_width:
                count += 1
        else:
            if trend_val - y[i] <= median_candle_width:
                count += 1
    return count



@njit
def dist_from_trendline_numba(y: np.ndarray, slope: float, intercept: float, percentile: float, support: bool) -> float:
    n = len(y)
    diffs = np.empty(n)
    for i in range(n):
        trend_val = slope * i + intercept
        diffs[i] = y[i] - trend_val if support else trend_val - y[i]

    # Supprimer les NaN à la main
    clean_diffs = []
    for i in range(n):
        if not np.isnan(diffs[i]):
            clean_diffs.append(diffs[i])

    if len(clean_diffs) == 0:
        return 0.0

    sorted_diffs = np.sort(np.array(clean_diffs))
    rank = int((percentile / 100.0) * (len(sorted_diffs) - 1))
    return abs(sorted_diffs[rank])


def _compute_window_score(df, percentile, start, end):
    df_window = df.iloc[start:end+1]
    
    # Conversion en ndarray pour Numba
    close_array = df_window['close'].to_numpy()

    # Calcul des lignes de tendance
    support_coefs, resistance_coefs = fit_trendlines(close_array)
    slope_final_resistance, intercept_final_resistance = resistance_coefs
    slope_final_support, intercept_final_support = support_coefs

    # Pré-calcul du candle width en ndarray
    open_array = df_window['open'].to_numpy()
    body_sizes = np.abs(open_array - close_array)
    median_width = np.nanpercentile(body_sizes, percentile)

    # Utilisation des fonctions numba-optimisées
    candles_close_to_trendline_support = candles_close_to_trendline_numba(
        close_array, slope_final_support, intercept_final_support, median_width, support=True)

    candles_close_to_trendline_resistance = candles_close_to_trendline_numba(
        close_array, slope_final_resistance, intercept_final_resistance, median_width, support=False)

    dist_from_trendline_support = dist_from_trendline_numba(
        close_array, slope_final_support, intercept_final_support, percentile, support=True)

    dist_from_trendline_resistance = dist_from_trendline_numba(
        close_array, slope_final_resistance, intercept_final_resistance, percentile, support=False)

    return {
        'support_line': {'slope': slope_final_support, 'intercept': intercept_final_support},
        'resistance_line': {'slope': slope_final_resistance, 'intercept': intercept_final_resistance},
        'candles_close_to_trendline': {
            'support': candles_close_to_trendline_support,
            'resistance': candles_close_to_trendline_resistance
        },
        'dist_from_trendline': {
            'support': dist_from_trendline_support,
            'resistance': dist_from_trendline_resistance
        },
        'range': {'start': df.index[start], 'end': df.index[end]},
        'candles_number': end - start + 1
    }


def compute_trendline_scores_parallel(df, percentile, min_window=20, n_jobs=-1):
    n = len(df)
    tasks = [
        (start, end)
        for start in range(n)
        for end in range(start + min_window, n)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_window_score)(df, percentile, start, end)
        for start, end in tqdm(tasks, desc="Analyse en parallèle")
    )

    return pd.DataFrame(results)



def compute_time_overlap(range1, range2):
    latest_start = max(range1['start'], range2['start'])
    earliest_end = min(range1['end'], range2['end'])
    overlap = (earliest_end - latest_start).total_seconds()
    duration1 = (range1['end'] - range1['start']).total_seconds()
    duration2 = (range2['end'] - range2['start']).total_seconds()
    if overlap <= 0:
        return 0.0
    return overlap / min(duration1, duration2)

def compute_canal_score(row, w_proximity, w_bougies, w_duration):
    r1 = row['range']
    touches = row['candles_close_to_trendline']
    distances = row['dist_from_trendline']

    proximity_support = 1 / (1 + distances['support'])
    proximity_resistance = 1 / (1 + distances['resistance'])
    avg_proximity = (proximity_support + proximity_resistance) / 2

    touch_score = (touches['support'] + touches['resistance']) / 2

    duration = (r1['end'] - r1['start']).total_seconds() / 3600  # en heures

    score = (
        w_proximity * avg_proximity +
        w_bougies * touch_score +
        w_duration * duration
    )
    return score


def has_three_spaced_contacts(y, slope, intercept, tolerance, min_spacing=5):
    touches = []

    for i in range(len(y)):
        expected = slope * i + intercept
        if abs(y[i] - expected) <= tolerance:
            if not touches or i - touches[-1] >= min_spacing:
                touches.append(i)
                if len(touches) >= 3:
                    return True

    return False


def is_wedge(
    support_line,
    resistance_line,
    range_window,
    y_close=None,
    slope_diff_threshold=10,
    min_abs_slope=5,
    require_convergence=True,
    max_end_to_start_ratio=0.3,
    require_contacts=True,
    contact_tolerance_factor=1.0,
    min_spacing=5
):
    """
    Détecte si un canal est un biseau (wedge), selon :
    - pentes convergentes et inclinées
    - convergence spatiale mesurée
    - minimum 3 contacts espacés sur chaque ligne (optionnel)
    """

    s_slope = support_line['slope']
    r_slope = resistance_line['slope']
    s_intercept = support_line['intercept']
    r_intercept = resistance_line['intercept']

    # 1. Les deux lignes doivent avoir une pente de même signe
    if s_slope * r_slope <= 0:
        return False

    # 2. Les pentes doivent être suffisamment inclinées
    if abs(s_slope) < min_abs_slope or abs(r_slope) < min_abs_slope:
        return False

    # 3. Les pentes doivent être significativement différentes
    if abs(s_slope - r_slope) < slope_diff_threshold:
        return False

    # 4. Convergence : distance entre les lignes doit diminuer
    if require_convergence:
        duration = (range_window['end'] - range_window['start']).total_seconds() / 3600
        x_start = 0
        x_end = duration

        support_start = s_slope * x_start + s_intercept
        support_end = s_slope * x_end + s_intercept

        resistance_start = r_slope * x_start + r_intercept
        resistance_end = r_slope * x_end + r_intercept

        dist_start = resistance_start - support_start
        dist_end = resistance_end - support_end

        if abs(dist_end) >= abs(dist_start):
            return False

        if abs(dist_end) / abs(dist_start) > max_end_to_start_ratio:
            return False

    # 5. Vérification des contacts espacés
    if require_contacts and y_close is not None:
        # tolérance basée sur la médiane des variations
        tolerance = contact_tolerance_factor * np.nanpercentile(np.abs(np.diff(y_close)), 50)

        support_ok = has_three_spaced_contacts(y_close, s_slope, s_intercept, tolerance, min_spacing)
        resistance_ok = has_three_spaced_contacts(y_close, r_slope, r_intercept, tolerance, min_spacing)

        if not (support_ok and resistance_ok):
            return False

    return True
