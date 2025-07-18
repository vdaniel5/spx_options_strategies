# SPX_strategy_recommender.py (Fixed + Enhanced Logic v3)

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from scipy.stats import norm

import logging
logging.basicConfig(
    filename='spx_strategy.log',
    level=logging.DEBUG,  # <- changed from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------- CONFIG ------------------------
TRADIER_BASE_URL = "https://api.tradier.com/v1"
TRADIER_HEADERS = {
    "Authorization": "your-key-here",
    "Accept": "application/json"
}
# Strategy parameters
MIN_VOLUME = 100
MIN_OPEN_INTEREST = 350
IV_RANK_THRESHOLD_LOW = 0.25
IV_RANK_THRESHOLD_HIGH = 0.75
MIN_DTE = 3
MAX_DTE = 45
MAX_SPREAD_WIDTH = 10
LOOKBACK_DAYS = 150 # For IV Rank and technicals
ADX_THRESHOLD = 25 # Minimum ADX for a valid trend

# ------------------- HELPERS ------------------

def get_spx_price():
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": "SPX"}
    try:
        logging.info('Fetching SPX price')
        resp = requests.get(url, headers=TRADIER_HEADERS, params=params)
        logging.debug(f'API Response: {resp.text[:500]}')
        resp.raise_for_status()
        data = resp.json()
        return float(data['quotes']['quote']['last'])
    except Exception as e:
        print(f"Error getting SPX price: {e}")
        return 4500.0

def get_all_expirations():
    url = f"{TRADIER_BASE_URL}/markets/options/expirations"
    try:
        logging.info('Fetching all expiration dates')
        resp = requests.get(url, headers=TRADIER_HEADERS, params={"symbol": "SPX", "includeAllRoots": "true"})
        logging.debug(f'API Response: {resp.text[:500]}')
        resp.raise_for_status()
        dates = resp.json().get('expirations', {}).get('date', [])
        dates = [dates] if isinstance(dates, str) else dates
        today = datetime.utcnow().date()
        return sorted([d for d in dates if MIN_DTE <= (datetime.strptime(d, "%Y-%m-%d").date() - today).days <= MAX_DTE])
    except Exception as e:
        print(f"Error getting expirations: {e}")
        return []


def get_options_chain(exp_date):
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    try:
        logging.info('Fetching options chain')
        resp = requests.get(url, headers=TRADIER_HEADERS, params={"symbol": "SPX", "expiration": exp_date, "greeks": "true"})
        logging.debug(f'API Response: {resp.text[:500]}')
        resp.raise_for_status()
        opts = resp.json().get('options', {}).get('option', [])
        df = pd.DataFrame(opts)
        if 'greeks' in df.columns:
            greeks_df = pd.json_normalize(df['greeks'])
            iv_col = 'mid_iv' if 'mid_iv' in greeks_df.columns else 'iv'
            greeks_df.rename(columns={iv_col: 'implied_volatility'}, inplace=True)
            df = pd.concat([df.drop(columns=['greeks']), greeks_df], axis=1)
        for col in ['strike', 'bid', 'ask', 'volume', 'open_interest', 'delta', 'theta', 'vega', 'implied_volatility']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['mid'] = (df['bid'] + df['ask']) / 2
        df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['volume'] >= MIN_VOLUME) & (df['open_interest'] >= MIN_OPEN_INTEREST)]
        return df[df['option_type'] == 'call'], df[df['option_type'] == 'put']
    except Exception as e:
        print(f"Error getting options chain for {exp_date}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def analyze_timeframe_trend(df):
    if len(df) < 50:
        return 'neutral'
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    adx = ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
    macd = MACD(df['close'], 26, 12, 9)
    sma_20 = SMAIndicator(df['close'], 20).sma_indicator()
    sma_50 = SMAIndicator(df['close'], 50).sma_indicator()
    is_bullish = (macd.macd().iloc[-1] > macd.macd_signal().iloc[-1] and df['close'].iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1])
    is_bearish = (macd.macd().iloc[-1] < macd.macd_signal().iloc[-1] and df['close'].iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1])
    if adx.iloc[-1] >= ADX_THRESHOLD:
        return 'bullish' if is_bullish else 'bearish' if is_bearish else 'neutral'
    return 'weak_bullish' if is_bullish else 'weak_bearish' if is_bearish else 'neutral'
    
def get_market_bias_and_vol(verbose=True):
    logging.info('Determining market bias and volatility')
    timeframes = {'daily': 200}
    trends = {}
    hist_data = {}

    for interval, days in timeframes.items():
        try:
            url = f"{TRADIER_BASE_URL}/markets/history"
            params = {
                "symbol": "SPX",
                "interval": interval,
                "start": (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d'),
                "end": datetime.utcnow().strftime('%Y-%m-%d')
            }
            resp = requests.get(url, headers=TRADIER_HEADERS, params=params)
            logging.debug(f'API Response: {resp.text[:500]}')
            resp.raise_for_status()
            key = 'day' if interval == 'daily' else 'data'
            df = pd.DataFrame(resp.json()['history'][key])

            for col in ['close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            hist_data[interval] = df
            trends[interval] = analyze_timeframe_trend(df)
        except Exception as e:
            print(f"Error fetching {interval} data: {e}")
            trends[interval] = 'neutral'

    # Fetch SPY volume data for volume confirmation
    try:
        spy_url = f"{TRADIER_BASE_URL}/markets/history"
        spy_params = {
            "symbol": "SPY",
            "interval": "daily",
            "start": (datetime.utcnow() - timedelta(days=200)).strftime('%Y-%m-%d'),
            "end": datetime.utcnow().strftime('%Y-%m-%d')
        }
        spy_resp = requests.get(spy_url, headers=TRADIER_HEADERS, params=spy_params)
        logging.debug(f'SPY Volume API Response: {spy_resp.text[:500]}')
        spy_resp.raise_for_status()
        spy_df = pd.DataFrame(spy_resp.json()['history']['day'])
        spy_df['volume'] = pd.to_numeric(spy_df['volume'], errors='coerce')
        spy_df.dropna(subset=['volume'], inplace=True)
    except Exception as e:
        print(f"Error fetching SPY volume data: {e}")
        spy_df = pd.DataFrame()

    final_bias = 'neutral'
    bias_reason = "No definitive signals found."

    daily_df = hist_data.get('daily')
    daily_df['close'] = pd.to_numeric(daily_df['close'], errors='coerce')
    daily_df['high'] = pd.to_numeric(daily_df['high'], errors='coerce')
    daily_df['low'] = pd.to_numeric(daily_df['low'], errors='coerce')
    daily_df.dropna(inplace=True)

    score = 0
    trend = analyze_timeframe_trend(daily_df)

    # Trend direction from MACD + SMA (already incorporated in analyze_timeframe_trend)
    if trend in ['bullish', 'bearish']:
        score += 2
    elif trend in ['weak_bullish', 'weak_bearish']:
        score += 1

    # ADX value
    adx_val = ADXIndicator(daily_df['high'], daily_df['low'], daily_df['close'], 14).adx().iloc[-1]
    if adx_val > 25:
        score += 1
    elif adx_val > 20:
        score += 0.5

    # SPY volume confirmation
    if not spy_df.empty:
        volume_ma = spy_df['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = spy_df['volume'].iloc[-1]
        if current_volume > 1.05 * volume_ma:
            score += 1
        elif current_volume > volume_ma:
            score += 0.5

    # RSI-based momentum and mean reversion
    rsi_val = RSIIndicator(daily_df['close'], 14).rsi().iloc[-1]
    if rsi_val > 70:
        score -= 1  # mean reversion bearish
    elif rsi_val < 30:
        score += 1  # mean reversion bullish
    elif rsi_val > 55:
        score += 0.5  # mild bullish momentum
    elif rsi_val < 45:
        score -= 0.5  # mild bearish momentum

    # Price above 50-day SMA
    sma_50 = SMAIndicator(daily_df['close'], 50).sma_indicator()
    if daily_df['close'].iloc[-1] > sma_50.iloc[-1]:
        score += 0.5
    else:
        score -= 0.5

    # Final bias decision
    if score >= 3:
        final_bias = 'bullish' if 'bullish' in trend else 'bearish'
        bias_reason = f"Strong {final_bias.upper()} bias based on multiple indicators."
    elif 1.5 <= score < 3:
        final_bias = 'weak_bullish' if 'bullish' in trend else 'weak_bearish'
        bias_reason = f"Moderate {final_bias.upper()} bias from mixed signals."
    else:
        final_bias = 'neutral'
        bias_reason = "Insufficient conviction across indicators."

    if verbose:
        print(f"Final Bias: {final_bias.upper()} | Reason: {bias_reason}")

    daily_df['log_return'] = np.log(daily_df['close'] / daily_df['close'].shift(1))
    historical_vol = (daily_df['log_return'].rolling(window=20).std() * np.sqrt(252)).dropna().tolist()
    logging.debug(f"Bias score breakdown → Trend: {trend}, ADX: {adx_val:.2f}, RSI: {rsi_val:.2f}, Volume: {current_volume:.0f} vs MA: {volume_ma:.0f}, Price > SMA50: {daily_df['close'].iloc[-1] > sma_50.iloc[-1]}")
    logging.debug(f"Total Score: {score:.2f}")
    return final_bias, historical_vol
    
def calculate_iv_rank(current_iv, historical_ivs):
    """Calculate IV rank - what percentile current IV is in historical range"""
    if not historical_ivs or len(historical_ivs) < 20: return 0.5
    historical_ivs = np.array(historical_ivs)
    return sum(1 for iv in historical_ivs if iv < current_iv) / len(historical_ivs)

# --- Functions for filtering and building recommendations ---

def filter_options_by_delta(df, target_delta, is_put, tolerance=0.05):
    if df.empty: return None
    df_clean = df.dropna(subset=['delta', 'mid']).copy()
    if df_clean.empty: return None
    df_clean['delta_diff'] = (df_clean['delta'] + abs(target_delta)).abs() if is_put else (df_clean['delta'] - target_delta).abs()
    within_tolerance = df_clean[df_clean['delta_diff'] <= tolerance]
    if not within_tolerance.empty:
        return within_tolerance.loc[within_tolerance['delta_diff'].idxmin()]
    return df_clean.loc[df_clean['delta_diff'].idxmin()]

def filter_credit_spread_improved(df, is_put, spot_price):
    if df.empty: return None, None
    short_delta = -0.20 if is_put else 0.20
    candidates = df[df['strike'] <= spot_price * 0.95] if is_put else df[df['strike'] >= spot_price * 1.05]
    if candidates.empty: return None, None
    short_leg = filter_options_by_delta(candidates, abs(short_delta), is_put)
    if short_leg is None: return None, None
    short_strike = short_leg['strike']
    if is_put:
        long_candidates = df[(df['strike'] < short_strike) & (df['strike'] >= short_strike - MAX_SPREAD_WIDTH)]
    else:
        long_candidates = df[(df['strike'] > short_strike) & (df['strike'] <= short_strike + MAX_SPREAD_WIDTH)]
    if long_candidates.empty: return short_leg, None
    long_leg = filter_options_by_delta(long_candidates, 0.05, is_put)
    return short_leg, long_leg

def filter_debit_spread(df, is_call, spot_price):
    if df.empty: return None, None
    if is_call:
        long_leg = filter_options_by_delta(df[df['strike'] <= spot_price * 1.02], 0.60, False)
        if long_leg is None: return None, None
        short_candidates = df[(df['strike'] > long_leg['strike']) & (df['strike'] <= long_leg['strike'] + MAX_SPREAD_WIDTH)]
        short_leg = filter_options_by_delta(short_candidates, 0.30, False)
    else:
        long_leg = filter_options_by_delta(df[df['strike'] >= spot_price * 0.98], 0.50, True)
        if long_leg is None: return None, None
        short_candidates = df[(df['strike'] < long_leg['strike']) & (df['strike'] >= long_leg['strike'] - MAX_SPREAD_WIDTH)]
        short_leg = filter_options_by_delta(short_candidates, 0.25, True)
    return long_leg, short_leg

def build_recommendation_improved(strategy_name, spot_price, expiration, legs, decisions):
    dte = (datetime.strptime(expiration, '%Y-%m-%d').date() - datetime.utcnow().date()).days
    rec = {'Strategy': strategy_name, 'Expiration': expiration, 'DTE': dte, 'Spot Price': round(spot_price, 2), 'Decision Path': decisions}
    if strategy_name == "Iron Condor":
        put_short, put_long, call_short, call_long = legs
        spread_width = abs(put_short['strike'] - put_long['strike'])
        premium = (put_short['mid'] - put_long['mid']) + (call_short['mid'] - call_long['mid'])
        rec.update({'Put Spread': f"{put_long['strike']}/{put_short['strike']}", 'Call Spread': f"{call_short['strike']}/{call_long['strike']}", 'Spread Width': spread_width, 'Premium': round(premium, 2), 'Max Profit': round(premium, 2), 'Max Loss': round(spread_width - premium, 2), 'Breakeven': f"{round(put_short['strike'] - premium, 2)} and {round(call_short['strike'] + premium, 2)}"})
    else:
        short_leg, long_leg = legs
        is_credit = 'Credit' in strategy_name
        spread_width = abs(long_leg['strike'] - short_leg['strike'])
        premium = (short_leg['mid'] - long_leg['mid']) if is_credit else (long_leg['mid'] - short_leg['mid'])
        max_profit = premium if is_credit else spread_width - premium
        max_loss = spread_width - premium if is_credit else premium
        breakeven = (short_leg['strike'] - premium) if ('Put' in strategy_name and is_credit) or ('Call' in strategy_name and not is_credit) else ((long_leg['strike'] - premium) if 'Put' in strategy_name else (short_leg['strike'] + premium))
        rec.update({'Short Strike': short_leg['strike'], 'Long Strike': long_leg['strike'], 'Spread Width': spread_width, 'Premium': round(premium, 2), 'Max Profit': round(max_profit, 2), 'Max Loss': round(max_loss, 2), 'Breakeven': round(breakeven, 2), 'Short Delta': round(short_leg.get('delta', 0), 3), 'Long Delta': round(long_leg.get('delta', 0), 3)})
    rec['Risk/Reward Ratio'] = round(rec['Max Profit'] / rec['Max Loss'], 3) if rec['Max Loss'] > 0 else 'N/A'
    return rec

def validate_trade_quality(legs, bias=None):
    min_liq = min(
        (leg['volume'] if pd.notna(leg['volume']) else 0) +
        (leg['open_interest'] if pd.notna(leg['open_interest']) else 0)
        for leg in legs
    )
    if min_liq < (MIN_VOLUME + MIN_OPEN_INTEREST) / 4:
        return False, f"Poor liquidity (min leg liquidity: {min_liq})"

    rec = build_recommendation_improved("Validation", 0, "2025-01-01", legs, [])

    premium = rec.get('Premium', 0)
    if bias and bias.startswith('weak_'):
        min_premium = 0.30  # relaxed for weak bias
    else:
        min_premium = 0.50
    if premium < min_premium:
        return False, f"Premium too small: ${premium:.2f}"

    max_loss = rec.get('Max Loss', 0)
    if max_loss <= 0:
        return False, "Invalid max loss"

    rr_ratio = rec.get('Risk/Reward Ratio', 0)
    is_credit = 'Credit' in rec['Strategy'] or 'Condor' in rec['Strategy']
    if bias and bias.startswith('weak_'):
        rr_thresh = 0.10 if is_credit else 0.6
    else:
        rr_thresh = 0.15 if is_credit else 0.8

    if rr_ratio < rr_thresh:
        return False, f"Poor Risk/Reward: {rr_ratio:.3f}"

    return True, "Trade quality validated"

def calculate_strategy_score(rec, bias, iv_rank, dte):
    score = 5.0
    rr = rec.get('Risk/Reward Ratio', 0)
    if isinstance(rr, (int, float)):
        score += (min(rr, 1.5) * 2) - 1
    premium = rec.get('Premium', 0)
    if premium > 3.0: score += 1.0
    elif premium < 1.0: score -= 1.0
    score += 1.5 - abs(dte - 30) / 15
    strat = rec['Strategy']
    strat = rec['Strategy']
    if (bias == 'bullish' and ('Put' in strat or 'Debit' in strat)) or \
       (bias == 'bearish' and ('Call' in strat or 'Debit' in strat)) or \
       (bias == 'neutral' and 'Condor' in strat):
        score += 1.5
    elif bias != 'neutral':
        score -= 1.5
    return max(0, min(10, score))
    
def print_recommendation_improved(rec):
    if 'Status' in rec:
        print(f"\n❌ {rec['Status']}: {rec['Message']}")
        return
    print("\n✅ STRATEGY RECOMMENDATION")
    print("=" * 50)
    print(f"Strategy: {rec['Strategy']} (Score: {rec.get('Score', 'N/A'):.1f}/10)")
    print(f"Expiration: {rec['Expiration']} ({rec['DTE']} DTE)")
    print(f"Underlying Price: ${rec['Spot Price']}")
    print("-" * 25)
    if rec['Strategy'] == "Iron Condor":
        print(f"Put Spread: {rec['Put Spread']}\nCall Spread: {rec['Call Spread']}")
    else:
        print(f"Position: Short ${rec['Short Strike']}, Long ${rec['Long Strike']}")
    print(f"Spread Width: ${rec['Spread Width']}\n" + "-" * 25)
    print("RISK/REWARD:")
    print(f"Net Premium: ${rec['Premium']}")
    # FIX: Corrected "Credit Received" text for debit spreads
    profit_text = "Credit Received" if "Credit" in rec['Strategy'] or "Condor" in rec['Strategy'] else "Debit Paid"
    print(f"Max Profit: ${rec['Max Profit']}")
    print(f"Max Loss: ${rec['Max Loss']}")
    print(f"Risk/Reward Ratio: {rec['Risk/Reward Ratio']}")
    print(f"Breakeven(s): {rec['Breakeven']}\n" + "-" * 25)
    print("DECISION PATH:")
    for i, decision in enumerate(rec.get('Decision Path', []), 1):
        print(f"{i}. {decision}")
    print("=" * 50)

def recommend_strategy_enhanced(verbose=True):
    logging.info('Starting strategy recommendation')
    spot_price = get_spx_price()
    bias, historical_vol = get_market_bias_and_vol(verbose=verbose)

    if verbose:
        print("-" * 50)

    expirations = get_all_expirations()
    if not expirations:
        return {'Status': 'Error', 'Message': 'No valid expirations found'}

    viable_strategies = []
    for exp_date in expirations:
        calls, puts = get_options_chain(exp_date)
        if calls.empty or puts.empty:
            continue

        atm_ivs = pd.concat([
            puts.iloc[(puts['strike'] - spot_price).abs().argsort()[:1]]['implied_volatility'],
            calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]]['implied_volatility']
        ])
        atm_iv = atm_ivs.mean()
        if pd.isna(atm_iv):
            continue

        iv_rank = calculate_iv_rank(atm_iv, historical_vol)
        dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.utcnow().date()).days
        if verbose:
            print(f"\nAnalyzing {exp_date} (DTE: {dte}) | ATM IV: {atm_iv:.1%} | IV Rank: {iv_rank:.2f}")

        strategies_to_try = []
        if iv_rank > IV_RANK_THRESHOLD_HIGH:
            if bias == 'neutral':
                strategies_to_try.append(('iron_condor', 'Iron Condor'))
            strategies_to_try.append(('put_credit', 'Put Credit Spread'))
            strategies_to_try.append(('call_credit', 'Call Credit Spread'))
        elif iv_rank < IV_RANK_THRESHOLD_LOW:
            if bias == 'bullish':
                strategies_to_try.append(('call_debit', 'Call Debit Spread'))
            if bias == 'bearish':
                strategies_to_try.append(('put_debit', 'Put Debit Spread'))
        else:
            if bias in ('bullish', 'weak_bullish'):
                strategies_to_try.extend([
                    ('put_credit', 'Put Credit Spread'),
                    ('call_debit', 'Call Debit Spread')
                ])
            if bias in ('bearish', 'weak_bearish'):
                strategies_to_try.extend([
                    ('call_credit', 'Call Credit Spread'),
                    ('put_debit', 'Put Debit Spread')
                ])
            if bias == 'neutral':
                strategies_to_try.append(('iron_condor', 'Iron Condor'))

        for strategy_type, strategy_name in strategies_to_try:
            legs = None
            if strategy_type == 'put_credit':
                short, long = filter_credit_spread_improved(puts, True, spot_price)
                if short is not None and long is not None:
                    legs = (short, long)
            elif strategy_type == 'call_credit':
                short, long = filter_credit_spread_improved(calls, False, spot_price)
                if short is not None and long is not None:
                    legs = (short, long)
            elif strategy_type == 'call_debit':
                long, short = filter_debit_spread(calls, True, spot_price)
                if long is not None and short is not None:
                    legs = (short, long)
            elif strategy_type == 'put_debit':
                long, short = filter_debit_spread(puts, False, spot_price)
                if long is not None and short is not None:
                    legs = (short, long)
            elif strategy_type == 'iron_condor':
                put_short, put_long = filter_credit_spread_improved(puts, True, spot_price)
                call_short, call_long = filter_credit_spread_improved(calls, False, spot_price)
                if all(x is not None for x in [put_short, put_long, call_short, call_long]):
                    if put_short['strike'] < call_short['strike']:
                        legs = (put_short, put_long, call_short, call_long)

            if legs:
                is_valid, validation_msg = validate_trade_quality(legs, bias=bias)
                if is_valid:
                    decisions = [
                        f"Bias: {bias.upper()}",
                        f"IV Rank: {iv_rank:.2f}",
                        f"Attempting {strategy_name}"
                    ]
                    result = build_recommendation_improved(strategy_name, spot_price, exp_date, legs, decisions)
                    if result:
                        score = calculate_strategy_score(result, bias, iv_rank, dte)
                        result['Score'] = score
                        viable_strategies.append(result)
                        if verbose:
                            print(f"✅ Found viable {strategy_name} (Score: {score:.1f})")

    return max(viable_strategies, key=lambda x: x['Score']) if viable_strategies else {'Status': 'No Strategy', 'Message': 'No suitable, high-quality strategy found.'}


if __name__ == '__main__':
    print("SPX Options Strategy Recommender (Enhanced)")
    print("=" * 50)
    recommendation = recommend_strategy_enhanced(verbose=True)
    print_recommendation_improved(recommendation)