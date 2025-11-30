# JPXトレード分析 Streamlit アプリ（単一ファイル版）
# ------------------------------------------------------------
# 目的:
#  - ユーザー提供の取引TSVを読み込み、日本株(JPX)前提で分析し可視化
#  - 指定のサマリー (1)全体 (2)貸借/制度別 (3)エントリー月×保持日数（区分け有/無） を表示
#  - KPI、エクイティカーブ、ドローダウン、ヒートマップをモダンUIで
#
# 前提/仕様:
#  - 営業日数: JPXカレンダー (pandas_market_calendars) を用い、両端含めでカウント
#  - 総合リターン: トレードリターンをイグジット日昇順で複利連鎖
#  - 年率換算(CAGR): [最初のエントリー日, 最後のイグジット日] の暦日を母数
#  - シャープ/ソルティノ年率化: tradeリターン系列を基準に freq = 250 / 平均保持営業日数
#  - PF: 金額ベース(損益列が無い場合は価格×数量で推定)
#  - 欠損/異常: イグジット<エントリー を検知し、除外/入替の選択肢
#
# 使い方:
#  1) 必要パッケージをインストール:
#     pip install streamlit pandas numpy pandas_market_calendars plotly
#  2) 実行:
#     streamlit run app.py
#  3) 画面左のサイドバーでTSVをアップロード（タブ区切り）。
#     サンプルとして /mnt/data/分析用.tsv がある場合は自動読込を試行します。
#
# 入力想定列 (一部は無くても可・自動推定):
#  - エントリー日, イグジット日, リターン率, 平均購入価格, 平均売却価格, 損益, 数量,
#    銘柄コード, 銘柄名, 市場, 貸借, 口座区分, など
#  備考: リターン率が % 文字や 40 のような百分率で入っていても自動で 0.40 に正規化します。
# ------------------------------------------------------------

from __future__ import annotations
import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except Exception:
    HAS_MCAL = False

# ------------------------------
# ユーティリティ
# ------------------------------

@st.cache_data(show_spinner=False)
def load_tsv(file_or_path) -> pd.DataFrame:
    if file_or_path is None:
        return pd.DataFrame()
    if hasattr(file_or_path, 'read'):  # UploadedFile
        data = file_or_path.read()
        return pd.read_csv(io.BytesIO(data), sep='	')
    # path-like
    return pd.read_csv(file_or_path, sep='	')

@st.cache_data(show_spinner=False)
def load_table(file_or_path) -> pd.DataFrame:
    """TSV/CSV自動判別読み込み。第一候補は区切り自動推定、失敗時TSVで再試行。"""
    if file_or_path is None:
        return pd.DataFrame()

    def _read_buf(buf):
        try:
            return pd.read_csv(buf, sep=None, engine='python')  # 区切り自動推定
        except Exception:
            try:
                if hasattr(buf, 'seek'):
                    buf.seek(0)
            except Exception:
                pass
            return pd.read_csv(buf, sep='	')

    if hasattr(file_or_path, 'read'):
        data = file_or_path.read()
        bio = io.BytesIO(data)
        return _read_buf(bio)
    else:
        try:
            return pd.read_csv(file_or_path, sep=None, engine='python')
        except Exception:
            return pd.read_csv(file_or_path, sep='	')


def _to_number(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return np.nan
    s = s.replace(',', '')
    percent = False
    if s.endswith('%'):
        percent = True
        s = s[:-1]
    try:
        val = float(s)
    except Exception:
        return np.nan
    if percent:
        val = val / 100.0
    return val


def _find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    for k in keywords:
        for c in df.columns:
            if k in str(c):
                return c
    return None


def normalize_dataframe(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """列名推定と型整備。返り値: (df, colmap, warnings)
    必須/推奨列は以下を推定:
      entry_date, exit_date, ret, buy_px, sell_px, qty, pnl, market, credit, account, code, name
    """
    df = raw.copy()
    warnings: List[str] = []

    # 1) 列マッピング推定
    colmap = {
        'entry_date': _find_col(df, ['エントリー', '購入日', '約定買']),
        'exit_date': _find_col(df, ['イグジット', '売却日', '約定売']),
        'ret': _find_col(df, ['リターン', '損益率', '騰落率']),
        'buy_px': _find_col(df, ['平均購入', '買値', '取得単価']),
        'sell_px': _find_col(df, ['平均売却', '売値', '売却単価']),
        'qty': _find_col(df, ['数量', '株数', '口数']),
        'pnl': _find_col(df, ['損益', '実現損益']),
        'market': _find_col(df, ['市場', '東']),
        'credit': _find_col(df, ['貸借', '制度']),
        'account': _find_col(df, ['口座', '特定', '一般', 'NISA']),
        'code': _find_col(df, ['銘柄コード', 'コード']),
        'name': _find_col(df, ['銘柄名', '名称']),
    }

    # 2) 型変換
    if colmap['entry_date']:
        df[colmap['entry_date']] = pd.to_datetime(df[colmap['entry_date']], errors='coerce')
    else:
        warnings.append('エントリー日が見つかりません。列名に「エントリー/購入日」等が含まれているか確認してください。')

    if colmap['exit_date']:
        df[colmap['exit_date']] = pd.to_datetime(df[colmap['exit_date']], errors='coerce')
    else:
        warnings.append('イグジット日が見つかりません。列名に「イグジット/売却日」等が含まれているか確認してください。')

    for k in ['ret', 'buy_px', 'sell_px', 'qty', 'pnl']:
        c = colmap[k]
        if c and c in df.columns:
            df[c] = df[c].map(_to_number)

    # リターン率の自動補正 (百分率入力検出)
    if colmap['ret'] and colmap['ret'] in df.columns:
        r = df[colmap['ret']]
        # 1より大きい絶対値の要素が多数 -> 百分率とみなして/100
        if r.dropna().abs().gt(1.5).mean() > 0.5:
            df[colmap['ret']] = r / 100.0

    # 損益 推定列 (なければ作る)
    if not colmap['pnl']:
        if colmap['buy_px'] and colmap['sell_px'] and colmap['qty']:
            df['__pnl_estimate'] = (df[colmap['sell_px']] - df[colmap['buy_px']]) * df[colmap['qty']]
            colmap['pnl'] = '__pnl_estimate'
        else:
            warnings.append('損益列が見つからず、価格×数量でも推定できません。PFや金額系の一部指標はNaNになります。')

    # リターン率 推定列 (なければ作る)
    if not colmap['ret']:
        if colmap['buy_px'] and colmap['sell_px']:
            df['__ret_estimate'] = (df[colmap['sell_px']] / df[colmap['buy_px']]) - 1.0
            colmap['ret'] = '__ret_estimate'
        elif colmap['pnl'] and colmap['buy_px'] and colmap['qty']:
            # 損益 / (買値×数量)
            denom = (df[colmap['buy_px']] * df[colmap['qty']]).replace(0, np.nan)
            df['__ret_estimate'] = df[colmap['pnl']] / denom
            colmap['ret'] = '__ret_estimate'
        else:
            warnings.append('リターン率が見つからず計算もできません。多くの指標がNaNになります。')

    return df, colmap, warnings


# ------------------------------
# JPX 営業日カウント
# ------------------------------

def build_jpx_business_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if not HAS_MCAL:
        raise RuntimeError('pandas_market_calendars がインストールされていません。pip install pandas_market_calendars')
    jpx = mcal.get_calendar('JPX')
    sched = jpx.schedule(start_date=start.normalize(), end_date=end.normalize())
    # index は tz-aware。naiveにして日単位でユニーク化
    biz = pd.DatetimeIndex(sched.index.tz_localize(None).normalize().unique())
    return biz


def count_business_days_inclusive(biz: pd.DatetimeIndex, s: pd.Timestamp, e: pd.Timestamp) -> int:
    if pd.isna(s) or pd.isna(e):
        return np.nan
    if e < s:
        # 呼び出し元で入替/除外を決める。ここでは0扱いにしない。
        return np.nan
    arr = biz.values.astype('datetime64[D]')
    s_d = np.datetime64(s.normalize(), 'D')
    e_d = np.datetime64(e.normalize(), 'D')
    left = arr.searchsorted(s_d, side='left')
    right = arr.searchsorted(e_d, side='right')
    return int(max(0, right - left))


def add_business_days_and_buckets(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    if not colmap['entry_date'] or not colmap['exit_date']:
        df['営業日数'] = np.nan
        df['保持日数バケット'] = '不明'
        return df
    s_min = df[colmap['entry_date']].min()
    e_max = df[colmap['exit_date']].max()
    if pd.isna(s_min) or pd.isna(e_max):
        df['営業日数'] = np.nan
        df['保持日数バケット'] = '不明'
        return df
    biz = build_jpx_business_days(s_min, e_max)
    df = df.copy()
    df['営業日数'] = [
        count_business_days_inclusive(biz, s, e)
        for s, e in zip(df[colmap['entry_date']], df[colmap['exit_date']])
    ]

    def bucket(n):
        try:
            n = float(n)
        except Exception:
            return '不明'
        if n < 20:
            return '1ヶ月未満'
        elif n < 40:
            return '1～2か月'
        elif n < 60:
            return '2～3か月'
        else:
            return '3か月以上'

    df['保持日数バケット'] = df['営業日数'].map(bucket)

    # エントリー月キー
    if colmap['entry_date']:
        df['エントリー月'] = df[colmap['entry_date']].dt.strftime('%Y-%m')
    else:
        df['エントリー月'] = '不明'
    return df


# ------------------------------
# 指標計算
# ------------------------------

def equity_curve_and_dd(trades: pd.DataFrame, colmap: Dict[str, str]) -> Tuple[pd.Series, pd.Series]:
    """イグジット日昇順で複利曲線とDDを返す。"""
    rcol = colmap['ret']
    dcol = colmap['exit_date']
    if not rcol or not dcol:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    x = trades[[dcol, rcol]].dropna().sort_values(dcol)
    if x.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    eq = (1.0 + x[rcol].astype(float)).cumprod()
    eq.index = x[dcol].values
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return eq, dd


def cagr(total_return: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if pd.isna(start_date) or pd.isna(end_date) or end_date <= start_date:
        return np.nan
    days = (end_date - start_date).days
    try:
        return (1.0 + float(total_return)) ** (365.0 / days) - 1.0
    except Exception:
        return np.nan


def sharpe_sortino(trades: pd.DataFrame, colmap: Dict[str, str]) -> Tuple[float, float, float]:
    """(Sharpe年率, Sortino年率, freq) を返す。rf=0, tradeリターンベース。
    例外処理: マイナスリターンが1件も無い場合は Sortino=+inf/0/-inf を返す。
    """
    rcol = colmap['ret']
    if not rcol or rcol not in trades.columns:
        return np.nan, np.nan, np.nan
    r = trades[rcol].dropna().astype(float)
    if r.empty:
        return np.nan, np.nan, np.nan
    avg_hold = trades['営業日数'].dropna().mean() if '営業日数' in trades.columns else np.nan
    freq = (250.0 / float(avg_hold)) if (avg_hold and avg_hold > 0) else np.nan

    mean_r = r.mean()
    std_r = r.std(ddof=1)
    has_downside = (r < 0).any()

    if np.isnan(freq) or freq <= 0:
        return np.nan, np.nan, freq

    scale = math.sqrt(freq)
    sharpe = (mean_r / std_r) * scale if std_r and std_r > 0 else np.nan

    if not has_downside:
        # ダウンサイド偏差=0の極限。平均が正→+inf, 0→0, 負→-inf
        sortino = np.inf if mean_r > 0 else (0.0 if mean_r == 0 else -np.inf)
    else:
        downside = np.where(r < 0, r, 0.0)
        downside_dev = math.sqrt(np.mean(np.square(downside)))
        sortino = (mean_r / downside_dev) * scale if downside_dev and downside_dev > 0 else np.nan

    return sharpe, sortino, freq


def profit_factor(trades: pd.DataFrame, colmap: Dict[str, str]) -> float:
    pcol = colmap['pnl']
    if not pcol or pcol not in trades.columns:
        return np.nan
    p = trades[pcol].dropna().astype(float)
    if p.empty:
        return np.nan
    gains = p[p > 0].sum()
    losses = p[p < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else (0.0 if gains == 0 else np.nan)
    return float(gains / abs(losses))


def build_cashflows(trades: pd.DataFrame, colmap: Dict[str, str]) -> List[Tuple[pd.Timestamp, float]]:
    """各トレードの買付(マイナス)・売却(プラス)のキャッシュフローを日付別に集計。"""
    req = ['entry_date', 'exit_date', 'buy_px', 'sell_px', 'qty']
    for k in req:
        if not colmap.get(k):
            return []
    flows: Dict[pd.Timestamp, float] = {}
    for _, row in trades.iterrows():
        try:
            if pd.notna(row[colmap['entry_date']]) and pd.notna(row[colmap['buy_px']]) and pd.notna(row[colmap['qty']]):
                amt = -float(row[colmap['buy_px']]) * float(row[colmap['qty']])
                d = pd.to_datetime(row[colmap['entry_date']]).normalize()
                flows[d] = flows.get(d, 0.0) + amt
            if pd.notna(row[colmap['exit_date']]) and pd.notna(row[colmap['sell_px']]) and pd.notna(row[colmap['qty']]):
                amt = float(row[colmap['sell_px']]) * float(row[colmap['qty']])
                d = pd.to_datetime(row[colmap['exit_date']]).normalize()
                flows[d] = flows.get(d, 0.0) + amt
        except Exception:
            continue
    if not flows:
        return []
    items = sorted(flows.items(), key=lambda x: x[0])
    return items


def xnpv(rate: float, cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    if rate <= -0.999999:
        return np.inf
    dates = [d for d, _ in cashflows]
    amounts = [a for _, a in cashflows]
    t0 = dates[0]
    total = 0.0
    for d, a in zip(dates, amounts):
        days = (d - t0).days
        total += a / ((1.0 + rate) ** (days / 365.0))
    return total


def xirr_bisection(cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    """単純な二分法でXIRRを推定。符号が一方向のみならNaN。"""
    if not cashflows or len(cashflows) < 2:
        return np.nan
    amts = [a for _, a in cashflows]
    if not (any(a < 0 for a in amts) and any(a > 0 for a in amts)):
        return np.nan
    low, high = -0.9999, 10.0  # up to 1000%/y 初期
    f_low = xnpv(low, cashflows)
    f_high = xnpv(high, cashflows)
    # 拡張: 符号が同じなら上限を広げる
    expand = 0
    while f_low * f_high > 0 and expand < 20:
        high *= 1.5
        f_high = xnpv(high, cashflows)
        expand += 1
        if high > 1e6:
            break
    if f_low * f_high > 0:
        return np.nan
    for _ in range(100):
        mid = (low + high) / 2.0
        f_mid = xnpv(mid, cashflows)
        if abs(f_mid) < 1e-9:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def avg_gain_loss_ratio(trades: pd.DataFrame, colmap: Dict[str, str]) -> Tuple[float, float, float]:
    rcol = colmap['ret']
    if not rcol or rcol not in trades.columns:
        return np.nan, np.nan, np.nan
    r = trades[rcol].dropna().astype(float)
    if r.empty:
        return np.nan, np.nan, np.nan
    gains = r[r > 0]
    losses = r[r < 0]
    avg_gain = gains.mean() if not gains.empty else np.nan
    avg_loss = losses.mean() if not losses.empty else np.nan
    ratio = (avg_gain / abs(avg_loss)) if (avg_gain is not np.nan and avg_loss not in [0, np.nan]) else np.nan
    return avg_gain, avg_loss, ratio


def calc_metrics(trades: pd.DataFrame, colmap: Dict[str, str]) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {
            'トレード数': 0,
            '総合リターン(%)': np.nan,
            '年率換算リターン(%)': np.nan,
            '平均リターン(%)': np.nan,
            '勝率': np.nan,
            'プロフィットファクター': np.nan,
            '最大ドローダウン(%)': np.nan,
            'シャープレシオ': np.nan,
            'ソルティノレシオ': np.nan,
            '利益平均(%)': np.nan,
            '損失平均(%)': np.nan,
            '損益比': np.nan,
            '総買付金額(円)': np.nan,
            '総売却金額(円)': np.nan,
            '総損益金額(円)': np.nan,
            '投下資金ROI(%)': np.nan,
            '資金加重年率(XIRR%)': np.nan,
        }

    # 複利エクイティとDD
    eq, dd = equity_curve_and_dd(trades, colmap)
    total_return = (eq.iloc[-1] - 1.0) if not eq.empty else np.nan

    # 期間
    start_date = trades[colmap['entry_date']].min() if colmap['entry_date'] else pd.NaT
    end_date = trades[colmap['exit_date']].max() if colmap['exit_date'] else pd.NaT

    # 勝率など
    rcol = colmap['ret']
    r = trades[rcol].dropna().astype(float) if rcol in trades.columns else pd.Series(dtype=float)
    win_rate = (r > 0).mean() if not r.empty else np.nan
    avg_ret = r.mean() if not r.empty else np.nan

    # PF
    pf = profit_factor(trades, colmap)

    # MaxDD
    max_dd = dd.min() if not dd.empty else np.nan

    # Sharpe/Sortino
    sharpe, sortino, freq = sharpe_sortino(trades, colmap)

    # Avg gain/loss & ratio
    avg_gain, avg_loss, gl_ratio = avg_gain_loss_ratio(trades, colmap)

    # CAGR
    cagr_val = cagr(total_return, start_date, end_date)

    # 金額系
    buy_amt = np.nan
    sell_amt = np.nan
    try:
        if colmap['buy_px'] and colmap['qty'] and colmap['buy_px'] in trades.columns and colmap['qty'] in trades.columns:
            buy_amt = (trades[colmap['buy_px']].astype(float) * trades[colmap['qty']].astype(float)).sum()
        if colmap['sell_px'] and colmap['qty'] and colmap['sell_px'] in trades.columns and colmap['qty'] in trades.columns:
            sell_amt = (trades[colmap['sell_px']].astype(float) * trades[colmap['qty']].astype(float)).sum()
    except Exception:
        pass

    if colmap['pnl'] and colmap['pnl'] in trades.columns:
        pnl_sum = trades[colmap['pnl']].astype(float).sum()
    elif pd.notna(buy_amt) and pd.notna(sell_amt):
        pnl_sum = float(sell_amt) - float(buy_amt)
    else:
        pnl_sum = np.nan

    roi = (pnl_sum / buy_amt * 100.0) if (pd.notna(pnl_sum) and pd.notna(buy_amt) and buy_amt != 0) else np.nan

    # XIRR
    try:
        cfs = build_cashflows(trades, colmap)
        xirr_val = xirr_bisection(cfs) * 100.0 if cfs else np.nan
    except Exception:
        xirr_val = np.nan

    return {
        'トレード数': int(len(trades)),
        '総合リターン(%)': float(total_return * 100.0) if pd.notna(total_return) else np.nan,
        '年率換算リターン(%)': float(cagr_val * 100.0) if pd.notna(cagr_val) else np.nan,
        '平均リターン(%)': float(avg_ret * 100.0) if pd.notna(avg_ret) else np.nan,
        '勝率': float(win_rate) if pd.notna(win_rate) else np.nan,
        'プロフィットファクター': float(pf) if pd.notna(pf) else np.nan,
        '最大ドローダウン(%)': float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        'シャープレシオ': float(sharpe) if pd.notna(sharpe) else np.nan,
        'ソルティノレシオ': float(sortino) if pd.notna(sortino) else np.nan,
        '利益平均(%)': float(avg_gain * 100.0) if pd.notna(avg_gain) else np.nan,
        '損失平均(%)': float(avg_loss * 100.0) if pd.notna(avg_loss) else np.nan,
        '損益比': float(gl_ratio) if pd.notna(gl_ratio) else np.nan,
        '総買付金額(円)': float(buy_amt) if pd.notna(buy_amt) else np.nan,
        '総売却金額(円)': float(sell_amt) if pd.notna(sell_amt) else np.nan,
        '総損益金額(円)': float(pnl_sum) if pd.notna(pnl_sum) else np.nan,
        '投下資金ROI(%)': float(roi) if pd.notna(roi) else np.nan,
        '資金加重年率(XIRR%)': float(xirr_val) if pd.notna(xirr_val) else np.nan,
    }

    # 複利エクイティとDD
    eq, dd = equity_curve_and_dd(trades, colmap)
    total_return = (eq.iloc[-1] - 1.0) if not eq.empty else np.nan

    # 期間
    start_date = trades[colmap['entry_date']].min() if colmap['entry_date'] else pd.NaT
    end_date = trades[colmap['exit_date']].max() if colmap['exit_date'] else pd.NaT

    # 勝率など
    rcol = colmap['ret']
    r = trades[rcol].dropna().astype(float) if rcol in trades.columns else pd.Series(dtype=float)
    win_rate = (r > 0).mean() if not r.empty else np.nan
    avg_ret = r.mean() if not r.empty else np.nan

    # PF
    pf = profit_factor(trades, colmap)

    # MaxDD
    max_dd = dd.min() if not dd.empty else np.nan

    # Sharpe/Sortino
    sharpe, sortino, freq = sharpe_sortino(trades, colmap)

    # Avg gain/loss & ratio
    avg_gain, avg_loss, gl_ratio = avg_gain_loss_ratio(trades, colmap)

    # CAGR
    cagr_val = cagr(total_return, start_date, end_date)

    return {
        'トレード数': int(len(trades)),
        '総合リターン(%)': float(total_return * 100.0) if pd.notna(total_return) else np.nan,
        '年率換算リターン(%)': float(cagr_val * 100.0) if pd.notna(cagr_val) else np.nan,
        '平均リターン(%)': float(avg_ret * 100.0) if pd.notna(avg_ret) else np.nan,
        '勝率': float(win_rate) if pd.notna(win_rate) else np.nan,
        'プロフィットファクター': float(pf) if pd.notna(pf) else np.nan,
        '最大ドローダウン(%)': float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        'シャープレシオ': float(sharpe) if pd.notna(sharpe) else np.nan,
        'ソルティノレシオ': float(sortino) if pd.notna(sortino) else np.nan,
        '利益平均(%)': float(avg_gain * 100.0) if pd.notna(avg_gain) else np.nan,
        '損失平均(%)': float(avg_loss * 100.0) if pd.notna(avg_loss) else np.nan,
        '損益比': float(gl_ratio) if pd.notna(gl_ratio) else np.nan,
    }


# ------------------------------
# 集計テーブル
# ------------------------------

def round_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """表示用に浮動小数点列のみを丸める（四捨五入）。元データは変更しない。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    float_cols = out.select_dtypes(include=['float64', 'float32', 'float16']).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].round(decimals)
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """表示用の丸めを適用したCSV（UTF-8 BOM付）をbytesで返す。空なら空bytes。"""
    try:
        if df is None or getattr(df, 'empty', True):
            return b''
        disp = round_for_display(df)
        return disp.to_csv(index=False).encode('utf-8-sig')
    except Exception:
        # 失敗時も最低限エクスポート（丸め無し）
        try:
            return df.to_csv(index=False).encode('utf-8-sig')
        except Exception:
            return b''

def summarize_by_bucket(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    if '保持日数バケット' not in df.columns:
        return pd.DataFrame()
    rows = []
    for bucket, g in df.groupby('保持日数バケット'):
        met = calc_metrics(g, colmap)
        met['保持日数バケット'] = bucket
        rows.append(met)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 表示順序を固定
    bucket_order = ['1ヶ月未満', '1～2か月', '2～3か月', '3か月以上', '不明']
    out['__order'] = out['保持日数バケット'].map({b:i for i,b in enumerate(bucket_order)}).fillna(999).astype(int)
    out = out.sort_values(['__order']).drop(columns='__order')
    # 列順
    out = out[['保持日数バケット'] + [c for c in out.columns if c != '保持日数バケット']]
    return out

def summarize_overall(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    met = calc_metrics(df, colmap)
    return pd.DataFrame([met])


def summarize_by_credit(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ccol = colmap['credit']
    if not ccol or ccol not in df.columns:
        return pd.DataFrame()
    rows = []
    for k, g in df.groupby(ccol):
        met = calc_metrics(g, colmap)
        met['区分'] = k
        rows.append(met)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[['区分'] + [c for c in out.columns if c != '区分']]
    return out


def summarize_by_month_bucket(df: pd.DataFrame, colmap: Dict[str, str], with_credit: bool) -> pd.DataFrame:
    if 'エントリー月' not in df.columns or '保持日数バケット' not in df.columns:
        return pd.DataFrame()
    group_cols = ['エントリー月', '保持日数バケット']
    if with_credit and colmap['credit'] and colmap['credit'] in df.columns:
        group_cols.append(colmap['credit'])

    rows = []
    for keys, g in df.groupby(group_cols):
        met = calc_metrics(g, colmap)
        if isinstance(keys, tuple):
            for i, k in enumerate(group_cols):
                met[k] = keys[i]
        else:
            met[group_cols[0]] = keys
        rows.append(met)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 表示順
    disp_cols = group_cols + [c for c in out.columns if c not in group_cols]
    out = out[disp_cols]
    return out.sort_values(group_cols)


# ------------------------------
# 追加ユーティリティ（TWR/ベンチマーク/閾値処理）
# ------------------------------

def apply_return_epsilon(df: pd.DataFrame, colmap: Dict[str, str], eps_pct: float):
    """
    |r| < ε(%) を 0 にする。新列 'ret_eps' を作り、colmap['ret'] を差し替えたコピーを返す。
    """
    rcol = colmap.get('ret')
    if not rcol or rcol not in df.columns:
        return df, dict(colmap)
    eps = float(eps_pct) / 100.0
    out = df.copy()
    r = out[rcol].astype(float)
    r = r.mask(r.abs() < eps, 0.0)
    out['ret_eps'] = r
    cmap = dict(colmap)
    cmap['ret'] = 'ret_eps'
    return out, cmap



def get_bizdays_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    JPXカレンダーが使えるならそれを利用、ダメなら pandas の平日ベース。
    """
    try:
        return build_jpx_business_days(start, end)
    except Exception:
        return pd.bdate_range(start=start.normalize(), end=end.normalize())

def build_daily_equity(trades: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    """複利エクイティをJPX営業日へ前方補完した日次系列を返す。"""
    eq_exit, _ = equity_curve_and_dd(trades, colmap)
    if eq_exit is None or eq_exit.empty:
        return pd.Series(dtype=float)
    s = trades[colmap['entry_date']].min() if colmap.get('entry_date') else eq_exit.index.min()
    e = trades[colmap['exit_date']].max() if colmap.get('exit_date') else eq_exit.index.max()
    if pd.isna(s) or pd.isna(e):
        return pd.Series(dtype=float)
    biz = get_bizdays_range(s, e)
    if len(biz) == 0:
        return pd.Series(dtype=float)
    # 初期値=1.0 を最初の営業日に設定、イグジット日の観測値を上書き
    daily = pd.Series(index=biz, dtype=float)
    daily.iloc[0] = 1.0
    # 同一日複数決済があっても最後の値でOK（eq_exitは昇順累積）
    eq_tmp = eq_exit.copy()
    eq_tmp.index = pd.to_datetime(eq_tmp.index).tz_localize(None).normalize()
    eq_tmp = eq_tmp[~eq_tmp.index.duplicated(keep='last')]
    daily.loc[eq_tmp.index.intersection(daily.index)] = eq_tmp.values
    daily = daily.sort_index().ffill()
    return daily


def daily_returns_from_equity(eq: pd.Series) -> pd.Series:
    if eq is None or eq.empty:
        return pd.Series(dtype=float)
    r = eq.pct_change().fillna(0.0)
    r.index = pd.to_datetime(r.index)
    return r

# --- 重複対応の近似ポートフォリオTWR（日次） ---

def approximate_portfolio_daily_returns(trades: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    """
    各トレードの総リターン R を保有営業日 N に等配分（logベース）。
    同日にアクティブなトレードは等加重（1/本数）で合成。
    """
    e_col, x_col, r_col = colmap.get('entry_date'), colmap.get('exit_date'), colmap.get('ret')
    if not (e_col and x_col and r_col):
        return pd.Series(dtype=float)
    d = trades[[e_col, x_col, r_col]].dropna().copy()
    if d.empty:
        return pd.Series(dtype=float)

    s = d[e_col].min().normalize()
    e = d[x_col].max().normalize()
    biz = get_bizdays_range(s, e)
    if len(biz) == 0:
        return pd.Series(dtype=float)

    active_count = pd.Series(0.0, index=biz)
    segments = []
    for _, row in d.iterrows():
        s_i = pd.to_datetime(row[e_col]).normalize()
        e_i = pd.to_datetime(row[x_col]).normalize()
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        days_i = get_bizdays_range(s_i, e_i)
        if len(days_i) == 0:
            continue

        R = float(row[r_col])
        log_r = math.log(1.0 + R) / float(len(days_i))  # 日次log等配分
        per_day = math.exp(log_r) - 1.0
        segments.append((days_i, per_day))
        active_count.loc[days_i.intersection(active_count.index)] += 1.0

    port_r = pd.Series(0.0, index=biz)
    for seg_days, per_day in segments:
        idx = seg_days.intersection(port_r.index)
        denom = active_count.loc[idx].replace(0, np.nan)
        port_r.loc[idx] = port_r.loc[idx] + (per_day / denom)
    return port_r.fillna(0.0)


def build_daily_equity_overlap_aware(trades: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    r = approximate_portfolio_daily_returns(trades, colmap)
    if r is None or r.empty:
        return pd.Series(dtype=float)
    return (1.0 + r).cumprod()


def get_daily_equity_by_mode(trades: pd.DataFrame, colmap: Dict[str, str], mode: str) -> pd.Series:
    """
    mode に '重複' を含むと近似ポートTWR、含まなければ従来の連続複利（イグジット連結）。
    """
    if mode and '重複' in mode:
        return build_daily_equity_overlap_aware(trades, colmap)
    # 従来の（イグジット日だけ）エクイティ → 営業日に前方補完
    return build_daily_equity(trades, colmap)


def calc_calendar_year_twr(daily_ret: pd.Series) -> pd.Series:
    """
    日次リターン系列からカレンダー年のTWRを算出
    """
    if daily_ret is None or daily_ret.empty:
        return pd.Series(dtype=float)
    yr = daily_ret.groupby(daily_ret.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    yr.index.name = 'Year'
    return yr


def calc_trailing_twr_table(daily_eq: pd.Series, windows_days: List[Tuple[str, int]] = None) -> pd.DataFrame:
    if windows_days is None:
        windows_days = [('1Y', 365), ('3Y', 365*3), ('5Y', 365*5)]
    rows = []
    if daily_eq is None or daily_eq.empty:
        return pd.DataFrame(columns=['Window', '期間日数', 'TWR(%)', '年率換算(%)'])
    for name, days in windows_days:
        end = daily_eq.index[-1]
        start = end - pd.Timedelta(days=int(days))
        sub = daily_eq[daily_eq.index >= start]
        if len(sub) < 2:
            twr = np.nan; ann = np.nan; span = np.nan
        else:
            twr = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
            span = (sub.index[-1] - sub.index[0]).days
            ann = (1.0 + twr) ** (365.0 / span) - 1.0 if span > 0 else np.nan
        rows.append({'Window': name, '期間日数': span if not pd.isna(span) else np.nan,
                     'TWR(%)': twr*100.0 if pd.notna(twr) else np.nan,
                     '年率換算(%)': ann*100.0 if pd.notna(ann) else np.nan})
    return pd.DataFrame(rows)


def normalize_benchmark(df: pd.DataFrame) -> pd.Series:
    """ベンチマークCSVを日次リターンへ正規化。許容列: date/日付 と price/nav/close/終値/価格/基準価額"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    # 列検出
    def find_col(keys):
        for k in keys:
            for c in df.columns:
                if k.lower() in str(c).lower():
                    return c
        return None
    dcol = find_col(['date', '日付'])
    pcol = find_col(['price', 'nav', 'close', '終値', '価格', '基準価額'])
    if not dcol or not pcol:
        return pd.Series(dtype=float)
    dd = df.copy()
    dd[dcol] = pd.to_datetime(dd[dcol], errors='coerce')
    dd[pcol] = dd[pcol].map(_to_number)
    dd = dd.dropna(subset=[dcol, pcol]).sort_values(dcol)
    dd = dd.drop_duplicates(subset=[dcol], keep='last')
    price = dd.set_index(dcol)[pcol].astype(float)
    # 価格→日次リターン
    ret = price.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    return ret


def active_stats_df(you_r: pd.Series, bm_r: pd.Series) -> pd.DataFrame:
    common = you_r.index.intersection(bm_r.index)
    if len(common) < 10:
        return pd.DataFrame()
    y = you_r.loc[common].astype(float)
    b = bm_r.loc[common].astype(float)
    # TE/IR（年率）
    diff = y - b
    te = diff.std(ddof=1) * math.sqrt(250.0)
    ar = (y.mean() - b.mean()) * 250.0  # 年率の超過平均
    ir = (ar / te) if te and te > 0 else np.nan
    # β/α（単回帰）
    var_b = b.var(ddof=1)
    cov_yb = np.cov(b, y, ddof=1)[0,1]
    beta = (cov_yb / var_b) if var_b and var_b > 0 else np.nan
    alpha_daily = y.mean() - (beta * b.mean() if pd.notna(beta) else 0.0)
    alpha_ann = (1.0 + alpha_daily) ** 250.0 - 1.0
    return pd.DataFrame([{
        'トラッキングエラー(年率% )': te * 100.0 if pd.notna(te) else np.nan,
        '情報比率(IR)': ir,
        'β': beta,
        'α(年率%)': alpha_ann * 100.0 if pd.notna(alpha_ann) else np.nan,
    }])

# ===== エントリー月サマリー =====

def ensure_entry_month_col(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """
    エントリー日から 'エントリー月'（YYYY-MM）を作成。
    """
    out = df.copy()
    dcol = colmap.get('entry_date')
    if not dcol or dcol not in out.columns:
        return out
    m = pd.to_datetime(out[dcol], errors='coerce')
    out['エントリー月'] = m.dt.to_period('M').astype(str)
    return out

def summarize_by_entry_month(df: pd.DataFrame, colmap: Dict[str, str], with_credit: bool = False) -> pd.DataFrame:
    """
    エントリー月ごと（必要なら貸借/制度別）に calc_metrics を適用してサマリー化。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    tmp = ensure_entry_month_col(df, colmap)
    if 'エントリー月' not in tmp.columns or tmp['エントリー月'].isna().all():
        return pd.DataFrame()

    credit_col = colmap.get('credit') if with_credit else None
    group_keys = ['エントリー月'] + ([credit_col] if credit_col else [])

    rows = []
    for keys, g in tmp.groupby(group_keys):
        met = calc_metrics(g, colmap)
        # groupbyキーの取り出し（単キー/複キーの両対応）
        if isinstance(keys, tuple):
            month = keys[0]
            cred = keys[1] if len(keys) > 1 else None
        else:
            month = keys
            cred = None
        met['エントリー月'] = month
        if credit_col:
            met['貸借/制度'] = cred
        rows.append(met)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 月順に並べる
    try:
        out['__order'] = pd.to_datetime(out['エントリー月'] + '-01', errors='coerce')
        if '貸借/制度' in out.columns:
            out = out.sort_values(['__order', '貸借/制度'])
        else:
            out = out.sort_values(['__order'])
        out = out.drop(columns='__order')
    except Exception:
        pass

    # 列順の調整（先頭にキー列）
    lead = ['エントリー月']
    if '貸借/制度' in out.columns:
        lead.append('貸借/制度')
    out = out[lead + [c for c in out.columns if c not in lead]]

    return out


# ------------------------------
# UI / メイン
# ------------------------------

st.set_page_config(page_title='JPXトレード分析', layout='wide')
st.title('JPXトレード分析（TSV）')

with st.sidebar:
    st.header('設定 / データ読込')
    uploaded = st.file_uploader('TSV/CSVファイルを選択', type=['tsv','csv'])

    df = pd.DataFrame()
    bm_df = None
    auto_info = ''
    if uploaded is not None:
        df = load_table(uploaded)
    else:
        # サンプルパス（存在すれば読み込み）
        try:
            for p in ['/mnt/data/分析用.tsv', '/mnt/data/2025-08-16T13-05_export.csv']:
                try:
                    df = load_table(p)
                    auto_info = f'（{p} を自動読込）'
                    break
                except Exception:
                    continue
        except Exception:
            pass

    st.caption(f'読み込み行数: {len(df):,} {auto_info}')

    st.divider()
    st.header('異常値処理')
    anomaly_policy = st.radio('イグジット日 < エントリー日の行の取り扱い', ['除外', 'エントリー/イグジット入替'], index=0)

if df.empty:
    st.info('左のサイドバーからTSVをアップロードしてください。タブ区切りが必要です。')
    st.stop()

# 正規化
ndf, colmap, warns = normalize_dataframe(df)

# 異常検知 & 処理
anom_mask = pd.Series(False, index=ndf.index)
if colmap['entry_date'] and colmap['exit_date']:
    anom_mask = (ndf[colmap['exit_date']] < ndf[colmap['entry_date']])
    anom_rows = ndf[anom_mask]
    if not anom_rows.empty:
        st.sidebar.warning(f'イグジット<エントリー の行が {len(anom_rows)} 件見つかりました。')
        if anomaly_policy == '除外':
            ndf = ndf.loc[~anom_mask].copy()
        else:
            tmp_s = ndf.loc[anom_mask, colmap['entry_date']].copy()
            ndf.loc[anom_mask, colmap['entry_date']] = ndf.loc[anom_mask, colmap['exit_date']]
            ndf.loc[anom_mask, colmap['exit_date']] = tmp_s

# 営業日数/バケット
try:
    ndf = add_business_days_and_buckets(ndf, colmap)
except Exception as e:
    st.error(f'営業日数の計算でエラー: {e}')
    ndf['営業日数'] = np.nan
    ndf['保持日数バケット'] = '不明'

# フィルタUI（推定された補助列ベース）
with st.sidebar:
    st.header('フィルタ')
    # 期間
    if colmap['exit_date'] and ndf[colmap['exit_date']].notna().any():
        dt_min = pd.to_datetime(ndf[colmap['exit_date']]).min()
        dt_max = pd.to_datetime(ndf[colmap['exit_date']]).max()
        d1, d2 = st.date_input('イグジット期間', [dt_min.date(), dt_max.date()])
        if d1 and d2:
            mask = (ndf[colmap['exit_date']].dt.date >= d1) & (ndf[colmap['exit_date']].dt.date <= d2)
            ndf = ndf.loc[mask].copy()

    # 銘柄、区分
    if colmap['code'] and colmap['code'] in ndf.columns:
        codes = st.multiselect('銘柄コード', sorted(ndf[colmap['code']].dropna().unique().tolist()))
        if codes:
            ndf = ndf[ndf[colmap['code']].isin(codes)]

    if colmap['market'] and colmap['market'] in ndf.columns:
        mkts = st.multiselect('市場', sorted(ndf[colmap['market']].dropna().unique().tolist()))
        if mkts:
            ndf = ndf[ndf[colmap['market']].isin(mkts)]

    if colmap['credit'] and colmap['credit'] in ndf.columns:
        creds = st.multiselect('貸借/制度', sorted(ndf[colmap['credit']].dropna().unique().tolist()))
        if creds:
            ndf = ndf[ndf[colmap['credit']].isin(creds)]

    if colmap['account'] and colmap['account'] in ndf.columns:
        accts = st.multiselect('口座区分', sorted(ndf[colmap['account']].dropna().unique().tolist()))
        if accts:
            ndf = ndf[ndf[colmap['account']].isin(accts)]

    st.caption('※ フィルタは上から順にAND条件で適用されます。')

    st.header('数値設定')
    mode = st.selectbox(
        'リターン集計モード',
        ['重複対応TWR（近似）', '連続フル複利（重複無視）'],
        index=0,
        help='ベンチマーク比較や年間TWRは「重複対応TWR（近似）」を推奨。'
    )
    eps_pct = st.slider('微小損益の閾値 ε（% を0扱い）', 0.0, 0.5, 0.05, step=0.01)

    st.caption('例：0.05% 未満の損益は0とみなして統計を計算')

    st.header('ベンチマーク（任意）')
    bm_uploaded = st.file_uploader('ベンチマークCSV/TSV（列例: date, price/nav/close）', type=['csv','tsv'], key='bm')
    if bm_uploaded is not None:
        try:
            bm_df = load_table(bm_uploaded)
            st.caption(f'ベンチマーク行数: {len(bm_df):,}')
        except Exception as _e:
            st.warning('ベンチマークの読み込みに失敗しました。ファイル形式をご確認ください。')

# 警告表示
if warns:
    with st.expander('⚠️ データ警告・注記'):  # ユーザー確認用
        for w in warns:
            st.write('-', w)

# ------------------------------
# サマリー (1) 全体
# ------------------------------
# ε閾値を適用した分析用データ/列マップを用意
ndf_eff, colmap_eff = apply_return_epsilon(ndf, colmap, eps_pct if 'eps_pct' in globals() else 0.0)

st.subheader('① 取引結果 全体成績サマリー')
# ε適用後のDF/colmapを使っている前提
# メインの日次エクイティ（モード準拠）
main_eq_daily = get_daily_equity_by_mode(ndf_eff, colmap_eff, mode)

# TWR（=累積リターン）を事前計算して見出しに埋め込む
twr_total_pct = np.nan
if main_eq_daily is not None and not main_eq_daily.empty:
    twr_total = float(main_eq_daily.iloc[-1] / main_eq_daily.iloc[0] - 1.0)
    twr_total_pct = twr_total * 100.0


all_metrics = summarize_overall(ndf_eff, colmap_eff)

# 総合リターン/年率/最大DD/TWR を日次エクイティから上書き
try:
    if main_eq_daily is not None and not main_eq_daily.empty:
        total = float(main_eq_daily.iloc[-1] / main_eq_daily.iloc[0] - 1.0)
        span_days = int((main_eq_daily.index[-1] - main_eq_daily.index[0]).days)
        cagr_main = (1.0 + total) ** (365.0 / span_days) - 1.0 if span_days > 0 else np.nan
        dd_main = (main_eq_daily / main_eq_daily.cummax() - 1.0).min()
        all_metrics.loc[0, 'TWR(%)'] = total * 100.0
        all_metrics.loc[0, '総合リターン(%)'] = total * 100.0
        all_metrics.loc[0, '年率換算リターン(%)'] = cagr_main * 100.0 if pd.notna(cagr_main) else np.nan
        all_metrics.loc[0, '最大ドローダウン(%)'] = dd_main * 100.0 if pd.notna(dd_main) else np.nan
except Exception:
    pass

# KPIカード
k1, k2, k3, k4, k5, k6 = st.columns(6)
fmt = lambda x, p=2: ('' if pd.isna(x) else f'{x:.{p}f}')
with k1: st.metric('総合リターン(%)', fmt(all_metrics.iloc[0]['総合リターン(%)']))
with k2: st.metric('年率換算リターン(%)', fmt(all_metrics.iloc[0]['年率換算リターン(%)']))
with k3: st.metric('勝率', fmt(all_metrics.iloc[0]['勝率'], 2))
with k4: st.metric('PF', fmt(all_metrics.iloc[0]['プロフィットファクター'], 3))
with k5: st.metric('最大DD(%)', fmt(all_metrics.iloc[0]['最大ドローダウン(%)']))
with k6: st.metric('平均リターン(%)', fmt(all_metrics.iloc[0]['平均リターン(%)']))

# 列順（TWRと総損益金額を前方に）
lead = ['トレード数','TWR(%)','総合リターン(%)','年率換算リターン(%)',
        '勝率','プロフィットファクター','最大ドローダウン(%)','シャープレシオ','ソルティノレシオ',
        '平均リターン(%)','利益平均(%)','損失平均(%)','損益比','総損益金額(円)']
cols = [c for c in lead if c in all_metrics.columns] + [c for c in all_metrics.columns if c not in lead]
all_metrics = all_metrics[cols]

# --- 年次TWR（カレンダー年）をあなた単独で表示 ---
try:
    if main_eq_daily is not None and not main_eq_daily.empty:
        you_r_daily_alone = daily_returns_from_equity(main_eq_daily)
    yr_you_alone = calc_calendar_year_twr(you_r_daily_alone)  # {Year: TWR}
    if not yr_you_alone.empty:
        yr_tbl = (yr_you_alone * 100.0).round(2).rename('年次TWR(%)').reset_index()
        st.write('年次TWR（カレンダー年・あなた単独）')
        st.dataframe(yr_tbl, use_container_width=True)
except Exception:
    pass

# エクイティ&DD（選択モードに準拠）
if main_eq_daily is not None and not main_eq_daily.empty:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=main_eq_daily.index, y=main_eq_daily.values, mode='lines', name='Equity'))
    title_eq = 'エクイティカーブ（重複対応TWR）' if '重複' in mode else 'エクイティカーブ（連続複利）'
    fig1.update_layout(height=320, title=title_eq)
    st.plotly_chart(fig1, use_container_width=True)

    dd_series = main_eq_daily / main_eq_daily.cummax() - 1.0
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100.0, mode='lines', name='Drawdown'))
    fig2.update_layout(height=220, title='ドローダウン(%)')
    st.plotly_chart(fig2, use_container_width=True)

st.dataframe(round_for_display(all_metrics), use_container_width=True)
 
# CSVダウンロード（全体成績サマリー）
try:
    csv_overall = to_csv_bytes(all_metrics)
    st.download_button(
        'CSVダウンロード（全体成績サマリー）',
        data=csv_overall,
        file_name='summary_overall.csv',
        mime='text/csv'
    )
except Exception:
    pass

# ------------------------------
# 年度別サマリー（カレンダー年ベース）
# ------------------------------
def summarize_by_year(trades: pd.DataFrame, colmap: Dict[str, str], main_eq_daily: pd.Series = None) -> pd.DataFrame:
    """
    トレード群をイグジット年（exit_date.year）でグルーピングして calc_metrics を実行。
    さらに main_eq_daily（日次エクイティ）が渡されれば、年次TWR(%)を補完して返す。
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    ycol = colmap.get('exit_date')
    if not ycol or ycol not in trades.columns:
        return pd.DataFrame()

    tmp = trades.copy()
    tmp['_exit_year'] = pd.to_datetime(tmp[ycol], errors='coerce').dt.year
    rows = []
    for year, g in tmp.groupby('_exit_year'):
        if pd.isna(year):
            continue
        met = calc_metrics(g, colmap)
        met['Year'] = int(year)
        rows.append(met)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values('Year', ascending=False).reset_index(drop=True)

    # 補完：main_eq_daily があればカレンダー年の TWR(%) を計算してマージ
    try:
        if main_eq_daily is not None and not main_eq_daily.empty:
            you_r_daily = daily_returns_from_equity(main_eq_daily)
            yr_twr = calc_calendar_year_twr(you_r_daily)  # Series index=Year value=fraction
            yr_df = pd.DataFrame({'Year': [int(y) for y in yr_twr.index], '年次TWR(%)': (yr_twr.values * 100.0)})
            out = out.merge(yr_df, on='Year', how='left')
    except Exception:
        # 失敗しても行の出力自体は止めない
        pass

    # 並べ替えのための列順整備（年, トレード数, 年次TWR, TWR/総損益など）
    pref = ['Year', 'トレード数', '年次TWR(%)', 'TWR(%)', '総合リターン(%)', '年率換算リターン(%)',
            '勝率', 'プロフィットファクター', '最大ドローダウン(%)', 'シャープレシオ',
            'ソルティノレシオ', '利益平均(%)', '損失平均(%)', '損益比', '総損益金額(円)']
    cols = [c for c in pref if c in out.columns] + [c for c in out.columns if c not in pref]
    out = out[cols]
    return out

# UI 表示ブロック
st.subheader('年度別サマリー（カレンダー年ベース）')

try:
    # main_eq_daily は既に作っている想定（無ければ作る）
    if 'main_eq_daily' in globals() and main_eq_daily is not None:
        med = main_eq_daily
    else:
        med = get_daily_equity_by_mode(ndf_eff, colmap_eff, mode)

    year_summary = summarize_by_year(ndf_eff, colmap_eff, main_eq_daily=med)
    if year_summary.empty:
        st.info('年度別サマリーを算出できません。exit(日付)列の存在を確認してください。')
    else:
        st.dataframe(round_for_display(year_summary), use_container_width=True)

        # 年次TWR(%) の棒グラフ（ある場合）
        try:
            if '年次TWR(%)' in year_summary.columns:
                figy = px.bar(year_summary.sort_values('Year'), x='Year', y='年次TWR(%)',
                              title='年度別 年次TWR(%)', labels={'年次TWR(%)':'年次TWR(%)'})
                st.plotly_chart(figy, use_container_width=True)
        except Exception:
            pass

        # 総損益金額(円) の年次推移（ある場合）
        try:
            if '総損益金額(円)' in year_summary.columns:
                figp = px.bar(year_summary.sort_values('Year'), x='Year', y='総損益金額(円)',
                              title='年度別 総損益金額(円)')
                st.plotly_chart(figp, use_container_width=True)
        except Exception:
            pass

        csv = to_csv_bytes(year_summary)
        st.download_button('CSVダウンロード（年度別サマリー）', data=csv,
                           file_name='summary_by_year.csv', mime='text/csv')
except Exception as e:
    st.warning(f'年度別サマリーの計算で問題が発生しました: {e}')


# ------------------------------
# サマリー (2) 貸借/制度 別
# ------------------------------
st.subheader('② 貸借/制度 別 成績サマリー')
by_credit = summarize_by_credit(ndf_eff, colmap_eff)
if by_credit.empty:
    st.info('貸借/制度の列が見つからないため、このサマリーは表示できません。')
else:
    st.dataframe(round_for_display(by_credit), use_container_width=True)
    csv = by_credit.to_csv(index=False).encode('utf-8-sig')
    st.download_button('CSVダウンロード（貸借/制度別）', data=csv, file_name='summary_credit.csv', mime='text/csv')

# ------------------------------
# ④ エントリー月(2a) 成績サマリー
# ------------------------------
st.subheader('④ エントリー月 成績サマリー（区分けなし）')
by_month = summarize_by_entry_month(ndf_eff, colmap_eff, with_credit=False)
if by_month.empty:
    st.info('エントリー月サマリーを計算できるデータが不足しています（エントリー日が必要）。')
else:
    st.dataframe(round_for_display(by_month), use_container_width=True)
    # 参考可視化（平均リターン%）
    try:
        figm = px.bar(by_month, x='エントリー月', y='平均リターン(%)',
                      title='平均リターン(%) by エントリー月')
        st.plotly_chart(figm, use_container_width=True)
    except Exception:
        pass
    csv = by_month.to_csv(index=False).encode('utf-8-sig')
    st.download_button('CSVダウンロード（エントリー月・区分けなし）',
                       data=csv, file_name='summary_entry_month.csv', mime='text/csv')

st.subheader('④ エントリー月 成績サマリー（貸借/制度の区分けあり）')
by_month_c = summarize_by_entry_month(ndf_eff, colmap_eff, with_credit=True)
if by_month_c.empty:
    st.info('貸借/制度の列が見つからないため、区分けありサマリーは表示できません。')
else:
    st.dataframe(round_for_display(by_month_c), use_container_width=True)
    # 参考可視化（平均リターン%）
    try:
        figmc = px.bar(by_month_c, x='エントリー月', y='平均リターン(%)', color='貸借/制度',
                       barmode='group', title='平均リターン(%) by エントリー月 × 貸借/制度')
        st.plotly_chart(figmc, use_container_width=True)
    except Exception:
        pass
    csv = by_month_c.to_csv(index=False).encode('utf-8-sig')
    st.download_button('CSVダウンロード（エントリー月・区分けあり）',
                       data=csv, file_name='summary_entry_month_credit.csv', mime='text/csv')


# ------------------------------
# サマリー (2b) 保持日数バケット 別
# ------------------------------
st.subheader('②-追加）保持日数バケット 別 成績サマリー')
by_bucket = summarize_by_bucket(ndf_eff, colmap_eff)
if by_bucket.empty:
    st.info('保持日数バケット列が見つからないため、このサマリーは表示できません。')
else:
    st.dataframe(round_for_display(by_bucket), use_container_width=True)
    # 可視化は try/except で囲む（except を必ず書く）
    try:
        figb = px.bar(by_bucket, x='保持日数バケット', y='平均リターン(%)',
                      title='平均リターン(%) by 保持日数バケット')
        st.plotly_chart(figb, use_container_width=True)
    except Exception:
        pass
    csv = by_bucket.to_csv(index=False).encode('utf-8-sig')
    st.download_button('CSVダウンロード（保持日数バケット別）',
                       data=csv, file_name='summary_bucket.csv', mime='text/csv')

# ------------------------------
# サマリー (3) エントリー月 × 保持日数
# ------------------------------
st.subheader('③ エントリー月 × 保持日数（区分けなし）')
by_m_b = summarize_by_month_bucket(ndf_eff, colmap_eff, with_credit=False)
if by_m_b.empty:
    st.info('エントリー月または保持日数バケットが見つからないため、このサマリーは表示できません。')
else:
    st.dataframe(round_for_display(by_m_b), use_container_width=True)
    csv = by_m_b.to_csv(index=False).encode('utf-8-sig')
    st.download_button('CSVダウンロード（③ 区分けなし）', data=csv,
                       file_name='summary_month_bucket.csv', mime='text/csv')

st.subheader('③ エントリー月 × 保持日数（貸借/制度の区分けあり）')
by_m_b_c = summarize_by_month_bucket(ndf_eff, colmap_eff, with_credit=True)
if by_m_b_c.empty:
    st.info('列が不足しているため、このサマリーは表示できません。')
else:
    st.dataframe(round_for_display(by_m_b_c), use_container_width=True)
    csv = by_m_b_c.to_csv(index=False).encode('utf-8-sig')
    st.download_button('CSVダウンロード（③ 区分けあり）', data=csv,
                       file_name='summary_month_bucket_credit.csv', mime='text/csv')


# ------------------------------
# ベンチマーク比較（TWRベース）
# ------------------------------
st.subheader('ベンチマーク比較（TWRベース）')
try:
    you_eq_daily = get_daily_equity_by_mode(ndf_eff, colmap_eff, mode)
    you_r_daily = daily_returns_from_equity(you_eq_daily)

    if bm_df is None or bm_df.empty:
        st.info('ベンチマークCSV/TSVをサイドバーから読み込むと比較を表示します。')
    else:
        bm_r_daily = normalize_benchmark(bm_df)
        if you_r_daily.empty or bm_r_daily.empty:
            st.info('日次データが不足しています。ベンチマークCSVの列（date, price/nav/close）をご確認ください。')
        else:
            common = you_r_daily.index.intersection(bm_r_daily.index)
            r_you = you_r_daily.loc[common]
            r_bm  = bm_r_daily.loc[common]

            # 年次TWR
            y_you = calc_calendar_year_twr(r_you)
            y_bm  = calc_calendar_year_twr(r_bm)
            ydf = pd.DataFrame({'あなた(%)': y_you*100.0, 'ベンチマーク(%)': y_bm*100.0})
            ydf['超過(%)'] = ydf['あなた(%)'] - ydf['ベンチマーク(%)']
            ydf.index.name = 'Year'
            st.write('年次TWR（カレンダー年）')
            st.dataframe(round_for_display(ydf.reset_index()), use_container_width=True)
            try:
                figy = px.bar(ydf.reset_index(), x='Year', y=['あなた(%)','ベンチマーク(%)'],
                              barmode='group', title='年次TWR')
                st.plotly_chart(figy, use_container_width=True)
            except Exception:
                pass

            # トレーリング（1/3/5年）
            eq_you = (1.0 + r_you).cumprod()
            eq_bm  = (1.0 + r_bm).cumprod()
            twr_you_tbl = calc_trailing_twr_table(eq_you)
            twr_bm_tbl  = calc_trailing_twr_table(eq_bm)
            if not twr_you_tbl.empty and not twr_bm_tbl.empty:
                tdf = twr_you_tbl.merge(twr_bm_tbl, on='Window', suffixes=('(あなた)','(BM)'))
                if '年率換算(%)(あなた)' in tdf.columns and '年率換算(%)(BM)' in tdf.columns:
                    tdf['超過(年率% )'] = tdf['年率換算(%)(あなた)'] - tdf['年率換算(%)(BM)']
                st.write('トレーリング（1/3/5年）TWRと年率換算')
                st.dataframe(round_for_display(tdf), use_container_width=True)

            # アクティブ統計
            adf = active_stats_df(r_you, r_bm)
            if not adf.empty:
                st.write('アクティブ統計（あなた vs ベンチマーク）')
                st.dataframe(round_for_display(adf), use_container_width=True)

            # 累積曲線
            try:
                cum = pd.DataFrame({'あなた': eq_you, 'ベンチマーク': eq_bm})
                figc = go.Figure()
                for c in cum.columns:
                    figc.add_trace(go.Scatter(x=cum.index, y=cum[c], mode='lines', name=c))
                figc.update_layout(height=320, title='累積リターン（共通日付）')
                st.plotly_chart(figc, use_container_width=True)
            except Exception:
                pass
except Exception as e:
    st.warning(f'ベンチマーク比較の計算で問題が発生しました: {e}')

# ------------------------------
# 原データプレビュー（生データ & 整備済みデータ）  <-- ここをコピペ
# ------------------------------
st.subheader("原データプレビュー（生データ / 整備済）")

# 1) 生データの候補を見つけて表示（アップロード直後に使っている変数名を自動探索）
raw_df = None
raw_name = None
for cand in ('uploaded', 'uploaded_file', 'raw_df', 'raw', 'df_uploaded', 'df'):
    try:
        obj = eval(cand)
    except Exception:
        continue
    # UploadedFile オブジェクトなら読み直して DataFrame 化
    if hasattr(obj, "read") and callable(obj.read):
        try:
            # attempt to reset stream
            try:
                obj.seek(0)
            except Exception:
                pass
            tmp = load_table(obj)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                raw_df = tmp
                raw_name = cand
                break
        except Exception:
            continue
    elif isinstance(obj, pd.DataFrame):
        raw_df = obj
        raw_name = cand
        break

if raw_df is not None:
    with st.expander(f"アップロード生データプレビュー ({raw_name})", expanded=False):
        st.write(f"行数: {raw_df.shape[0]:,} 列数: {raw_df.shape[1]:,}")
        st.dataframe(raw_df.head(100), use_container_width=True)
        csv = raw_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button('CSVダウンロード（生データ 先頭100行）', data=csv,
                           file_name='raw_preview.csv', mime='text/csv')
else:
    st.info("生データ（raw/df 等）が検出できません。ファイル読み込み箇所の変数名をご確認ください。")

# 2) 整備済みデータ（ndf / ndf_eff / df_clean など）も表示
clean_df = None
clean_name = None
for cand in ('ndf_eff', 'ndf', 'df_clean', 'clean_df'):
    try:
        obj = eval(cand)
    except Exception:
        continue
    if isinstance(obj, pd.DataFrame):
        clean_df = obj
        clean_name = cand
        break

if clean_df is not None:
    with st.expander(f"整備済みデータプレビュー ({clean_name})", expanded=False):
        st.write(f"行数: {clean_df.shape[0]:,} 列数: {clean_df.shape[1]:,}")
        # 表示は丸め関数を使って見やすく
        try:
            st.dataframe(round_for_display(clean_df.head(100)), use_container_width=True)
        except Exception:
            st.dataframe(clean_df.head(100), use_container_width=True)
        csv2 = clean_df.head(100).to_csv(index=False).encode('utf-8-sig')
        st.download_button('CSVダウンロード（整備済 先頭100行）', data=csv2,
                           file_name='clean_preview.csv', mime='text/csv')
else:
    st.info("整備済みデータ（ndf 等）が検出できません。正規化処理後の変数名をご確認ください。")
# ------------------------------
# （end of preview block）
# ------------------------------
