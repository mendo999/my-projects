# -*- coding: utf-8 -*-
"""
取引実績TSV/CSV → 管理TSV 変換ツール（数値列を上書き：カンマ/％を除去 + 保有日数）
- 指定列（平均購入価格, 平均売却価格, 損益, リターン率）を数値化して上書き出力
- 新機能：保有日数（取引営業日ベース、両端含む。例：同一営業日なら 1 日）
  * 可能なら JPX 取引所カレンダー（pandas_market_calendars）を使用
  * 次に jpholiday（日本の祝日）を使用
  * いずれも無い場合は平日（Mon-Fri）のみで近似
"""
from __future__ import annotations
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

import pandas as pd
import numpy as np

OUTPUT_COLUMNS_MATCH = [
    "No", "銘柄コード", "銘柄名", "権利月", "市場", "貸借", "保持期間", "購入月", "売却月",
    "口座区分", "数量", "エントリー日", "購入金額合計", "イグジット日",
    "売却金額合計", "損益", "リターン率", "保有日数"
]

OUTPUT_COLUMNS_UNMATCH = [
    "理由", "約定日", "銘柄コード", "銘柄名", "市場名称", "口座区分", "取引区分", "売買区分",
    "信用区分", "数量［株］", "単価［円］", "受渡金額［円］", "備考"
]

SRC = {
    "code": "銘柄コード",
    "name": "銘柄名",
    "market": "市場名称",
    "account": "口座区分",
    "trade_category": "取引区分",
    "trade_type": "売買区分",
    "margin_type": "信用区分",
    "qty": "数量［株］",
    "price": "単価［円］",
    "amount": "受渡金額［円］",
    "date": "約定日",
}

def sniff_delimiter_and_encoding(path: Path) -> Tuple[str, str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
        enc = "utf-8"
    comma = text.count(",")
    tab = text.count("\t")
    sep = "," if comma >= tab else "\t"
    return sep, enc

def _to_number(series: pd.Series) -> pd.Series:
    """Convert strings like '130,430.00' or '(1,234)' to float safely."""
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).fillna("").str.strip()
    # (1,234) -> -1,234
    s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
    # 全角コンマ/半角コンマ、通貨記号を削除
    s = s.str.replace(r'[,\uFF0C¥$]', '', regex=True)
    # 空文字を NaN に変換（警告対応：infer_objects追加）
    s = s.replace('', np.nan).infer_objects(copy=False)
    return pd.to_numeric(s, errors="coerce")


def _quantize(val: Any, places: int) -> Optional[Decimal]:
    if pd.isna(val):
        return None
    try:
        d = Decimal(str(val))
        q = Decimal('1').scaleb(-places)
        return d.quantize(q, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def _format_decimal(val: Optional[Decimal]) -> str:
    if val is None:
        return ""
    return format(val.normalize(), 'f') if val.as_tuple().exponent != 0 else str(val)

def _percent_to_decimal(series: pd.Series) -> pd.Series:
    """Convert '40.1%' or '40.1 %' to 0.401. If already numeric, keep as is."""
    s = series.astype(str).fillna("").str.strip()
    # 全角％を半角に
    s = s.str.replace('％', '%', regex=False)
    # remove percent sign
    has_pct = s.str.contains('%')
    s_clean = s.str.replace('%', '', regex=False)
    # remove commas etc
    s_clean = s_clean.str.replace(r'[,\uFF0C]', '', regex=True)
    # 空文字を NaN に変換（警告対応：infer_objects追加）
    s_clean = s_clean.replace('', np.nan).infer_objects(copy=False)
    num = pd.to_numeric(s_clean, errors="coerce")
    # where original had %, divide by 100
    num = np.where(has_pct, num / 100.0, num)
    # convert to pandas Series with float dtype
    return pd.Series(num, index=series.index, dtype="float64")

def _count_trading_days(start, end) -> Optional[int]:
    """
    取引営業日ベースの保有日数（両端含む）。
    優先: JPX (pandas_market_calendars) → jpholiday → 平日(Mon-Fri)のみ
    """
    if pd.isna(start) or pd.isna(end):
        return np.nan
    try:
        s = pd.to_datetime(start).normalize()
        e = pd.to_datetime(end).normalize()
    except Exception:
        return np.nan
    if s > e:
        return np.nan

    # 1) pandas_market_calendars の JPX カレンダー
    try:
        import pandas_market_calendars as pmc  # type: ignore
        cal = pmc.get_calendar('JPX')
        valid = cal.valid_days(s, e)  # inclusive
        return int(len(valid))
    except Exception:
        pass

    # 2) jpholiday + 平日(Mon-Fri)
    try:
        import jpholiday  # type: ignore
        rng = pd.date_range(s, e, freq='D')
        cnt = 0
        for d in rng:
            if d.weekday() >= 5:  # Sat/Sun
                continue
            if jpholiday.is_holiday(d.date()):
                continue
            cnt += 1
        return int(cnt)
    except Exception:
        pass

    # 3) 近似: 平日(Mon-Fri)のみ（祝日はカウント）
    # numpy.busday_count は [start, end) の営業日数（祝日定義なし）→ inclusive にするため end+1日
    try:
        c = np.busday_count(s.date(), (e + pd.Timedelta(days=1)).date())
        return int(c)
    except Exception:
        return np.nan


def _normalize_code(code: Any) -> str:
    s = str(code).strip()
    if not s.isdigit():
        return ""
    return s.zfill(4)[-4:]


def _clean_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().replace('', np.nan)
    parsed = pd.to_datetime(s, errors='coerce')
    return parsed.dt.strftime('%Y/%m/%d')


def _get_margin_short_dates(raw_df: pd.DataFrame) -> set[str]:
    """信用新規×売建が存在する約定日（買付除外用）を取得。"""
    if raw_df is None or raw_df.empty:
        return set()
    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    mask = (
        df.get(SRC['trade_category'], '').astype(str).str.strip() == '信用新規'
    ) & (
        df.get(SRC['trade_type'], '').astype(str).str.strip() == '売建'
    )
    if not mask.any():
        return set()
    dates = _clean_date(df.loc[mask, SRC['date']])
    return set(dates.dropna().tolist())


def _load_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    # 正規化
    df = df.copy()
    df.columns = df.columns.str.strip()

    df[SRC['code']] = df.get(SRC['code'], '').apply(_normalize_code)
    df[SRC['trade_category']] = df.get(SRC['trade_category'], '')
    df[SRC['margin_type']] = df.get(SRC['margin_type'], '')
    df[SRC['trade_type']] = df.get(SRC['trade_type'], '')

    # 数値変換
    df[SRC['qty']] = _to_number(df.get(SRC['qty'], pd.Series(dtype=object)))
    df[SRC['price']] = _to_number(df.get(SRC['price'], pd.Series(dtype=object)))
    df[SRC['amount']] = _to_number(df.get(SRC['amount'], pd.Series(dtype=object)))

    # 日付整形
    df[SRC['date']] = _clean_date(df.get(SRC['date'], pd.Series(dtype=object)))

    # フィルタ条件
    cond = (
        df[SRC['trade_category']].astype(str).str.strip() == '現物'
    ) & (
        df[SRC['margin_type']].astype(str).str.strip() == '-'
    ) & (
        df[SRC['trade_type']].astype(str).str.strip().isin(['買付', '売付'])
    )
    return df.loc[cond].copy()


def _calc_amount(row: pd.Series) -> float:
    amount = row.get(SRC['amount'])
    if pd.notna(amount):
        return float(abs(amount))
    qty = row.get(SRC['qty'])
    price = row.get(SRC['price'])
    if pd.notna(qty) and pd.notna(price):
        return float(abs(qty * price))
    return float('nan')


def _partial_amount(lot: Dict[str, Any], qty: float, prefer_price: bool) -> float:
    """
    部分約定の金額を計算する。
    prefer_price=True の場合、単価×数量を優先（売り側の要望に対応）。
    """
    if prefer_price and pd.notna(lot.get('price')):
        return float(abs(lot['price'] * qty))
    # 比率配分（受渡金額がある場合）
    if pd.notna(lot.get('amount')):
        base = float(abs(lot['amount']))
        ratio = qty / lot['qty'] if lot['qty'] else 0
        return base * ratio
    if pd.notna(lot.get('price')):
        return float(abs(lot['price'] * qty))
    return float('nan')


def _prepare_lots(df: pd.DataFrame, trade_type: str, exclude_dates: set[str]) -> List[Dict[str, Any]]:
    lots = []
    for _, row in df.iterrows():
        qty = row.get(SRC['qty'])
        if pd.isna(qty) or qty == 0:
            continue
        date_str = row.get(SRC['date'], "")
        if trade_type == '買付' and date_str in exclude_dates:
            # 信用新規売建と同日買付は除外
            continue
        lots.append({
            "date": date_str,
            "code": row.get(SRC['code'], ""),
            "name": row.get(SRC['name'], ""),
            "market": row.get(SRC['market'], ""),
            "account": row.get(SRC['account'], ""),
            "qty": float(qty),
            "amount": _calc_amount(row),
            "price": row.get(SRC['price']),
            "raw": row,
        })
    return sorted(lots, key=lambda x: (x['date'] or '', x['raw'].name))


def _match_trades(df: pd.DataFrame, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_filtered = _load_and_filter(df)
    margin_short_dates = _get_margin_short_dates(raw_df)

    buys = df_filtered[df_filtered[SRC['trade_type']] == '買付']
    sells = df_filtered[df_filtered[SRC['trade_type']] == '売付']

    matched_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []

    # グループキー: 銘柄コード + 口座区分
    keys = set(zip(df_filtered[SRC['code']], df_filtered[SRC['account']]))

    for code, account in keys:
        buy_lots = [l for l in _prepare_lots(buys[(buys[SRC['code']] == code) & (buys[SRC['account']] == account)], '買付', margin_short_dates)]
        sell_lots = [l for l in _prepare_lots(sells[(sells[SRC['code']] == code) & (sells[SRC['account']] == account)], '売付', margin_short_dates)]

        bi = si = 0
        while bi < len(buy_lots) and si < len(sell_lots):
            b = buy_lots[bi]
            s = sell_lots[si]
            match_qty = min(b['qty'], s['qty'])
            if match_qty <= 0:
                break

            buy_amt = _partial_amount(b, match_qty, prefer_price=True)
            sell_amt = _partial_amount(s, match_qty, prefer_price=True)

            matched_rows.append({
                "銘柄コード": code,
                "銘柄名": b['name'] or s['name'],
                "市場": b['market'] or s['market'],
                "口座区分": account,
                "数量": match_qty,
                "エントリー日": b['date'],
                "購入金額合計": buy_amt,
                "イグジット日": s['date'],
                "売却金額合計": sell_amt,
            })

            b['qty'] -= match_qty
            s['qty'] -= match_qty
            if b['qty'] <= 1e-9:
                bi += 1
            if s['qty'] <= 1e-9:
                si += 1

        # 売り先行（買い無し）
        for rest in sell_lots[si:]:
            unmatched_rows.append({
                "理由": "売り先行 買い未登録",
                "約定日": rest['date'],
                "銘柄コード": code,
                "銘柄名": rest['name'],
                "市場名称": rest['market'],
                "口座区分": account,
                "取引区分": "現物",
                "売買区分": "売付",
                "信用区分": "-",
                "数量［株］": rest['qty'],
                "単価［円］": rest['price'],
                "受渡金額［円］": rest['amount'],
                "備考": "売付が買付数量を超過" ,
            })

        # 買い残（売却未実行）: converted にも出力する
        for rest in buy_lots[bi:]:
            matched_rows.append({
                "銘柄コード": code,
                "銘柄名": rest['name'],
                "市場": rest['market'],
                "口座区分": account,
                "数量": rest['qty'],
                "エントリー日": rest['date'],
                "購入金額合計": rest['amount'],
                "イグジット日": "",
                "売却金額合計": np.nan,
            })

            unmatched_rows.append({
                "理由": "買い不足 残株",
                "約定日": rest['date'],
                "銘柄コード": code,
                "銘柄名": rest['name'],
                "市場名称": rest['market'],
                "口座区分": account,
                "取引区分": "現物",
                "売買区分": "買付",
                "信用区分": "-",
                "数量［株］": rest['qty'],
                "単価［円］": rest['price'],
                "受渡金額［円］": rest['amount'],
                "備考": "売却未実行" ,
            })

    match_df = pd.DataFrame(matched_rows)
    unmatch_df = pd.DataFrame(unmatched_rows)
    return match_df, unmatch_df


def _build_output(match_df: pd.DataFrame, include_holding_days: bool) -> pd.DataFrame:
    if match_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS_MATCH)

    rows = []
    for _, r in match_df.iterrows():
        qty = r["数量"]
        buy_total = r["購入金額合計"]
        sell_total = r["売却金額合計"]

        pnl = None if pd.isna(sell_total) or pd.isna(buy_total) else Decimal(str(sell_total)) - Decimal(str(buy_total))
        rate = None
        if pnl is not None and buy_total and buy_total != 0:
            rate = (pnl / Decimal(str(buy_total))).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

        hold_days = _count_trading_days(r["エントリー日"], r["イグジット日"]) if include_holding_days else np.nan

        rows.append({
            "銘柄コード": r["銘柄コード"],
            "銘柄名": r["銘柄名"],
            "権利月": "",
            "市場": r["市場"],
            "貸借": "-",
            "保持期間": "",
            "購入月": "",
            "売却月": "",
            "口座区分": r["口座区分"],
            "数量": qty,
            "エントリー日": r["エントリー日"],
            "購入金額合計": int(round(buy_total)) if pd.notna(buy_total) else "",
            "イグジット日": r["イグジット日"],
            "売却金額合計": int(round(sell_total)) if pd.notna(sell_total) else "",
            "損益": int(pnl) if pnl is not None else "",
            "リターン率": _format_decimal(rate) if rate is not None else "",
            "保有日数": hold_days if include_holding_days else "",
        })

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS_MATCH)
    # ソート: エントリー日 → イグジット日 → 銘柄コード（安定ソート）
    out_df["__entry_dt"] = pd.to_datetime(out_df["エントリー日"], errors="coerce")
    out_df["__exit_dt"] = pd.to_datetime(out_df["イグジット日"], errors="coerce")
    out_df = out_df.sort_values(by=["__entry_dt", "__exit_dt", "銘柄コード"], kind="mergesort")
    out_df = out_df.drop(columns=["__entry_dt", "__exit_dt"])

    out_df["No"] = range(1, len(out_df) + 1)
    return out_df


def convert_dataframe(df: pd.DataFrame, include_holding_days: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    match_df, unmatch_df = _match_trades(df, df)
    out_df = _build_output(match_df, include_holding_days)
    return out_df, unmatch_df

def convert_file(input_path: str | Path, output_dir: Optional[str | Path]=None, output_filename: Optional[str]=None, include_holding_days: bool=True) -> Path:
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{input_path.stem}_converted.tsv"

    sep, enc = sniff_delimiter_and_encoding(input_path)
    df = pd.read_csv(input_path, sep=sep, encoding=enc)
    out_df, unmatch_df = convert_dataframe(df, include_holding_days=include_holding_days)

    out_path = output_dir / output_filename
    out_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig", float_format='%.10g')

    # 未マッチ出力
    unmatch_path = output_dir / f"{input_path.stem}_unmatch.tsv"
    if not unmatch_df.empty:
        unmatch_df.to_csv(unmatch_path, sep="\t", index=False, encoding="utf-8-sig", float_format='%.10g')
    else:
        # 空でもヘッダ付きで出しておく方が運用しやすいケースが多いため生成
        pd.DataFrame(columns=OUTPUT_COLUMNS_UNMATCH).to_csv(unmatch_path, sep="\t", index=False, encoding="utf-8-sig")

    return out_path

# ---- Simple GUI ----
@dataclass
class AppState:
    input_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    include_holding_days: bool = True


class ConverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("楽天証券取引結果変換ツール")

        # macOS対応: ウィンドウサイズを拡大、リサイズ可能に変更
        self.geometry("700x350")
        self.resizable(True, True)  # リサイズ可能にする

        # プラットフォーム検出（macOS用の調整）
        import platform
        is_mac = platform.system() == "Darwin"

        # グリッドの列重み設定（レイアウト安定化）
        self.columnconfigure(0, weight=1)  # メインエントリー列に重み
        self.columnconfigure(1, weight=0)  # 中間列
        self.columnconfigure(2, weight=0)  # ボタン列

        self.state = AppState()

        # macOS用パディング調整
        padx, pady = (10, 8) if is_mac else (8, 6)

        tk.Label(self, text="入力ファイル（TSV/CSV）").grid(row=0, column=0, sticky="w", padx=padx, pady=pady)
        self.in_var = tk.StringVar()
        tk.Entry(self, textvariable=self.in_var, width=72).grid(row=1, column=0, columnspan=2, padx=padx, sticky="we")
        tk.Button(self, text="参照...", command=self.browse_input).grid(row=1, column=2, padx=padx, sticky="e")

        tk.Label(self, text="出力先フォルダ（未指定なら入力と同じ）").grid(row=2, column=0, sticky="w", padx=padx,
                                                                         pady=pady)
        self.out_var = tk.StringVar()
        tk.Entry(self, textvariable=self.out_var, width=72).grid(row=3, column=0, columnspan=2, padx=padx, sticky="we")
        tk.Button(self, text="変更...", command=self.browse_output_dir).grid(row=3, column=2, padx=padx, sticky="e")

        # Option: include 保有日数
        self.hold_var = tk.IntVar(value=1)
        tk.Checkbutton(self, text="保有日数（取引営業日, 両端含む）を出力する", variable=self.hold_var).grid(row=4,
                                                                                                           column=0,
                                                                                                           columnspan=2,
                                                                                                           sticky="w",
                                                                                                           padx=padx,
                                                                                                           pady=(6, 0))

        tk.Button(self, text="変換を実行", command=self.run_convert, width=20).grid(row=5, column=0, pady=16)
        tk.Button(self, text="終了", command=self.destroy, width=10).grid(row=5, column=2, sticky="e", padx=padx)

        # ダークモード対応: システム背景色を取得して適切な文字色を設定
        try:
            # 背景色を取得してダークモード判定
            bg_color = self.cget('bg')
            if bg_color in ('systemWindowBackgroundColor', 'SystemButtonFace') or is_mac:
                # macOSまたはシステム色の場合、システム標準の文字色を使用
                tips_color = "systemLabelColor" if is_mac else "#000000"
            else:
                # その他の場合は明度から判定（簡易実装）
                tips_color = "#666666"
        except:
            # エラー時のフォールバック
            tips_color = "#666666"

        tips = (
            "・指定列を数値化して上書き出力します（平均購入価格/平均売却価格/損益/リターン率）。\n"
            "・リターン率は小数（例: 0.401）。Excelで%表示にする場合はセルの表示形式で % を選択してください。\n"
            "・保有日数は可能なら JPX カレンダー、無ければ jpholiday、最終的に平日カウントで近似します（両端含む）。\n"
            "・出力名は <入力ファイル名>_converted.tsv"
        )

        # ダークモード対応のラベル作成
        tips_label = tk.Label(self, text=tips, justify="left", fg=tips_color)
        tips_label.grid(row=6, column=0, columnspan=3, padx=padx, pady=(4, 8), sticky="w")

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="入力ファイルを選択",
            filetypes=[("TSV/CSV ファイル", "*.tsv *.csv"), ("すべてのファイル", "*.*")]
        )
        if path:
            self.state.input_path = Path(path)
            self.in_var.set(path)
            self.state.output_dir = self.state.input_path.parent
            self.out_var.set(str(self.state.output_dir))

    def browse_output_dir(self):
        initdir = str(self.state.output_dir) if self.state.output_dir else ""
        path = filedialog.askdirectory(initialdir=initdir, title="出力フォルダを選択")
        if path:
            self.state.output_dir = Path(path)
            self.out_var.set(path)

    def run_convert(self):
        try:
            # Entryの値を優先して状態を更新（手入力にも対応）
            in_path_str = self.in_var.get().strip()
            if in_path_str:
                self.state.input_path = Path(in_path_str)
            if not self.state.input_path:
                raise ValueError("入力ファイルを指定してください。")
            if not self.state.input_path.exists():
                raise FileNotFoundError(f"入力ファイルが存在しません: {self.state.input_path}")

            out_dir_str = self.out_var.get().strip()
            if out_dir_str:
                self.state.output_dir = Path(out_dir_str)

            output_dir = self.state.output_dir or self.state.input_path.parent
            include_hold = bool(self.hold_var.get())

            out_path = convert_file(self.state.input_path, output_dir, include_holding_days=include_hold)
            messagebox.showinfo("完了", f"変換が完了しました。\n\n出力: {out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"変換に失敗しました。\n\n{e}")

def main():
    if len(sys.argv) >= 2:
        in_path = Path(sys.argv[1])
        out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
        include_hold = True
        if len(sys.argv) >= 4:
            include_hold = sys.argv[3].lower() not in ("0","false","off","no")
        out = convert_file(in_path, out_dir, include_holding_days=include_hold)
        print(f"出力: {out}")
        return

    app = ConverterApp()
    app.mainloop()

if __name__ == "__main__":
    main()
