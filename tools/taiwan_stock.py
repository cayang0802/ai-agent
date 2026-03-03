import logging
from datetime import datetime

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_COMMON_NAME_TO_CODE: dict[str, str] = {
    "台積電": "2330",
    "鴻海": "2317",
    "聯發科": "2454",
    "廣達": "2382",
    "台達電": "2308",
}


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s:
        return ""
    if s in ("TWII", "加權", "加權指數", "^TWII", "T00"):
        return "TWII"
    if symbol.strip() in _COMMON_NAME_TO_CODE:
        return _COMMON_NAME_TO_CODE[symbol.strip()]
    return s


def _build_ex_ch_list(normalized: str) -> list[str]:
    if normalized == "TWII":
        return ["tse_t00.tw"]
    if normalized.isdigit():
        return [f"tse_{normalized}.tw", f"otc_{normalized}.tw"]
    if normalized.endswith(".TWO"):
        return [f"otc_{normalized[:-4]}.tw"]
    if normalized.endswith(".TW"):
        return [f"tse_{normalized[:-3]}.tw"]
    return [f"tse_{normalized.lower()}.tw", f"otc_{normalized.lower()}.tw"]


def _safe_float(value: str | None) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_date(date_str: str | None) -> str:
    if not date_str:
        return ""
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return date_str


def _fetch_mis_quote(ex_ch_list: list[str]) -> dict | None:
    url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
    params = {"ex_ch": "|".join(ex_ch_list), "json": "1", "delay": "0"}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    for item in data.get("msgArray", []):
        if _safe_float(item.get("z")) is not None or _safe_float(item.get("y")) is not None:
            return item
    return None


@tool
def get_taiwan_stock(symbol: str) -> str:
    """查詢台股最新股價與指數。輸入台股代號，例如：2330（台積電）、2317（鴻海）、TWII（加權指數）。"""
    try:
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return "請輸入有效的台股代號。"

        ex_ch_list = _build_ex_ch_list(normalized)
        logger.info("get_taiwan_stock symbol=%r -> ex_ch=%r", symbol, ex_ch_list)
        quote = _fetch_mis_quote(ex_ch_list)

        if not quote:
            return f"找不到代號「{symbol}」的台股資料，請確認代號是否正確（例如 2330、TWII）。"

        name = quote.get("n") or quote.get("nf") or symbol
        code = quote.get("c") or normalized
        price = _safe_float(quote.get("z")) or _safe_float(quote.get("o")) or _safe_float(quote.get("y"))
        prev = _safe_float(quote.get("y"))
        if price is None:
            return f"代號「{symbol}」目前無可用報價（可能尚未開盤或暫停交易）。"

        change = (price - prev) if prev is not None else 0.0
        pct = ((change / prev) * 100) if prev not in (None, 0) else 0.0
        date_str = _format_date(quote.get("d"))
        time_str = quote.get("t") or quote.get("%") or ""

        lines = [
            f"【{name}】 {code}",
            f"最新價：{price:.2f}",
            f"漲跌：{change:+.2f}（{pct:+.2f}%）",
            f"時間：{date_str} {time_str}".strip(),
        ]
        result = "\n".join(lines)
        logger.info("get_taiwan_stock result: %s", result)
        return result
    except Exception as e:
        logger.exception("get_taiwan_stock failed: %s", e)
        return f"查詢失敗：{e}"
