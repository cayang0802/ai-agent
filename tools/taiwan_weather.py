import logging
import os

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_CWA_API_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"

_COUNTY_ALIAS: dict[str, str] = {
    "台北": "臺北市",
    "臺北": "臺北市",
    "北市": "臺北市",
    "台北市": "臺北市",
    "新北": "新北市",
    "新北市": "新北市",
    "桃園": "桃園市",
    "桃園市": "桃園市",
    "台中": "臺中市",
    "臺中": "臺中市",
    "中市": "臺中市",
    "台中市": "臺中市",
    "台南": "臺南市",
    "臺南": "臺南市",
    "南市": "臺南市",
    "台南市": "臺南市",
    "高雄": "高雄市",
    "高雄市": "高雄市",
    "基隆": "基隆市",
    "基隆市": "基隆市",
    "新竹": "新竹市",
    "新竹市": "新竹市",
    "新竹縣": "新竹縣",
    "苗栗": "苗栗縣",
    "苗栗縣": "苗栗縣",
    "彰化": "彰化縣",
    "彰化縣": "彰化縣",
    "南投": "南投縣",
    "南投縣": "南投縣",
    "雲林": "雲林縣",
    "雲林縣": "雲林縣",
    "嘉義": "嘉義市",
    "嘉義市": "嘉義市",
    "嘉義縣": "嘉義縣",
    "屏東": "屏東縣",
    "屏東縣": "屏東縣",
    "宜蘭": "宜蘭縣",
    "宜蘭縣": "宜蘭縣",
    "花蓮": "花蓮縣",
    "花蓮縣": "花蓮縣",
    "台東": "臺東縣",
    "臺東": "臺東縣",
    "台東縣": "臺東縣",
    "臺東縣": "臺東縣",
    "澎湖": "澎湖縣",
    "澎湖縣": "澎湖縣",
    "金門": "金門縣",
    "金門縣": "金門縣",
    "連江": "連江縣",
    "連江縣": "連江縣",
    "馬祖": "連江縣",
}


def _resolve_county(location: str) -> tuple[str | None, str | None]:
    loc = location.strip()
    county = _COUNTY_ALIAS.get(loc)
    if county:
        return county, None
    for alias, full in _COUNTY_ALIAS.items():
        if loc.startswith(alias):
            rest = loc[len(alias):].strip()
            return full, rest or None
    return None, loc


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return None if f in (-99.0, -999.0, 9999.0) else f
    except (TypeError, ValueError):
        return None


def _fetch_weather(county: str | None, station: str | None, api_key: str) -> dict | None:
    url = f"{_CWA_API_BASE}/O-A0003-001"
    resp = requests.get(
        url,
        params={"Authorization": api_key, "format": "JSON", "limit": "500"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("success") != "true":
        return None

    stations: list = data.get("records", {}).get("Station", [])

    if county:
        stations = [s for s in stations if s.get("GeoInfo", {}).get("CountyName") == county]
    if station:
        matched = [s for s in stations if station in s.get("StationName", "")]
        if matched:
            stations = matched

    for s in stations:
        we = s.get("WeatherElement", {})
        if _safe_float(we.get("AirTemperature")) is not None or we.get("Weather"):
            return s
    return stations[0] if stations else None


def _format_station(s: dict) -> str:
    geo = s.get("GeoInfo", {})
    we = s.get("WeatherElement", {})
    obs_time = s.get("ObsTime", {}).get("DateTime", "")

    county = geo.get("CountyName", "")
    town = geo.get("TownName", "")
    name = s.get("StationName", "")
    location_str = f"{county}{town}" if (county or town) else name

    weather = we.get("Weather") or we.get("Now", {}).get("Weather") or "—"
    temp = _safe_float(we.get("AirTemperature"))
    humidity = _safe_float(we.get("RelativeHumidity"))
    wind_speed = _safe_float(we.get("WindSpeed"))
    wind_dir = _safe_float(we.get("WindDirection"))
    precip = _safe_float(we.get("Now", {}).get("Precipitation")) if isinstance(we.get("Now"), dict) else None

    lines = [f"【{location_str}】測站：{name}"]
    lines.append(f"天氣：{weather}")
    if temp is not None:
        lines.append(f"氣溫：{temp:.1f} °C")
    if humidity is not None:
        lines.append(f"相對濕度：{humidity:.0f}%")
    if wind_speed is not None:
        dir_str = f"，風向 {wind_dir:.0f}°" if wind_dir is not None else ""
        lines.append(f"風速：{wind_speed:.1f} m/s{dir_str}")
    if precip is not None and precip >= 0:
        lines.append(f"降雨量（現在）：{precip:.1f} mm")
    if obs_time:
        lines.append(f"觀測時間：{obs_time[:16].replace('T', ' ')}")
    return "\n".join(lines)


@tool
def get_taiwan_weather(location: str) -> str:
    """查詢台灣指定地點目前的天氣觀測資料。輸入縣市名稱或地點，例如：台北、高雄、新竹、花蓮、台中大里。"""
    api_key = os.getenv("CWA_API_KEY", "").strip()
    if not api_key:
        return "找不到 CWA_API_KEY，請在 .env 加入中央氣象署 API 金鑰。"

    try:
        county, station = _resolve_county(location)
        logger.info(
            "get_taiwan_weather location=%r -> county=%r, station=%r",
            location, county, station,
        )
        result = _fetch_weather(county, station, api_key)
        if not result:
            return f"找不到「{location}」的天氣觀測資料，請確認地名是否正確（例如：台北、高雄、花蓮）。"

        output = _format_station(result)
        logger.info("get_taiwan_weather result: %s", output)
        return output
    except Exception as e:
        logger.exception("get_taiwan_weather failed: %s", e)
        return f"查詢失敗：{e}"
