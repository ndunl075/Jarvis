"""Current weather conditions via Open-Meteo (no API key required).

Location resolution order when no explicit location arg is given:
  1. cfg.weather.latitude / longitude — user-configured or previously
     persisted from IP detection; highest priority.
  2. ipapi.co auto-detect — used only when both coords are None. Result
     is written back to cfg.weather and persisted via save_fn so the next
     app launch skips ipapi.co entirely.

When a location string arg is provided:
  - Geocodes via Open-Meteo's free geocoding API (no key required).
  - On success: uses those coords and includes the city name in the response.
  - On failure (no results, timeout, network error): falls back to configured
    default with a spoken acknowledgment.

Open-Meteo API: https://open-meteo.com/
Open-Meteo Geocoding: https://open-meteo.com/en/docs/geocoding-api
WMO weather codes: https://open-meteo.com/en/docs/wmo-weather-code
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from jarvis.tools.registry import ToolResult

log = logging.getLogger(__name__)

# WMO weather interpretation codes (Open-Meteo / WMO standard).
_WMO: dict[int, str] = {
    0:  "clear skies",
    1:  "mainly clear",
    2:  "partly cloudy",
    3:  "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "moderate showers",
    82: "heavy showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}

_IPAPI_URL = "https://ipapi.co/json/"
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT = 3.0
_GEO_TIMEOUT = 5.0


def _geocode_queries(location: str) -> list[str]:
    """Return candidate query strings to try in order.

    If the location contains a comma (e.g. "Reston, Virginia"), a second
    attempt with just the city name is appended so city+state inputs work
    even when the geocoding API returns no results for the full string.
    """
    queries = [location]
    if "," in location:
        city_only = location.split(",", 1)[0].strip()
        if city_only:
            queries.append(city_only)
    return queries


class WeatherArgs(BaseModel):
    location: str | None = None


class WeatherTool:
    name: str = "get_weather"
    description: str = (
        "Reports current weather conditions. Optionally accepts a 'location' "
        "argument (city name or city+state). Without it, uses the configured "
        "default location. Use this whenever the user asks about weather, "
        "temperature, forecast, rain, snow, or similar weather topics."
    )
    args_schema: type[BaseModel] = WeatherArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        weather_config: object,
        save_fn: Callable[[], None] | None = None,
    ) -> None:
        self._config = weather_config
        # Called after IP-detection to persist coords so the next run skips ipapi.co.
        self._save_fn = save_fn

    async def execute(self, args: WeatherArgs) -> ToolResult:
        try:
            async with httpx.AsyncClient() as client:
                if args.location is not None:
                    geo = await self._geocode(client, args.location)
                    if geo is None:
                        lat, lon = await self._resolve_location(client)
                        result = await self._fetch_weather(client, lat, lon)
                        if result.success and result.output:
                            return ToolResult(
                                success=True,
                                output=(
                                    f"I couldn't find {args.location}. "
                                    f"Reporting weather for the default location, sir. "
                                    f"{result.output}"
                                ),
                            )
                        return result
                    lat, lon, city_name = geo
                    return await self._fetch_weather(client, lat, lon, city=city_name)
                lat, lon = await self._resolve_location(client)
                return await self._fetch_weather(client, lat, lon)
        except Exception as exc:
            log.warning("weather fetch failed: %s", exc)
            return ToolResult(
                success=False,
                output="I couldn't reach the weather service, sir.",
            )

    async def _geocode(
        self, client: httpx.AsyncClient, location: str
    ) -> tuple[float, float, str] | None:
        """Return (lat, lon, city_name) from the geocoding API, or None on failure.

        Tries the full query first. If no results and the query contains a
        comma (e.g. "Reston, Virginia"), retries with just the part before
        the first comma so city+state formats work reliably.
        """
        for query in _geocode_queries(location):
            try:
                r = await client.get(
                    _GEOCODING_URL,
                    params={"name": query, "count": 1},
                    timeout=_GEO_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                results = data.get("results") or []
                if results:
                    first = results[0]
                    return (
                        float(first["latitude"]),
                        float(first["longitude"]),
                        str(first["name"]),
                    )
                log.info("geocoding: no results for %r", query)
            except Exception as exc:
                log.warning("geocoding failed for %r: %s", query, exc)
                return None  # don't retry after a network error
        return None

    async def _resolve_location(self, client: httpx.AsyncClient) -> tuple[float, float]:
        cfg_lat = getattr(self._config, "latitude", None)
        cfg_lon = getattr(self._config, "longitude", None)
        if cfg_lat is not None and cfg_lon is not None:
            return float(cfg_lat), float(cfg_lon)
        log.info("weather: auto-detecting location from IP via ipapi.co")
        r = await client.get(_IPAPI_URL, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        # Persist so subsequent runs skip ipapi.co.
        try:
            self._config.latitude = lat  # type: ignore[union-attr]
            self._config.longitude = lon  # type: ignore[union-attr]
            if self._save_fn is not None:
                self._save_fn()
        except Exception:
            log.warning("weather: could not persist detected location")
        return lat, lon

    async def _fetch_weather(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        *,
        city: str | None = None,
    ) -> ToolResult:
        unit = getattr(self._config, "unit", "fahrenheit")
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": unit,
        }
        r = await client.get(_OPEN_METEO_URL, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        current = data["current"]
        temp = current["temperature_2m"]
        code = int(current["weather_code"])
        conditions = _WMO.get(code, "unknown conditions")
        if city is not None:
            return ToolResult(
                success=True,
                output=f"It's {temp:.0f} degrees and {conditions} in {city}, sir.",
            )
        return ToolResult(
            success=True,
            output=f"It's {temp:.0f} degrees and {conditions}, sir.",
        )
