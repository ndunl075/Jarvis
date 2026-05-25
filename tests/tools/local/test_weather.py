"""Tests for jarvis.tools.local.weather.WeatherTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.tools.local.weather import WeatherArgs, WeatherTool, _WMO, _geocode_queries


def _cfg(lat=None, lon=None, unit="fahrenheit"):
    cfg = MagicMock()
    cfg.latitude = lat
    cfg.longitude = lon
    cfg.unit = unit
    return cfg


def _meteo_response(temp: float, code: int) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "current": {
            "temperature_2m": temp,
            "weather_code": code,
            "wind_speed_10m": 5.0,
        }
    }
    return resp


def _ipapi_response(lat: float, lon: float) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"latitude": lat, "longitude": lon, "city": "Test City"}
    return resp


def _geocoding_response(lat: float, lon: float, name: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "results": [{"latitude": lat, "longitude": lon, "name": name}]
    }
    return resp


def _geocoding_no_results() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"results": []}
    return resp


def _make_client(responses: list) -> MagicMock:
    """AsyncMock client whose .get() returns responses in order."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=responses)
    return client


# --- happy paths: no location arg ---


async def test_configured_location_calls_meteo_with_coords():
    tool = WeatherTool(weather_config=_cfg(lat=40.0, lon=-74.0))
    client = _make_client([_meteo_response(72.0, 0)])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    assert result.success
    assert "72" in (result.output or "")
    assert "clear skies" in (result.output or "")
    assert "sir" in (result.output or "").lower()


async def test_ip_fallback_calls_ipapi_then_meteo():
    """When lat/lon are None, ipapi.co is called first, then Open-Meteo."""
    tool = WeatherTool(weather_config=_cfg())
    client = _make_client([
        _ipapi_response(51.5, -0.1),
        _meteo_response(55.0, 61),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    assert result.success
    output = result.output or ""
    assert "55" in output
    assert "light rain" in output


async def test_ip_location_cached_across_calls():
    """Second execute() must NOT call ipapi.co again."""
    tool = WeatherTool(weather_config=_cfg())
    ipapi = _ipapi_response(40.7, -74.0)
    meteo1 = _meteo_response(70.0, 1)
    meteo2 = _meteo_response(65.0, 2)
    client = _make_client([ipapi, meteo1, meteo2])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await tool.execute(WeatherArgs())
        await tool.execute(WeatherArgs())
    # 3 total calls: 1 ipapi + 2 meteo
    assert client.get.call_count == 3


async def test_celsius_unit_passed_to_meteo():
    tool = WeatherTool(weather_config=_cfg(lat=48.0, lon=2.0, unit="celsius"))
    client = _make_client([_meteo_response(22.0, 0)])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    assert result.success
    params = client.get.call_args.kwargs.get("params", {})
    assert params.get("temperature_unit") == "celsius"


# --- location arg: happy path ---


async def test_location_arg_geocodes_then_fetches_weather():
    """Providing a city name geocodes it and reports weather for that city."""
    tool = WeatherTool(weather_config=_cfg(lat=39.7, lon=-84.2))
    client = _make_client([
        _geocoding_response(51.5074, -0.1278, "London"),
        _meteo_response(58.0, 2),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs(location="London"))
    assert result.success
    output = result.output or ""
    assert "58" in output
    assert "partly cloudy" in output
    assert "London" in output
    assert "sir" in output.lower()
    # Geocoding was called, then Open-Meteo — 2 API calls total.
    assert client.get.call_count == 2


async def test_location_arg_includes_city_in_spoken_output():
    """Output for a named location must include the city, e.g. 'in London, sir.'"""
    tool = WeatherTool(weather_config=_cfg(lat=39.7, lon=-84.2))
    client = _make_client([
        _geocoding_response(48.8566, 2.3522, "Paris"),
        _meteo_response(68.0, 1),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs(location="Paris"))
    assert "in Paris" in (result.output or "")


async def test_no_location_arg_does_not_include_city_in_output():
    """Without a location arg the output must NOT include 'in <city>'."""
    tool = WeatherTool(weather_config=_cfg(lat=39.7, lon=-84.2))
    client = _make_client([_meteo_response(72.0, 0)])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    output = result.output or ""
    assert " in " not in output
    assert output.endswith("sir.")


# --- location arg: geocoding failure / no results ---


async def test_geocoding_no_results_falls_back_to_default():
    """Empty geocoding results → spoken acknowledgment + default-location weather."""
    tool = WeatherTool(weather_config=_cfg(lat=39.7187, lon=-84.1736))
    client = _make_client([
        _geocoding_no_results(),
        _meteo_response(75.0, 0),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs(location="Atlantis"))
    assert result.success
    output = result.output or ""
    assert "couldn't find" in output.lower()
    assert "Atlantis" in output
    assert "default location" in output.lower()
    assert "75" in output


async def test_geocoding_timeout_falls_back_to_default():
    """Geocoding timeout is caught internally → same fallback path as no results."""
    tool = WeatherTool(weather_config=_cfg(lat=39.7187, lon=-84.1736))
    client = _make_client([
        Exception("connection timed out"),
        _meteo_response(60.0, 3),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs(location="Atlantis"))
    assert result.success
    output = result.output or ""
    assert "couldn't find" in output.lower()
    assert "60" in output


# --- WMO codes ---


def test_wmo_key_codes_are_mapped():
    for code in (0, 1, 2, 3, 45, 61, 63, 65, 71, 95):
        assert code in _WMO, f"WMO code {code} missing from mapping"


async def test_unknown_wmo_code_returns_unknown_conditions():
    tool = WeatherTool(weather_config=_cfg(lat=0.0, lon=0.0))
    client = _make_client([_meteo_response(80.0, 999)])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    assert result.success
    assert "unknown conditions" in (result.output or "")


# --- error / timeout paths ---


async def test_network_error_returns_spoken_fallback():
    tool = WeatherTool(weather_config=_cfg(lat=0.0, lon=0.0))
    client = MagicMock()
    client.get = AsyncMock(side_effect=Exception("connection refused"))
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    assert not result.success
    assert "couldn't reach" in (result.output or "").lower()
    assert "sir" in (result.output or "").lower()


async def test_ipapi_failure_returns_spoken_fallback():
    """If IP detection fails, the spoken fallback is returned."""
    tool = WeatherTool(weather_config=_cfg())
    client = MagicMock()
    client.get = AsyncMock(side_effect=Exception("timeout"))
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs())
    assert not result.success
    assert "couldn't reach" in (result.output or "").lower()


# --- geocoding retry (city+state format) ---


def test_geocode_queries_plain_city_returns_single_entry():
    assert _geocode_queries("London") == ["London"]


def test_geocode_queries_city_state_returns_two_entries():
    queries = _geocode_queries("Reston, Virginia")
    assert queries == ["Reston, Virginia", "Reston"]


def test_geocode_queries_city_state_strips_whitespace():
    queries = _geocode_queries("Dayton,  Ohio")
    assert queries[1] == "Dayton"


async def test_geocoding_city_state_retries_city_only_when_first_fails():
    """Full 'City, State' query returns no results → retries with city name only."""
    tool = WeatherTool(weather_config=_cfg(lat=39.7, lon=-84.2))
    client = _make_client([
        _geocoding_no_results(),          # "Reston, Virginia" → empty
        _geocoding_response(38.9586, -77.357, "Reston"),  # "Reston" → found
        _meteo_response(65.0, 1),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs(location="Reston, Virginia"))
    assert result.success
    output = result.output or ""
    assert "Reston" in output
    assert "65" in output
    assert client.get.call_count == 3  # geocode x2 + meteo x1


async def test_geocoding_city_state_falls_back_when_both_queries_fail():
    """Both geocoding queries return empty → falls back to default location."""
    tool = WeatherTool(weather_config=_cfg(lat=39.7187, lon=-84.1736))
    client = _make_client([
        _geocoding_no_results(),  # "Atlantis, GA" → empty
        _geocoding_no_results(),  # "Atlantis" → empty
        _meteo_response(75.0, 0),
    ])
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await tool.execute(WeatherArgs(location="Atlantis, GA"))
    assert result.success
    output = result.output or ""
    assert "couldn't find" in output.lower()
    assert "75" in output


# --- metadata ---


def test_tool_name_and_description():
    tool = WeatherTool(weather_config=_cfg())
    assert tool.name == "get_weather"
    assert "weather" in tool.description.lower()
    assert "location" in tool.description.lower()
    assert tool.requires_confirmation is False
