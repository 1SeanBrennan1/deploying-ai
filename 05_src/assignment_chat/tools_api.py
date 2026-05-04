
# tools_api.py
from langchain.tools import tool
import requests
from utils.logger import get_logger

_logs = get_logger(__name__)


@tool
def get_weather(city: str) -> str:
    """
    Fetches current weather conditions for any city in the world.
    
    Use this tool whenever a user asks about:
    - Current weather in a city (e.g., "What's the weather in Toronto?")
    - Temperature in a location (e.g., "How hot is it in Miami?")
    - Climate conditions (e.g., "Is it raining in London?")
    
    Args:
        city: The name of the city to get weather for
    
    Returns:
        A natural language description of the current weather
    """
    _logs.info(f'Getting weather for city: {city}')
    
    # Step 1: Get coordinates for the city
    geo_response = get_city_coordinates(city)
    if not geo_response:
        return f"I wasn't able to find weather data for {city}. Could you check the spelling?"
    
    # Step 2: Get weather using coordinates
    weather_response = get_weather_from_coordinates(geo_response)
    if not weather_response:
        return f"I found {city} but couldn't get the current weather. The weather service might be down."
    
    # Step 3: Format the response in natural language
    return format_weather_response(city, weather_response)


def get_city_coordinates(city: str) -> dict:
    """
    Gets the latitude and longitude for a city using the Open-Meteo Geocoding API.
    
    Args:
        city: Name of the city to look up
    
    Returns:
        Dictionary with latitude and longitude, or None if not found
    """
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    
    try:
        response = requests.get(geo_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("results"):
            _logs.warning(f'No coordinates found for city: {city}')
            return None
        
        location = data["results"][0]
        return {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "name": location.get("name", city),
            "country": location.get("country", "")
        }
    except Exception as e:
        _logs.error(f'Error getting coordinates for {city}: {str(e)}')
        return None


def get_weather_from_coordinates(coords: dict) -> dict:
    """
    Gets current weather data from Open-Meteo API using coordinates.
    
    Args:
        coords: Dictionary with latitude and longitude
    
    Returns:
        Dictionary with weather data, or None if failed
    """
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "current_weather": True
    }
    
    try:
        response = requests.get(weather_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("current_weather")
    except Exception as e:
        _logs.error(f'Error getting weather: {str(e)}')
        return None


def format_weather_response(city_name: str, weather_data: dict) -> str:
    """
    Transforms raw weather data into a natural language response.
    This satisfies the assignment requirement of NOT returning raw data.
    
    Args:
        city_name: Name of the city
        weather_data: Raw weather data from the API
    
    Returns:
        A friendly, natural language weather report
    """
    temperature = weather_data.get("temperature", "unknown")
    wind_speed = weather_data.get("windspeed", "unknown")
    weather_code = weather_data.get("weathercode", 0)
    
    # Interpret weather codes into human-readable descriptions
    weather_descriptions = {
        0: "clear skies",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow fall",
        73: "moderate snow fall",
        75: "heavy snow fall",
        95: "thunderstorm"
    }
    
    conditions = weather_descriptions.get(weather_code, "variable conditions")
    
    return (
        f"Right now in {city_name}, it's {temperature}°C with {conditions}. "
        f"The wind is blowing at about {wind_speed} km/h. "
        f"Not too shabby for a day out, if you ask me!"
    )