import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

weather = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["timezone"],
                },
            }
}
time = {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current year, month, date and time in a given IANA timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Time zone expressed in IANA time zone identifier. e.g. Asia/Tokyo",
                        },
                    },
                    "required": ["location"],
                },
            }
}

# Simplified weather data
WEATHER_DATA = {
    "tokyo": {"temperature": "10", "unit": "celsius"},
    "san francisco": {"temperature": "72", "unit": "fahrenheit"},
    "paris": {"temperature": "22", "unit": "celsius"}
}

# Simplified timezone data
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}

def get_current_weather(location):
    """Get the current weather for a given location"""
    print(f"get_current_weather called with location: {location}")
    location_lower = location.lower()

    for key in WEATHER_DATA:
        if key in location_lower:
            print(f"Weather data found for {key}")
            weather = WEATHER_DATA[key]
            return json.dumps({
                "location": location_lower,
                "temperature": weather["temperature"],
                "unit": weather["unit"]
            })

    print(f"No weather data found for {location_lower}")
    return json.dumps({"location": location_lower, "temperature": "unknown"})

def get_current_datetime(timezone):
    """Get the current time for a given timezone"""
    print(f"get_current_time called with timezone: {timezone}")

    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as e:
        print("ZoneInfoNotFoundError")
        return json.dumps({
                "timezone": timezone,
                "current_time": "Please specify valid timezone as IANA tzdata"
        })

    current_time = datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S")
    print(f"current time: {current_time}")
    return json.dumps({
        "timezone": timezone,
        "current_time": current_time
    })
