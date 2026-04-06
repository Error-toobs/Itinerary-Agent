from pydantic import BaseModel
from typing import List, Dict, Optional


# 🔹 Each day's plan
class DayPlan(BaseModel):
    attractions: List[str]
    food: List[str]
    souvenir_shops: List[str]
    lodging: List[str]


# 🔹 Each city → multiple dates
# Example: "2026-02-20": DayPlan
CitySchedule = Dict[str, DayPlan]


# 🔹 Each item in list → one city with its schedule
# Example: { "Gilgit": { "2026-02-20": {...} } }
CityItinerary = Dict[str, CitySchedule]


# 🔹 Request model (user input in natural language)
class ItineraryRequest(BaseModel):
    user_input: str


# 🔹 Final response model
class ItineraryResponse(BaseModel):
    success: bool
    message: str

    # Main output (your structured itinerary)
    itinerary: Optional[List[CityItinerary]] = None

    # Optional warning (budget etc.)
    warning: Optional[str] = None