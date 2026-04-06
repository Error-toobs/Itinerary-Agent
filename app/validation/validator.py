import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pyexpat.errors import messages
from typing import List, Optional, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import prompt


# ==========================================================
# CONFIG
# ==========================================================

load_dotenv() 
api_key = os.environ.get("API_KEY")

llm = ChatOpenAI(
    model="openai/gpt-5.2",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

MIN_BUDGET = 100000
MAX_BUDGET = 2000000
ALLOWED_TRANSPORT = ["car", "plane", "bus"]



# ===============================
# City Segment
# ===============================

@dataclass
class CitySegment:
    city: str
    number_of_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    transport_from_previous: Optional[str] = None
    preferences: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "city": self.city,
            "number_of_days": self.number_of_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "transport_from_previous": self.transport_from_previous,
            "preferences": self.preferences,
        }


# ===============================
# Global Travel State
# ===============================

@dataclass
class TravelState:
    starting_city: Optional[str] = None
    adults: Optional[int] = None
    kids: Optional[int] = None
    food : bool = False
    souvenir_shopping: bool = False

    budget: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"amount": None, "currency": None}
    )

    total_start_date: Optional[str] = None
    total_end_date: Optional[str] = None

    segments: List[CitySegment] = field(default_factory=list)

    def to_dict(self):
        return {
            "starting_city": self.starting_city,
            "adults": self.adults,
            "kids": self.kids,
            "food": self.food, 
            "souvenir_shopping": self.souvenir_shopping, 
            "budget": self.budget,
            "total_start_date": self.total_start_date,
            "total_end_date": self.total_end_date,
            "segments": [segment.to_dict() for segment in self.segments],
        }

# ==========================================================
# CORE ENGINE
# ==========================================================

class TravelPlanner:

    def __init__(self):
        self.state = TravelState()
        self.messages = []

    # -----------------------------
    # Merge New Info
    # -----------------------------
    def merge_state(self, new_data: dict):

        # -------- Global fields --------
        global_fields = [
            "starting_city",
            "adults",
            "kids",
            "food",
            "souvenir_shopping",
            "total_start_date",
            "total_end_date",
        ]

        for field in global_fields:
            if field in new_data and new_data[field] not in (None, ""):
                setattr(self.state, field, new_data[field])

        # -------- Budget --------
        if "budget" in new_data:
            for k, v in new_data["budget"].items():
                if v not in (None, ""):
                    self.state.budget[k] = v

        # -------- Segments --------
        if "segments" in new_data:
            for seg_data in new_data["segments"]:

                city_name = seg_data.get("city")
                if not city_name:
                    continue

                # Check if segment exists
                existing_segment = next(
                    (s for s in self.state.segments if s.city.lower() == city_name.lower()),
                    None,
                )

                if not existing_segment:
                    existing_segment = CitySegment(city=city_name)
                    self.state.segments.append(existing_segment)

                # Update segment fields
                for field in [
                    "number_of_days",
                    "start_date",
                    "end_date",
                    "transport_from_previous",
                    "preferences",
                ]:
                    if field in seg_data and seg_data[field] not in (None, ""):
                        setattr(existing_segment, field, seg_data[field])
        
    # -----------------------------
    #Check segment
    # -----------------------------
    def add_segment_if_not_exists(self, city_name: str):
        existing = [s.city.lower() for s in self.state.segments]
        if city_name.lower() not in existing:
            self.state.segments.append(CitySegment(city=city_name))
    
    # -----------------------------
    # Smart Auto-Compute Segment Dates
    # -----------------------------

    def auto_compute_segment_dates(self):
        errors = []
        s = self.state

        if not s.total_start_date:
            return errors

        if not all(seg.number_of_days for seg in s.segments):
            return errors

        try:
            trip_start = datetime.strptime(s.total_start_date, "%Y-%m-%d")
        except ValueError:
            errors.append("Trip start date format is invalid.")
            return errors

        current_date = trip_start

        for seg in s.segments:
            seg.start_date = current_date.strftime("%Y-%m-%d")
            seg.end_date = (
                current_date + timedelta(days=seg.number_of_days - 1)
            ).strftime("%Y-%m-%d")

            current_date = current_date + timedelta(days=seg.number_of_days)

        return errors

    # -----------------------------
    # Validate Gloabl fields
    # -----------------------------
    def validate_global_fields(self):
        errors = []
        s = self.state

        if not s.starting_city:
            errors.append("Starting city is required.")

        if s.adults is None:
            errors.append("Number of adults is required.")
        elif s.adults <= 0:
            errors.append("At least one adult is required.")

        if s.kids is None:
            errors.append("Number of kids is required.")
        elif s.kids < 0:
            errors.append("Number of kids cannot be negative.")

        if not s.budget["amount"]:
            errors.append("Budget amount is required.")
        elif not (MIN_BUDGET <= s.budget["amount"] <= MAX_BUDGET):
            errors.append(f"Budget must be between {MIN_BUDGET} and {MAX_BUDGET}.")

        if not s.total_start_date:
            errors.append("Trip start date is required.")

        if not s.total_end_date:
            errors.append("Trip end date is required.")

        return errors
    
    # -----------------------------
    # Validate segment structure
    # -----------------------------
    def validate_segments_structure(self):
        errors = []

        if not self.state.segments:
            errors.append("At least one destination city is required.")
            return errors

        for idx, seg in enumerate(self.state.segments):

            if not seg.city:
                errors.append(f"Segment {idx+1}: City name missing.")

            if seg.number_of_days is not None and seg.number_of_days <= 0:
                errors.append(f"{seg.city}: Number of days must be positive.")

            if seg.transport_from_previous:
                if seg.transport_from_previous.lower() not in ALLOWED_TRANSPORT:
                    errors.append(
                        f"{seg.city}: Transport must be one of {', '.join(ALLOWED_TRANSPORT)}."
                    )

        return errors

    # -----------------------------
    # Validate Total Day Consistency 
    # -------------------------------
    def validate_total_day_consistency(self):
        errors = []
        s = self.state

        if not s.total_start_date or not s.total_end_date:
            return errors

        if not s.segments:
            return errors

        try:
            trip_start = datetime.strptime(s.total_start_date, "%Y-%m-%d")
            trip_end = datetime.strptime(s.total_end_date, "%Y-%m-%d")
        except ValueError:
            errors.append("Trip date format is invalid.")
            return errors
        
        if trip_end < trip_start:
            errors.append("Trip end date cannot be before start date.")
            return errors
        
        total_trip_days = (trip_end - trip_start).days + 1

        allocated_days = sum(
            seg.number_of_days for seg in s.segments if seg.number_of_days
        )

        if allocated_days:
            if allocated_days > total_trip_days:
                errors.append(
                    f"Total allocated city days ({allocated_days}) "
                    f"exceed trip duration ({total_trip_days})."
                )
            elif total_trip_days - allocated_days > 1:
                errors.append(
                    f"There are {total_trip_days - allocated_days} unplanned days in the trip."
                )

        return errors
    
    # -----------------------------
    # Business Rules Validation
    # -----------------------------
    def validate_business_rules(self):
        errors = []

        # Prevent 365+ days per segment
        for seg in self.state.segments:
            if seg.number_of_days and seg.number_of_days > 365:
                errors.append(f"{seg.city}: Stay cannot exceed 365 days.")

        return errors
    
    # -----------------------------
    # Final validate() Method
    # -----------------------------
    def validate(self):

        errors = []

        errors.extend(self.validate_global_fields())
        errors.extend(self.validate_segments_structure())

        # Compute before checking consistency
        errors.extend(self.auto_compute_segment_dates())

        errors.extend(self.validate_total_day_consistency())
        errors.extend(self.validate_business_rules())

        return errors

    def process_input(self, user_input: str):

        # 1️⃣ Convert current state to JSON
        current_state_json = json.dumps(self.state.to_dict(), indent=2)

        # 2️⃣ Normal triple string (NOT f-string)
        system_prompt = prompt
      
        # 3️⃣ Replace placeholder with actual JSON
        system_prompt = system_prompt.replace("__STATE_JSON__", current_state_json)

        messages = [SystemMessage(content=system_prompt)]

        # Conversation history
        # for m in self.messages:
        #     if m["role"] == "user":
        #         messages.append(HumanMessage(content=m["content"]))
        #     else:
        #         messages.append(SystemMessage(content=m["content"]))

        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)

        raw_content = response.content.strip()

        if not raw_content:
            return "Temporary system issue. Please try again.", False

        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError:
            # One automatic retry
            retry_response = llm.invoke(messages)
            raw_retry = retry_response.content.strip()

            try:
                result = json.loads(raw_retry)
            except json.JSONDecodeError:
                return "I’m having trouble formatting the response. Please try again.", False

        # self.messages.append({"role": "user", "content": user_input})
        # self.messages.append(
        #     {"role": "assistant", "content": result["assistant_message"]}
        # )

        self.merge_state(result["updated_travel_info"])

        return result["assistant_message"], result["trip_complete"]
    
    def get_final_json(self):
        return json.dumps(self.state.to_dict(), indent=2)


# ==========================================================
# SIMPLE CLI LOOP (Optional)
# ==========================================================

if __name__ == "__main__":

    print("Hi there! Welcome to your Pakistan travel planning assistant. I'm here to help you plan an amazing trip between cities in Pakistan. To get started, could you tell me about your travel plans?")
    planner = TravelPlanner()

    while True:
        user_input = input("\nYou: ")

        reply, complete = planner.process_input(user_input)

        errors = planner.validate()

        print("\nAssistant:", reply)

        if errors:
            print("\nIssues Found in plan, please clarify: ")
            for e in errors:
                print("-", e)

        if complete and not errors:
            print("\nFinal Structured JSON:")
            print(planner.get_final_json())
            break