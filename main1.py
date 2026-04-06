
from json.tool import main
import json
from app.services.Integrate import ItineraryGenerator
from app.orchestration.orchestrator import MultiCityOrchestrator
from app.validation.validator import TravelPlanner


def main():
    print("Hi there! Welcome to your Pakistan travel planning assistant. I'm here to help you plan an amazing trip between cities in Pakistan. To get started, could you tell me about your travel plans?")
    planner = TravelPlanner()
    trip_data = None

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
            result_json = planner.get_final_json()
            print(result_json)
            # convert the JSON string back to a dict before processing
            trip_data = json.loads(result_json)
            break

    o = MultiCityOrchestrator()
    o.user_input_structured = trip_data  # Store structured input for later use in budget comparison
    # ensure we pass a dict, not the raw string
    iti = o.run_trip(trip_data)
    print("\nGenerated Itineraries for Each Segment:")
    for i, segment_iti in enumerate(iti):
        print(f"\nSegment {i+1} Itinerary:\n{segment_iti}")
    print(f"\nTotal Estimated Budget for Optional Places: {o.budget}")

    if o.budget > trip_data.get("budget", {}).get("amount", float("inf")):
        print("\nWarning: The estimated budget for optional places exceeds your specified budget!")

if __name__ == "__main__":
    main()