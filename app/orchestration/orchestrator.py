import json

# from transformers import Any
from app.services.Integrate import ItineraryGenerator
from app.validation.validator import TravelPlanner

class MultiCityOrchestrator:

    
    def __init__(self):
        self.segment_engine = ItineraryGenerator()
        self.segment_graph = self.segment_engine.create_graph()
        self.budget = 0.00
        self.user_input_structured = None
        self.itineraries = []

    def run_trip(self, trip_json):
        all_segment_results = []
        segments = trip_json["segments"]
        
        # Get global food and souvenir preferences (apply to ALL segments)
        global_food = trip_json.get("food", False)
        global_souvenirs = trip_json.get("souvenir_shopping", False)

        for i, segment in enumerate(segments):

            arrival_context = None

            # if i > 0:
            #     transport = segment.get("transport_from_previous")
            #     arrival_context = self.compute_arrival(transport)
            print(type(segment.get("start_date")))

            # Construct state with ALL required fields for ItineraryState
            state = {
                "user_query": f"Plan a {segment['number_of_days']}-day trip to {segment['city']} with preferences: {', '.join(segment.get('preferences', []))}",
                "parsed_location": segment["city"],
                "parsed_days": segment["number_of_days"],
                "parsed_preferences": segment.get("preferences", []),
                "query_parse_error": None,
                "start_date": segment['start_date'],  # Pass start date for the segment
                "end_date": segment['end_date'],      # Pass end date for the segment
                "include_food": global_food,  # ✅ Use GLOBAL flag
                "include_souvenirs": global_souvenirs,  # ✅ Use GLOBAL flag
                "retrieved_attractions": [],
                "retrieval_metadata": {}, 
                "place_coordinates": {},
                "draft_itinerary": None,
                "retry_count": 0,
                "clusters": {},
                "clustered_optional_places": {}
            }

            result = self.segment_graph.invoke(state)
            self.budget += (result.get("budget_needed") or 0)  # Accumulate budget from each segment
            all_segment_results.append({segment["city"]: result["draft_itinerary"]})

        self.itineraries = all_segment_results
        return all_segment_results


# print("Hi there! Welcome to your Pakistan travel planning assistant. I'm here to help you plan an amazing trip between cities in Pakistan. To get started, could you tell me about your travel plans?")
# planner = TravelPlanner()
# trip_data = None

# while True:
#     user_input = input("\nYou: ")

#     reply, complete = planner.process_input(user_input)

#     errors = planner.validate()

#     print("\nAssistant:", reply)

#     if errors:
#         print("\nIssues Found in plan, please clarify: ")
#         for e in errors:
#             print("-", e)

#     if complete and not errors:
#         print("\nFinal Structured JSON:")
#         result_json = planner.get_final_json()
#         print(result_json)
#         # convert the JSON string back to a dict before processing
#         trip_data = json.loads(result_json)
#         break

# o = MultiCityOrchestrator()
# o.user_input_structured = trip_data  # Store structured input for later use in budget comparison
# # ensure we pass a dict, not the raw string
# iti = o.run_trip(trip_data)
# print("\nGenerated Itineraries for Each Segment:")
# for i, segment_iti in enumerate(iti):
#     print(f"\nSegment {i+1} Itinerary:\n{segment_iti}")
# print(f"\nTotal Estimated Budget for Optional Places: {o.budget}")

# if o.budget > trip_data.get("budget", {}).get("amount", float("inf")):
#     print("\nWarning: The estimated budget for optional places exceeds your specified budget!")