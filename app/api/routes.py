from fastapi import APIRouter
from app.models.schema import ItineraryRequest, ItineraryResponse
from app.services.Integrate import ItineraryGenerator
from app.orchestration.orchestrator import MultiCityOrchestrator
from app.validation.validator import TravelPlanner
import json

router = APIRouter()


@router.get("/")
def health_check():
    return {"message": "Itinerary API is running 🚀"}


@router.post("/generate-itinerary", response_model=ItineraryResponse)
def generate_itinerary(request: ItineraryRequest):
    try:
        planner = TravelPlanner()

        # Step 1: Process user input (single-shot instead of loop)
        reply, complete = planner.process_input(request.user_input)

        errors = planner.validate()

        # If validation fails → return message
        if errors:
            return ItineraryResponse(
                success=False,
                message="Validation errors: " + ", ".join(errors),
                itinerary=None,
                warning=None
            )

        # Step 2: Get structured JSON from planner
        result_json = planner.get_final_json()
        trip_data = json.loads(result_json)

        # Step 3: Run orchestrator
        orchestrator = MultiCityOrchestrator()
        orchestrator.user_input_structured = trip_data

        itineraries = orchestrator.run_trip(trip_data)

        # Step 4: Budget warning check
        warning_msg = None
        user_budget = trip_data.get("budget", {}).get("amount", float("inf"))

        if orchestrator.budget > user_budget:
            warning_msg = "The estimated budget for optional places exceeds your specified budget!"

        # Step 5: Return structured response
        return ItineraryResponse(
            success=True,
            message="Itinerary generated successfully",
            itinerary=itineraries,
            warning=warning_msg
        )

    except Exception as e:
        return ItineraryResponse(
            success=False,
            message=str(e),
            itinerary=None,
            warning=None
        )