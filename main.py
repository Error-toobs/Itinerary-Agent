from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Pakistan Itinerary Planner API",
    description="AI-powered travel itinerary generator with multi-city planning",
    version="1.0.0"
)

# include all routes
app.include_router(router)


# optional root check (already in routes, but fine to keep here too)
@app.get("/")
def root():
    return {"message": "Welcome to the Itinerary Planner API 🚀"}
