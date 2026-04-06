from __future__ import annotations
import difflib
import json
import os
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional
from networkx import config
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict
from sklearn.cluster import DBSCAN
from app.utils.helper import ensure_date, next_date, parse_date_from_text
from config.settings import llm, DF_ATTRACTIONS, DF_FOOD, DF_SOUVENIRS, DF_LODGING

# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def cluster_attractions(place_coordinates: Dict[str, Dict[str, float]],
                        max_distance_km: float = 40.0):
    """
    Clusters attractions using DBSCAN with haversine metric.
    Returns:
        Dict[int, List[str]] → {cluster_id: [place_names]}
    """
    if not place_coordinates:
        return {}

    names = list(place_coordinates.keys())
    coords = np.array([
        [radians(v["lat"]), radians(v["lng"])]
        for v in place_coordinates.values()
    ])

    # Convert km to radians (earth radius 6371 km)
    epsilon = max_distance_km / 6371.0

    db = DBSCAN(
        eps=epsilon,
        min_samples=1,
        metric="haversine"
    ).fit(coords)

    clusters = {}
    for label, name in zip(db.labels_, names):
        clusters.setdefault(label, []).append(name)

    return clusters

def compute_centroid(
    place_names: List[str],
    place_coordinates: Dict[str, Dict[str, float]]
) -> Dict[str, float]:

    lat_sum = 0
    lng_sum = 0
    count = 0

    for name in place_names:
        coords = place_coordinates.get(name)
        if coords:
            lat_sum += coords["lat"]
            lng_sum += coords["lng"]
            count += 1

    if count == 0:
        return {"lat": 0.0, "lng": 0.0}

    return {
        "lat": lat_sum / count,
        "lng": lng_sum / count
    }

def auto_balance_clusters(
    clusters: Dict[int, List[str]],
    place_coordinates: Dict[str, Dict[str, float]],
    target_days: int
) -> Dict[int, List[str]]:
    """
    Balances clustered attractions so that attractions are
    more evenly distributed across the requested number of days.

    Strategy:
    - If clusters > target_days → merge closest clusters
    - If clusters < target_days → keep as is (LLM can handle empty days)
    - If cluster sizes are highly uneven → move nearest attractions
      from largest cluster to smallest cluster
    """

    if not clusters:
        return clusters

    # Convert to mutable structure
    clusters = {k: list(v) for k, v in clusters.items()}

    # ------------------------------------------------------------
    # STEP 1: Merge clusters if more clusters than days
    # ------------------------------------------------------------
    while len(clusters) > target_days:

        cluster_ids = list(clusters.keys())

        # Find two closest clusters by centroid distance
        min_dist = float("inf")
        pair_to_merge = None

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):

                c1 = cluster_ids[i]
                c2 = cluster_ids[j]

                centroid1 = compute_centroid(clusters[c1], place_coordinates)
                centroid2 = compute_centroid(clusters[c2], place_coordinates)

                dist = haversine_km(
                    centroid1["lat"], centroid1["lng"],
                    centroid2["lat"], centroid2["lng"]
                )

                if dist < min_dist:
                    min_dist = dist
                    pair_to_merge = (c1, c2)

        # Merge closest pair
        c1, c2 = pair_to_merge
        clusters[c1].extend(clusters[c2])
        del clusters[c2]

    # ------------------------------------------------------------
    # STEP 2: Evenly redistribute if highly uneven
    # ------------------------------------------------------------

    changed = True
    while changed:
        changed = False

        # Sort clusters by size
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: len(x[1])
        )

        smallest_id, smallest_list = sorted_clusters[0]
        largest_id, largest_list = sorted_clusters[-1]

        if len(largest_list) - len(smallest_list) <= 1:
            break  # Balanced enough

        # Move closest attraction from largest to smallest
        smallest_centroid = compute_centroid(
            smallest_list, place_coordinates
        )

        best_candidate = None
        min_dist = float("inf")

        for attraction in largest_list:

            coords = place_coordinates.get(attraction)
            if not coords:
                continue

            dist = haversine_km(
                coords["lat"], coords["lng"],
                smallest_centroid["lat"], smallest_centroid["lng"]
            )

            if dist < min_dist:
                min_dist = dist
                best_candidate = attraction

        if best_candidate:
            clusters[largest_id].remove(best_candidate)
            clusters[smallest_id].append(best_candidate)
            changed = True

    return clusters

def nearest_from_pool(
    anchor_coords: List[Dict[str, float]],
    pool_df: pd.DataFrame,
    top_n: int = 5,
    max_radius_km: float = 80.0,
) -> List[Dict[str, Any]]:
    """
    Given a list of anchor coordinates (the trip's attractions), find up to
    `top_n` rows from `pool_df` that are nearest to ANY anchor and within
    `max_radius_km`.  Results are sorted by ascending min-distance.

    Each returned dict has an extra "_nearest_km" field.
    """
    if pool_df.empty or not anchor_coords:
        return []

    scored: List[tuple[float, Dict]] = []
    for row in pool_df.to_dict("records"):
        try:
            rlat = float(row["latitude"])
            rlng = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        min_dist = min(
            haversine_km(a["lat"], a["lng"], rlat, rlng) for a in anchor_coords
        )
        if min_dist <= max_radius_km:
            entry = dict(row)
            entry["_nearest_km"] = round(min_dist, 2)
            scored.append((min_dist, entry))

    scored.sort(key=lambda x: x[0])
    return [item for _, item in scored[:top_n]]


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ItineraryState(TypedDict):
    # Input
    user_query: str

    # Query parser output
    parsed_days:        Optional[int]
    parsed_location:    Optional[str]
    parsed_preferences: Optional[List[str]]
    query_parse_error:  Optional[str]
    start_date:       Optional[str]  # ISO format date string, e.g. "2024-12-20"
    end_date:         Optional[str]  # ISO format date string, e.g. "2024-12-27"

    # Feature flags — derived from query intent
    include_food:      bool   # True only when query explicitly asks for food spots
    include_souvenirs: bool   # True only when query explicitly asks for souvenir shops

    # Semantic search output
    retrieved_attractions: List[Dict[str, Any]]
    retrieval_metadata:    Dict[str, Any]

    # Enrichment output: { place_name: {"lat": float, "lng": float} }
    place_coordinates: Dict[str, Dict[str, float]]

    # # Optional-place selection output
    # nearby_food:      List[Dict[str, Any]]   # populated only if include_food
    # nearby_souvenirs: List[Dict[str, Any]]   # populated only if include_souvenirs
    # nearby_lodging:   List[Dict[str, Any]]   # always populated when data exists

    # Generator output
    draft_itinerary: Optional[str]

    # Control
    retry_count: int

    clusters: Dict[int, List[str]]
    clustered_optional_places: Dict[int, Dict[str, List[Dict[str, Any]]]]
    budget_needed: float

def find_close_district(district_name):
    target = str(district_name).strip().lower()
    for df in [DF_ATTRACTIONS]:
        vals = df["district"].astype(str).str.strip().str.lower().tolist()
        matches = difflib.get_close_matches(target, vals, n=5, cutoff=0.6)
        if matches:
            # print(f"Did you mean '{matches[0]}'?")
            return matches[0]
    return 0

# ---------------------------------------------------------------------------
# Tool factory — bakes attraction coordinates into the closure
# ---------------------------------------------------------------------------

def make_nearby_tool(place_coordinates: Dict[str, Dict[str, float]]):
    """
    Returns a LangChain tool that lets the LLM query how close any two
    attractions are.  Used exclusively for clustering attractions by day.
    Food / souvenir / lodging proximity is handled deterministically in
    select_optional_places — NOT by this tool.
    """

    @tool
    def find_nearby_places(anchor_place: str, radius_km: float = 60.0) -> str:
        """
        Return a JSON list of candidate attractions within `radius_km` km of
        `anchor_place`, sorted by ascending distance.

        Call this on several attractions to discover geographic clusters, then
        assign clustered attractions to the same day to minimise travel.

        Args:
            anchor_place: Exact _key of the anchor attraction.
            radius_km:    Search radius in kilometres (default 60).
        """
        anchor = place_coordinates.get(anchor_place)
        if anchor is None:
            return json.dumps(
                {
                    "error": f"'{anchor_place}' not found.",
                    "available_places": list(place_coordinates.keys()),
                }
            )

        results = []
        for name, coords in place_coordinates.items():
            if name == anchor_place:
                continue
            dist = haversine_km(
                anchor["lat"], anchor["lng"], coords["lat"], coords["lng"]
            )
            if dist <= radius_km:
                results.append({"place": name, "distance_km": round(dist, 2)})

        results.sort(key=lambda x: x["distance_km"])
        return json.dumps(
            results if results else {"message": "No places found within radius."}
        )

    return find_nearby_places


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ItineraryGenerator:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_and_embed_attractions()

    def _load_and_embed_attractions(self) -> None:
        self.df = DF_ATTRACTIONS.copy()
        self.df["search_text"] = self.df.apply(
            lambda r: f"{r['_key']} {r['category']} {r['Desc']} {r['district']}",
            axis=1,
        )
        # print("Embedding attraction data …")
        self.embeddings = self.embedding_model.encode(
            self.df["search_text"].tolist(), show_progress_bar=True
        )

    # ── Graph ──────────────────────────────────────────────────────────────

    def create_graph(self):
        wf = StateGraph(ItineraryState)

        wf.add_node("semantic_searcher",      self.semantic_search)
        wf.add_node("data_enricher",          self.enrich_data)
        wf.add_node("select_optional_places", self.select_optional_places)
        wf.add_node("itinerary_generator",    self.generate_itinerary)

        wf.set_entry_point("semantic_searcher")
        wf.add_edge("semantic_searcher",      "data_enricher")
        wf.add_edge("data_enricher",          "select_optional_places")
        wf.add_edge("select_optional_places", "itinerary_generator")
        wf.add_edge("itinerary_generator",    END)

        return wf.compile()

    # -----------------------------------------------------------------------
    # Node 1 — Query parser
    #   Extracts trip parameters AND detects optional-feature flags.
    # -----------------------------------------------------------------------

    def parse_query(self, state: ItineraryState) -> ItineraryState:
        prompt = f"""Analyse this travel query and return ONLY valid JSON — no markdown fences.

Query: "{state['user_query']}"

Return exactly this JSON structure:
{{
  "days": <integer>,
  "location": "<city or region>",
  "preferences": ["<keyword>", ...],
  "include_food": <true if the traveller explicitly asks for food spots/restaurants/eateries, else false>,
  "include_souvenirs": <true if the traveller explicitly asks for souvenir/handicraft/gift shops, else false>
}}
"""
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content
            raw = raw.strip().strip("```json").strip("```").strip()
            p = json.loads(raw)
            return {
                **state,
                "parsed_days":        int(p.get("days", 3)),
                "parsed_location":    str(p.get("location", "")),
                "parsed_preferences": p.get("preferences", []),
                "include_food":       bool(p.get("include_food", False)),
                "include_souvenirs":  bool(p.get("include_souvenirs", False)),
                "query_parse_error":  None,
            }
        except Exception as exc:
            print(f"[parse_query] Fallback — {exc}")
            return {
                **state,
                "parsed_days":        3,
                "parsed_location":    state["user_query"],
                "parsed_preferences": ["sightseeing"],
                "include_food":       False,
                "include_souvenirs":  False,
                "query_parse_error":  str(exc),
            }

    # -----------------------------------------------------------------------
    # Node 2 — Semantic search
    #   Searches only the attractions sheet.
    # ---------------------------------------------------------------------

    def semantic_search(self, state: ItineraryState) -> ItineraryState:
        location = state.get("parsed_location")
        if location:
            location = find_close_district(location)
        if location:            state["parsed_location"] = location
        prefs     = state.get("parsed_preferences") or []
        
        query_txt = f"{state['parsed_location']} {' '.join(prefs)}"
        q_emb     = self.embedding_model.encode([query_txt])
        sims      = np.dot(self.embeddings, q_emb.T).flatten()
        # top_idx   = np.argsort(sims)[- state.get("parsed_days", 3) * 4:][::-1]

        # attractions = [
        #     {**self.df.iloc[i].to_dict(), "similarity_score": float(sims[i])}
        #     for i in top_idx
        # ]

        top_idx = np.argsort(sims)[::-1]

        location = (state.get("parsed_location") or "").strip().lower()

        filtered_attractions = []

        for i in top_idx:
            row = self.df.iloc[i]

            if location and row["district"] != location:
                continue

            filtered_attractions.append(
                {**row.to_dict(), "similarity_score": float(sims[i])}
            )

            if len(filtered_attractions) >= state.get("parsed_days", 3) * 4:
                break

        return {
            **state,
            "retrieved_attractions": filtered_attractions,
            "retrieval_metadata": {
                "total_results": len(filtered_attractions),
                "avg_score": float(np.mean([a["similarity_score"] for a in filtered_attractions]))
                if filtered_attractions else 0.0,
            },
        }


    # -----------------------------------------------------------------------
    # Node 3 — Data enricher
    #   Single responsibility: extract lat/lng from retrieved attractions
    #   into `place_coordinates`.  Does NOT touch optional data.
    # -----------------------------------------------------------------------

    def enrich_data(self, state: ItineraryState) -> ItineraryState:
        place_coordinates: Dict[str, Dict[str, float]] = {}
        enriched = []

        for attraction in state["retrieved_attractions"]:
            name = str(attraction.get("_key", "unknown"))
            try:
                place_coordinates[name] = {
                    "lat": float(attraction["latitude"]),
                    "lng": float(attraction["longitude"]),
                }
            except (KeyError, TypeError, ValueError):
                pass

            attraction.setdefault("estimated_duration_hrs", 2)
            enriched.append(attraction)

        # print(
        #     f"[enrich_data] Coordinates stored for "
        #     f"{len(place_coordinates)} / {len(enriched)} attractions."
        # )

        return {
            **state,
            "retrieved_attractions": enriched,
            "place_coordinates":     place_coordinates,
        }

    # -----------------------------------------------------------------------
    # Node 4 — Select optional places
    #   Single responsibility: proximity-filter food, souvenirs, and lodging
    #   against the retrieved attraction coordinates.
    #
    #   Rules:
    #     • Food      → only when include_food is True
    #     • Souvenirs → only when include_souvenirs is True
    #     • Lodging   → always (travellers always need accommodation)
    # -----------------------------------------------------------------------

    def select_optional_places(self, state: ItineraryState) -> ItineraryState:

        # clusters = cluster_attractions(state["place_coordinates"], max_distance_km=40.0)
        clusters = cluster_attractions(
            state["place_coordinates"],
            max_distance_km=10.0
        )

        clusters = auto_balance_clusters(
            clusters,
            state["place_coordinates"],
            target_days=state["parsed_days"]
        )

        clustered_optional = {}
        budget_needed = 0  # Initialize budget accumulator

        for cluster_id, place_names in clusters.items():

            # Get anchor coordinates for this cluster only
            anchors = [
                state["place_coordinates"][name]
                for name in place_names
                if name in state["place_coordinates"]
            ]

            cluster_data = {}

            # Food (only if requested)
            if state.get("include_food"):
                cluster_data["food"] = nearest_from_pool(
                    anchors, DF_FOOD, top_n=3, max_radius_km=40.0
                )
            else:
                cluster_data["food"] = []

            # Souvenirs
            if state.get("include_souvenirs"):
                cluster_data["souvenirs"] = nearest_from_pool(
                    anchors, DF_SOUVENIRS, top_n=2, max_radius_km=40.0
                )
            else:
                cluster_data["souvenirs"] = []

            # Lodging (always)
            #picking 5 so that do budget optimization on it. 
            cluster_data["lodging_candidates"] = nearest_from_pool(
            anchors, DF_LODGING, top_n=5, max_radius_km=60.0)

            lodging_candidates = cluster_data.get("lodging_candidates", [])

            if lodging_candidates:

                # Sort by price (ascending)
                lodging_candidates.sort(
                    key=lambda x: float(x.get("price", float("inf")))
                )
                
                # print(f" Cluster {cluster_id} lodging candidates (sorted by price):")
                # for lodging in lodging_candidates:
                #     print(f"  - {lodging['_key']} (Price: {lodging.get('price', 'N/A')}, Distance: {lodging.get('_nearest_km', 'N/A')} km)")

                # Select cheapest
                cluster_data["lodging"] = lodging_candidates
                budget_needed+= float(lodging_candidates[0].get("price", float("inf")))
            else:
                cluster_data["lodging"] = []

            # # Lodging (before budget optimization)
            # cluster_data["lodging"] = nearest_from_pool(
            #     anchors, DF_LODGING, top_n=3, max_radius_km=60.0
            # )

            clustered_optional[cluster_id] = cluster_data

        
        all_food = []
        all_souvenirs = []
        all_lodging = []

        for cdata in clustered_optional.values():
            all_food.extend(cdata.get("food", []))
            all_souvenirs.extend(cdata.get("souvenirs", []))
            all_lodging.extend(cdata.get("lodging", []))

        return {
            **state,
            "clusters": clusters,
            "clustered_optional_places": clustered_optional,
            "nearby_food": all_food,
            "nearby_souvenirs": all_souvenirs,
            "nearby_lodging": all_lodging,
            "budget_needed": budget_needed,
        }

    # -----------------------------------------------------------------------
    # Node 5 — Itinerary generator
    #   Agentic LLM that:
    #     1. Uses find_nearby_places (attractions only) to cluster by day
    #     2. Slots food spots as brief en-route stops when include_food
    #     3. Slots souvenir shops as short detours when include_souvenirs
    #     4. Assigns the nearest lodging per day's cluster
    # -----------------------------------------------------------------------

    def generate_itinerary(self, state: ItineraryState) -> ItineraryState:
        """
        Zero-hallucination itinerary generator.
        LLM decides only ordering of clusters.
        All place names inserted deterministically.
        """

        clusters = state.get("clusters", {})
        clustered_optional = state.get("clustered_optional_places", {})

        if not clusters:
            return {**state, "draft_itinerary": "{}"}

        # -------------------------------------------------------
        # STEP 1: Ask LLM only to order clusters
        # -------------------------------------------------------

        cluster_ids = sorted(clusters.keys())

        system_prompt = """
    You are a travel planner.

    Your task:
    Given numbered clusters, return ONLY a JSON array
    representing the order of clusters for the itinerary.

    Example:
    [0, 1, 2]

    Rules:
    - Use each cluster exactly once.
    - Do not add extra numbers.
    - Do not skip any cluster.
    - Return raw JSON only.
    """

        user_prompt = f"""
    We have {len(cluster_ids)} clusters for a {state['parsed_days']}-day trip.

    Cluster IDs:
    {cluster_ids}

    Return the best visiting order.
    """

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        try:
            cluster_order = json.loads(response.content)
        except Exception:
            cluster_order = cluster_ids  # fallback safe ordering

        # -------------------------------------------------------
        # STEP 2: Build itinerary deterministically
        # -------------------------------------------------------

        final_output = {}

        exact_date = state.get("start_date")
        if not exact_date:
            raise ValueError(
                "Missing start_date in state; please provide a start_date (e.g. '2025-02-24' or '24 Feb 2025')."
            )

        try:
            exact_date = ensure_date(exact_date)
        except TypeError:
            exact_date = parse_date_from_text(exact_date)

        for day_index, cluster_id in enumerate(cluster_order):

            attractions = clusters.get(cluster_id, [])
            optional = clustered_optional.get(cluster_id, {})

            food = optional.get("food", [])
            souvenirs = optional.get("souvenirs", [])
            lodging = optional.get("lodging", [])

            # Extract names safely
            # attraction_names = attractions
            attraction_names = attractions.get("_key", []) if isinstance(attractions, dict) else attractions


            food_names = [f.get("_key") for f in food if f.get("_key")]
            souvenir_names = [s.get("_key") for s in souvenirs if s.get("_key")]
            lodging_names = [l.get("_key") for l in lodging if l.get("_key")]



            final_output[exact_date.isoformat()] = {
                "attractions": attraction_names,
                "food": food_names,
                "souvenir_shops": souvenir_names,
                "lodging": lodging_names [:2]  # include up to 2 lodging options
            }
            new_date=next_date(exact_date)
            exact_date=new_date

            # final_output[f"day_{day_index + 1}"] = {
            #     "attractions": attraction_names,
            #     "food": food_names,
            #     "souvenir_shops": souvenir_names,
            #     "lodging": lodging_names [:2]  # include up to 2 lodging options
            # }
        
        allowed_names = set()

        for c in clusters.values():
            allowed_names.update(c.get("_key", []) if isinstance(c, dict) else c)

        for opt in clustered_optional.values():
            for category in opt.values():
                for place in category:
                    if "_key" in place:
                        allowed_names.add(place["_key"])
        
        for day in final_output.values():
            for category in day:
                for name in day[category]:
                    if name not in allowed_names:
                        raise ValueError("Hallucination detected")

        return {
            **state,
            # "draft_itinerary": json.dumps(final_output, indent=4)
            "draft_itinerary": final_output
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # print("=" * 65)
    # print("Initialising Itinerary Generator …")
    generator = ItineraryGenerator()
    graph     = generator.create_graph()

    test_queries = [
        # Food & souvenirs both requested
        # "Plan a 3-day trip to Lahore. Include famous food spots and "
        # "add souvenir shops on the way.",

        # Neither food nor souvenirs
        # "I want a 2-day trip to Hunza Valley focused on lakes and mountains.",

        # Only food
        "Give me a 2-day Islamabad itinerary with adventurous spots and add famous food spots."

        #tricky
        # "Give me a 5 -day northern place itinerary with adventurous spots"
    ]

    for query in test_queries:
        # print("\n" + "=" * 65)
        # print(f"QUERY: {query}")
        # print("=" * 65)

        initial_state: ItineraryState = {
            "user_query":            query,
            "parsed_days":           None,
            "parsed_location":       None,
            "parsed_preferences":    None,
            "query_parse_error":     None,
            "include_food":          False,
            "include_souvenirs":     False,
            "retrieved_attractions": [],
            "retrieval_metadata":    {},
            "place_coordinates":     {},
            "nearby_food":           [],
            "nearby_souvenirs":      [],
            "nearby_lodging":        [],
            "draft_itinerary":       None,
            "retry_count":           0,
        }

        result = graph.invoke(initial_state)

        # print(f"\n📍 Location       : {result['parsed_location']}")
        # print(f"📅 Days           : {result['parsed_days']}")
        # print(f"🎯 Preferences    : {result['parsed_preferences']}")
        # print(f"🍽  Food requested : {result['include_food']}  "
        #       f"({len(result['nearby_food'])} spot(s) selected)")
        # print(f"🛍  Souv. requested: {result['include_souvenirs']}  "
        #       f"({len(result['nearby_souvenirs'])} shop(s) selected)")
        # print(f"🏨 Lodging options : {len(result['nearby_lodging'])}")
        # print(f"🗺  Coords mapped   : {len(result['place_coordinates'])} attractions")
        # print(f"\n{'─'*65}\n📋 ITINERARY:\n{result['draft_itinerary']}")
        # print("\n" + "─" * 65)


if __name__ == "__main__":
    main() 