# integrate_h was not getting any data so claude compl code
from __future__ import annotations
import pickle
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path
import difflib
import json
import os
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional
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
from config.settings import llm, DF_FOOD, DF_SOUVENIRS, DF_LODGING

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def cluster_attractions(place_coordinates: Dict[str, Dict[str, float]],
                        max_distance_km: float = 40.0):
    if not place_coordinates:
        return {}

    names = list(place_coordinates.keys())
    coords = np.array([
        [radians(v["lat"]), radians(v["lng"])]
        for v in place_coordinates.values()
    ])

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

    return {"lat": lat_sum / count, "lng": lng_sum / count}


def auto_balance_clusters(
    clusters: Dict[int, List[str]],
    place_coordinates: Dict[str, Dict[str, float]],
    target_days: int
) -> Dict[int, List[str]]:

    if not clusters:
        return clusters

    clusters = {k: list(v) for k, v in clusters.items()}

    # STEP 1: Merge clusters if more clusters than days
    while len(clusters) > target_days:
        cluster_ids = list(clusters.keys())
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

        c1, c2 = pair_to_merge
        clusters[c1].extend(clusters[c2])
        del clusters[c2]

    # STEP 1B: Split clusters if fewer clusters than target_days
    while len(clusters) < target_days:
        largest_cluster_id = max(clusters, key=lambda k: len(clusters[k]))
        largest_cluster = clusters[largest_cluster_id]

        if len(largest_cluster) <= 1:
            break

        mid = len(largest_cluster) // 2
        new_cluster_1 = largest_cluster[:mid]
        new_cluster_2 = largest_cluster[mid:]

        clusters[largest_cluster_id] = new_cluster_1
        new_cluster_id = max(clusters.keys()) + 1
        clusters[new_cluster_id] = new_cluster_2

    # STEP 2: Evenly redistribute if highly uneven
    changed = True
    while changed:
        changed = False
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]))
        smallest_id, smallest_list = sorted_clusters[0]
        largest_id, largest_list = sorted_clusters[-1]

        if len(largest_list) - len(smallest_list) <= 1:
            break

        smallest_centroid = compute_centroid(smallest_list, place_coordinates)
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
    parsed_days:        Optional[int]
    parsed_location:    Optional[str]
    parsed_preferences: Optional[List[str]]
    query_parse_error:  Optional[str]
    start_date:         Optional[str]
    end_date:           Optional[str]
    include_food:       bool
    include_souvenirs:  bool
    retrieved_attractions: List[Dict[str, Any]]
    retrieval_metadata:    Dict[str, Any]
    place_coordinates:     Dict[str, Dict[str, float]]
    draft_itinerary:       Optional[str]
    retry_count:           int
    clusters:              Dict[int, List[str]]
    clustered_optional_places: Dict[int, Dict[str, List[Dict[str, Any]]]]
    budget_needed:         float


# ---------------------------------------------------------------------------
# FIX: find_close_district now returns the ORIGINAL (un-lowercased) city value
# so it can be used for both display AND comparison consistently.
# ---------------------------------------------------------------------------

def find_close_district(district_name: str, df: pd.DataFrame) -> Optional[str]:
    """
    Fuzzy-match the user's location string against the 'city' column.
    Returns the matched city value in its ORIGINAL case as stored in the df,
    or None if no match is found.
    """
    if not district_name:
        return None

    target = str(district_name).strip().lower()

    # Build a mapping: lowercase → original value
    city_series = df["city"].dropna().astype(str).str.strip()
    lowercase_to_original: Dict[str, str] = {
        v.lower(): v for v in city_series.unique()
    }

    lowercase_vals = list(lowercase_to_original.keys())
    matches = difflib.get_close_matches(target, lowercase_vals, n=5, cutoff=0.6)

    if matches:
        # Return the ORIGINAL casing so equality check in semantic_search works
        return lowercase_to_original[matches[0]]

    return None


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def make_nearby_tool(place_coordinates: Dict[str, Dict[str, float]]):
    @tool
    def find_nearby_places(anchor_place: str, radius_km: float = 60.0) -> str:
        """
        Return a JSON list of candidate attractions within `radius_km` km of
        `anchor_place`, sorted by ascending distance.
        """
        anchor = place_coordinates.get(anchor_place)
        if anchor is None:
            return json.dumps({
                "error": f"'{anchor_place}' not found.",
                "available_places": list(place_coordinates.keys()),
            })

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

    EMBEDDING_FILE = "data/attractions_embeddings.npy"
    DATAFRAME_FILE = "data/attractions_processed.pkl"

    SUPABASE_URL_CLASS = "https://vtnqhxylpkbhugbnjxnc.supabase.co"
    SUPABASE_TABLE     = "Attractions"

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_and_embed_attractions()

    # -----------------------------------------------------------------------
    # Supabase fetch with pagination
    # -----------------------------------------------------------------------

    def _fetch_from_supabase(self) -> pd.DataFrame:
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_key:
            raise EnvironmentError(
                "SUPABASE_KEY environment variable is not set."
            )

        client: Client = create_client(self.SUPABASE_URL_CLASS, supabase_key)

        all_rows: list[dict] = []
        page_size = 1_000
        offset    = 0

        print("Fetching attractions from Supabase …")

        while True:
            response = (
                client.table(self.SUPABASE_TABLE)
                .select("*")
                .range(offset, offset + page_size - 1)
                .execute()
            )

            batch = response.data or []
            all_rows.extend(batch)
            print(f"  fetched {len(all_rows)} rows so far …")

            if len(batch) < page_size:
                break
            offset += page_size

        print(f"Supabase fetch complete — {len(all_rows)} attractions loaded.")
        df = pd.DataFrame(all_rows)

        # ── DEBUG: print a sample so you can verify column names & values ──
        print("[DEBUG] Columns from Supabase:", df.columns.tolist())
        if not df.empty:
            print("[DEBUG] Sample city values:", df["city"].dropna().unique()[:10])

        return df

    # -----------------------------------------------------------------------
    # Load / embed
    # -----------------------------------------------------------------------

    def _load_and_embed_attractions(self) -> None:

        embedding_path = Path(self.EMBEDDING_FILE)
        dataframe_path = Path(self.DATAFRAME_FILE)

        if embedding_path.exists() and dataframe_path.exists():
            print("Loading cached embeddings …")
            self.embeddings = np.load(embedding_path)
            with open(dataframe_path, "rb") as f:
                self.df = pickle.load(f)
            print("Embeddings loaded successfully.")

            # ── DEBUG: confirm city values in cached df ──
            print("[DEBUG] Sample city values (cache):", self.df["city"].dropna().unique()[:10])
            return

        print("Fetching data from Supabase and generating embeddings …")
        self.df = self._fetch_from_supabase()

        self.df["search_text"] = (
            self.df.get("_key",      pd.Series("", index=self.df.index)).fillna("").astype(str)
            + " "
            + self.df.get("Desc",     pd.Series("", index=self.df.index)).fillna("").astype(str)
            + " "
            + self.df.get("category", pd.Series("", index=self.df.index)).fillna("").astype(str)
            + " "
            + self.df.get("city",     pd.Series("", index=self.df.index)).fillna("").astype(str)
        )

        self.embeddings = self.embedding_model.encode(
            self.df["search_text"].tolist(),
            show_progress_bar=True,
        )

        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, self.embeddings)
        with open(dataframe_path, "wb") as f:
            pickle.dump(self.df, f)

        print("Embeddings generated and cached.")

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
    # Node 1 — Semantic search
    # -----------------------------------------------------------------------

    def semantic_search(self, state: ItineraryState) -> ItineraryState:
        location = state.get("parsed_location")

        # find_close_district now returns ORIGINAL casing (e.g. "Islamabad")
        if location:
            matched = find_close_district(location, self.df)
            if matched:
                location = matched
                state["parsed_location"] = location

        prefs     = state.get("parsed_preferences") or []
        query_txt = f"{location or ''} {' '.join(prefs)}".strip()
        q_emb     = self.embedding_model.encode([query_txt])
        sims      = np.dot(self.embeddings, q_emb.T).flatten()
        top_idx   = np.argsort(sims)[::-1]

        # ── FIX: compare both sides in the same case ──
        # location is now original-cased; normalise both sides to be safe.
        location_lower = location.strip().lower() if location else ""

        filtered_attractions = []

        for i in top_idx:
            row = self.df.iloc[i]

            if location_lower:
                row_city = str(row.get("city", "")).strip().lower()
                if row_city != location_lower:
                    continue

            filtered_attractions.append(
                {**row.to_dict(), "similarity_score": float(sims[i])}
            )

            if len(filtered_attractions) >= state.get("parsed_days", 3) * 4:
                break

        # ── DEBUG ──
        print(f"[DEBUG] parsed_location='{location}', location_lower='{location_lower}'")
        print(f"[DEBUG] retrieved {len(filtered_attractions)} attractions")

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
    # Node 2 — Data enricher
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

        return {
            **state,
            "retrieved_attractions": enriched,
            "place_coordinates":     place_coordinates,
        }

    # -----------------------------------------------------------------------
    # Node 3 — Select optional places
    # -----------------------------------------------------------------------

    def select_optional_places(self, state: ItineraryState) -> ItineraryState:

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
        budget_needed = 0

        for cluster_id, place_names in clusters.items():
            anchors = [
                state["place_coordinates"][name]
                for name in place_names
                if name in state["place_coordinates"]
            ]

            cluster_data = {}

            if state.get("include_food"):
                cluster_data["food"] = nearest_from_pool(
                    anchors, DF_FOOD, top_n=3, max_radius_km=40.0
                )
            else:
                cluster_data["food"] = []

            if state.get("include_souvenirs"):
                cluster_data["souvenirs"] = nearest_from_pool(
                    anchors, DF_SOUVENIRS, top_n=2, max_radius_km=40.0
                )
            else:
                cluster_data["souvenirs"] = []

            cluster_data["lodging_candidates"] = nearest_from_pool(
                anchors, DF_LODGING, top_n=5, max_radius_km=60.0
            )

            lodging_candidates = cluster_data.get("lodging_candidates", [])

            if lodging_candidates:
                lodging_candidates.sort(
                    key=lambda x: float(x.get("price", float("inf")))
                )
                cluster_data["lodging"] = lodging_candidates
                budget_needed += float(lodging_candidates[0].get("price", float("inf")))
            else:
                cluster_data["lodging"] = []

            clustered_optional[cluster_id] = cluster_data

        all_food      = []
        all_souvenirs = []
        all_lodging   = []

        for cdata in clustered_optional.values():
            all_food.extend(cdata.get("food", []))
            all_souvenirs.extend(cdata.get("souvenirs", []))
            all_lodging.extend(cdata.get("lodging", []))

        return {
            **state,
            "clusters":                  clusters,
            "clustered_optional_places": clustered_optional,
            "nearby_food":               all_food,
            "nearby_souvenirs":          all_souvenirs,
            "nearby_lodging":            all_lodging,
            "budget_needed":             budget_needed,
        }

    # -----------------------------------------------------------------------
    # Node 4 — Itinerary generator
    # -----------------------------------------------------------------------

    def generate_itinerary(self, state: ItineraryState) -> ItineraryState:

        clusters           = state.get("clusters", {})
        clustered_optional = state.get("clustered_optional_places", {})

        if not clusters:
            return {**state, "draft_itinerary": "{}"}

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
            {"role": "user",   "content": user_prompt}
        ])

        try:
            cluster_order = json.loads(response.content)
        except Exception:
            cluster_order = cluster_ids

        final_output = {}

        exact_date = state.get("start_date")
        if not exact_date:
            raise ValueError(
                "Missing start_date in state; please provide a start_date."
            )

        try:
            exact_date = ensure_date(exact_date)
        except TypeError:
            exact_date = parse_date_from_text(exact_date)

        for day_index, cluster_id in enumerate(cluster_order):

            attractions = clusters.get(cluster_id, [])
            optional    = clustered_optional.get(cluster_id, {})

            food      = optional.get("food", [])
            souvenirs = optional.get("souvenirs", [])
            lodging   = optional.get("lodging", [])

            attraction_names = (
                attractions.get("_key", [])
                if isinstance(attractions, dict)
                else attractions
            )

            food_names     = [f.get("_key") for f in food     if f.get("_key")]
            souvenir_names = [s.get("_key") for s in souvenirs if s.get("_key")]
            lodging_names  = [l.get("_key") for l in lodging   if l.get("_key")]

            final_output[exact_date.isoformat()] = {
                "attractions":   attraction_names,
                "food":          food_names,
                "souvenir_shops": souvenir_names,
                "lodging":       lodging_names[:2],
            }

            exact_date = next_date(exact_date)

        allowed_names = set()

        for c in clusters.values():
            allowed_names.update(
                c.get("_key", []) if isinstance(c, dict) else c
            )

        for opt in clustered_optional.values():
            for category in opt.values():
                for place in category:
                    if "_key" in place:
                        allowed_names.add(place["_key"])

        for day in final_output.values():
            for category in day:
                for name in day[category]:
                    if name not in allowed_names:
                        raise ValueError(f"Hallucination detected: '{name}'")

        return {**state, "draft_itinerary": final_output}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    generator = ItineraryGenerator()
    graph     = generator.create_graph()

    test_queries = [
        "Give me a 2-day Islamabad itinerary with adventurous spots and add famous food spots."
    ]

    for query in test_queries:
        initial_state: ItineraryState = {
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

        print(f"\n📍 Location        : {result['parsed_location']}")
        print(f"📅 Days            : {result['parsed_days']}")
        print(f"🎯 Preferences     : {result['parsed_preferences']}")
        print(f"🍽  Food requested  : {result['include_food']}  ({len(result.get('nearby_food', []))} spot(s))")
        print(f"🛍  Souv. requested : {result['include_souvenirs']}  ({len(result.get('nearby_souvenirs', []))} shop(s))")
        print(f"🏨 Lodging options  : {len(result.get('nearby_lodging', []))}")
        print(f"🗺  Coords mapped    : {len(result['place_coordinates'])} attractions")
        print(f"\n{'─'*65}\n📋 ITINERARY:\n{json.dumps(result['draft_itinerary'], indent=2)}")


if __name__ == "__main__":
    main()