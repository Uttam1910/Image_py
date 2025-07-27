from flask import Flask, request, jsonify
import os
import logging
import requests
from urllib.parse import quote
from google.cloud import vision
import requests_cache

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Install caching for external requests (cache expires in 1 hour)
requests_cache.install_cache('api_cache', backend='sqlite', expire_after=3600)

# Set Google Cloud Vision credentials using the correct file name (service-account.json)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(BASE_DIR, "service-account.json")

def get_wikidata_info(wikidata_id):
    """Fetch structured data from Wikidata"""
    if not wikidata_id:
        return {}
    
    try:
        # Fetch main entity data
        params = {
            "action": "wbgetentities",
            "ids": wikidata_id,
            "format": "json",
            "props": "claims|descriptions|labels",
            "languages": "en"
        }
        response = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=5)
        if response.status_code != 200:
            return {}
        
        entity_data = response.json().get("entities", {}).get(wikidata_id, {})
        claims = entity_data.get("claims", {})
        
        # Extract relevant properties
        info = {
            "description": entity_data.get("descriptions", {}).get("en", {}).get("value", ""),
            "inception_date": extract_claim_value(claims.get("P571", [])),
            "architectural_styles": extract_entity_labels(claims.get("P149", [])),
            "country": extract_entity_labels(claims.get("P17", [])),
            "official_website": extract_claim_value(claims.get("P856", []), is_url=True)
        }
        
        return info
    except Exception as e:
        logging.error(f"Wikidata query failed: {str(e)}")
        return {}

def extract_claim_value(claims, is_url=False):
    """Extract value from Wikidata claims"""
    if not claims:
        return None
    try:
        value = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")
        if is_url:
            return value
        if isinstance(value, dict) and "time" in value:
            return value["time"].lstrip("+").split("T")[0]
        return value
    except Exception:
        return None

def extract_entity_labels(claims):
    """Resolve Wikidata entity IDs to their English labels"""
    if not claims:
        return []
    
    entity_ids = [c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") for c in claims]
    entity_ids = list(filter(None, set(entity_ids)))
    
    if not entity_ids:
        return []
    
    try:
        params = {
            "action": "wbgetentities",
            "ids": "|".join(entity_ids),
            "format": "json",
            "props": "labels",
            "languages": "en"
        }
        response = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=5)
        if response.status_code != 200:
            return []
        
        labels = {}
        for qid, data in response.json().get("entities", {}).items():
            labels[qid] = data.get("labels", {}).get("en", {}).get("value", qid)
        
        return [labels.get(eid, eid) for eid in entity_ids]
    except Exception:
        return []

def get_location_details(latitude, longitude):
    """Get human-readable location details using OpenStreetMap"""
    try:
        headers = {'User-Agent': 'LandmarkInfoService/1.0 (contact@example.com)'}
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            address = response.json().get("address", {})
            return {
                "country": address.get("country"),
                "city": address.get("city") or address.get("town") or address.get("village")
            }
    except Exception as e:
        logging.error(f"Geocoding failed: {str(e)}")
    return {}

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        # Process image with Google Cloud Vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=request.files['image'].read())
        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations
        
        if not landmarks:
            return jsonify({"error": "No landmarks detected"}), 404

        landmark = landmarks[0]
        landmark_name = landmark.description
        locations = [{"latitude": loc.lat_lng.latitude, "longitude": loc.lat_lng.longitude}
                     for loc in landmark.locations if loc.lat_lng]

        # Fetch Wikipedia data
        wiki_data = {}
        try:
            wiki_response = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(landmark_name)}",
                timeout=5
            )
            if wiki_response.status_code == 200:
                wiki_data = wiki_response.json()
        except Exception as e:
            logging.error(f"Wikipedia query failed: {str(e)}")

        # Fetch Wikidata information
        wikidata_id = wiki_data.get("wikibase_item")
        wikidata_info = get_wikidata_info(wikidata_id) if wikidata_id else {}

        # Prepare location details
        location_details = {}
        if locations:
            loc = locations[0]
            location_details = {
                "coordinates": loc,
                **get_location_details(loc["latitude"], loc["longitude"])
            }
            # Add Google Maps link
            location_details["maps_link"] = f"https://maps.google.com/?q={loc['latitude']},{loc['longitude']}"

        # Build comprehensive response
        result = {
            "name": landmark_name,
            "description": wikidata_info.get("description") or wiki_data.get("extract") or "No description available",
            "historical_context": {
                "inception_date": wikidata_info.get("inception_date"),
                "architectural_styles": wikidata_info.get("architectural_styles"),
                "significance": wiki_data.get("description")
            },
            "location": location_details,
            "official_website": wikidata_info.get("official_website"),
            "references": {
                "wikipedia": wiki_data.get("content_urls", {}).get("desktop", {}).get("page"),
                "wikidata": f"https://www.wikidata.org/wiki/{wikidata_id}" if wikidata_id else None,
                "google_maps": location_details.get("maps_link")
            }
        }

        return jsonify(result)

    except Exception as e:
        logging.exception("Processing failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


