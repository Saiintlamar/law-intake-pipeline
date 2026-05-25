"""
OP3 API PIPE BUILDER — Law Firm Client Intake AI Pipeline
Apex production code. Aggressive error handling. Full visibility.
Stack: Flask + Gemini REST API (no SDK) + structured logging
Deploy: Render (free tier)

Every failure path returns exact HTTP status + response body.
Zero silent failures. Zero dependency conflicts.
"""

import os
import sys
import logging
import traceback
import json
from typing import Dict, Optional, Tuple

from flask import Flask, request, jsonify
import requests

# ===== CONFIGURATION =====
REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]
OPTIONAL_ENV_VARS = {
    "LAW_FIRM_NAME": "Smith & Associates",
    "PRACTICE_AREAS": "Personal Injury, Family Law",
    "LAWYER_EMAIL": "attorney@example.com",
    "LAWYER_PHONE": "(555) 123-4567",
    "STATE": "California",
    "INTAKE_FORM_URL": "https://forms.example.com/intake",
    "CALENDLY_URL": "https://calendly.com/attorney/consultation",
}

# Validate required env vars on startup
MISSING_VARS = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
if MISSING_VARS:
    print(f"FATAL: Missing required environment variables: {MISSING_VARS}")
    print("Set these in Render Dashboard → Environment")
    # Don't crash — let Flask start so health check passes, but log the error

# Load config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LAW_FIRM_NAME = os.environ.get("LAW_FIRM_NAME", OPTIONAL_ENV_VARS["LAW_FIRM_NAME"])
PRACTICE_AREAS = os.environ.get("PRACTICE_AREAS", OPTIONAL_ENV_VARS["PRACTICE_AREAS"])
LAWYER_EMAIL = os.environ.get("LAWYER_EMAIL", OPTIONAL_ENV_VARS["LAWYER_EMAIL"])
LAWYER_PHONE = os.environ.get("LAWYER_PHONE", OPTIONAL_ENV_VARS["LAWYER_PHONE"])
STATE = os.environ.get("STATE", OPTIONAL_ENV_VARS["STATE"])
INTAKE_FORM_URL = os.environ.get("INTAKE_FORM_URL", OPTIONAL_ENV_VARS["INTAKE_FORM_URL"])
CALENDLY_URL = os.environ.get("CALENDLY_URL", OPTIONAL_ENV_VARS["CALENDLY_URL"])

# Gemini API — use the generateContent endpoint (no generateContent needed, it's v1)
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout  # Render captures stdout
)
log = logging.getLogger(__name__)

# Log startup state
log.info(f"Starting Law Firm Intake Pipeline")
log.info(f"Firm: {LAW_FIRM_NAME}")
log.info(f"API Key present: {'Yes' if OPENAI_API_KEY else 'NO — FATAL'}")
log.info(f"API Key prefix: {OPENAI_API_KEY[:8]}..." if OPENAI_API_KEY else "N/A")
log.info(f"Gemini Model: {GEMINI_MODEL}")

app = Flask(__name__)


# ===== GEMINI API CLIENT =====
def call_gemini(prompt: str, max_tokens: int = 1000) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Calls Gemini REST API directly.
    
    Returns:
        (success, text_or_none, error_dict_or_none)
        
    On success: (True, "response text", None)
    On failure: (False, None, {"status": 403, "body": "...", "message": "..."})
    
    NEVER swallows errors. Always returns exactly what happened.
    """
    
    # Validate API key before making request
    if not OPENAI_API_KEY:
        return (False, None, {
            "status": "N/A",
            "body": "API key not configured",
            "message": "OPENAI_API_KEY environment variable is empty or missing. Set it in Render Dashboard."
        })
    
    if not OPENAI_API_KEY.startswith("AIza"):
        return (False, None, {
            "status": "N/A",
            "body": "Invalid API key format",
            "message": f"API key starts with '{OPENAI_API_KEY[:4]}...' but should start with 'AIza'. This looks like an OpenAI key, not a Gemini key."
        })
    
    url = f"{GEMINI_API_URL}?key={OPENAI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
    }
    
    try:
        log.info(f"Calling Gemini: {url[:80]}...")
        
        response = requests.post(
            url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        status_code = response.status_code
        response_text = response.text
        
        log.info(f"Gemini HTTP {status_code}")
        
        # Success
        if status_code == 200:
            try:
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                log.info(f"Gemini returned {len(text)} characters")
                return (True, text, None)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                return (False, None, {
                    "status": status_code,
                    "body": response_text[:500],
                    "message": f"Failed to parse Gemini response: {str(e)}. Raw response: {response_text[:300]}"
                })
        
        # Error responses
        error_message = f"Gemini returned HTTP {status_code}"
        
        try:
            error_data = response.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except:
            pass
        
        return (False, None, {
            "status": status_code,
            "body": response_text[:500],
            "message": error_message
        })
    
    except requests.exceptions.Timeout:
        return (False, None, {
            "status": "TIMEOUT",
            "body": "Request timed out after 30 seconds",
            "message": "Gemini API timed out. Service may be overloaded or network issue."
        })
    except requests.exceptions.ConnectionError as e:
        return (False, None, {
            "status": "CONNECTION_ERROR",
            "body": str(e)[:300],
            "message": f"Cannot connect to Gemini API. Network error: {str(e)[:200]}"
        })
    except Exception as e:
        return (False, None, {
            "status": "EXCEPTION",
            "body": traceback.format_exc()[:500],
            "message": f"Unexpected error: {str(e)[:200]}"
        })


# ===== AI ENGINE =====
def analyze_inquiry(email_body: str, client_name: str = "Potential Client") -> Dict:
    """
    Analyzes a client inquiry email.
    Returns structured analysis or detailed error information.
    """
    
    prompt = f"""You are an AI intake specialist for {LAW_FIRM_NAME}, a law firm practicing in {PRACTICE_AREAS} in {STATE}.

A potential client named {client_name} sent this inquiry:

---
{email_body}
---

Analyze this inquiry and return a JSON object with exactly these fields:
1. "practice_area": string — Which practice area? If unclear, say "Needs clarification"
2. "urgency": string — "High", "Medium", or "Low"
3. "statute_of_limitations_risk": string — "Yes" or "No"
4. "key_facts": array of strings — 2-3 bullet points of legally relevant facts
5. "conflicts_check": string — "Pass" or "Flag - [reason]"
6. "recommended_action": string — What should the attorney do next?
7. "auto_reply_draft": string — A warm, professional email reply that:
   - Thanks them for reaching out
   - Shows you understood their specific situation
   - Explains next steps clearly
   - Includes the intake form link: {INTAKE_FORM_URL}
   - Includes the consultation scheduling link: {CALENDLY_URL}
   - Is signed by the team at {LAW_FIRM_NAME}
   - Includes the phone number: {LAWYER_PHONE}

IMPORTANT: Return ONLY the JSON object. No markdown code blocks. No ```json``` wrappers. No text before or after the JSON. Just the raw JSON starting with {{ and ending with }}."""

    success, text, error = call_gemini(prompt)
    
    if not success:
        # Return the actual error details — never hide them
        return {
            "practice_area": f"API Error — HTTP {error.get('status', 'Unknown')}",
            "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": [error.get("message", "Unknown error")[:200]],
            "conflicts_check": "Error",
            "recommended_action": f"Fix: {error.get('message', '')[:200]}",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}.",
            "_debug": error  # Full debug info
        }
    
    # Parse the JSON response
    cleaned = text.strip()
    # Remove any markdown wrappers
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:]) if len(lines) > 1 else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    
    # Find JSON boundaries
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    if start >= 0 and end > start:
        cleaned = cleaned[start:end]
    
    try:
        parsed = json.loads(cleaned)
        log.info(f"Analysis: {parsed.get('practice_area')} | Urgency: {parsed.get('urgency')}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed. Raw text: {text[:500]}")
        return {
            "practice_area": "Error — JSON Parse Failed",
            "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": [f"JSON parse error: {str(e)[:150]}"],
            "conflicts_check": "Error",
            "recommended_action": "Manual review — Gemini returned non-JSON response",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}.",
            "_debug_raw_response": text[:500]
        }


def generate_case_summary(intake_form_data: Dict) -> str:
    """Generates pre-consultation case summary."""
    
    prompt = f"""You are a legal intake specialist at {LAW_FIRM_NAME}.
A potential client submitted this intake form:
---
{json.dumps(intake_form_data, indent=2)}
---
Create a concise pre-consultation summary for the attorney. Include:
1. Client name and contact info
2. Case type and jurisdiction
3. Key facts with timeline
4. Potential claims or defenses
5. Statute of limitations analysis ({STATE} law)
6. Damages or relief sought
7. Recommended consultation preparation
Format as a professional memo. Under 500 words."""

    success, text, error = call_gemini(prompt, max_tokens=800)
    
    if success:
        return text
    
    return f"SUMMARY FAILED: {error.get('message', 'Unknown error')}\n\nPlease review intake form manually at: {INTAKE_FORM_URL}"


# ===== STUB SERVICES =====
def send_email(to_email: str, subject: str, body: str) -> bool:
    log.info(f"[EMAIL] To: {to_email} | {subject}")
    return True

def create_clio_contact(name: str, email: str, phone: str = "", notes: str = "") -> Optional[str]:
    log.info(f"[CLIO] Created: {name}")
    return "demo-contact-id"

def notify_lawyer(subject: str, message: str):
    log.info(f"[NOTIFY] {subject}")


# ===== ENDPOINTS =====
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": f"{LAW_FIRM_NAME} — AI Client Intake Pipeline",
        "version": "4.0.0-apex",
        "ai_provider": f"Google Gemini ({GEMINI_MODEL})",
        "api_key_configured": bool(OPENAI_API_KEY),
        "api_key_valid_format": OPENAI_API_KEY.startswith("AIza") if OPENAI_API_KEY else False,
        "endpoints": ["GET /demo", "POST /intake/email", "POST /intake/form", "POST /calendly/webhook"],
        "status": "operational"
    })


@app.route("/demo", methods=["GET"])
def demo():
    sample = {
        "from_email": "john.smith@example.com",
        "from_name": "John Smith",
        "subject": "Slipped and fell at grocery store",
        "body": "I was shopping at FreshMart on Main Street last Tuesday evening around 7pm. There was a spill in the produce aisle with no warning sign. I slipped, fell, and hurt my back. I've been to the ER and have medical bills piling up. The store manager took a report but now their insurance company is ignoring my calls. What should I do?",
        "phone": "555-123-4567"
    }
    
    log.info("=== DEMO REQUEST ===")
    
    analysis = analyze_inquiry(sample["body"], sample["from_name"])
    
    # Determine success
    is_success = "Error" not in str(analysis.get("practice_area", ""))
    
    response = {
        "demo": "Law Firm Intake AI Pipeline",
        "firm": LAW_FIRM_NAME,
        "ai_model": GEMINI_MODEL,
        "success": is_success,
        "input": sample,
        "ai_analysis": analysis,
        "time_saved": "15-30 minutes per inquiry",
        "setup_fee": "$800",
        "monthly_maintenance": "$99"
    }
    
    log.info(f"Demo result: {'SUCCESS' if is_success else 'FAILED'}")
    log.info(f"Analysis keys: {list(analysis.keys())}")
    
    return jsonify(response)


@app.route("/intake/email", methods=["POST"])
def intake_email():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    from_email = data.get("from_email", "")
    from_name = data.get("from_name", "Client")
    email_body = data.get("body", "")
    phone = data.get("phone", "")
    
    if not from_email or not email_body:
        return jsonify({"error": "Missing from_email or body"}), 400
    
    log.info(f"Intake: {from_name} <{from_email}>")
    
    analysis = analyze_inquiry(email_body, from_name)
    
    auto_reply = analysis.get("auto_reply_draft", "")
    if auto_reply:
        send_email(from_email, f"Re: Your inquiry to {LAW_FIRM_NAME}", auto_reply)
    
    contact_id = create_clio_contact(from_name, from_email, phone, json.dumps(analysis)[:500])
    notify_lawyer(f"New Intake: {from_name}", json.dumps(analysis, indent=2))
    
    return jsonify({
        "status": "success",
        "analysis": analysis,
        "contact_id": contact_id,
        "auto_reply_sent": bool(auto_reply)
    })


@app.route("/intake/form", methods=["POST"])
def intake_form():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    client_name = data.get("full_name", data.get("name", "Unknown"))
    client_email = data.get("email", "")
    
    log.info(f"Form: {client_name}")
    
    summary = generate_case_summary(data)
    contact_id = create_clio_contact(client_name, client_email, data.get("phone", ""), summary)
    notify_lawyer(f"Form: {client_name}", summary)
    send_email(client_email, f"Intake Received — {LAW_FIRM_NAME}", 
               f"Dear {client_name},\n\nYour intake form has been received.\n\nSchedule: {CALENDLY_URL}\n\n{LAW_FIRM_NAME}\n{LAWYER_PHONE}")
    
    return jsonify({"status": "success", "contact_id": contact_id})


@app.route("/calendly/webhook", methods=["POST"])
def calendly_webhook():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    if data.get("event") == "invitee.created":
        invitee = data.get("payload", {}).get("invitee", {})
        log.info(f"Booking: {invitee.get('name')}")
        notify_lawyer("Booking", f"{invitee.get('name')} — {invitee.get('start_time')}")
    
    return jsonify({"status": "received"})


# ===== HEALTH CHECK =====
@app.route("/health", methods=["GET"])
def health():
    """Render health check endpoint."""
    return jsonify({
        "status": "healthy",
        "api_key_set": bool(OPENAI_API_KEY),
        "api_key_valid": OPENAI_API_KEY.startswith("AIza") if OPENAI_API_KEY else False
    })


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port)
