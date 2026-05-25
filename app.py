"""
OP3 API PIPE BUILDER — Law Firm Client Intake AI Pipeline
Uses Gemini REST API directly (no OpenAI SDK).
Zero dependency conflicts.
"""

import os
import logging
import traceback
from typing import Dict, Optional
import json

from flask import Flask, request, jsonify
import requests

# ===== CONFIGURATION =====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LAW_FIRM_NAME = os.environ.get("LAW_FIRM_NAME", "Smith & Associates")
PRACTICE_AREAS = os.environ.get("PRACTICE_AREAS", "Personal Injury, Family Law")
LAWYER_EMAIL = os.environ.get("LAWYER_EMAIL", "attorney@example.com")
LAWYER_PHONE = os.environ.get("LAWYER_PHONE", "(555) 123-4567")
STATE = os.environ.get("STATE", "California")
INTAKE_FORM_URL = os.environ.get("INTAKE_FORM_URL", "https://forms.example.com/intake")
CALENDLY_URL = os.environ.get("CALENDLY_URL", "https://calendly.com/attorney/consultation")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)


# ===== AI ENGINE (RAW GEMINI API — NO OPENAI SDK) =====
def call_gemini(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """Calls Gemini REST API directly. No SDK needed."""
    
    url = f"{GEMINI_API_URL}?key={OPENAI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract text from Gemini response
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text
        
    except requests.exceptions.Timeout:
        log.error("Gemini API timeout")
        return None
    except requests.exceptions.HTTPError as e:
        log.error(f"Gemini HTTP error: {e.response.status_code} — {e.response.text[:500]}")
        return None
    except Exception as e:
        log.error(f"Gemini call failed: {traceback.format_exc()}")
        return None


def analyze_inquiry(email_body: str, client_name: str = "Potential Client") -> Dict:
    prompt = f"""You are an AI intake specialist for {LAW_FIRM_NAME}, a law firm practicing in {PRACTICE_AREAS} in {STATE}.

A potential client named {client_name} sent this inquiry:

---
{email_body}
---

Analyze this inquiry and return a JSON object with:
1. practice_area: Which practice area? If unclear, say "Needs clarification"
2. urgency: "High", "Medium", or "Low"
3. statute_of_limitations_risk: "Yes" or "No"
4. key_facts: 2-3 bullet points summarizing legally relevant facts
5. conflicts_check: "Pass" or "Flag - [reason]"
6. recommended_action: What should the attorney do next?
7. auto_reply_draft: A warm, professional email reply that thanks them, shows understanding, explains next steps, links to intake form ({INTAKE_FORM_URL}) and scheduling ({CALENDLY_URL}), signed by {LAW_FIRM_NAME}, phone {LAWYER_PHONE}

Return ONLY valid JSON. No markdown. No code blocks. No extra text."""

    raw = call_gemini(prompt)
    
    if not raw:
        return {
            "practice_area": "Error - API Call Failed",
            "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": ["Gemini API call failed. Check API key and quota."],
            "conflicts_check": "Error",
            "recommended_action": "Manual review required — AI service unavailable",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}."
        }
    
    # Clean and parse JSON
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find('{')
    end = raw.rfind('}') + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    
    try:
        parsed = json.loads(raw)
        log.info(f"Analysis success: {parsed.get('practice_area', 'Unknown')}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed. Raw: {raw[:500]}")
        return {
            "practice_area": "Error - JSON Parse",
            "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": [f"JSON parsing failed: {str(e)[:150]}"],
            "conflicts_check": "Error",
            "recommended_action": "Manual review — AI response was not valid JSON",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}."
        }


def generate_case_summary(intake_form_data: Dict) -> str:
    prompt = f"""You are a legal intake specialist at {LAW_FIRM_NAME}.
A potential client submitted this intake form:
---
{intake_form_data}
---
Create a concise pre-consultation summary for the attorney. Include:
1. Client name and contact info
2. Case type and jurisdiction
3. Key facts (timeline)
4. Potential claims or defenses
5. Statute of limitations analysis ({STATE} law)
6. Damages or relief sought
7. Recommended consultation preparation
Format as a professional memo. Under 500 words."""

    result = call_gemini(prompt, max_tokens=800)
    return result if result else f"Summary generation failed. Please review intake form manually."


# ===== STUB SERVICES (DEMO MODE) =====
def send_email(to_email: str, subject: str, body: str) -> bool:
    log.info(f"[EMAIL] To: {to_email} | Subject: {subject}")
    return True

def create_clio_contact(name: str, email: str, phone: str = "", notes: str = "") -> Optional[str]:
    log.info(f"[CLIO] Would create contact: {name}")
    return "demo-contact-id"

def notify_lawyer(subject: str, message: str):
    log.info(f"[NOTIFY] {subject}")


# ===== ENDPOINTS =====
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": f"{LAW_FIRM_NAME} — AI Client Intake Pipeline",
        "version": "3.0.0",
        "ai_provider": "Google Gemini (REST API — no SDK)",
        "endpoints": {
            "GET /demo": "Run demo",
            "POST /intake/email": "Process inquiry",
            "POST /intake/form": "Process intake form",
            "POST /calendly/webhook": "Handle booking"
        },
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
    
    log.info("Running demo...")
    analysis = analyze_inquiry(sample["body"], sample["from_name"])
    
    return jsonify({
        "demo": "Law Firm Intake AI Pipeline",
        "firm": LAW_FIRM_NAME,
        "ai_provider": "Google Gemini REST API",
        "input": sample,
        "ai_analysis": analysis,
        "time_saved": "15-30 minutes per inquiry",
        "setup_fee": "$800",
        "monthly_maintenance": "$99"
    })


@app.route("/intake/email", methods=["POST"])
def intake_email():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    from_email = data.get("from_email", "")
    from_name = data.get("from_name", "Potential Client")
    email_body = data.get("body", "")
    phone = data.get("phone", "")
    
    if not from_email or not email_body:
        return jsonify({"error": "Missing from_email or body"}), 400
    
    analysis = analyze_inquiry(email_body, from_name)
    send_email(from_email, f"Re: Your inquiry to {LAW_FIRM_NAME}", analysis.get("auto_reply_draft", ""))
    contact_id = create_clio_contact(from_name, from_email, phone, f"AI Analysis: {json.dumps(analysis)[:500]}")
    notify_lawyer(f"New Intake: {from_name} — {analysis.get('practice_area', 'Unknown')}", json.dumps(analysis, indent=2))
    
    return jsonify({"status": "success", "analysis": analysis, "contact_id": contact_id})


@app.route("/intake/form", methods=["POST"])
def intake_form():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    client_name = data.get("full_name", data.get("name", "Unknown"))
    client_email = data.get("email", "")
    
    summary = generate_case_summary(data)
    contact_id = create_clio_contact(client_name, client_email, data.get("phone", ""), summary)
    notify_lawyer(f"Intake Form: {client_name}", summary)
    send_email(client_email, f"Intake Received — {LAW_FIRM_NAME}", f"Dear {client_name},\n\nYour intake form has been received.\n\nSchedule: {CALENDLY_URL}\n\n{LAW_FIRM_NAME}\n{LAWYER_PHONE}")
    
    return jsonify({"status": "success", "contact_id": contact_id})


@app.route("/calendly/webhook", methods=["POST"])
def calendly_webhook():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    if data.get("event") == "invitee.created":
        invitee = data.get("payload", {}).get("invitee", {})
        notify_lawyer("Consultation Booked", f"{invitee.get('name')} booked for {invitee.get('start_time')}")
    
    return jsonify({"status": "received"})


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
