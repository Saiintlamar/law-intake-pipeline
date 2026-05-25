"""
OP3 API PIPE BUILDER — Law Firm Client Intake AI Pipeline
Apex production code. Groq API (free tier). No quota issues.
Stack: Flask + Groq REST API + structured logging
Deploy: Render (free tier)
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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LAW_FIRM_NAME = os.environ.get("LAW_FIRM_NAME", "Smith & Associates")
PRACTICE_AREAS = os.environ.get("PRACTICE_AREAS", "Personal Injury, Family Law")
LAWYER_EMAIL = os.environ.get("LAWYER_EMAIL", "attorney@example.com")
LAWYER_PHONE = os.environ.get("LAWYER_PHONE", "(555) 123-4567")
STATE = os.environ.get("STATE", "California")
INTAKE_FORM_URL = os.environ.get("INTAKE_FORM_URL", "https://forms.example.com/intake")
CALENDLY_URL = os.environ.get("CALENDLY_URL", "https://calendly.com/attorney/consultation")

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

log.info(f"Starting Law Firm Intake Pipeline — Groq Edition")
log.info(f"Firm: {LAW_FIRM_NAME}")
log.info(f"API Key present: {'Yes' if OPENAI_API_KEY else 'NO'}")
log.info(f"API Key prefix: {OPENAI_API_KEY[:8]}..." if OPENAI_API_KEY else "N/A")
log.info(f"Groq Model: {GROQ_MODEL}")

app = Flask(__name__)


def call_groq(prompt: str, max_tokens: int = 1000) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Calls Groq REST API. Returns (success, text, error_dict)."""
    
    if not OPENAI_API_KEY:
        return (False, None, {"status": "N/A", "body": "", "message": "API key not configured. Set OPENAI_API_KEY in Render."})
    
    if not OPENAI_API_KEY.startswith("gsk_"):
        return (False, None, {"status": "N/A", "body": "", "message": f"Key starts with '{OPENAI_API_KEY[:4]}...' — should start with 'gsk_'. Is this a Groq key?"})
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    
    try:
        log.info("Calling Groq API...")
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            timeout=30
        )
        
        log.info(f"Groq HTTP {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            log.info(f"Groq returned {len(text)} chars")
            return (True, text, None)
        
        return (False, None, {"status": response.status_code, "body": response.text[:500], "message": f"Groq HTTP {response.status_code}: {response.text[:200]}"})
    
    except requests.exceptions.Timeout:
        return (False, None, {"status": "TIMEOUT", "body": "", "message": "Groq API timed out."})
    except Exception as e:
        return (False, None, {"status": "EXCEPTION", "body": traceback.format_exc()[:300], "message": str(e)[:200]})


def analyze_inquiry(email_body: str, client_name: str = "Potential Client") -> Dict:
    prompt = f"""You are an AI intake specialist for {LAW_FIRM_NAME}, a law firm practicing in {PRACTICE_AREAS} in {STATE}.

A potential client named {client_name} sent this inquiry:
---
{email_body}
---

Analyze and return a JSON object with:
1. "practice_area": Which practice area? If unclear, say "Needs clarification"
2. "urgency": "High", "Medium", or "Low"
3. "statute_of_limitations_risk": "Yes" or "No"
4. "key_facts": 2-3 bullet points of legally relevant facts
5. "conflicts_check": "Pass" or "Flag - [reason]"
6. "recommended_action": What should the attorney do next?
7. "auto_reply_draft": A warm, professional email that thanks them, shows understanding, links to intake form ({INTAKE_FORM_URL}) and scheduling ({CALENDLY_URL}), signed by {LAW_FIRM_NAME}, phone {LAWYER_PHONE}

Return ONLY valid JSON. No markdown. No code blocks."""

    success, text, error = call_groq(prompt)
    
    if not success:
        return {
            "practice_area": f"API Error — HTTP {error.get('status', 'Unknown')}",
            "urgency": "Medium", "statute_of_limitations_risk": "No",
            "key_facts": [error.get("message", "Unknown error")[:200]],
            "conflicts_check": "Error",
            "recommended_action": f"Fix: {error.get('message', '')[:200]}",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We will respond shortly. For immediate assistance, call {LAWYER_PHONE}.",
            "_debug": error
        }
    
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:]) if len(lines) > 1 else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    if start >= 0 and end > start:
        cleaned = cleaned[start:end]
    
    try:
        parsed = json.loads(cleaned)
        log.info(f"Analysis: {parsed.get('practice_area')} | Urgency: {parsed.get('urgency')}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed. Raw: {text[:500]}")
        return {
            "practice_area": "Error — JSON Parse Failed",
            "urgency": "Medium", "statute_of_limitations_risk": "No",
            "key_facts": [f"JSON parse error: {str(e)[:150]}"],
            "conflicts_check": "Error",
            "recommended_action": "Manual review — AI returned non-JSON response",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We will respond shortly. Call {LAWYER_PHONE}.",
            "_debug_raw": text[:500]
        }


def generate_case_summary(intake_form_data: Dict) -> str:
    prompt = f"""You are a legal intake specialist at {LAW_FIRM_NAME}.
A client submitted this intake form:
---
{json.dumps(intake_form_data, indent=2)}
---
Create a pre-consultation summary. Include: client info, case type, key facts with timeline, potential claims/defenses, statute of limitations analysis ({STATE} law), damages sought, and recommended preparation. Under 500 words."""
    
    success, text, error = call_groq(prompt, max_tokens=800)
    return text if success else f"SUMMARY FAILED: {error.get('message', 'Unknown')}"


def send_email(to_email: str, subject: str, body: str) -> bool:
    log.info(f"[EMAIL] To: {to_email} | {subject}")
    return True

def create_clio_contact(name: str, email: str, phone: str = "", notes: str = "") -> Optional[str]:
    log.info(f"[CLIO] Created: {name}")
    return "demo-contact-id"

def notify_lawyer(subject: str, message: str):
    log.info(f"[NOTIFY] {subject}")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": f"{LAW_FIRM_NAME} — AI Client Intake Pipeline",
        "version": "5.0.0-groq",
        "ai_provider": f"Groq ({GROQ_MODEL})",
        "api_key_configured": bool(OPENAI_API_KEY),
        "api_key_valid_format": OPENAI_API_KEY.startswith("gsk_") if OPENAI_API_KEY else False,
        "status": "operational"
    })

@app.route("/demo", methods=["GET"])
def demo():
    sample = {
        "from_email": "john.smith@example.com",
        "from_name": "John Smith",
        "body": "I was shopping at FreshMart on Main Street last Tuesday evening around 7pm. There was a spill in the produce aisle with no warning sign. I slipped, fell, and hurt my back. I've been to the ER and have medical bills piling up. The store manager took a report but now their insurance company is ignoring my calls. What should I do?",
        "phone": "555-123-4567"
    }
    
    log.info("=== DEMO REQUEST ===")
    analysis = analyze_inquiry(sample["body"], sample["from_name"])
    is_success = "Error" not in str(analysis.get("practice_area", ""))
    log.info(f"Demo: {'SUCCESS' if is_success else 'FAILED'}")
    
    return jsonify({
        "demo": "Law Firm Intake AI Pipeline",
        "firm": LAW_FIRM_NAME,
        "ai_model": GROQ_MODEL,
        "ai_provider": "Groq",
        "success": is_success,
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
    from_name = data.get("from_name", "Client")
    email_body = data.get("body", "")
    phone = data.get("phone", "")
    if not from_email or not email_body:
        return jsonify({"error": "Missing from_email or body"}), 400
    
    log.info(f"Intake: {from_name}")
    analysis = analyze_inquiry(email_body, from_name)
    send_email(from_email, f"Re: Your inquiry", analysis.get("auto_reply_draft", ""))
    contact_id = create_clio_contact(from_name, from_email, phone, json.dumps(analysis)[:500])
    notify_lawyer(f"New Intake: {from_name}", json.dumps(analysis, indent=2))
    
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
        notify_lawyer("Booking", f"{invitee.get('name')} — {invitee.get('start_time')}")
    return jsonify({"status": "received"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "api_key_set": bool(OPENAI_API_KEY), "api_key_valid": OPENAI_API_KEY.startswith("gsk_") if OPENAI_API_KEY else False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
