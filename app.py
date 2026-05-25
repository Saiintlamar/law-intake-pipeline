"""
OP3 API PIPE BUILDER — Law Firm Client Intake AI Pipeline
Fixed: Lazy-load OpenAI client to avoid startup crash.
"""

import os
import logging
from typing import Dict, Optional
import json

from flask import Flask, request, jsonify
import requests
from openai import OpenAI

# ===== CONFIGURATION =====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LAW_FIRM_NAME = os.environ.get("LAW_FIRM_NAME", "Smith & Associates")
PRACTICE_AREAS = os.environ.get("PRACTICE_AREAS", "Personal Injury, Family Law")
LAWYER_EMAIL = os.environ.get("LAWYER_EMAIL", "attorney@example.com")
LAWYER_PHONE = os.environ.get("LAWYER_PHONE", "(555) 123-4567")
STATE = os.environ.get("STATE", "California")
INTAKE_FORM_URL = os.environ.get("INTAKE_FORM_URL", "https://forms.example.com/intake")
CALENDLY_URL = os.environ.get("CALENDLY_URL", "https://calendly.com/attorney/consultation")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)


def get_client():
    """Lazy-load OpenAI client — only when needed."""
    return OpenAI(api_key=OPENAI_API_KEY)


def analyze_inquiry(email_body: str, client_name: str = "Potential Client") -> Dict:
    prompt = f"""
You are an AI intake specialist for {LAW_FIRM_NAME}, a law firm practicing in {PRACTICE_AREAS} in {STATE}.
A potential client named {client_name} sent this inquiry:
---
{email_body}
---
Analyze this inquiry and return a JSON object with:
1. practice_area: Which practice area? If unclear, say "Needs clarification"
2. urgency: "High", "Medium", or "Low"
3. statute_of_limitations_risk: "Yes" or "No"
4. key_facts: 2-3 bullet points
5. conflicts_check: "Pass" or "Flag - [reason]"
6. recommended_action: What should the attorney do next?
7. auto_reply_draft: A warm, professional email reply that thanks them, shows understanding, explains next steps, links to intake form ({INTAKE_FORM_URL}) and scheduling ({CALENDLY_URL}), signed by {LAW_FIRM_NAME}, phone {LAWYER_PHONE}
Return ONLY valid JSON. No markdown. No code blocks.
"""
    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        result = response.choices[0].message.content
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except Exception as e:
        log.error(f"AI analysis failed: {e}")
        return {
            "practice_area": "Error", "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": ["Analysis failed"], "conflicts_check": "Error",
            "recommended_action": "Manual review",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}."
        }


def generate_case_summary(intake_form_data: Dict) -> str:
    prompt = f"""
You are a legal intake specialist at {LAW_FIRM_NAME}.
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
8. Conflict check result
Format as a professional memo. Under 500 words.
"""
    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"Case summary failed: {e}")
        return "Summary generation failed. Manual review required."


def send_email(to_email: str, subject: str, body: str) -> bool:
    log.info(f"[DEMO] Would send email to {to_email}: {subject}")
    return True


def create_clio_contact(name: str, email: str, phone: str = "", notes: str = "") -> Optional[str]:
    log.info(f"[DEMO] Would create Clio contact: {name}")
    return "demo-contact-id"


def notify_lawyer(subject: str, message: str):
    send_email(LAWYER_EMAIL, subject, message)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": f"{LAW_FIRM_NAME} — AI Client Intake Pipeline",
        "version": "1.0.0",
        "endpoints": {
            "POST /intake/email": "Process client email inquiry",
            "POST /intake/form": "Process submitted intake form",
            "POST /calendly/webhook": "Handle consultation booking",
            "GET /demo": "Run demo with sample data"
        },
        "status": "operational"
    })


@app.route("/intake/email", methods=["POST"])
def intake_email():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    from_email = data.get("from_email", "")
    from_name = data.get("from_name", "Potential Client")
    email_body = data.get("body", "")
    phone = data.get("phone", "")
    
    if not from_email or not email_body:
        return jsonify({"error": "Missing from_email or body"}), 400
    
    log.info(f"New intake: {from_name} ({from_email})")
    analysis = analyze_inquiry(email_body, from_name)
    
    auto_reply = analysis.get("auto_reply_draft", "Thank you for contacting us.")
    send_email(from_email, f"Re: Your inquiry to {LAW_FIRM_NAME}", auto_reply)
    
    notes = f"Initial inquiry: {email_body[:300]}...\n\nAI Analysis:\nPractice Area: {analysis.get('practice_area')}\nUrgency: {analysis.get('urgency')}\nKey Facts: {', '.join(analysis.get('key_facts', []))}\nSOL Risk: {analysis.get('statute_of_limitations_risk')}\nConflicts: {analysis.get('conflicts_check')}"
    contact_id = create_clio_contact(from_name, from_email, phone, notes)
    
    notify_lawyer(
        f"New Client Intake: {from_name} — {analysis.get('practice_area', 'Unknown')}",
        f"NEW INTAKE\nName: {from_name}\nEmail: {from_email}\nPhone: {phone}\nPractice Area: {analysis.get('practice_area')}\nUrgency: {analysis.get('urgency')}\nSOL Risk: {analysis.get('statute_of_limitations_risk')}\n\nKey Facts:\n" + "\n".join(f"- {f}" for f in analysis.get('key_facts', [])) + f"\n\nAction: {analysis.get('recommended_action')}\n\nAuto-reply sent: Yes\nClio Contact: {contact_id}"
    )
    
    return jsonify({"status": "success", "analysis": analysis, "contact_id": contact_id, "auto_reply_sent": True})


@app.route("/intake/form", methods=["POST"])
def intake_form():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    client_name = data.get("full_name", data.get("name", "Unknown"))
    client_email = data.get("email", "")
    
    log.info(f"Intake form: {client_name}")
    summary = generate_case_summary(data)
    contact_id = create_clio_contact(client_name, client_email, data.get("phone", ""), f"Intake form completed.\n\nAI Case Summary:\n{summary}")
    
    notify_lawyer(
        f"Intake Form Completed: {client_name}",
        f"INTAKE FORM SUBMITTED\nClient: {client_name}\nEmail: {client_email}\n\nCASE SUMMARY:\n{summary}\n\nClio Contact: {contact_id}\nSchedule: {CALENDLY_URL}"
    )
    
    send_email(client_email, f"Your intake form has been received — {LAW_FIRM_NAME}",
        f"Dear {client_name},\n\nThank you for completing our intake form. Your case information has been received.\n\nSchedule your consultation: {CALENDLY_URL}\n\nQuestions? Call {LAWYER_PHONE}.\n\nBest regards,\n{LAW_FIRM_NAME}")
    
    return jsonify({"status": "success", "contact_id": contact_id, "next_step": "Consultation scheduling"})


@app.route("/calendly/webhook", methods=["POST"])
def calendly_webhook():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    event = data.get("event", "")
    if event == "invitee.created":
        invitee = data.get("payload", {}).get("invitee", {})
        client_name = invitee.get("name", "Unknown")
        client_email = invitee.get("email", "")
        event_time = invitee.get("start_time", "")
        
        log.info(f"Consultation booked: {client_name} on {event_time}")
        notify_lawyer(f"Consultation Booked: {client_name}", f"CONSULTATION BOOKED\nClient: {client_name}\nEmail: {client_email}\nWhen: {event_time}\n\nReview pre-consultation summary in Clio.")
        send_email(client_email, f"Consultation Confirmed — {LAW_FIRM_NAME}", f"Dear {client_name},\n\nYour consultation is confirmed for {event_time}.\n\nPlease have relevant documents ready. To reschedule, use the link in your confirmation email.\n\nWe look forward to speaking with you.\n\n{LAW_FIRM_NAME}\n{LAWYER_PHONE}")
    
    return jsonify({"status": "received"})


@app.route("/demo", methods=["GET"])
def demo():
    sample = {
        "from_email": "john.smith@example.com",
        "from_name": "John Smith",
        "subject": "Slipped and fell at grocery store",
        "body": "I was shopping at FreshMart on Main Street last Tuesday around 7pm. There was a spill in the produce aisle with no warning sign. I slipped, fell, and hurt my back. I've been to the ER and have medical bills piling up. The store manager took a report but their insurance company is ignoring my calls. What should I do?",
        "phone": "555-123-4567"
    }
    analysis = analyze_inquiry(sample["body"], sample["from_name"])
    
    return jsonify({
        "demo": "Law Firm Intake AI Pipeline",
        "firm": LAW_FIRM_NAME,
        "input": sample,
        "ai_analysis": analysis,
        "what_happens_next": [
            "1. Auto-reply sent to client with intake form link",
            "2. Contact created in Clio with AI analysis notes",
            "3. Lawyer receives notification with case summary",
            "4. Client fills intake form -> case summary generated",
            "5. Consultation booked via Calendly",
            "6. Lawyer walks into consultation fully briefed"
        ],
        "time_saved": "15-30 minutes per inquiry",
        "setup_fee": "$800",
        "monthly_maintenance": "$99"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Law Firm Intake AI Pipeline — {LAW_FIRM_NAME}")
    print(f"Port: {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
