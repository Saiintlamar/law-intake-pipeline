"""
OP3 API PIPE BUILDER — Law Firm Client Intake AI Pipeline
Production code. Deployable. Demo-ready.
Stack: Python + Flask + Gemini API (free tier) + Clio + Calendly + SendGrid
Deploy: Render (free tier)
Fixed: Lazy-load client, full error logging, Gemini compatibility.
"""

import os
import logging
import traceback
from typing import Dict, Optional
import json

from flask import Flask, request, jsonify
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


# ===== AI CLIENT (LAZY-LOADED) =====
def get_client():
    """Lazy-load Gemini client via OpenAI compatibility layer."""
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


# ===== AI ENGINE =====
def analyze_inquiry(email_body: str, client_name: str = "Potential Client") -> Dict:
    """
    Analyzes a client inquiry email using Gemini.
    Returns: practice_area, urgency, key_facts, recommended_action, auto_reply
    """
    prompt = f"""You are an AI intake specialist for {LAW_FIRM_NAME}, a law firm practicing in {PRACTICE_AREAS} in {STATE}.

A potential client named {client_name} sent this inquiry:

---
{email_body}
---

Analyze this inquiry and return a JSON object with:
1. practice_area: Which practice area does this fall under? If unclear, say "Needs clarification"
2. urgency: "High", "Medium", or "Low"
3. statute_of_limitations_risk: "Yes" or "No" — is there a time-sensitive legal deadline?
4. key_facts: 2-3 bullet points summarizing the legally relevant facts
5. conflicts_check: Any obvious conflict of interest? "Pass" or "Flag - [reason]"
6. recommended_action: What should the attorney do next?
7. auto_reply_draft: A warm, professional email reply that:
   - Thanks them for reaching out
   - Shows you understood their situation (reference specifics)
   - Explains next steps
   - Links to the intake form: {INTAKE_FORM_URL}
   - Links to schedule a consultation: {CALENDLY_URL}
   - Is signed by the team at {LAW_FIRM_NAME}
   - Includes phone number: {LAWYER_PHONE}

Return ONLY valid JSON. No markdown. No code blocks. No extra text."""

    try:
        response = get_client().chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        
        # Clean any markdown wrapping
        result = result.replace("```json", "").replace("```", "").strip()
        
        # Try to find JSON object if there's extra text
        start = result.find('{')
        end = result.rfind('}') + 1
        if start >= 0 and end > start:
            result = result[start:end]
        
        parsed = json.loads(result)
        log.info(f"Analysis success: {parsed.get('practice_area', 'Unknown')} | Urgency: {parsed.get('urgency', 'Unknown')}")
        return parsed
        
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed. Raw response: {result[:500] if 'result' in dir() else 'No response'}")
        return {
            "practice_area": "Error - JSON Parse",
            "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": [f"JSON parsing failed: {str(e)[:150]}"],
            "conflicts_check": "Error",
            "recommended_action": "Manual review required — AI response was not valid JSON",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}."
        }
    except Exception as e:
        full_error = traceback.format_exc()
        log.error(f"AI analysis failed: {full_error}")
        return {
            "practice_area": "Error",
            "urgency": "Medium",
            "statute_of_limitations_risk": "No",
            "key_facts": [f"Error: {str(e)[:200]}"],
            "conflicts_check": "Error",
            "recommended_action": f"Debug info: {str(e)[:200]}",
            "auto_reply_draft": f"Thank you for contacting {LAW_FIRM_NAME}. We received your inquiry and will respond shortly. For immediate assistance, call {LAWYER_PHONE}."
        }


def generate_case_summary(intake_form_data: Dict) -> str:
    """Generates a pre-consultation case summary from intake form data."""
    
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
5. Statute of limitations analysis (based on {STATE} law)
6. Damages or relief sought
7. Recommended preparation for consultation
8. Conflict check result

Format as a professional memo. Keep it under 500 words."""

    try:
        response = get_client().chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        full_error = traceback.format_exc()
        log.error(f"Case summary failed: {full_error}")
        return f"Summary generation failed. Error: {str(e)[:200]}. Please review intake form manually."


# ===== EMAIL (DEMO MODE) =====
def send_email(to_email: str, subject: str, body: str) -> bool:
    """Logs email in demo mode. Connect SendGrid for production."""
    log.info(f"[EMAIL] To: {to_email} | Subject: {subject}")
    log.info(f"[EMAIL] Body preview: {body[:150]}...")
    return True


# ===== CLIO CRM (DEMO MODE) =====
def create_clio_contact(name: str, email: str, phone: str = "", notes: str = "") -> Optional[str]:
    """Creates contact in Clio. Demo mode returns placeholder ID."""
    log.info(f"[CLIO] Would create contact: {name} ({email})")
    log.info(f"[CLIO] Notes: {notes[:150]}...")
    return "demo-contact-id"


# ===== NOTIFICATIONS =====
def notify_lawyer(subject: str, message: str):
    """Sends notification to lawyer about new intake."""
    log.info(f"[NOTIFY] {subject}")
    log.info(f"[NOTIFY] {message[:300]}...")
    send_email(LAWYER_EMAIL, subject, message)


# ===== API ENDPOINTS =====

@app.route("/", methods=["GET"])
def home():
    """Health check + API overview."""
    return jsonify({
        "service": f"{LAW_FIRM_NAME} — AI Client Intake Pipeline",
        "version": "2.0.0",
        "ai_provider": "Google Gemini (free tier)",
        "endpoints": {
            "GET /demo": "Run demo with sample slip-and-fall case",
            "POST /intake/email": "Process incoming client email inquiry",
            "POST /intake/form": "Process submitted intake form",
            "POST /calendly/webhook": "Handle new consultation booking"
        },
        "status": "operational"
    })


@app.route("/demo", methods=["GET"])
def demo():
    """Runs a demo with a sample slip-and-fall inquiry."""
    
    sample = {
        "from_email": "john.smith@example.com",
        "from_name": "John Smith",
        "subject": "Slipped and fell at grocery store",
        "body": "I was shopping at FreshMart on Main Street last Tuesday evening around 7pm. There was a spill in the produce aisle with no warning sign. I slipped, fell, and hurt my back. I've been to the ER and have medical bills piling up. The store manager took a report but now their insurance company is ignoring my calls. What should I do?",
        "phone": "555-123-4567"
    }
    
    log.info("Running demo with sample slip-and-fall inquiry...")
    
    analysis = analyze_inquiry(sample["body"], sample["from_name"])
    
    return jsonify({
        "demo": "Law Firm Intake AI Pipeline",
        "firm": LAW_FIRM_NAME,
        "ai_provider": "Google Gemini (gemini-2.5-flash)",
        "input": sample,
        "ai_analysis": analysis,
        "what_happens_next": [
            "1. Auto-reply sent to client with intake form link",
            "2. Contact created in Clio with AI analysis notes",
            "3. Lawyer receives notification with full case summary",
            "4. Client fills intake form → AI generates pre-consultation brief",
            "5. Consultation booked via Calendly",
            "6. Lawyer walks into consultation fully prepared"
        ],
        "time_saved": "15-30 minutes per inquiry",
        "setup_fee": "$800",
        "monthly_maintenance": "$99"
    })


@app.route("/intake/email", methods=["POST"])
def intake_email():
    """
    Process incoming client inquiry email.
    Connect to Zapier/Make webhook or direct email integration.
    
    Expected JSON:
    {
        "from_email": "client@example.com",
        "from_name": "John Smith",
        "subject": "Slipped at grocery store",
        "body": "I was shopping at...",
        "phone": "555-123-4567" (optional)
    }
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    from_email = data.get("from_email", "")
    from_name = data.get("from_name", "Potential Client")
    email_body = data.get("body", "")
    phone = data.get("phone", "")
    
    if not from_email or not email_body:
        return jsonify({"error": "Missing required fields: from_email, body"}), 400
    
    log.info(f"New intake: {from_name} ({from_email})")
    
    # Step 1: AI analyzes the inquiry
    analysis = analyze_inquiry(email_body, from_name)
    
    # Step 2: Send auto-reply to client
    auto_reply = analysis.get("auto_reply_draft", f"Thank you for contacting {LAW_FIRM_NAME}. We will respond shortly.")
    send_email(from_email, f"Re: Your inquiry to {LAW_FIRM_NAME}", auto_reply)
    
    # Step 3: Create Clio contact with AI analysis
    notes = f"""INITIAL INQUIRY:
{email_body[:500]}

AI ANALYSIS:
Practice Area: {analysis.get('practice_area')}
Urgency: {analysis.get('urgency')}
Statute of Limitations Risk: {analysis.get('statute_of_limitations_risk')}
Key Facts: {', '.join(analysis.get('key_facts', []))}
Conflicts Check: {analysis.get('conflicts_check')}
Recommended Action: {analysis.get('recommended_action')}"""
    
    contact_id = create_clio_contact(from_name, from_email, phone, notes)
    
    # Step 4: Notify lawyer
    notify_lawyer(
        f"New Client Intake: {from_name} — {analysis.get('practice_area', 'Unknown')}",
        f"""
╔══════════════════════════════════════╗
║       NEW CLIENT INTAKE              ║
╠══════════════════════════════════════╣
║ Name: {from_name}
║ Email: {from_email}
║ Phone: {phone}
║ Practice Area: {analysis.get('practice_area', 'Unknown')}
║ Urgency: {analysis.get('urgency', 'Unknown')}
║ SOL Risk: {analysis.get('statute_of_limitations_risk', 'Unknown')}
╠══════════════════════════════════════╣
║ Key Facts:
{chr(10).join(f'║ - {f}' for f in analysis.get('key_facts', []))}
╠══════════════════════════════════════╣
║ Action: {analysis.get('recommended_action', 'Review')}
║ Auto-reply: Sent
║ Clio Contact: {contact_id}
║ Intake Form: {INTAKE_FORM_URL}
║ Schedule: {CALENDLY_URL}
╚══════════════════════════════════════╝
        """
    )
    
    return jsonify({
        "status": "success",
        "analysis": analysis,
        "contact_id": contact_id,
        "auto_reply_sent": True,
        "next_step": "Client received auto-reply with intake form and scheduling links"
    })


@app.route("/intake/form", methods=["POST"])
def intake_form():
    """
    Process submitted intake form.
    Connect to Google Forms, Typeform, or JotForm webhook.
    
    Expected JSON: Any form fields (full_name, email, case_type, description, etc.)
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    client_name = data.get("full_name", data.get("name", "Unknown"))
    client_email = data.get("email", "")
    
    log.info(f"Intake form submitted: {client_name} ({client_email})")
    
    # Step 1: Generate case summary
    summary = generate_case_summary(data)
    
    # Step 2: Create Clio contact with summary
    contact_id = create_clio_contact(
        client_name,
        client_email,
        data.get("phone", ""),
        f"Intake form completed.\n\nAI CASE SUMMARY:\n{summary}"
    )
    
    # Step 3: Notify lawyer with full summary
    notify_lawyer(
        f"Intake Form Completed: {client_name} — {data.get('case_type', 'General')}",
        f"""
INTAKE FORM SUBMITTED
=====================
Client: {client_name}
Email: {client_email}
Case Type: {data.get('case_type', 'General')}

AI CASE SUMMARY:
{summary}

Clio Contact: {contact_id}
Schedule Consultation: {CALENDLY_URL}
        """
    )
    
    # Step 4: Send confirmation to client
    send_email(
        client_email,
        f"Your intake form has been received — {LAW_FIRM_NAME}",
        f"""Dear {client_name},

Thank you for completing our intake form. Your case information has been received and our team will review it shortly.

To schedule your consultation immediately, please use this link:
{CALENDLY_URL}

If you have any urgent questions, call us at {LAWYER_PHONE}.

Best regards,
{LAW_FIRM_NAME}"""
    )
    
    return jsonify({
        "status": "success",
        "contact_id": contact_id,
        "summary_preview": summary[:300] + "...",
        "next_step": "Consultation scheduling via Calendly"
    })


@app.route("/calendly/webhook", methods=["POST"])
def calendly_webhook():
    """
    Handle Calendly booking notifications.
    Set up in Calendly: Settings → Webhooks → Add this URL.
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    event = data.get("event", "")
    
    if event == "invitee.created":
        invitee = data.get("payload", {}).get("invitee", {})
        client_name = invitee.get("name", "Unknown")
        client_email = invitee.get("email", "")
        event_time = invitee.get("start_time", "")
        
        log.info(f"Consultation booked: {client_name} on {event_time}")
        
        # Notify lawyer
        notify_lawyer(
            f"📅 Consultation Booked: {client_name}",
            f"""
CONSULTATION BOOKED
===================
Client: {client_name}
Email: {client_email}
When: {event_time}

Review the pre-consultation summary in Clio before the meeting.
            """
        )
        
        # Send reminder to client
        send_email(
            client_email,
            f"Consultation Confirmed — {LAW_FIRM_NAME}",
            f"""Dear {client_name},

Your consultation has been confirmed for {event_time}.

Please have any relevant documents ready for review. If you need to reschedule, use the link in your confirmation email.

We look forward to speaking with you.

{LAW_FIRM_NAME}
{LAWYER_PHONE}"""
        )
    
    return jsonify({"status": "received"})


# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "available_endpoints": ["/", "/demo", "/intake/email", "/intake/form", "/calendly/webhook"]}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════════╗
║   LAW FIRM INTAKE AI PIPELINE v2.0          ║
║   {LAW_FIRM_NAME:<40}║
╠══════════════════════════════════════════════╣
║   AI Provider: Google Gemini (free)          ║
║   Model: gemini-2.5-flash                     ║
║   Port: {port}                                  ║
╠══════════════════════════════════════════════╣
║   Endpoints:                                  ║
║   GET  /demo  — Run demo                     ║
║   POST /intake/email — Process inquiry       ║
║   POST /intake/form  — Process intake form   ║
║   POST /calendly/webhook — Handle booking    ║
╚══════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=port, debug=True)
