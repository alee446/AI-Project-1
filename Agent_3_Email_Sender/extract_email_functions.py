import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage
from config import GEMINI_API_KEY
from pathlib import Path
# ------------------------ Configuring APIs -------------------------

api_key_gemini = GEMINI_API_KEY

# ------------------------ Chat LLM initialization -------------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=300,
    google_api_key=api_key_gemini,
    convert_system_message_to_human=True
)
# -------------------------------------
# 🔧 Extract JSON from response text
# -------------------------------------
def extract_json_from_text(text: str) -> str | None:
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"({.*})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# -------------------------------------
# 🧹 Clean greetings and sign-offs from body
# -------------------------------------
def clean_email_body(body: str) -> str:
    lines = body.strip().splitlines()

    # Remove greeting
    if lines and re.match(r"^(hi|dear)\b", lines[0].strip(), re.IGNORECASE):
        lines = lines[1:]

    # Remove sign-off
    if lines and re.search(r"(regards|sincerely|thanks|thank you|best)", lines[-1], re.IGNORECASE):
        lines = lines[:-1]

    return "\n".join(line.strip() for line in lines if line.strip())

# -------------------------------------
# 🧠 Build the prompt to send to Gemini
# -------------------------------------
def build_extraction_prompt(user_prompt: str) -> str:
    return f"""
You are a JSON-generating assistant that extracts structured email data from user instructions.

Extract ONLY the following fields:
- "receiver_name"
- "receiver_email"
- "subject" (can be inferred)
- "body" (short, core message only — ❌ no greeting, ❌ no sign-off)
- "tone" ("friendly" or "business") YOU ARE STRICTLY SAID TO USE ONLY "friendly" or "business" only lower cased

✅ Return output as **pure JSON only** — no markdown, no commentary, no greeting or closing.
❌ DO NOT generate a full email. DO NOT include lines like "Hi..." or "Best regards".

---

✅ Good Example:
Instruction:
"Send a friendly thank-you email to Sarah at sarah@example.com for yesterday’s demo."

Response:
{{
  "receiver_name": "Sarah",
  "receiver_email": "sarah@example.com",
  "subject": "Thank you for the product demo",
  "body": "Thank you for the demo yesterday. I really appreciate your time.",
  "tone": "friendly"
}}

---

❌ Bad Example:
"Hi Sarah,\nThank you for the demo...\nBest regards,\n[Your Name]"

---

If any required info is missing, DO NOT return JSON. Just ask for the missing detail.

Now extract from:
\"\"\"{user_prompt}\"\"\"
"""

# -------------------------------------
# 🧩 Main: Extract email data using Gemini
# -------------------------------------


def extract_email_data(user_prompt: str) -> dict | str:
    prompt = build_extraction_prompt(user_prompt)
    response = chat.invoke([HumanMessage(content=prompt)])
    response_text = response.content

    json_text = extract_json_from_text(response_text)
    if not json_text:
        return response_text  # Not valid JSON — probably missing info

    try:
        email_data = json.loads(json_text)
        email_data["body"] = clean_email_body(email_data["body"])

        # ✅ Save to a file
        save_path = Path("extracted_email_data.json")
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(email_data, f, indent=4, ensure_ascii=False)

        return email_data
    except json.JSONDecodeError:
        return f"❌ Gemini returned malformed JSON:\n{response_text}"

