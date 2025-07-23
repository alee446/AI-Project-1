# ------------------ Importing Necessary Packages ----------------------
import smtplib
from email.message import EmailMessage
import json
import streamlit as st
SENDER_EMAIL = st.secrets["SENDER_EMAIL"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]
def load_extracted_email_data(file_path: str = "extracted_email_data.json") -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("❌ Email data JSON not found. Please run the extraction step first.")

def send_email(_: str = "") -> str:
    try:
        # ✅ Load receiver email and subject from JSON
        data = load_extracted_email_data()
        receiver_email = data.get("receiver_email")
        subject = data.get("subject")

        if not receiver_email or not subject:
            return "❌ Missing receiver email or subject in extracted data."

        # ✅ Read the latest email body (from edited/generated email)
        with open("email.txt", "r", encoding="utf-8") as f:
            email_body = f.read()

        # Create the email
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.set_content(email_body)

        # Send via SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print("✅ Email sent successfully.")
        return "✅ Email sent successfully."

    except Exception as e:
        return f"❌ Failed to send email: {e}"


