# ------------------------ Importing Packages -------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
import re
import webbrowser
import urllib.parse
import smtplib
from email.message import EmailMessage
from extract_email_functions import extract_email_data
import json
import streamlit as st
# ------------------------ Configuring APIs -------------------------

api_key_gemini = st.secrets["GEMINI_API_KEY"]

# ------------------------ Chat LLM initialization -------------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=300,
    google_api_key=api_key_gemini,
    convert_system_message_to_human=True
)

# ------------------------ Friendly Email Body Generator -------------------------

def friendly_email_generator(subject: str, details: str, recipient: str, word_count: int) -> str:
    prompt = f"""Write a friendly and polite email with the subject: '{subject}'. 
The email should cover the following important details in a clear and approachable tone:
{details}

the length of email has to be {word_count}

The recipient's name is {recipient}.
Only return the email body and a warm closing from 'Ali Sina'. Use bullet points to organize the information for better readability.

Ensure:
- The tone is courteous, informal but respectful.
- The structure is easy to follow with bullet points.
- No placeholders like [Date] or [Name] are left.
- Do NOT include subject lines or any extra text — just the body and closing.

This email will be sent as-is, so complete all parts.
"""

    response = chat.invoke(prompt)
    return response.content.strip()

# ------------------------ Business Email Body Generator -------------------------

def business_email_generator(subject: str, details: str, recipient: str, word_count: int) -> str:
    prompt = f"""Compose a professional business email with the subject: '{subject}'.

Recipient's name: {recipient if recipient else 'Recipient'}
Sender's name: Ali Sina

Use the following input as raw details for the body:
\"\"\"
{details}
\"\"\"

the length of the email should be {word_count}

Structure the email into three paragraphs:
1. **Introduction** – briefly greet the recipient and state the purpose of the email.
2. **Body** – extract the key points from the raw input and present them as clear, well-formatted bullet points.
3. **Conclusion** – thank the recipient, offer assistance or next steps, and close politely.

Guidelines:
- Maintain a formal, respectful tone suitable for business.
- Use bullet points for clarity in the body.
- Do NOT include subject lines or placeholders like [Name] or [Date].
- Return ONLY the body of the email (starting after "Dear...") and the closing signature.

The output will be sent as-is, so ensure it is polished and complete. and 
"""
    response = chat.invoke(prompt)
    return response.content.strip()

# ----------------------- Loading Json -------------------------

def load_extracted_email_data(file_path: str = "extracted_email_data.json") -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("❌ Extracted email data not found. Please run the extraction step first.")

# ------------------------ Generating Function ------------------------------
def generate_email(query: str) -> str:
    # 🚨 Fail-safe: detect bad input from LLM instead of raw user instruction
    if "@" not in query.lower() or "email" not in query.lower():
        return "❌ The input doesn't look like a valid prompt. Please provide a clear user instruction."

    # Extract and save data
    extract_email_data(query)

    # Load extracted JSON safely
    result = load_extracted_email_data()
    print("✅ Loaded extracted data:", result)

    required_keys = ["receiver_name", "subject", "body", "tone"]
    if any(k not in result or not result[k] for k in required_keys):
        return "❌ Missing some fields in the extracted data. Please check your input prompt."

    receiver_name = result["receiver_name"]
    subject_refined = result["subject"]
    details = result["body"]
    tone = result["tone"]
    word_count = 200

    # Generate body
    if tone.lower() == "friendly":
        body = friendly_email_generator(subject_refined, details, receiver_name, word_count)
    elif tone.lower() == "business":
        body = business_email_generator(subject_refined, details, receiver_name, word_count)
    else:
        return "❌ Invalid email tone. Please use 'friendly' or 'business'."

    # Save body
    try:
        with open("email.txt", "w", encoding="utf-8") as f:
            f.write(body)
    except Exception as e:
        return f"⚠️ Failed to save email: {e}"

    return body


#print(generate_email(query= "write an email to my friend Haider to the email aleeghulami2024@gmail.com saying that I accept to attend his saturday evening party"))