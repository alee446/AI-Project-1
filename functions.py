
# ------------------------ Importing Packages -------------------------
from config import GEMINI_API_KEY, TAVILY_API_KEY, APP_PASSWORD, SENDER_EMAIL
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from langchain_community.tools.tavily_search.tool import TavilySearchResults
import re
import webbrowser
import urllib.parse
import smtplib
from email.message import EmailMessage

# ------------------------ Configuring APIs -------------------------

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
api_key_gemini = GEMINI_API_KEY
app_password = APP_PASSWORD

# ------------------------ Web Search Function-------------------------

tavily_tool = TavilySearchResults()

def web_search_func(query: str, message_log: list = None) -> str:
    raw_results = tavily_tool.run(query)

    top_contents = ""
    if isinstance(raw_results, list):
        top_contents = "\n\n".join([r.get("content", "") for r in raw_results[:3]])

    message_log = chat_memory.chat_memory.messages

    history_str = ""
    if message_log:
        history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in message_log[-4:]])

    prompt = f"""
You are an AI assistant summarizing online search results for the user.

User's recent conversation:
{history_str}

Here are the search results:
{top_contents}

Please write a short, clear, helpful response based on this.
"""
    response = chat.invoke(prompt)
    return response.content

# ------------------------ Open App Function -------------------------

APP_PATHS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "microsoft browser": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "opera browser": r"C:\Users\Ali Sena\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Opera Browser",
    "pycharm": r"C:\Program Files\JetBrains\PyCharm Community Edition 2024.2.3\bin\pycharm64.exe",
    "v s code": r"C:\Users\Ali Sena\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd",
    "file explorer": r"C:\Windows\explorer.exe",
    "whatsapp": r"C:\Program Files\WindowsApps\5319275A.WhatsAppDesktop_2.2526.2.0_x64__cv1g1gvanyjgm\WhatsApp.exe",
    "calculator": r"start calc",
    "acrobat reader": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Acrobat Reader",
    "microsoft word": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Word 2016",
    "powerpoint": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\PowerPoint 2016",
    "excel": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Excel",
    "calendar": r"start outlookcal:",
    "media player": r"start mswindowsvideo:",
    "map": r"start wmplayer"
}

def open_app(app_name: str) -> str:
    app_name = app_name.lower().strip()
    path = APP_PATHS.get(app_name)

    if not path:
        return f"❌ App '{app_name}' not configured yet."

    try:
        # If it's a system command like "start calc"
        if path.startswith("start "):
            os.system(path)
            return f"✅ Opened {app_name.title()} via system command."

        else:
            os.startfile(path)
            return f"✅ Opened {app_name.title()} via startfile command."

    except Exception as e:
        return f"❌ Failed to open {app_name}: {str(e)}"

# ------------------------ Open Website Function -------------------------

import os

# Define a simple mapping
WEBSITE_MAP = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://www.twitter.com",
    "instagram": "https://www.instagram.com",
    "chatgpt": "https://www.openai.com",
    "github": "https://www.github.com",
    "gmail": "https://mail.google.com",
    "map": "https://www.google.com/maps",
    "calendar": "https://calendar.google.com",
}

def open_website_func(website):
    website = website.lower()
    for name in WEBSITE_MAP:
        if name in website:
            os.system(f'start {WEBSITE_MAP[name]}')
            return f"{website} opened successfuly"
    print("Sorry, I don't recognize that website.")

# ------------------------ Search Google Function -------------------------

def search_google(query: str) -> str:
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        webbrowser.open(url)
        return f"Searching Google for: {query}"
    except Exception as e:
        return f"Failed to search Google: {str(e)}"

# ------------------------ Search YouTube Function -------------------------

def search_youtube(query: str) -> str:
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.youtube.com/results?search_query={encoded_query}"
        webbrowser.open(url)
        return f"Searching YouTube for: {query}"
    except Exception as e:
        return f"Failed to search YouTube: {str(e)}"

# ------------------------ Send Email Function -------------------------

# ------------------------ Send Email Function -------------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=2048,
    google_api_key=api_key_gemini,
    convert_system_message_to_human=True
)

def refine_subject(subject, recepient: None):
    prompt = f"""
Take the following input and rewrite it into a clear, professional subject line for an email. 
Only return the subject itself — no extra explanation, no placeholders, and no formatting outside of the subject. 
This subject will be used directly in a real email. 
Make sure it is complete, polished, and appropriate. 
The sender's name is Ali Sina. use only the pronoun for myself: I, me, my etc
recepient: {recepient}
Original subject: {subject}
"""

    result = chat.invoke(prompt)
    return result.content.strip()

def generate_email_body(subject: str, details: str, recepient: str) -> str:
    prompt = f"""Write a professional email with the subject: '{subject}'. 
The email should include the following important details: {details}
The recepient's name is {recepient}
You are expected to generate ONLY the BODY of this email and the closing section. 
Make it clear, polite, and well-formatted. 
Only return the email body — do not include any extra text or instructions. 
If the recipient's name is not specified, use 'Recipient'. 
The sender's name is Ali Sina. 
Do not leave any placeholders — this email will be sent as-is, so complete all parts.
Texts like : [Date of Absence - e.g., May 15th] is strictly prohibited 
"""

    response = chat.invoke(prompt)
    return response.content.strip()

def send_email(to=None, receiver_name=None, subject=None, details=None):
    if to is None:
        to = input("Enter the receiver's email: ")
    if receiver_name is None:
        receiver_name = input("Enter the recipient's name: ")
    if subject is None:
        subject = input("Enter the subject: ")
    if details is None:
        details = input("Enter any important details: ")

    try:
        subject_refined = refine_subject(subject, receiver_name)
        body = generate_email_body(subject_refined, details, receiver_name)

        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to
        msg["Subject"] = subject_refined.replace('\n', ' ').replace('\r', ' ').strip()
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        return f"✅ Email sent to {to} with subject: {subject_refined}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"


