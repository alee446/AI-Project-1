# ----------------------- Importing Necessary Packages -----------------
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------ Configuring APIs -------------------------

api_key_gemini = st.secrets["GEMINI_API_KEY"]

# ------------------------ Chat LLM initialization -------------------------

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=300,
    google_api_key=api_key_gemini,
    convert_system_message_to_human=True
)

# ------------------- Editing Email Function ---------------------
def editor(user_input):
    # Reading the email generated
    try:
        with open("email.txt", "r", encoding="utf-8") as f:
            email_content = f.read()
        print("✅ Email content loaded successfully.")
    except FileNotFoundError:
        return "❌ No email file found. Please generate one first."

    # Making the Prompt
    prompt = f"""
You are an email editor who edits email content based on the user demand. 
DOs:
- change the email based on how the user wants. 
- make changes to the point of the user desire. 
DON'Ts:
- do not generate a complete new email UNLESS you are told do. 
- don't over complicate the changes and make sure it is suitable based on the rest of the content
- don't write any extra text or commentary other than the changed email. 
- don't write like (okay, here you go the changed email or do you want further changes )

Here is the email content:
{email_content}

Here is the prompt to change according to:
{user_input}
"""
    # Generating Response
    try:
        result = chat.invoke(prompt)
        if not result or not result.content:
            return "⚠️ The model returned no content."

        new_email = result.content.strip()

        # ✅ Save updated email back to email.txt
        with open("email.txt", "w", encoding="utf-8") as f:
            f.write(new_email)
        print("💾 Updated email saved to email.txt.")

        return new_email

    except Exception as e:
        return f"❌ Error: {e}"
