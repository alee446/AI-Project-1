import streamlit as st

# ---------- Page Configuration ----------
st.set_page_config(page_title="AI Agent Suite", page_icon="🧠", layout="centered")

# ---------- Custom Divider (Defined Once) ----------
def rainbow_divider():
    st.markdown("""
    <hr style='height: 4px;
                background: linear-gradient(to right, #ff6ec4, #7873f5, #4ade80, #facc15, #fb923c, #f43f5e);
                border: none;
                border-radius: 6px;
                margin: 30px 0;'>
    """, unsafe_allow_html=True)

# ---------- Title and Subtitle ----------
st.title("ALVEO AI")
st.subheader("Where Smart Agents Meet Real-World Tasks.")

# ---------- Introduction ----------
st.markdown("""
Welcome to **Alveo AI** – your suite of intelligent agents.  
A unified collection of powerful AI tools designed to streamline your daily tasks. From managing documents and staying ahead of the news to automating emails and getting real-time guidance — everything you need, in one intelligent workspace.
""")

# ---------- Agent Overview ----------
rainbow_divider()
st.markdown("### Available AI Agents")
st.markdown("""
**Nexa AI**  
Ask questions and extract insights from your uploaded documents with intelligent document understanding.

**News Flow AI**  
Get real-time summaries and stay informed with the latest trending news powered by AI.

**AutoMail AI**  
Generate, summarize, and manage your emails effortlessly with automated email intelligence.

**Guide Chatbot**  
Your intelligent assistant for instant help, guidance, and answers—anytime you need them.
""")

# ---------- Navigation Instructions ----------
rainbow_divider()
st.markdown("### Navigation")
st.markdown("""
Use the sidebar to access different AI agents.

You also have access to a **Guide Chatbot**, which can assist you throughout the application.  
For any inquiries about the app or AI agents, simply connect with the chatbot.

Each tool is designed to streamline tasks, enhance productivity, and support your daily workflows—whether you're working, learning, or exploring.
""")

# ---------- About the Developer ----------
rainbow_divider()
st.markdown("""
<h4 style='text-align: center;'>About the Developer</h4>
<p style='text-align: center; font-size: 15px;'>
<strong>Ali Seena</strong> is a self-taught AI Engineer and product developer with a passion for creating intelligent tools that empower users, automate tasks, and enhance productivity.
With a focus on AI Agent Engineering, he builds future-ready applications that combine innovation, practicality, and user experience.
</p>
<p style='text-align: center; font-size: 14px;'>
Ali aims to bridge the gap between people and intelligent systems — making AI accessible, helpful, and personal.
</p>
""", unsafe_allow_html=True)

# ---------- Contact Button and Form ----------
with st.expander("Contact the Developer"):
    st.markdown("**Let's connect! Fill in your details below.**")

    name = st.text_input("Your Name")
    profession = st.text_input("Your Profession or Role")
    email = st.text_input("Email Address")
    preference = st.selectbox(
        "What would you like to connect for?",
        ["General Feedback", "Collaboration", "Hiring", "Questions about the App", "Other"]
    )

    if st.button("Submit"):
        if name and email:
            st.success(f"Thank you, {name}! I'll get back to you soon via {email}.")
        else:
            st.warning("Please fill in at least your name and email.")

# ---------- Footer ----------

st.markdown("""
<p style='text-align: center; font-size: 14px; color: grey;'>
Crafted by <strong>Ali Seena</strong> • Powered by AI
</p>
""", unsafe_allow_html=True)
