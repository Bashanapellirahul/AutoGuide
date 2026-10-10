import streamlit as st

st.set_page_config(
    page_title="AutoGuide AI",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 AutoGuide AI")
st.subheader("Your Vehicle Safety Assistant")

st.write(
    "Ask questions about vehicle safety features, "
    "braking systems, airbags, and vehicle recalls."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input(
    "Ask a vehicle safety question..."
)

if user_question:

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        response = (
            "Thanks for your question! "
            "The NHTSA retrieval system is not connected yet. "
            "Once connected, I will answer using official "
            "vehicle safety information and provide citations."
        )

        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })