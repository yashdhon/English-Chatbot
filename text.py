import streamlit as st

# SESSION STATE
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# LOGIN PAGE
if not st.session_state.authenticated:

    st.title("Login")

    password = st.text_input(
        "Enter Password",
        type="password"
    )

    if st.button("Login"):

        if password == "1234":

            st.session_state.authenticated = True

            st.rerun()

        else:

            st.error("Wrong Password")

# CHATBOT PAGE
else:

    st.title("AI Chatbot")

    query = st.text_input("Ask me")

    if st.button("Send"):

        st.success(f"You asked: {query}")