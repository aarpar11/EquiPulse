import streamlit as st

# Configure page settings
st.set_page_config(page_title="EquiPulse Tools", page_icon="🫀", layout="centered")

st.title("My Python Web App")
st.write("Enter input data below to execute your Python logic.")

# Input widget
user_input = st.text_input("Enter something:", placeholder="e.g., Patient ID or parameter")

# Action button
if st.button("Run Python Logic"):
    if user_input.strip():
        # Call your custom functions here
        result = f"Processed: {user_input}"
        st.success(result)
    else:
        st.warning("Please enter a value before running.")