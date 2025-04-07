# auth.py
import streamlit as st
import bcrypt
from pymongo import MongoClient
import time

# ✅ MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
client = MongoClient(MONGO_URI)

db = client["bird_app"]
users_collection = db["users"]

# ✅ Hash Password
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

# ✅ Verify Password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password)

# ✅ Login Function
def login():
    st.subheader("🔐 Login")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Login"):
        user = users_collection.find_one({"email": email})
        if user and verify_password(password, user["password"]):
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.success("✅ Login successful!")
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Invalid email or password")

# ✅ Register Function
def register():
    st.subheader("📝 Register")
    name = st.text_input("👤 Name")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Register"):
        if users_collection.find_one({"email": email}):
            st.error("⚠️ Email already exists. Please log in.")
        else:
            hashed_password = hash_password(password)
            users_collection.insert_one({
                "name": name,
                "email": email,
                "password": hashed_password
            })
            st.success("✅ Registration successful! Please log in.")
            time.sleep(2)
            st.rerun()

# ✅ Logout Function
def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = ""
    st.success("✅ Logged out successfully!")
    time.sleep(2)
    st.rerun()
