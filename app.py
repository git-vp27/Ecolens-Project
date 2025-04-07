import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

import librosa
import tempfile

from bird_info import bird_info

from datetime import datetime
import pytz
from pymongo import MongoClient
from auth import login, register, logout
import warnings
import base64

warnings.filterwarnings("ignore")

# ✅ MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")

db = client["bird_app"]
checklist_collection = db["checklist"]
users_collection = db["users"]

timezone = pytz.timezone("Asia/Kolkata")

# ✅ Set page config
st.set_page_config(page_title="Bird Species Detection", layout="wide")

# ✅ Initialize session state for page and auth
if "page" not in st.session_state:
    st.session_state.page = "home"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# ✅ Load the model (cached for efficiency)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Ecolens40_model.h5')
    # model = tf.keras.models.load_model('best15_bird_model.h5')
    return model

model = load_model()


class_labels = [
    "Ashy-Prinia", "Asian-Green-Bee-Eater", "Black-Drongo", "Black-Winged-Kite", "Black-Winged-Stilt",
    "Brown-Headed-Barbet", "Cattle-Egret", "Common-Kingfisher", "Common-Myna", "Common-Rosefinch",
    "Common-Tailorbird", "Coppersmith-Barbet", "Forest-Wagtail", "Gray-Wagtail", "Hoopoe",
    "House-Crow", "Indian-Grey-Hornbill", "Indian-Paradise-Flycatcher", "Indian-Peacock",
    "Indian-Pitta", "Indian-Roller", "Indian-Silverbill", "Jungle-Babbler", "Jungle-Owlet",
    "Long-Tailed-Shrike", "Northern-Lapwing", "Oriental-Magpie-Robin", "Pied-Kingfisher",
    "Red-Avadavat", "Red-Naped-Ibis", "Red-Wattled-Lapwing", "River-Tern", "Ruddy-Shelduck",
    "Rufous-Treepie", "Sarus-Crane", "Shikra", "Tickells-Blue-Flycatcher", "White-Breasted-Kingfisher",
    "White-Breasted-Waterhen", "White-Wagtail"
]

# === Load Labels ===
def load_audio_labels(label_path="CustomClassifier_Labels.txt"):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f]

# === Load TFLite Model ===
def load_audio_model(model_path="CustomClassifier.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# === Preprocess Audio to Fixed Length Vector ===
def preprocess_audio(file, target_len=144000, sr=48000):
    samples, _ = librosa.load(file, sr=sr, mono=True)
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)))
    else:
        samples = samples[:target_len]
    samples = samples / np.max(np.abs(samples))  # Normalize
    return np.expand_dims(samples.astype(np.float32), axis=0)  # (1, 144000)

# === Prediction Function ===
def predict_bird(file, model, labels):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_data = preprocess_audio(file)
    
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    
    probs = model.get_tensor(output_details[0]['index'])[0]
    top_idx = np.argmax(probs)
    return labels[top_idx], probs[top_idx]


# ✅ Sidebar - Show User Email if Authenticated
if st.session_state.authenticated:
    st.sidebar.info(f"Logged in as: **{st.session_state.user_email}**")
st.sidebar.header("About")
st.sidebar.write("""
**Ecolens** helps you identify bird species from images and audio recordings.
- **Identify Birds:** Upload an image or capture one using your camera to predict bird species.
- **Audio Detection:** Upload bird sounds to recognize species by their vocalizations.
- **Track Sightings:** Save, manage, and review your bird sightings with ease.
- **Explore Bird Info:** Learn more about each bird's habitat, diet, and conservation status.
- **Our Users:** Birdwatchers, Researchers, Students, Nature Enthusiasts, Conservationists.
""")
st.sidebar.markdown("---")

# ✅ Sidebar - Navigation with Page Control
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
if not st.session_state.authenticated:
    if st.sidebar.button("🔐 Login"):
        st.session_state.page = "login"
    if st.sidebar.button("📝 Register"):
        st.session_state.page = "register"
else:
    if st.sidebar.button("🚪 Logout"):
        logout()

# ✅ Page Router
if st.session_state.page == "home":
    st.title("Ecolens 🦅")
    st.subheader("Bird Species Detection App")

    feature = st.selectbox("Select a Feature", [
        "Bird Species Prediction Using Image",
        "Bird Species Prediction Using Audio",
        "Checklist (Record Bird Sightings)",
        "Recent Sightings"
    ])

    # ✅ Bird Species Prediction Using Image
    if feature == "Bird Species Prediction Using Image":
        # Select Model
        # model_option = st.radio("Select the Bird Model:", ["25 Species", "15 Species"])
    
        # if model_option == "25 Species":
        #     model = load_model_25()
        #     class_labels = class_labels_25
        # else:
        #     model = load_model_15()
        #     class_labels = class_labels_15

            
        st.subheader("Upload or Capture an Image")
        option = st.radio("Choose an option:", ["Upload an Image", "Use Camera"])

        uploaded_file = None
        if option == "Upload an Image":
            uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
        elif option == "Use Camera":
            uploaded_file = st.camera_input("Take a photo")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess image
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Prediction
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            predicted_species = class_labels[predicted_class]

            st.success(f"✅ Predicted Bird Species: **{predicted_species}**")
            st.info(f"🎯 Confidence: **{confidence:.2f}%**")

            species_info = bird_info.get(predicted_species, "ℹ️ No information available.")
            st.subheader(f"{predicted_species}:")
            st.markdown(f"""
            - **Scientific Name:** {species_info.get('Scientific Name', 'N/A')}
            - **Description:** {species_info.get('Description', 'N/A')}
            - **Habitat:** {species_info.get('Habitat', 'N/A')}
            - **Food:** {species_info.get('Diet', 'N/A')}
            - **Conservation Status:** {species_info.get('Conservation Status', 'N/A')}
            """)

            # ✅ Encode image to base64 for storage
            buffered = uploaded_file.getvalue()
            encoded_image = base64.b64encode(buffered).decode('utf-8')

            # ✅ Get user input for checklist
            date = st.date_input("📅 Date of Sighting")
            time = datetime.now(timezone).strftime("%H:%M")
            st.text_input("🕐 Time of Sighting", time)
            location = st.text_input("📍 Location")

            # ✅ Check if user is authenticated before saving to checklist
            if st.button("Save to Checklist"):
                if not st.session_state.authenticated:
                    st.warning("❗️ You need to log in/register to save bird info in the checklist.")
                else:
                    checklist_collection.insert_one({
                        "user_email": st.session_state.user_email,
                        "species": predicted_species,
                        "date": str(date),
                        "time": str(time),
                        "location": location,
                        "image": encoded_image
                    })
                    st.success("✅ Bird sighting recorded successfully!")

    elif feature == "Bird Species Prediction Using Audio":
        audio_file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

        if audio_file:
            st.audio(audio_file)
            
            try:
                labels = load_audio_labels("CustomClassifier_Labels.txt")
                model = load_audio_model("CustomClassifier.tflite")
                
                bird, confidence = predict_bird(audio_file, model, labels)

                st.success(f"🔊 Predicted Bird: **{bird}**")

                normalized_name = bird.title().replace(" ", "-")

                species_info = bird_info.get(normalized_name, "ℹ️ No information available.")
                st.subheader(f"{bird}:")
                if isinstance(species_info, dict):
                    st.markdown(f"""
                    - **Scientific Name:** {species_info.get('Scientific Name', 'N/A')}
                    - **Description:** {species_info.get('Description', 'N/A')}
                    - **Habitat:** {species_info.get('Habitat', 'N/A')}
                    - **Food:** {species_info.get('Diet', 'N/A')}
                    - **Conservation Status:** {species_info.get('Conservation Status', 'N/A')}
                    """)
                else:
                    st.info(species_info)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        
    # ✅ Checklist (Record Bird Sightings)
    elif feature == "Checklist (Record Bird Sightings)":
        st.subheader("📋 Recorded Bird Sightings")

        if not st.session_state.authenticated:
            st.warning("❗️ You need to log in to view your checklist.")
        else:
            sightings = checklist_collection.find({"user_email": st.session_state.user_email})

            if checklist_collection.count_documents({"user_email": st.session_state.user_email}) == 0:
                st.info("No sightings recorded yet. Make a prediction first!")
            else:
                to_delete = None

                for sighting in sightings:
                    with st.expander(f"📌 {sighting['species']} - {sighting['date']}"):
                        st.write(f"📅 **Date:** {sighting['date']}")
                        st.write(f"🕐 **Time:** {sighting['time']}")
                        st.write(f"📍 **Location:** {sighting['location']}")

                        # ✅ Display image if available
                        if "image" in sighting:
                            image_data = base64.b64decode(sighting["image"])
                            st.image(image_data, caption="Sighting Image", use_container_width=True)

                        # ✅ Delete option for authenticated users
                        if st.button(f"🗑️ Delete {sighting['species']}", key=f"delete_{sighting['_id']}"):
                            to_delete = sighting["_id"]

                # ✅ If an entry is marked for deletion, remove it
                if to_delete is not None:
                    checklist_collection.delete_one({"_id": to_delete})
                    st.success("✅ Sighting deleted successfully!")
                    st.rerun()

    # ✅ Recent Sightings - Displays all sightings by different users
    elif feature == "Recent Sightings":
        st.subheader("🌍 Recent Bird Sightings")
        st.write("")
        
        sightings = checklist_collection.find().sort("date", -1).limit(20)

        if checklist_collection.count_documents({}) == 0:
            st.info("No recent sightings available.")
        else:
            for sighting in sightings:
                user_info = users_collection.find_one({"email": sighting["user_email"]})
                user_name = user_info.get("name", "Unknown User")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.write(f"{user_name}")
                col2.write(f"🦅 {sighting['species']}")
                col3.write(f"📅 {sighting['date']}")
                col4.write(f"🕐 {sighting['time']}")
                col5.write(f"📍 {sighting['location']}")

                st.markdown(
                    "<div style='border-bottom: 1px solid #ddd; margin: 5px 0;'></div>",
                    unsafe_allow_html=True
                )

# ✅ Show Login Page
elif st.session_state.page == "login":
    login()

# ✅ Show Register Page
elif st.session_state.page == "register":
    register()
