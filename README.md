# 🦅 Ecolens - Bird Species Prediction App

## Overview
**Ecolens** is an AI-powered bird species recognition app built using **Streamlit** that enables users to identify bird species using **images** or **audio recordings**. 
Designed for nature lovers, bird watchers, and researchers, Ecolens also offers features like a personalized **checklist** for tracking sightings and a **community sightings board** to explore birds spotted by others.

---

## Key Features

### 1. Bird Species Prediction Using Image
- Upload or capture bird images to predict species.
- Model trained on **12,000 images** across **40 bird species**.
- Uses **DenseNet121** architecture with **97.85% accuracy**.
- Displays rich bird details: description, habitat, diet, and conservation status.

### 2. Bird Species Prediction Using Audio
- Upload bird calls to identify species by sound.
- Model trained on **15 bird species** using **BirdNET Analyzer**.
- Achieves **99% accuracy** for clean and clear recordings.

### 3. Checklist – Track Personal Sightings
- Save sightings with bird name, date, time, and location.
- Organize and manage your birdwatching journey.

### 4. Recent Sightings – Community Dashboard
- Explore sightings recorded by other users.
- View bird name, locations, and timestamps.

---

## Model Details

### Image Classification
- **Model**: DenseNet121  
- **Training Data**: 12,000 images  
- **Accuracy**: 97.85%  
- **Classes**: 40 bird species  

```python
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
```
---
### 🎯 Audio Classification
- **Model**: BirdNET Analyzer
- **Classes**: 15 bird species
- **Accuracy**: 99%

#### Bird Name:
- Asian Green Bee Eater, Common Kingfisher, Common Myna, House Crow, Indian Grey Hornbill, Indian Peafowl, Indian Robin, Jungle Babbler, Pied Kingfisher,
  Purple Sunbird, Red Wattled Lapwing, Rufous Treepie, Shikra, Tickell's Blue Flycatcher, White Throated Kingfisher.

### Tech Stack
- **Frontend**: Streamlit
- **Model Training**: TensorFlow (Densenet121), BirdNET Analyzer
- **Database**: MongoDB
