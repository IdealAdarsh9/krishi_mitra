# 🌾 Krishi Mitra — AI Agriculture Assistant

<div align="center">

![Krishi Mitra Banner](https://img.shields.io/badge/🌾_Krishi_Mitra-AI_Agriculture_Assistant-green?style=for-the-badge)

[![HuggingFace Model](https://img.shields.io/badge/🤗_Model-Mistral--7B_QLoRA-yellow?style=flat-square)](https://huggingface.co/YOUR_USERNAME/krishi-mitra-mistral-7b-qlora)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-8k+_Q%26A_Pairs-blue?style=flat-square)](https://huggingface.co/datasets/YOUR_USERNAME/krishi-mitra-agriculture-qa)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-orange?style=flat-square)](https://YOUR_USERNAME-krishi-mitra.hf.space)
[![Open In Colab](https://img.shields.io/badge/Open_In-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](YOUR_COLAB_LINK)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**An end-to-end fine-tuned LLM chatbot for Indian farmers**  
Plant disease detection · Crop advice · Government schemes · English, Hindi & Chhattisgarhi

</div>

---

## 📌 Overview

Krishi Mitra *(कृषि मित्र — "Friend of Farmers")* is a domain-specific AI assistant built for Indian agriculture. It combines a **QLoRA fine-tuned Mistral-7B** with a **MobileNetV2 plant disease classifier** and a **multilingual translation pipeline** to give farmers instant, practical advice in their own language.

The entire pipeline — from dataset generation to deployment — was built from scratch, including automated Q&A generation using the Groq API, custom fine-tuning on Google Colab's free T4 GPU, and deployment on HuggingFace Spaces.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌿 **Plant Disease Detection** | Upload a photo → MobileNetV2 diagnoses disease from 38 classes (PlantVillage) |
| 💊 **Treatment Advice** | Organic + chemical treatment recommendations with dosage and safety precautions |
| 🏛️ **Government Schemes** | PM-KISAN, Fasal Bima Yojana, MSP, Kisan Credit Card explained simply |
| 🌐 **Multilingual** | Supports English, Hindi (हिंदी), and Chhattisgarhi (छत्तीसगढ़ी) |
| 👨‍🌾 **Dual Mode** | Farmer mode (simple language) and Agronomist mode (technical detail) |
| 📍 **Location-aware** | Crop + region context for central India (Chhattisgarh, MP) |

---

## 🏗️ System Architecture

```
User Input (Text / Image / Voice)
          │
          ▼
┌─────────────────────────────────────┐
│         Language Detection          │
│  English │ Hindi │ Chhattisgarhi   │
└─────────────────────────────────────┘
          │                    │
          │              ┌─────▼──────┐
          │              │ MobileNetV2 │  ← Plant photo
          │              │ Vision Model│
          │              └─────┬──────┘
          │                    │ Disease label + confidence
          ▼                    ▼
┌─────────────────────────────────────┐
│     Helsinki-NLP MarianMT           │
│     Translate → English             │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   Mistral-7B-Instruct + QLoRA       │
│   Fine-tuned on 8,000+ Agri Q&A    │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│     Translate Answer Back           │
│     English → Hindi / Chhattisgarhi │
└─────────────────────────────────────┘
          │
          ▼
      Final Response
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Base LLM** | Mistral-7B-Instruct-v0.2 |
| **Fine-tuning** | QLoRA — r=32, lora_alpha=64, PEFT |
| **Training hardware** | Google Colab T4 GPU (free tier) |
| **Dataset generation** | Groq API — llama-3.3-70b-versatile |
| **Disease detection** | MobileNetV2 — PlantVillage (38 classes) |
| **Translation** | Helsinki-NLP MarianMT (hi ↔ en) |
| **Language detection** | langdetect |
| **UI** | Gradio |
| **Deployment** | HuggingFace Spaces |

---

## 📊 Dataset

**8,000+ agriculture Q&A pairs** generated and collected from multiple sources:

| Source | Samples | Topics |
|---|---|---|
| Groq API (llama-3.3-70b) | ~5,000 | 40 agriculture topics |
| HuggingFace (AgriKnowBot, Dolly) | ~2,000 | General farming |
| PlantVillage disease labels | ~200 | 38 disease classes |
| Govt portal scraping (ICAR, FAO) | ~800 | Schemes, pest management |

**Coverage:**
- 🦠 18 plant diseases (paddy blast, soybean YMV, wheat rust, cotton bollworm...)
- 🌱 12 crop management topics (soil health, irrigation, seed treatment...)
- 🏛️ 7 government schemes (PM-KISAN, PMFBY, MSP, KCC, eNAM...)
- 🗺️ Central India focus (Chhattisgarh, Madhya Pradesh)

---

## 🚀 Quick Start

### Run locally

```bash
# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("aiwithadarsh/krishi-mitra-Mistral-7B-Instruct-v0.2", dtype="auto")
```

### Run in Colab

Click the badge to open the full pipeline notebook (dataset generation + fine-tuning):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gJ55QKp47GCeXNC8Ex1YIAfZOVuFNb_J#scrollTo=5IrlZkPlW0C3)

---

## 📁 Project Structure

```
krishi-mitra/
│
├── app.py                        # HuggingFace Spaces entry point
├── requirements.txt              # Python dependencies
├── .python-version               # Python 3.11 pin for Spaces
│
├── notebooks/
│   ├── 01_dataset_generation.ipynb   # Groq API Q&A generation + HF push
│   ├── 02_finetune_mistral.ipynb     # QLoRA fine-tuning on Colab T4
│   └── 03_chatbot_full.ipynb         # Full chatbot with image + multilingual
│
├── src/
│   ├── translator.py            # Language detection + MarianMT pipeline
│   ├── vision.py                # MobileNetV2 disease classifier
│   └── chatbot.py               # Mistral inference function
│
├── data/
│   └── sample_qa.json           # 50 sample Q&A pairs from the dataset
│
└── assets/
    ├── architecture.png         # System architecture diagram
    └── demo.gif                 # Chatbot demo screen recording
```

---

## 🧠 Fine-tuning Details

```python
# Model
base_model  = "mistralai/Mistral-7B-Instruct-v0.2"
method      = "QLoRA (4-bit NF4 quantization)"

# LoRA config
r           = 32
lora_alpha  = 64
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
epochs          = 3
learning_rate   = 2e-4
batch_size      = 2  # gradient accumulation × 4 = effective 8
optimizer       = "paged_adamw_8bit"
hardware        = "Google Colab T4 (free tier)"
training_time   = "~3 hours"
trainable_params = "~0.3% of 7B parameters"
```

---

## 💬 Example Conversations

**English — Plant disease:**
> 👨‍🌾 *"My paddy leaves have brown spots with yellow borders"*  
> 🤖 *"This looks like Paddy Blast (Magnaporthe oryzae). Spray Tricyclazole 75WP @ 0.6g/L water. Also apply Carbendazim as a preventive. Avoid excess nitrogen fertilizer..."*

**Hindi — Government scheme:**
> 👨‍🌾 *"पीएम किसान के लिए कैसे आवेदन करें?"*  
> 🤖 *"पीएम-किसान योजना में ₹6,000 सालाना मिलते हैं। pmkisan.gov.in पर जाएं, आधार और बैंक खाते की जानकारी दर्ज करें..."*

---

## 🗺️ Roadmap

- [x] Dataset generation pipeline (Groq API + HuggingFace)
- [x] QLoRA fine-tuning on Mistral-7B
- [x] Plant disease image detection (MobileNetV2)
- [x] Multilingual support (English, Hindi, Chhattisgarhi)
- [x] Gradio chatbot UI
- [x] HuggingFace Spaces deployment
- [ ] Voice input (Whisper STT)
- [ ] Weather API integration (IMD agromet)
- [ ] Android app (Gradio lite)
- [ ] Feedback loop + continuous retraining

---

## 📈 Results

| Metric | Value |
|---|---|
| Dataset size | 8,000+ Q&A pairs |
| Disease classes covered | 38 (PlantVillage) |
| Languages supported | 3 (EN, HI, HNE) |
| Training loss | ~0.85 after 3 epochs |
| Inference speed (GPU) | ~3 seconds/response |
| Model size (adapter only) | ~50 MB |

---

## 🙏 Acknowledgements

- [ICAR](https://icar.org.in) and [FAO](https://fao.org) for agriculture reference data
- [PlantVillage Dataset](https://plantvillage.psu.edu) for disease classification training data
- [Groq](https://groq.com) for fast LLM inference during dataset generation
- [HuggingFace](https://huggingface.co) for model hosting and Spaces deployment
- [Google Colab](https://colab.research.google.com) for free T4 GPU training

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ for Indian farmers

⭐ Star this repo if you found it useful!

</div>
