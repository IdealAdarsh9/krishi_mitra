# 🌾 Krishi Mitra — AI Agriculture Assistant

An end-to-end fine-tuned LLM chatbot for Indian farmers that answers 
questions about plant diseases, crop management, and government schemes 
in English, Hindi, and Chhattisgarhi.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/YOUR_USERNAME/krishi-mitra-mistral-7b-qlora)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/YOUR_USERNAME/krishi-mitra-agriculture-qa)
[![Live Demo](https://img.shields.io/badge/🚀-Live_Demo-green)](YOUR_SPACES_LINK)

---

## 🎯 What it does

- 🌿 **Plant disease detection** — upload a photo of a diseased plant and 
  get instant diagnosis + treatment advice
- 💬 **Multilingual Q&A** — ask in English, Hindi, or Chhattisgarhi
- 🏛️ **Government schemes** — PM-KISAN, Fasal Bima, MSP explained simply
- 👩‍🌾 **Dual mode** — simple language for farmers, technical for agronomists

## 🏗️ Architecture

![Architecture](assets/architecture.png)

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning | QLoRA (r=32, PEFT) |
| Dataset | 8,000+ Q&A pairs (self-generated via Groq API) |
| Disease detection | MobileNetV2 on PlantVillage (38 classes) |
| Translation | Helsinki-NLP MarianMT (hi↔en) |
| Training | Google Colab T4 GPU (free) |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

## 📊 Dataset

Generated 8,000+ agriculture Q&A pairs covering:
- 18 plant diseases (paddy blast, soybean YMV, wheat rust...)
- 12 crop management topics
- 7 government schemes (PM-KISAN, PMFBY, MSP...)
- Central India focus (Chhattisgarh, MP)

Dataset: [huggingface.co/datasets/aiwithadarsh/krishi-mitra-agriculture-qa](link)

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/krishi-mitra
cd krishi-mitra
pip install -r requirements.txt
```

Or open directly in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK)

## 📱 Demo

![Demo GIF](assets/demo.gif)

## 🗂️ Project Structure

\```
krishi-mitra/
├── notebooks/          # Colab notebooks for each stage
├── src/                # Core Python modules
├── data/               # Sample dataset
└── assets/             # Images and demo GIF
\```

## 🙏 Acknowledgements

- ICAR and FAO for agriculture reference data
- PlantVillage dataset for disease classification
- Groq for fast LLM inference during dataset generation
