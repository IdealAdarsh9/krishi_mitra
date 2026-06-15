"""
app.py — Krishi Mitra HuggingFace Spaces deployment
=====================================================
Replace YOUR_HF_USERNAME with your actual HuggingFace username
before pushing to Spaces.
"""

import os
import gc
import re
import json
import time
import torch
import gradio as gr
from PIL import Image
from langdetect import detect, DetectorFactory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MarianMTModel,
    MarianTokenizer,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
)
from peft import PeftModel
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BNB_CUDA_VERSION"] = "0"        # tells bnb to skip CUDA init

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change YOUR_HF_USERNAME to your actual username
# ─────────────────────────────────────────────────────────────────────────────

HF_USERNAME      = "aiwithadarsh"
BASE_MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_REPO     = f"{HF_USERNAME}/krishi-mitra-mistral-7b-qlora"
VISION_MODEL_ID  = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
HI_EN_MODEL_ID   = "Helsinki-NLP/opus-mt-hi-en"
EN_HI_MODEL_ID   = "Helsinki-NLP/opus-mt-en-hi"

# HuggingFace Spaces provides a GPU on paid tiers; free tier uses CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

DetectorFactory.seed = 42   # deterministic language detection


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MISTRAL + QLORA ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

def load_llm():
    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,         # torch_dtype → dtype
            trust_remote_code=True,
    )
    else:
        # CPU — no quantization, no bitsandbytes needed
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    base.config.use_cache = False
    llm = PeftModel.from_pretrained(base, ADAPTER_REPO)
    llm.eval()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    return llm, tok


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD TRANSLATION MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_translators():
    print("Loading translation models...")

    hi_en_tok   = MarianTokenizer.from_pretrained(HI_EN_MODEL_ID)
    hi_en_model = MarianMTModel.from_pretrained(HI_EN_MODEL_ID).to(DEVICE)
    hi_en_model.eval()

    en_hi_tok   = MarianTokenizer.from_pretrained(EN_HI_MODEL_ID)
    en_hi_model = MarianMTModel.from_pretrained(EN_HI_MODEL_ID).to(DEVICE)
    en_hi_model.eval()

    print("✅ Translation models loaded")
    return hi_en_tok, hi_en_model, en_hi_tok, en_hi_model


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD VISION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_vision():
    print("Loading plant disease classifier...")
    feat_ext  = AutoFeatureExtractor.from_pretrained(VISION_MODEL_ID)
    vis_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL_ID)
    vis_model = vis_model.to(DEVICE)
    vis_model.eval()
    print(f"✅ Vision model loaded — {vis_model.config.num_labels} classes")
    return feat_ext, vis_model


# ─── Load everything at startup ──────────────────────────────────────────────
print("="*60)
print("Krishi Mitra — loading models...")
print("="*60)

model, tokenizer           = load_llm()
hi_en_tok, hi_en_model, \
en_hi_tok, en_hi_model     = load_translators()
feat_ext, vis_model        = load_vision()

print("="*60)
print("All models ready!")
print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISEASE LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_MAP = {
    "Apple___Apple_scab":                             ("Apple scab",            "Apple"),
    "Apple___Black_rot":                              ("Black rot",             "Apple"),
    "Apple___Cedar_apple_rust":                       ("Cedar apple rust",      "Apple"),
    "Apple___healthy":                                ("Healthy",               "Apple"),
    "Corn_(maize)___Cercospora_leaf_spot":            ("Cercospora leaf spot",  "Maize"),
    "Corn_(maize)___Common_rust_":                    ("Common rust",           "Maize"),
    "Corn_(maize)___Northern_Leaf_Blight":            ("Northern leaf blight",  "Maize"),
    "Corn_(maize)___healthy":                         ("Healthy",               "Maize"),
    "Grape___Black_rot":                              ("Black rot",             "Grape"),
    "Grape___Esca_(Black_Measles)":                   ("Esca / Black measles",  "Grape"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":     ("Leaf blight",           "Grape"),
    "Grape___healthy":                                ("Healthy",               "Grape"),
    "Orange___Haunglongbing_(Citrus_greening)":       ("Citrus greening (HLB)", "Orange"),
    "Peach___Bacterial_spot":                         ("Bacterial spot",        "Peach"),
    "Peach___healthy":                                ("Healthy",               "Peach"),
    "Pepper,_bell___Bacterial_spot":                  ("Bacterial spot",        "Pepper"),
    "Pepper,_bell___healthy":                         ("Healthy",               "Pepper"),
    "Potato___Early_blight":                          ("Early blight",          "Potato"),
    "Potato___Late_blight":                           ("Late blight",           "Potato"),
    "Potato___healthy":                               ("Healthy",               "Potato"),
    "Squash___Powdery_mildew":                        ("Powdery mildew",        "Squash"),
    "Strawberry___Leaf_scorch":                       ("Leaf scorch",           "Strawberry"),
    "Strawberry___healthy":                           ("Healthy",               "Strawberry"),
    "Tomato___Bacterial_spot":                        ("Bacterial spot",        "Tomato"),
    "Tomato___Early_blight":                          ("Early blight",          "Tomato"),
    "Tomato___Late_blight":                           ("Late blight",           "Tomato"),
    "Tomato___Leaf_Mold":                             ("Leaf mold",             "Tomato"),
    "Tomato___Septoria_leaf_spot":                    ("Septoria leaf spot",    "Tomato"),
    "Tomato___Spider_mites Two-spotted_spider_mite":  ("Spider mites",          "Tomato"),
    "Tomato___Target_Spot":                           ("Target spot",           "Tomato"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":         ("Yellow leaf curl virus","Tomato"),
    "Tomato___Tomato_mosaic_virus":                   ("Mosaic virus",          "Tomato"),
    "Tomato___healthy":                               ("Healthy",               "Tomato"),
}

def clean_label(raw_label):
    if raw_label in DISEASE_MAP:
        return DISEASE_MAP[raw_label]
    parts   = raw_label.replace("___", "|").replace("_", " ").split("|")
    crop    = parts[0].strip() if len(parts) > 0 else "Unknown crop"
    disease = parts[1].strip() if len(parts) > 1 else "Unknown disease"
    return disease, crop


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_plant_image(pil_image, top_k=3):
    """Run plant disease classifier on a PIL image. Returns top-k predictions."""
    inputs = feat_ext(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = vis_model(**inputs).logits

    probs              = torch.nn.functional.softmax(logits, dim=-1)[0]
    topk_probs, topk_i = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(topk_probs.cpu().tolist(), topk_i.cpu().tolist()):
        raw        = vis_model.config.id2label[idx]
        disease, crop = clean_label(raw)
        results.append({
            "disease":    disease,
            "crop":       crop,
            "confidence": round(prob * 100, 1),
            "raw_label":  raw,
            "is_healthy": "healthy" in raw.lower(),
        })
    return results


def build_image_context(predictions):
    """Convert predictions into a context string for Mistral."""
    top = predictions[0]

    if top["confidence"] < 40:
        return "", None                 # too uncertain — ignore image

    if top["is_healthy"]:
        return (
            f"Image analysis: The {top['crop']} plant appears healthy "
            f"(confidence: {top['confidence']}%)."
        ), top

    ctx = (
        f"Image analysis: {top['crop']} plant shows signs of "
        f"{top['disease']} (confidence: {top['confidence']}%). "
    )
    if len(predictions) > 1 and predictions[1]["confidence"] > 20:
        ctx += (
            f"Alternative possibility: {predictions[1]['disease']} "
            f"({predictions[1]['confidence']}%)."
        )
    return ctx, top


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRANSLATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# Chhattisgarhi-specific words — langdetect can't distinguish from Hindi
CG_MARKERS = [
    "हावय", "हाबे", "काबर", "तेन", "करथे", "जाथे", "आथे",
    "हे गा", "कइसे", "काय", "मोला", "तोला", "ओला",
    "अइसे", "वइसे", "गइस", "आइस", "करिस", "देइस",
]

def detect_language(text):
    """Returns 'en' | 'hi' | 'hne' (Chhattisgarhi)"""
    if any(m in text for m in CG_MARKERS):
        return "hne"
    try:
        lang = detect(text)
        if lang in ("hi", "mr", "ne"):
            return "hi"
        return "en"
    except Exception:
        return "en"


def _translate(text, tok, mdl, max_length=512):
    if not text.strip():
        return text
    inputs = tok(
        [text], return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    ).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=max_length,
                           num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)


def to_english(text, source_lang):
    if source_lang == "en":
        return text
    if source_lang in ("hi", "hne"):
        return _translate(text, hi_en_tok, hi_en_model)
    return text


def hindi_to_chhattisgarhi(text):
    """Post-process Hindi translation into Chhattisgarhi dialect."""
    replacements = [
        ("है",          "हे"),
        ("हैं",         "हें"),
        ("करता है",     "करथे"),
        ("करते हैं",    "करथें"),
        ("जाता है",     "जाथे"),
        ("आता है",      "आथे"),
        ("होता है",     "होथे"),
        ("देता है",     "देथे"),
        ("लेता है",     "लेथे"),
        ("था",          "रिहिस"),
        ("थे",          "रिहिन"),
        ("वह",          "ओ"),
        ("यह",          "ए"),
        ("क्या",        "काय"),
        ("कैसे",        "कइसे"),
        ("क्यों",       "काबर"),
        ("मिट्टी",      "माटी"),
        ("बीज",         "बीजा"),
        ("उर्वरक",      "खाद"),
    ]
    for hi, cg in replacements:
        text = text.replace(hi, cg)
    return text


def from_english(text, target_lang):
    if target_lang == "en":
        return text
    if target_lang == "hi":
        return _translate(text, en_hi_tok, en_hi_model)
    if target_lang == "hne":
        hindi = _translate(text, en_hi_tok, en_hi_model)
        return hindi_to_chhattisgarhi(hindi)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 7. CORE INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def ask_krishi_mitra(question, context="", difficulty="simple",
                     image=None, max_new_tokens=400):
    """
    Full pipeline:
      detect lang → translate to EN → (optionally) diagnose image
      → run Mistral → translate answer back to user's language
    """

    # ── Language detection & translation ─────────────────────────────
    user_lang        = detect_language(question)
    english_question = to_english(question, user_lang)

    # ── Vision ────────────────────────────────────────────────────────
    image_context  = ""
    top_prediction = None

    if image is not None:
        try:
            pil_img = image.convert("RGB") if isinstance(image, Image.Image) \
                      else Image.fromarray(image).convert("RGB")
            preds                    = diagnose_plant_image(pil_img)
            image_context, top_prediction = build_image_context(preds)
        except Exception as e:
            print(f"Vision error: {e}")

    # ── Context assembly ──────────────────────────────────────────────
    full_context = ""
    if image_context:
        full_context += image_context + " "
    if context.strip():
        full_context += context.strip()

    # ── Auto-enhance question when confident diagnosis exists ─────────
    if (top_prediction
            and not top_prediction["is_healthy"]
            and top_prediction["confidence"] >= 60):
        english_question = (
            f"My {top_prediction['crop']} plant has been diagnosed with "
            f"{top_prediction['disease']}. "
            f"{english_question if english_question.strip() else 'What should I do?'}"
        )

    # ── System prompt ─────────────────────────────────────────────────
    if difficulty == "expert":
        system = (
            "You are Krishi Mitra, an expert agronomist and plant pathologist "
            "specializing in Indian agriculture. Provide technically accurate, "
            "evidence-based recommendations. Reference ICAR guidelines and "
            "approved pesticide schedules. Be specific about chemical names, "
            "dosages, and application timing."
        )
    else:
        system = (
            "You are Krishi Mitra (कृषि मित्र), a helpful farming assistant "
            "for Indian farmers. Give simple, practical advice in easy language. "
            "Suggest both organic and chemical treatment options. Mention "
            "government schemes and subsidies when applicable. Always include "
            "safety precautions for chemical use."
        )

    ctx_line = f"\nContext: {full_context}" if full_context.strip() else ""
    prompt   = (
        f"### System:\n{system}\n\n"
        f"### Instruction:{ctx_line}\n\n"
        f"### Question:\n{english_question}\n\n"
        f"### Answer:\n"
    )

    # ── Mistral inference ─────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full   = tokenizer.decode(output[0], skip_special_tokens=True)
    marker = "### Answer:"
    idx    = full.rfind(marker)
    english_answer = full[idx + len(marker):].strip() if idx != -1 else full.strip()

    # ── Translate answer back ─────────────────────────────────────────
    return from_english(english_answer, user_lang)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

def respond(message, image, history, location, crop, user_type, lang_override):
    """Gradio callback — wraps ask_krishi_mitra for the chat interface."""

    difficulty = "expert" if user_type == "Agronomist / Expert" else "simple"

    # Build location/crop context string
    context = ""
    if location.strip():
        context += f"Location: {location.strip()}. "
    if crop and crop not in ("Select crop", ""):
        context += f"Crop: {crop}."

    # Default prompt when only an image is uploaded with no text
    if not message.strip() and image is not None:
        defaults = {
            "Auto-detect":   "What disease does this plant have and what should I do?",
            "English":       "What disease does this plant have and what should I do?",
            "Hindi":         "इस पौधे में कौन सी बीमारी है और क्या करना चाहिए?",
            "Chhattisgarhi": "ए पौधा म कोन बीमारी हे अउ काय करे ला लागही?",
        }
        message = defaults.get(lang_override, defaults["Auto-detect"])

    # If user forced a language, override auto-detection
    if lang_override != "Auto-detect":
        lang_map    = {"English": "en", "Hindi": "hi", "Chhattisgarhi": "hne"}
        forced_lang = lang_map[lang_override]
        en_q        = to_english(message, forced_lang)
        en_answer   = ask_krishi_mitra(en_q, context, difficulty, image)
        return from_english(en_answer, forced_lang)

    return ask_krishi_mitra(message, context, difficulty, image)


# ─── Custom CSS ──────────────────────────────────────────────────────────────
CSS = """
#header { text-align: center; padding: 16px 0 8px 0; }
#header h1 { font-size: 2rem; margin: 0; }
#header p  { color: #666; margin: 4px 0 0 0; font-size: 0.95rem; }
.tip-box   { background: #f0faf0; border-left: 3px solid #3d8b40;
             padding: 10px 14px; border-radius: 6px; font-size: 0.88rem; }
"""

with gr.Blocks(css=CSS, title="Krishi Mitra — कृषि मित्र") as demo:

    # ── Header ────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
      <h1>🌾 Krishi Mitra — कृषि मित्र</h1>
      <p>AI-powered farming assistant · Plant diseases · Crop advice · Govt schemes</p>
      <p>Ask in <b>English</b>, <b>हिंदी</b>, or <b>छत्तीसगढ़ी</b></p>
    </div>
    """)

    with gr.Row():

        # ── Left panel — controls ─────────────────────────────────────
        with gr.Column(scale=1, min_width=260):

            image_input = gr.Image(
                label="📷 Plant photo (optional)",
                type="pil",
                sources=["upload", "webcam"],
            )

            lang_override = gr.Radio(
                choices=["Auto-detect", "English", "Hindi", "Chhattisgarhi"],
                value="Auto-detect",
                label="Language / भाषा / भाखा",
            )

            location = gr.Textbox(
                label="📍 Location / जगह",
                placeholder="e.g. Raipur, Chhattisgarh",
            )

            crop = gr.Dropdown(
                choices=[
                    "Select crop",
                    "Paddy / Rice / धान",
                    "Wheat / गेहूं",
                    "Soybean / सोयाबीन",
                    "Maize / मक्का",
                    "Cotton / कपास",
                    "Tomato / टमाटर",
                    "Chilli / मिर्च",
                    "Potato / आलू",
                    "Onion / प्याज",
                    "Groundnut / मूंगफली",
                    "Other / अन्य",
                ],
                value="Select crop",
                label="🌱 Crop / फसल",
            )

            user_type = gr.Radio(
                choices=["Farmer / किसान", "Agronomist / Expert"],
                value="Farmer / किसान",
                label="👤 I am a / मैं हूं",
            )

            gr.HTML("""
            <div class="tip-box">
              <b>📸 Photo tips:</b><br>
              • Photograph the affected leaf clearly<br>
              • Good lighting, no blur<br>
              • Fill the frame with the leaf<br>
              • One leaf per photo
            </div>
            """)

        # ── Right panel — chat ────────────────────────────────────────
        with gr.Column(scale=2):
            gr.ChatInterface(
                fn=respond,
                additional_inputs=[
                    image_input,
                    location,
                    crop,
                    user_type,
                    lang_override,
                ],
                examples=[
                    # English — text
                    ["My paddy leaves have brown spots, what disease is this?"],
                    # Hindi — text
                    ["मेरे धान की पत्तियों पर भूरे धब्बे हैं, क्या करूँ?"],
                    # Chhattisgarhi
                    ["मोर धान के पान म भूरा दाग हावय, काय करंव?"],
                    # English — scheme
                    ["How do I apply for PM-KISAN scheme?"],
                    # Hindi — scheme
                    ["पीएम किसान के लिए कैसे आवेदन करें?"],
                    # English — image prompt
                    ["What disease does this plant have and what should I do?"],
                    # English — general
                    ["What is the best fertilizer dose for soybean in kharif?"],
                    # Hindi — general
                    ["सोयाबीन के लिए कौन सा उर्वरक और कितनी मात्रा में दें?"],
                ],
                title="",
            )

    # ── Footer ────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="text-align:center; padding:16px 0 4px 0;
                color:#888; font-size:0.82rem;">
      Krishi Mitra · Fine-tuned on Mistral-7B with QLoRA ·
      <a href="https://huggingface.co/{HF_USERNAME}/krishi-mitra-mistral-7b-qlora"
         target="_blank">Model</a> ·
      <a href="https://huggingface.co/datasets/{HF_USERNAME}/krishi-mitra-agriculture-qa"
         target="_blank">Dataset</a> ·
      <a href="https://github.com/{HF_USERNAME}/krishi-mitra"
         target="_blank">GitHub</a>
    </div>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 9. LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # HuggingFace Spaces sets PORT automatically
    # share=False because Spaces provides the public URL itself
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
else:
    # When Spaces imports app.py as a module (not __main__),
    # demo is already defined at module level above.
    pass
