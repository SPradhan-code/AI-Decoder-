import sys
import os

# Ensure virtual environment site-packages is prioritized
venv_site_packages = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages"))
if os.path.exists(venv_site_packages):
    while venv_site_packages in sys.path:
        sys.path.remove(venv_site_packages)
    sys.path.insert(0, venv_site_packages)

import streamlit as st
import pandas as pd
import time
import hashlib
import backend

# Load variables from a local .env file (if present) into the process environment.
# This keeps the API key out of source control - it should live in a .env file
# in the project root that looks like:
#   GEMINI_API_KEY=your_real_key_here
# and that .env file should be listed in .gitignore so it never gets committed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except (ImportError, ModuleNotFoundError):
    # Fallback manual .env parser in case python-dotenv isn't installed
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val

# 1. Page Configuration
st.set_page_config(
    page_title="AI Insurance Decoder",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False

if "ready_to_process" not in st.session_state:
    st.session_state.ready_to_process = False

if "real_ai_data" not in st.session_state:
    st.session_state.real_ai_data = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "file_uploader_default"

if "camera_key" not in st.session_state:
    st.session_state.camera_key = "camera_default"

# Initialize Gemini API key from environment (populated from the .env file above).
# There is intentionally NO hardcoded fallback key here - if GEMINI_API_KEY is
# missing, the app runs in demo mode (using the static mock text) instead of
# silently calling Gemini with a baked-in credential.
api_key = os.environ.get("GEMINI_API_KEY")
st.session_state.gemini_api_key = api_key if api_key else None


# 2. Translation Dictionaries
TRANSLATIONS = {
    "English": {
        "title": "🛡️ AI Insurance Policy Decoder",
        "subtitle": "Demystify complex coverage clauses instantly. Make informed financial decisions with corporate precision.",
        "lang_label": "🌐 Preferred Language / भाषा / ভাষা",
        "add_file_section": "📁 Input Policy Document",
        "add_file_desc": "Select how you would like to submit your document:",
        "opt_upload": "Upload Document File",
        "opt_camera": "Take a Photo with Camera",
        "upload_help": "Drag & drop your policy PDF or image here (PNG, JPG)",
        "camera_help": "Position your document paper inside the camera view and click capture",
        "demo_loaded": "✨ Loaded Demo Document Coverage Analysis. Upload your policy above to replace.",
        "file_loaded": "✅ Successfully loaded and decoded policy: ",
        "dashboard_section": "📊 Policy Insights & Financial Dashboard",
        "col_highlights": "Highlights Summary",
        "box_covered": "🟢 What is covered (Yes)",
        "box_not_covered": "🔴 What is NOT covered (No)",
        "box_traps": "⚠️ Hidden Traps and Conditions",
        "covered_items": [
            "Emergency hospitalizations and ICU daily charges.",
            "Accidental vehicle body repairs and third-party property damages.",
            "Physician consultations and general prescription drugs (Tier-1)."
        ],
        "not_covered_items": [
            "Elective aesthetic treatments and cosmetic surgeries.",
            "Normal mechanical wear-and-tear or oil-deprived engine damages.",
            "Private hospital nursing fees and personal comfort amenities."
        ],
        "traps_items": [
            "30-day waiting period from policy start for all non-accidental illnesses.",
            "24-month waiting period for pre-specified ailments (hernia, cataracts).",
            "Treatments outside network hospitals incur a mandatory 20% co-pay."
        ],
        "table_title": "📊 Interactive Financial Cost Table (Tableau Style)",
        "table_desc": "Modify the Deductible, Co-pay Share %, or Bill Amount in the spreadsheet to instantly calculate your expenses.",
        "col_feature": "Medical or Car Feature",
        "col_limit": "Covered Limit ($)",
        "col_deductible": "Your Deductible Fee ($)",
        "col_copay": "Your Co-pay Share %",
        "col_bill": "Total Bill Amount ($)",
        "col_oop": "Your Final Out-of-Pocket Expense ($)",
        "col_savings": "Your Net Financial Savings ($)",
        "chat_section": "💬 Interactive Coverage Assistant",
        "chat_desc": "Ask any specific questions about coverage limits, claims, or terms.",
        "chat_placeholder": "Type your question here (e.g. Does this cover MRIs?)...",
        "suggested_title": "Suggested Questions:",
        "suggested_qs": [
            "What are the hidden traps?",
            "Does this cover emergencies?",
            "How do I make a claim?"
        ],
        "answers": {
            "What are the hidden traps?": "⚠️ **Hidden Traps Identified:**\n1. **General Illness Waiting Period:** 30 days initial waiting period from policy inception.\n2. **Special Disease Waiting Period:** 24 months for surgeries like joint replacements and hernia.\n3. **Pre-existing Diseases:** Strictly excluded for the first 48 months of active coverage.\n4. **Co-pay Penalty:** Out-of-network admissions demand a 20% co-pay on the total claim.",
            "Does this cover emergencies?": "🚨 **Emergency Coverage:**\nYes, emergency admissions are covered up to the maximum limit of your coverage tier. However, if you are treated at a non-network hospital, you must co-pay 20% of the bill, and you must notify the insurer within 24 hours of admission.",
            "How do I make a claim?": "📝 **Claims Guidelines:**\n- **Cashless Claim:** Treat at a network hospital. Present your card at the insurance desk. Fill out the pre-authorization form prior to or within 24 hours of admission.\n- **Reimbursement Claim:** Pay the bills yourself. Collect the discharge card, medical bills, diagnostic reports, and pharmacy receipts. Submit them online or via post within 30 days."
        },
        "default_features": [
            "ICU Bed Charges",
            "Inpatient Surgical Fees",
            "Emergency Ambulance Ride",
            "Advanced MRI/CT Scans",
            "Routine Consultation"
        ]
    },
    "Hindi": {
        "title": "🛡️ एआई बीमा पॉलिसी डिकोडर",
        "subtitle": "जटिल बीमा धाराओं को तुरंत समझें। कॉर्पोरेट परिशुद्धता के साथ वित्तीय निर्णय लें।",
        "lang_label": "🌐 पसंदीदा भाषा / Preferred Language / ভাষা",
        "add_file_section": "📁 बीमा दस्तावेज़ जोड़ें",
        "add_file_desc": "दस्तावेज़ जमा करने का तरीका चुनें:",
        "opt_upload": "दस्तावेज़ फ़ाइल अपलोड करें",
        "opt_camera": "कैमरे से फोटो लें",
        "upload_help": "अपनी पॉलिसी पीडीएफ या छवि (PNG, JPG) यहाँ खींचें या ब्राउज़ करें",
        "camera_help": "अपने दस्तावेज़ को कैमरे के सामने रखें और फ़ोटो खींचें",
        "demo_loaded": "✨ डेमो दस्तावेज़ विश्लेषण लोड किया गया। बदलने के लिए ऊपर अपनी पॉलिसी अपलोड करें।",
        "file_loaded": "✅ पॉलिसी सफलतापूर्वक लोड और डिकोड की गई: ",
        "dashboard_section": "📊 नीति अंतर्दृष्टि और वित्तीय डैशबोर्ड",
        "col_highlights": "हाइलाइट्स सारांश",
        "box_covered": "🟢 क्या कवर किया गया है (हाँ)",
        "box_not_covered": "🔴 क्या कवर नहीं किया गया है (नहीं)",
        "box_traps": "⚠️ छिपे हुए जाल और शर्तें",
        "covered_items": [
            "आपातकालीन अस्पताल में भर्ती और आईसीयू दैनिक शुल्क।",
            "आकस्मिक वाहन बॉडी मरम्मत और तीसरे पक्ष की संपत्ति का नुकसान।",
            "चिकित्सक परामर्श और जेनेरिक नुस्खे वाली दवाएं (टियर-1)।"
        ],
        "not_covered_items": [
            "वैकल्पिक सौंदर्य उपचार और कॉस्मेटिक सर्जरी।",
            "सामान्य यांत्रिक टूट-फूट या तेल की कमी से इंजन क्षति।",
            "निजी अस्पताल नर्सिंग शुल्क और व्यक्तिगत आराम सुविधाएं।"
        ],
        "traps_items": [
            "सभी गैर-आकस्मिक बीमारियों के लिए पॉलिसी शुरू होने से 30 दिन की प्रतीक्षा अवधि।",
            "पूर्व-निर्धारित बीमारियों (हर्निया, मोतियाबिंद) के लिए 24 महीने की प्रतीक्षा अवधि।",
            "नेटवर्क से बाहर के अस्पतालों में इलाज कराने पर अनिवार्य 20% सह-भुगतान लागू।"
        ],
        "table_title": "📊 इंटरैक्टिव वित्तीय लागत तालिका (झांकी शैली)",
        "table_desc": "अपने खर्चों की तुरंत गणना करने के लिए स्प्रेडशीट में डिडक्टिबल, सह-भुगतान प्रतिशत, या बिल राशि बदलें।",
        "col_feature": "चिकित्सा या कार सुविधा",
        "col_limit": "कवर की गई सीमा ($)",
        "col_deductible": "आपकी कटौती योग्य शुल्क ($)",
        "col_copay": "आपका सह-भुगतान हिस्सा %",
        "col_bill": "कुल बिल राशि ($)",
        "col_oop": "आपका अंतिम जेब से खर्च ($)",
        "col_savings": "आपकी शुद्ध वित्तीय बचत ($)",
        "chat_section": "💬 इंटरैक्टिव कवरेज सहायक",
        "chat_desc": "कवरेज सीमा, दावों या शर्तों के बारे में कोई भी विशिष्ट प्रश्न पूछें।",
        "chat_placeholder": "अपना प्रश्न यहाँ लिखें (जैसे: क्या यह एमआरआई कवर करता है?)...",
        "suggested_title": "सुझाए गए प्रश्न:",
        "suggested_qs": [
            "छिपे हुए जाल क्या हैं?",
            "क्या यह आपात स्थिति को कवर करता है?",
            "मैं दावा कैसे करूँ?"
        ],
        "answers": {
            "छिपे हुए जाल क्या हैं?": "⚠️ **छिपे हुए जाल:**\n1. **सामान्य बीमारी प्रतीक्षा अवधि:** पॉलिसी शुरू होने से पहले 30 दिनों की प्रतीक्षा अवधि।\n2. **विशेष बीमारी प्रतीक्षा अवधि:** जोड़ों के प्रतिस्थापन और हर्निया जैसी सर्जरी के लिए 24 महीने।\n3. **पहले से मौजूद बीमारियां:** सक्रिय कवरेज के पहले 48 महीनों के लिए पूरी तरह से बाहर रखी गई हैं।\n4. **सह-भुगतान शुल्क:** नेटवर्क से बाहर के दावों पर कुल दावा राशि का 20% सह-भुगतान देना होगा।",
            "क्या यह आपात स्थिति को कवर करता है?": "🚨 **आपातकालीन कवरेज:**\nहाँ, आपातकालीन स्थितियों को आपकी पॉलिसी टियर की अधिकतम सीमा तक कवर किया जाता है। हालाँकि, यदि नेटवर्क से बाहर इलाज होता है, तो 20% सह-भुगतान देना होगा और भर्ती होने के 24 घंटे के भीतर बीमाकर्ता को सूचित करना होगा।",
            "मैं दावा कैसे करूँ?": "📝 **दावा करने के नियम:**\n- **कैशलेस दावा:** नेटवर्क अस्पताल में इलाज कराएं। बीमा डेस्क पर कार्ड दिखाएं। भर्ती होने के 24 घंटे के भीतर प्री-ऑथ फॉर्म भरें।\n- **प्रतिपूर्ति दावा:** बिलों का भुगतान स्वयं करें। डिस्चार्ज कार्ड, चिकित्सा बिल, रिपोर्ट और रसीदें जमा करें। 30 दिनों के भीतर ऑनलाइन जमा करें।"
        },
        "default_features": [
            "आईसीयू बेड शुल्क",
            "अस्पताल में भर्ती सर्जरी शुल्क",
            "आपातकालीन एम्बुलेंस सवारी",
            "उन्नत एमआरआइ/सीटी स्कैन",
            "नियमित परामर्श"
        ]
    },
    "Bengali": {
        "title": "🛡️ এআই পলিসি ডিকোডার",
        "subtitle": "জটিল বীমা পলিসির শর্তাবলী অবিলম্বে বুঝুন। সঠিক আর্থিক সিদ্ধান্ত নিন অত্যন্ত সহজে।",
        "lang_label": "🌐 পছন্দের ভাষা / Preferred Language / ভাষা",
        "add_file_section": "📁 বীমা নথি যোগ করুন",
        "add_file_desc": "কীভাবে নথি জমা দিতে চান তা নির্বাচন করুন:",
        "opt_upload": "নথি ফাইল আপলোড করুন",
        "opt_camera": "ক্যামেরা দিয়ে ছবি তুলুন",
        "upload_help": "আপনার পলিসি পিডিএফ বা ছবি (PNG, JPG) এখানে টেনে আনুন বা ব্রাউজ করুন",
        "camera_help": "ক্যামেরার সামনে আপনার পলিসি নথি রাখুন এবং ছবি তুলুন",
        "demo_loaded": "✨ ডেমো নথি বিশ্লেষণ লোড করা হয়েছে। পরিবর্তন করতে উপরে আপনার পলিসি আপলোড করুন।",
        "file_loaded": "✅ পলিসি সফলভাবে লোড ও ডিকোড হয়েছে: ",
        "dashboard_section": "📊 পলিসি ইনসাইট ও আর্থিক ড্যাশবোর্ড",
        "col_highlights": "হাইলাইটস সংক্ষিপ্তসার",
        "box_covered": "🟢 কি কভার করা হয়েছে (হ্যাঁ)",
        "box_not_covered": "🔴 কি কভার করা হয়নি (না)",
        "box_traps": "⚠️ লুকানো ফাঁদ এবং শর্তাবলী",
        "covered_items": [
            "জরুরি হাসপাতালে ভর্তি এবং আইসিইউ-র দৈনন্দิน বেড চার্জ।",
            "দুর্ঘটনাজনিত যানবাহনের বডি মেরামত এবং ওয়ান থার্ড পার্টি সম্পত্তির ক্ষতি।",
            "চিকিৎসকের পরামর্শ এবং জেনেরিক প্রেসক্রিপশন ওষুধ (টিয়ার-১)।"
        ],
        "not_covered_items": [
            "ঐচ্ছিক প্রসাধন চিকিৎসা এবং কসমেটিক সার্জারি।",
            "স্বাভাবিক যান্ত্রিক ক্ষয়ক্ষতি বা লুব্রিকেন্টের অভাবজনিত ইঞ্জিনের ক্ষতি।",
            "বেসরকারি হাসপাতালে নার্সিং চার্জ এবং ব্যক্তিগত আরামদায়ক সুযোগ-সুবিধা।"
        ],
        "traps_items": [
            "দুর্ঘটনাজনিত রোগ ছাড়া অন্য সব রোগের ক্ষেত্রে পলিসি শুরুর প্রথম ৩০ দিন অপেক্ষার সময়কাল।",
            "নির্দিষ্ট রোগের (হার্নিয়া, ছানি) ক্ষেত্রে ২৪ মাস অপেক্ষার সময়কাল।",
            "নেটওয়ার্কের বাইরের হাসপাতালে চিকিৎসা নিলে ২০% বাধ্যতামূলক কো-পে প্রযোজ্য।"
        ],
        "table_title": "📊 ইন্টারেক্টিভ আর্থিক হিসাব তালিকা (ট্যাবলো স্টাইল)",
        "table_desc": "আপনার খরচ অবিলম্বে হিসাব করতে স্প্রেডশীটে ডিডাক্টিবল, কো-পে শতাংশ, বা বিলের পরিমাণ পরিবর্তন করুন।",
        "col_feature": "চিকিৎসা বা কার বৈশিষ্ট্য",
        "col_limit": "কভার সীমা ($)",
        "col_deductible": "আপনার ডিডাক্টিবল ফি ($)",
        "col_copay": "আপনার কো-পে শতাংশ %",
        "col_bill": "মোট বিলের পরিমাণ ($)",
        "col_oop": "আপনার চূড়ান্ত পকেট খরচ ($)",
        "col_savings": "আপনার নেট আর্থিক সঞ্চয় ($)",
        "chat_section": "💬 ইন্টারেক্টিভ কভারেজ অ্যাসিস্ট্যান্ট",
        "chat_desc": "কভারেজ সীমা, দাবি বা শর্তাবলী সম্পর্কে যেকোনো প্রশ্ন জিজ্ঞাসা করুন।",
        "chat_placeholder": "আপনার প্রশ্ন এখানে লিখুন (যেমন: এটি কি এমআরআই কভার করে?)...",
        "suggested_title": "প্রস্তাবিত প্রশ্নাবলী:",
        "suggested_qs": [
            "লুকানো ফাঁদগুলি কী কী?",
            "এটি কি জরুরি অবস্থা কভার করে?",
            "আমি কীভাবে দাবি করব?"
        ],
        "answers": {
            "লুকানো ফাঁদগুলি কী কী?": "⚠️ **লুকানো ফাঁদসমূহ:**\n১. **সাধারণ রোগের জন্য অপেক্ষার সময়কাল:** পলিসি শুরুর দিন থেকে প্রথম ৩০ দিন অপেক্ষার সময়কাল।\n২. **বিশেষ রোগের জন্য অপেক্ষার সময়কাল:** জয়েন্ট প্রতিস্থাপন বা হার্নিয়ার মতো অস্ত্রোপচারের জন্য ২৪ মাস।\n৩. **পূর্ব-বিদ্যমান রোগ:** সক্রিয় কভারেজের প্রথম ৪৮ মাসের জন্য সম্পূর্ণ বর্জিত।\n৪. **কো-পে জরিমানা:** নেটওয়ার্কের বাইরে চিকিৎসা নিলে মোট বিলের ২০% কো-পে দিতে হবে।",
            "এটি কি জরুরি অবস্থা কভার করে?": "🚨 **জরুরি কভারেজ:**\nহ্যাঁ, জরুরি ভর্তি পলিসির সর্বোচ্চ সীমা পর্যন্ত কভার করা হয়। তবে নেটওয়ার্কের বাইরে চিকিৎসা নিলে ২০% কো-পে জরিমানা প্রযোজ্য এবং ভর্তির ২৪ ঘণ্টার মধ্যে জানাতে হবে।",
            "আমি কীভাবে দাবি করব?": "📝 **দাবি জমা দেওয়ার নিয়ম:**\n- **ক্যাশলেস দাবি:** নেটওয়ার্কভুক্ত হাসপাতালে চিকিৎসা নিন। ওয়ান থার্ড বীমা ডেস্কে হেলথ কার্ড দেখান এবং ভর্তির ২৪ ঘণ্টার মধ্যে প্রাক-অনুমোদন ফর্ম পূরণ করুন।\n- **রিইমবার্সমেন্ট দাবি:** বিল নিজে পরিশোধ করুন। ডিসচার্জ কার্ড, বিল, রিপোর্ট ও রসিদ সংগ্রহ করুন এবং ৩০ দিনের মধ্যে অনলাইনে আবেদন করুন।"
        },
        "default_features": [
            "আইসিইউ বেড চার্জ",
            "ইনপেশেন্ট সার্জিক্যাল ফি",
            "জরুরি অ্যাম্বুলেন্স যাত্রা",
            "উнят এমআরআই/সিটি স্ক্যান",
            "রুটিন পরামর্শ"
        ]
    }
}

# 3. Custom CSS Injector
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Base Reset and Slate Colors */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 1) 0%, rgba(9, 12, 22, 1) 90.1%) !important;
    color: #F8FAFC !important;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Header UI */
.header-hero {
    text-align: center;
    padding: 3rem 1.5rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.header-hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    margin-bottom: 0.75rem;
}

.header-hero p {
    font-size: 1.15rem;
    color: #94A3B8;
    max-width: 720px;
    margin: 0 auto;
    font-weight: 400;
    line-height: 1.6;
}

/* Thin Section Dividers */
.section-divider {
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    margin: 3rem 0 !important;
    opacity: 0.5;
}

/* Premium Card Design with Glassmorphism and glows */
.premium-card {
    background: rgba(30, 41, 59, 0.3) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 1.75rem !important;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
}

.premium-card:hover {
    transform: translateY(-4px) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Glow highlights per category */
.covered-card { 
    border-top: 4px solid #10B981 !important; 
}
.covered-card:hover {
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
}

.not-covered-card { 
    border-top: 4px solid #F43F5E !important; 
}
.not-covered-card:hover {
    box-shadow: 0 10px 30px rgba(244, 63, 94, 0.15) !important;
    border-color: rgba(244, 63, 94, 0.4) !important;
}

.traps-card { 
    border-top: 4px solid #F59E0B !important; 
}
.traps-card:hover {
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.15) !important;
    border-color: rgba(245, 158, 11, 0.4) !important;
}

.premium-card-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: #F8FAFC;
}

.premium-card-list {
    margin: 0;
    padding-left: 1.25rem;
    color: #CBD5E1;
    font-size: 0.95rem;
    line-height: 1.7;
}

.premium-card-list li {
    margin-bottom: 0.85rem;
}

/* Interactive Tableau Grid Wrapper */
.grid-container {
    background: rgba(30, 41, 59, 0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 1rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}

/* Chat Block Wrapper */
.chat-container-card {
    background: rgba(30, 41, 59, 0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.75rem;
    margin-top: 1rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}

/* Custom Chat Message Styling */
[data-testid="stChatMessage"] {
    background-color: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
}

/* Customize Streamlit Buttons */
.stButton>button {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.15) 100%) !important;
    border-color: #38BDF8 !important;
    color: #38BDF8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2) !important;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #38BDF8 !important;
    background: rgba(30, 41, 59, 0.3) !important;
}

/* Custom styling for inputs */
.stTextInput input {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.15) !important;
}

</style>
""", unsafe_allow_html=True)



# 4. Top Header & Language Selection
# Place Language Selection Box at the very top
col_header_title, col_lang_select = st.columns([3, 1])

with col_lang_select:
    selected_language = st.selectbox(
        "🌐 Choose Language / भाषा / ভাষা",
        ["English", "Hindi", "Bengali"],
        index=0,
        label_visibility="collapsed"
    )

# Retrieve selected dictionary
t = TRANSLATIONS[selected_language]

# Track language changes and reset state
if "lang_prev" not in st.session_state:
    st.session_state.lang_prev = selected_language

if st.session_state.lang_prev != selected_language:
    # Clear session data
    if "df_data" in st.session_state:
        del st.session_state.df_data
    if "chat_messages" in st.session_state:
        del st.session_state.chat_messages
    if "live_covered" in st.session_state:
        del st.session_state.live_covered
    if "live_not_covered" in st.session_state:
        del st.session_state.live_not_covered
    if "live_traps" in st.session_state:
        del st.session_state.live_traps
    if "document_raw_text" in st.session_state:
        del st.session_state.document_raw_text
        
    # If no file uploaded, keep it ready so the demo loads in the new language
    if st.session_state.get("current_file") is None:
        st.session_state.ready_to_process = True
    else:
        st.session_state.ready_to_process = False
        
    st.session_state.lang_prev = selected_language
    st.rerun()



def get_document_context() -> str:
    if "live_covered" not in st.session_state:
        return "No policy document decoded yet."
    
    context = (
        f"Covered Items:\n" + "\n".join([f"- {item}" for item in st.session_state.live_covered]) + "\n\n"
        f"Not Covered Items:\n" + "\n".join([f"- {item}" for item in st.session_state.live_not_covered]) + "\n\n"
        f"Hidden Traps:\n" + "\n".join([f"- {item}" for item in st.session_state.live_traps]) + "\n\n"
    )
    if "df_data" in st.session_state:
        features_list = []
        for idx, row in st.session_state.df_data.iterrows():
            features_list.append(
                f"- {row['Feature']}: Limit={row['Covered Limit ($)']}, Deductible={row['Your Deductible Fee ($)']}, Copay={row['Your Co-pay Share %']}%, Bill={row['Total Bill Amount ($)']}"
            )
        context += f"Financial Table Details:\n" + "\n".join(features_list)
        
    raw_text = st.session_state.get("document_raw_text", "")
    if raw_text:
        context = f"Raw Policy Text:\n{raw_text}\n\nSummary Context:\n" + context
        
    return context

# Header Title and Subtitle block
st.markdown(f"""
<div class="header-hero">
    <h1>{t["title"]}</h1>
    <p>{t["subtitle"]}</p>
</div>
""", unsafe_allow_html=True)

# Define feature mappings for translation
default_features_mapping = {
    "English": [
        "ICU Bed Charges",
        "Inpatient Surgical Fees",
        "Emergency Ambulance Ride",
        "Advanced MRI/CT Scans",
        "Routine Consultation"
    ],
    "Hindi": [
        "आईसीयू बेड शुल्क",
        "अस्पताल में भर्ती सर्जरी शुल्क",
        "आपातकालीन एम्बुलेंस सवारी",
        "उन्नत एमआरआइ/सीटी स्कैन",
        "नियमित परामर्श"
    ],
    "Bengali": [
        "আইসিইউ বেড চার্জ",
        "ইনপেশেন্ট সার্জিক্যাল ফি",
        "জরুরি অ্যাম্বুলেন্স যাত্রা",
        "উন্নত এমআরআই/সিটি স্ক্যান",
        "রুটিন পরামর্শ"
    ]
}

# 5. Add Document Selection Choice (Two Ways to Add Files)
st.markdown(f"### {t['add_file_section']}")
st.write(t['add_file_desc'])

input_mode = st.radio(
    "Choose Input Method",
    [t["opt_upload"], t["opt_camera"]],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file_name = None
uploaded_file_hash = None

if input_mode == t["opt_upload"]:
    # Option 1: File Uploader
    uploaded_file = st.file_uploader(
        t["upload_help"],
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key=st.session_state.uploader_key
    )
    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        uploaded_file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
else:
    # Option 2: Camera Capture tool
    camera_file = st.camera_input(
        t["camera_help"],
        label_visibility="collapsed",
        key=st.session_state.camera_key
    )
    if camera_file is not None:
        uploaded_file_name = "captured_photo.png"
        uploaded_file_hash = hashlib.md5(camera_file.getvalue()).hexdigest()

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

# IMPORTANT: identity is based on the file's CONTENT (hash), not its name.
# Two different policy PDFs can easily share the same filename (e.g. both
# named "policy.pdf"), and comparing names only would make the app think
# the second upload is the same file as the first - keeping the old,
# already-decoded results on screen instead of re-analyzing the new one.
if uploaded_file_hash != st.session_state.current_file_hash:
    st.session_state.current_file = uploaded_file_name
    st.session_state.current_file_hash = uploaded_file_hash
    st.session_state.ready_to_process = False
    st.session_state.real_ai_data = False
    if "df_data" in st.session_state:
        del st.session_state.df_data
    if "live_covered" in st.session_state:
        del st.session_state.live_covered
    if "live_not_covered" in st.session_state:
        del st.session_state.live_not_covered
    if "live_traps" in st.session_state:
        del st.session_state.live_traps
    if "document_raw_text" in st.session_state:
        del st.session_state.document_raw_text
    st.session_state.chat_messages = []
    st.rerun()

# Trigger and Reset Buttons (File Flow Control)
if uploaded_file_name:
    col_trigger, col_reset = st.columns(2)
    with col_trigger:
        if st.button("🚀 Decode Policy Document", use_container_width=True):
            if not st.session_state.get("gemini_api_key"):
                st.error(
                    "⚠️ No Gemini API key is configured. Add `GEMINI_API_KEY=your_key_here` "
                    "to a `.env` file in the project's root folder (next to app.py), then restart the app."
                )
                st.stop()
            with st.spinner("🕵️‍♂️ AI is reading your policy line-by-line... Please wait."):
                try:
                    # Determine file source object
                    if input_mode == t["opt_upload"]:
                        file_object = uploaded_file
                    else:
                        file_object = camera_file
                    
                    file_bytes = file_object.getvalue() if file_object is not None else None
                    
                    # Call structured Gemini decoder
                    analysis_result = backend.call_gemini_decoder(
                        file_object=file_bytes,
                        file_name=uploaded_file_name,
                        target_language=selected_language,
                        api_key=st.session_state.gemini_api_key
                    )
                    
                    # Save result summaries to memory state
                    st.session_state.live_covered = analysis_result.covered_items
                    st.session_state.live_not_covered = analysis_result.not_covered_items
                    st.session_state.live_traps = analysis_result.traps_items
                    
                    # Load simulated bills directly into the cost table dataframe
                    st.session_state.df_data = pd.DataFrame({
                        "Feature": default_features_mapping[selected_language],
                        "Covered Limit ($)": [10000.0, 20000.0, 500.0, 1500.0, 150.0],
                        "Your Deductible Fee ($)": [100.0, 250.0, 50.0, 75.0, 20.0],
                        "Your Co-pay Share %": [10, 15, 0, 20, 10],
                        "Total Bill Amount ($)": [
                            analysis_result.icu_bill,
                            analysis_result.surgery_bill,
                            analysis_result.ambulance_bill,
                            analysis_result.mri_bill,
                            analysis_result.consultation_bill
                        ],
                        "Your Final Out-of-Pocket Expense ($)": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "Your Net Financial Savings ($)": [0.0, 0.0, 0.0, 0.0, 0.0]
                    })
                    
                    # Extract and save raw PDF text for reference context in chat
                    if uploaded_file_name.lower().endswith('.pdf'):
                        st.session_state.document_raw_text = backend.extract_pdf_text(file_bytes)
                    else:
                        st.session_state.document_raw_text = ""
                        
                    st.session_state.ready_to_process = True
                    st.session_state.real_ai_data = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Gemini decoding failed: {str(e)}")
    with col_reset:
        if st.button("🗑️ Clear & Upload New", use_container_width=True):
            if "df_data" in st.session_state:
                del st.session_state.df_data
            if "chat_messages" in st.session_state:
                del st.session_state.chat_messages
            if "live_covered" in st.session_state:
                del st.session_state.live_covered
            if "live_not_covered" in st.session_state:
                del st.session_state.live_not_covered
            if "live_traps" in st.session_state:
                del st.session_state.live_traps
            if "document_raw_text" in st.session_state:
                del st.session_state.document_raw_text
            st.session_state.ready_to_process = False
            st.session_state.real_ai_data = False
            st.session_state.current_file = None
            st.session_state.current_file_hash = None
            st.session_state.uploader_key = f"file_uploader_{time.time()}"
            st.session_state.camera_key = f"camera_{time.time()}"
            st.rerun()

# Display status notification banner
if uploaded_file_name:
    if st.session_state.ready_to_process:
        st.info(f"{t['file_loaded']} **{uploaded_file_name}**")
    else:
        st.info("📄 Document uploaded! Click the 'Decode Policy Document' button above to start.")
        st.stop()

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Initialize live list variables if not in session state
if "live_covered" not in st.session_state:
    st.session_state.live_covered = []
if "live_not_covered" not in st.session_state:
    st.session_state.live_not_covered = []
if "live_traps" not in st.session_state:
    st.session_state.live_traps = []

# Verify check for decoded file (Empty State Gate)
if not (st.session_state.get("ready_to_process") and st.session_state.get("real_ai_data")):
    st.info("👋 Welcome! Please upload an insurance policy document PDF or snap a photo using your camera above, then click '🚀 Decode Policy Document' to generate your plain-language financial dashboard.")
    st.stop()

# 6. Policy Insights & Financial Dashboard
st.markdown(f"### {t['dashboard_section']}")
st.write("")

# Layout: Split into Text Section (left) and Data Table Section (right)
col_text_details, col_data_table = st.columns([1, 1.2], gap="large")

with col_text_details:
    st.markdown(f"#### {t['col_highlights']}")
    
    # 🟢 Covered Yes Card
    covered_items_list = "".join([f"<li>{item}</li>" for item in st.session_state.live_covered])
    st.markdown(f"""
    <div class="premium-card covered-card" style="margin-bottom: 1.2rem;">
        <div class="premium-card-title">{t['box_covered']}</div>
        <ul class="premium-card-list">
            {covered_items_list}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 🔴 Not Covered No Card
    not_covered_items_list = "".join([f"<li>{item}</li>" for item in st.session_state.live_not_covered])
    st.markdown(f"""
    <div class="premium-card not-covered-card" style="margin-bottom: 1.2rem;">
        <div class="premium-card-title">{t['box_not_covered']}</div>
        <ul class="premium-card-list">
            {not_covered_items_list}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ⚠️ Hidden Traps Card
    traps_items_list = "".join([f"<li>{item}</li>" for item in st.session_state.live_traps])
    st.markdown(f"""
    <div class="premium-card traps-card">
        <div class="premium-card-title">{t['box_traps']}</div>
        <ul class="premium-card-list">
            {traps_items_list}
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_data_table:
    st.markdown(f"#### {t['table_title']}")
    st.write(t["table_desc"])
    
    # Initialize dataframe in session state
    if "df_data" not in st.session_state:
        st.session_state.df_data = pd.DataFrame({
            "Feature": default_features_mapping[selected_language],
            "Covered Limit ($)": [10000.0, 20000.0, 500.0, 1500.0, 150.0],
            "Your Deductible Fee ($)": [100.0, 250.0, 50.0, 75.0, 20.0],
            "Your Co-pay Share %": [10, 15, 0, 20, 10],
            "Total Bill Amount ($)": [1200.0, 5000.0, 450.0, 1000.0, 150.0],
            "Your Final Out-of-Pocket Expense ($)": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Your Net Financial Savings ($)": [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
    # Language switch detection & update
    if "lang_prev" not in st.session_state:
        st.session_state.lang_prev = selected_language

    if st.session_state.lang_prev != selected_language:
        prev_features = default_features_mapping[st.session_state.lang_prev]
        new_features = default_features_mapping[selected_language]
        for idx, feat in enumerate(st.session_state.df_data["Feature"]):
            if feat in prev_features:
                match_idx = prev_features.index(feat)
                st.session_state.df_data.at[idx, "Feature"] = new_features[match_idx]
        
        # Translate live memory variables to match language change
        st.session_state.live_covered = t["covered_items"]
        st.session_state.live_not_covered = t["not_covered_items"]
        st.session_state.live_traps = t["traps_items"]
        
        st.session_state.lang_prev = selected_language

    # Calculate computed columns
    st.session_state.df_data["Your Final Out-of-Pocket Expense ($)"] = (
        st.session_state.df_data["Your Deductible Fee ($)"] +
        (st.session_state.df_data["Total Bill Amount ($)"] * (st.session_state.df_data["Your Co-pay Share %"] / 100.0))
    )
    st.session_state.df_data["Your Net Financial Savings ($)"] = (
        st.session_state.df_data["Total Bill Amount ($)"] -
        st.session_state.df_data["Your Final Out-of-Pocket Expense ($)"]
    )

    # Display spreadsheet using st.data_editor
    column_config = {
        "Feature": st.column_config.TextColumn(label=t["col_feature"], width="medium"),
        "Covered Limit ($)": st.column_config.NumberColumn(label=t["col_limit"], format="$%d"),
        "Your Deductible Fee ($)": st.column_config.NumberColumn(label=t["col_deductible"], format="$%d"),
        "Your Co-pay Share %": st.column_config.NumberColumn(label=t["col_copay"], format="%d%%"),
        "Total Bill Amount ($)": st.column_config.NumberColumn(label=t["col_bill"], format="$%d"),
        "Your Final Out-of-Pocket Expense ($)": st.column_config.NumberColumn(label=t["col_oop"], format="$%d", disabled=True),
        "Your Net Financial Savings ($)": st.column_config.NumberColumn(label=t["col_savings"], format="$%d", disabled=True)
    }

    safe_file_id = str(st.session_state.get("current_file_hash") or st.session_state.current_file).replace(".", "_") if st.session_state.current_file else "demo_view"
    edited_df = st.data_editor(
        st.session_state.df_data,
        column_config=column_config,
        use_container_width=True,
        num_rows="dynamic",
        key=f"data_editor_{safe_file_id}"
    )

    # Sync back edits
    st.session_state.df_data = edited_df

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# 7. Easy Assistant Chat Box Section
st.markdown(f"### {t['chat_section']}")
st.write(t["chat_desc"])

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Suggestions Container
st.write(f"**{t['suggested_title']}**")
col_sug1, col_sug2, col_sug3 = st.columns(3)

with col_sug1:
    if st.button(t["suggested_qs"][0], use_container_width=True):
        q = t["suggested_qs"][0]
        if not st.session_state.get("gemini_api_key"):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            st.session_state.chat_messages.append({"role": "assistant", "content": t["answers"][q]})
            st.rerun()
        else:
            with st.spinner("💬 Gemini is thinking..."):
                try:
                    ans = backend.call_gemini_chat(
                        question=q,
                        document_context=get_document_context(),
                        target_language=selected_language,
                        api_key=st.session_state.gemini_api_key
                    )
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")
with col_sug2:
    if st.button(t["suggested_qs"][1], use_container_width=True):
        q = t["suggested_qs"][1]
        if not st.session_state.get("gemini_api_key"):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            st.session_state.chat_messages.append({"role": "assistant", "content": t["answers"][q]})
            st.rerun()
        else:
            with st.spinner("💬 Gemini is thinking..."):
                try:
                    ans = backend.call_gemini_chat(
                        question=q,
                        document_context=get_document_context(),
                        target_language=selected_language,
                        api_key=st.session_state.gemini_api_key
                    )
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")
with col_sug3:
    if st.button(t["suggested_qs"][2], use_container_width=True):
        q = t["suggested_qs"][2]
        if not st.session_state.get("gemini_api_key"):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            st.session_state.chat_messages.append({"role": "assistant", "content": t["answers"][q]})
            st.rerun()
        else:
            with st.spinner("💬 Gemini is thinking..."):
                try:
                    ans = backend.call_gemini_chat(
                        question=q,
                        document_context=get_document_context(),
                        target_language=selected_language,
                        api_key=st.session_state.gemini_api_key
                    )
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")

# Render chat messages
chat_box = st.container(height=260)
with chat_box:
    if not st.session_state.chat_messages:
        st.info("Ask any question below or click a suggestion button to start.")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Text Input and Voice Mic Input Section
col_txt_in, col_mic_btn = st.columns([5, 1.2])

with col_txt_in:
    user_query = st.text_input(
        "Chat Input Text",
        placeholder=t["chat_placeholder"],
        label_visibility="collapsed",
        key="txt_query_input"
    )

with col_mic_btn:
    mic_clicked = st.button("🎙️ Record Voice", use_container_width=True)

# Process voice recording mock
if mic_clicked:
    with st.spinner("🎙️ Recording... Speak your question now."):
        time.sleep(1.5)
    st.success("Voice recorded successfully! Decoding audio...")
    
    mock_voice_qs = {
        "English": "Does this policy cover prescription drugs?",
        "Hindi": "क्या यह पॉलिसी नुस्खे वाली दवाओं को कवर करती है?",
        "Bengali": "এই পলিসি কি প্রেসক্রিপশনের ওষুধ কভার করে?"
    }
    q_text = mock_voice_qs[selected_language]
    
    if not st.session_state.get("gemini_api_key"):
        mock_answers = {
            "English": "Yes, prescription drugs are covered at 90% for tier-1 and tier-2 generic formulations, subject to a 10% co-payment.",
            "Hindi": "हाँ, नुस्खे वाली दवाओं को टियर-1 और टियर-2 जेनेरिक दवाओं के लिए 90% तक कवर किया जाता है, जिसमें 10% का सह-भुगतान लागू होता है।",
            "Bengali": "হ্যাঁ, প্রেসক্রিপশনের ওষুধগুলি টিয়ার-১ এবং টিয়ার-২ জেনেরিক ফর্মুলেশনের জন্য ৯০% কভার করা হয়, ১০% কো-পে সাপেক্ষে।"
        }
        st.session_state.chat_messages.append({"role": "user", "content": f"🎙️ [Voice]: {q_text}"})
        st.session_state.chat_messages.append({"role": "assistant", "content": mock_answers[selected_language]})
        st.rerun()
    else:
        with st.spinner("💬 Gemini is thinking..."):
            try:
                ans = backend.call_gemini_chat(
                    question=q_text,
                    document_context=get_document_context(),
                    target_language=selected_language,
                    api_key=st.session_state.gemini_api_key
                )
                st.session_state.chat_messages.append({"role": "user", "content": f"🎙️ [Voice]: {q_text}"})
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e:
                st.error(f"Chat error: {str(e)}")

# Process text input
if user_query:
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    
    if user_query != st.session_state.last_query:
        st.session_state.last_query = user_query
        
        if not st.session_state.get("gemini_api_key"):
            # Check matching fallback if no API key is set
            lower_q = user_query.lower()
            ans = ""
            if "trap" in lower_q or "जाल" in lower_q or "ফাঁদ" in lower_q:
                ans = t["answers"][t["suggested_qs"][0]]
            elif "emergency" in lower_q or "आपात" in lower_q or "জরুরি" in lower_q:
                ans = t["answers"][t["suggested_qs"][1]]
            elif "claim" in lower_q or "दावा" in lower_q or "দাবি" in lower_q:
                ans = t["answers"][t["suggested_qs"][2]]
            else:
                if selected_language == "English":
                    ans = f"Analyzing policy context for your query: *'{user_query}'*.\n\nStandard guidelines apply. Please consult Section 6.2 for deductible details regarding specific services."
                elif selected_language == "Hindi":
                    ans = f"आपकी क्वेरी के लिए पॉलिसी संदर्भ का विश्लेषण: *'{user_query}'*.\n\nमानक दिशा-निर्देश लागू होते हैं। विशिष्ट सेवाओं के लिए कटौती योग्य विवरण के लिए कृपया धारा 6.2 देखें।"
                else:
                    ans = f"আপনার অনুসন্ধানের জন্য নীতি বিশ্লেষণের বিবরণ: *'{user_query}'*.\n\nসাধারণ নির্দেশিকা প্রযোজ্য। নির্দিষ্ট পরিষেবার ডিডাক্টিবল বিবরণের জন্য অনুগ্রহ করে ধারা ৬.২ দেখুন।"
            
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            st.rerun()
        else:
            with st.spinner("💬 Gemini is thinking..."):
                try:
                    ans = backend.call_gemini_chat(
                        question=user_query,
                        document_context=get_document_context(),
                        target_language=selected_language,
                        api_key=st.session_state.gemini_api_key
                    )
                    st.session_state.chat_messages.append({"role": "user", "content": user_query})
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")