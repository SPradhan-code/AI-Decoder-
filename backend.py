import sys
import os

# Ensure virtual environment site-packages is prioritized
venv_site_packages = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages"))
if os.path.exists(venv_site_packages):
    while venv_site_packages in sys.path:
        sys.path.remove(venv_site_packages)
    sys.path.insert(0, venv_site_packages)

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import pdfplumber
import PIL.Image
import io
import time
import random

# Errors that are worth retrying because they're transient / server-side,
# not a problem with the request itself.
RETRYABLE_STATUS_MARKERS = ("503", "UNAVAILABLE", "OVERLOADED", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(marker in msg for marker in RETRYABLE_STATUS_MARKERS)

def _call_with_retry(fn, max_retries: int = 4, base_delay: float = 2.0):
    """
    Calls fn() and retries with exponential backoff + jitter if the
    failure looks transient (e.g. Gemini's '503 UNAVAILABLE - model
    overloaded'). Non-transient errors (bad API key, invalid file, etc.)
    are raised immediately without wasting retries.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == max_retries:
                raise
            # Exponential backoff with a little jitter so we don't hammer
            # the API at exactly the same intervals on repeated attempts.
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
    raise last_exc

class PolicyAnalysis(BaseModel):
    covered_items: List[str] = Field(description="3 items showing what is covered, written in a clear, friendly, and simple language.")
    not_covered_items: List[str] = Field(description="3 items showing what is NOT covered, written in a clear, friendly, and simple language.")
    traps_items: List[str] = Field(description="3 items showing hidden traps, waiting periods, limits, or co-pay rules, written in simple language.")
    icu_bill: float = Field(description="A simulated default bill amount for ICU bed charges in dollars.")
    surgery_bill: float = Field(description="A simulated default bill amount for inpatient surgery fee in dollars.")
    ambulance_bill: float = Field(description="A simulated default bill amount for emergency ambulance ride in dollars.")
    mri_bill: float = Field(description="A simulated default bill amount for advanced MRI/CT scans in dollars.")
    consultation_bill: float = Field(description="A simulated default bill amount for routine consultation in dollars.")

def extract_pdf_text(file_object) -> str:
    """Extracts all text from a PDF file using pdfplumber."""
    try:
        if isinstance(file_object, bytes):
            file_object = io.BytesIO(file_object)
        with pdfplumber.open(file_object) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

def call_gemini_decoder(file_object, file_name: str, target_language: str, api_key: str) -> PolicyAnalysis:
    """Connects to the Gemini model to extract policy details as structured JSON using the user's API key."""
    # FIX: Correctly utilize the passed api_key variable
    client = genai.Client(api_key=api_key)
    
    is_pdf = file_name.lower().endswith('.pdf')
    contents = []
    
    if is_pdf:
        pdf_text = extract_pdf_text(file_object)
        if len(pdf_text.strip()) < 100:
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_object)
                max_pages = min(len(pdf), 8)
                for i in range(max_pages):
                    page = pdf[i]
                    pil_image = page.render_topil(scale=2)
                    contents.append(pil_image)
                contents.append("Here are the insurance policy document page images (scanned PDF).")
                pdf.close()
            except Exception as e:
                contents.append(f"Here is the insurance policy document text (extraction failed): {str(e)}\n\n{pdf_text}")
        else:
            contents.append(f"Here is the insurance policy document text:\n\n{pdf_text}")
    else:
        try:
            if isinstance(file_object, bytes):
                image = PIL.Image.open(io.BytesIO(file_object))
            else:
                image = PIL.Image.open(file_object)
            contents.append(image)
            contents.append("Here is the insurance policy document image.")
        except Exception as e:
            raise ValueError(f"Failed to open image file: {str(e)}")
            
    contents.append(
        f"You are a professional insurance analyzer. Please analyze the provided insurance policy document and extract "
        f"covered items, non-covered items, hidden conditions/traps, and reasonable simulated hospital bill numbers "
        f"for standard services. Translate all text summaries and features into '{target_language}'. "
        f"Provide exactly 3 bullet points for covered_items, 3 for not_covered_items, and 3 for traps_items."
    )
    
    try:
        response = _call_with_retry(lambda: client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PolicyAnalysis,
                temperature=0.1
            )
        ))
    except Exception as e:
        if _is_retryable(e):
            raise RuntimeError(
                "Gemini's servers are currently overloaded (503 UNAVAILABLE) and stayed "
                "unavailable after several automatic retries. This is on Google's end, not "
                "your document or API key - please wait a minute and click 'Decode Policy "
                "Document' again."
            ) from e
        raise

    return response.parsed

def call_gemini_chat(question: str, document_context: str, target_language: str, api_key: str) -> str:
    """Uses Gemini to chat about the policy document context using the user's API key."""
    # FIX: Correctly utilize the passed api_key variable with valid syntax
    client = genai.Client(api_key=api_key)
    
    prompt = (
        f"You are an AI Insurance Assistant. You have access to the following parsed insurance document context:\n"
        f"--- START CONTEXT ---\n"
        f"{document_context}\n"
        f"--- END CONTEXT ---\n\n"
        f"The user's preferred language is {target_language}. All your responses must be in {target_language} "
        f"and explain policy terms accurately, in simple terms. Be polite and helpful.\n\n"
        f"User question: {question}"
    )
    
    try:
        response = _call_with_retry(lambda: client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        ))
    except Exception as e:
        if _is_retryable(e):
            raise RuntimeError(
                "Gemini's servers are currently overloaded (503 UNAVAILABLE) and stayed "
                "unavailable after several automatic retries. Please wait a moment and "
                "ask your question again."
            ) from e
        raise

    return response.text