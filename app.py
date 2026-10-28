import io
import os

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from pypdf import PdfReader

# ---------------------------------------------------------------- config
LANGUAGES = {  # display name -> gTTS language code
    "Arabic": "ar", "Bengali": "bn", "Chinese (Simplified)": "zh-CN",
    "Dutch": "nl", "English": "en", "French": "fr", "German": "de",
    "Hindi": "hi", "Indonesian": "id", "Italian": "it", "Japanese": "ja",
    "Korean": "ko", "Polish": "pl", "Portuguese": "pt", "Russian": "ru",
    "Spanish": "es", "Swahili": "sw", "Tamil": "ta", "Telugu": "te",
    "Thai": "th", "Turkish": "tr", "Urdu": "ur", "Vietnamese": "vi",
}
DEFAULT_MODEL = "gemini-3.5-flash"
# Tried in order if the chosen model has been retired / is not found.
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.8-flash"]
CHUNK_SIZE = 4000          # characters per Gemini request
MAX_CHARS = 50000          # safety cap on input size
SUPPORTED_FILES = ["txt", "pdf", "csv", "xlsx", "xls"]


# ---------------------------------------------------------------- helpers
def get_api_key() -> str:
    """Look for the key in Streamlit secrets, then env, then sidebar input."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def extract_text(uploaded) -> str:
    """Extract plain text from TXT, PDF, CSV or Excel uploads."""
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        raw = uploaded.read()
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode the text file.")
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if not text.strip():
            raise ValueError("No extractable text found (scanned PDFs are not supported).")
        return text
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str).fillna("")
    elif name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(uploaded, sheet_name=None, dtype=str)
        df = pd.concat(sheets.values(), ignore_index=True).fillna("")
    else:
        raise ValueError("Unsupported file type.")
    return "\n".join(" ".join(c for c in row if c.strip()) for row in df.values.tolist())


def chunk_text(text: str, size: int = CHUNK_SIZE):
    """Split on paragraph/line boundaries so sentences stay intact."""
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > size and current:
            chunks.append(current)
            current = ""
        while len(line) > size:               # very long single line
            chunks.append(line[:size])
            line = line[size:]
        current += line
    if current.strip():
        chunks.append(current)
    return chunks


def _response_text(resp) -> str:
    """Get text from a response, even when resp.text is None."""
    try:
        if resp.text:
            return resp.text
    except Exception:
        pass
    parts = []
    for cand in resp.candidates or []:
        if cand.content and cand.content.parts:
            parts += [p.text for p in cand.content.parts if getattr(p, "text", None)]
    return "".join(parts)


def _empty_reason(resp) -> str:
    fb = getattr(resp, "prompt_feedback", None)
    if fb and getattr(fb, "block_reason", None):
        return f"the prompt was blocked ({fb.block_reason})"
    if resp.candidates:
        return f"finish reason: {resp.candidates[0].finish_reason}"
    return "no candidates were returned"


def _generate(client, model: str, prompt: str) -> str:
    cfg = {
        "safety_settings": [
            types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
            for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ],
    }
    # Thinking tokens can eat the output budget; translation doesn't need them.
    if "2.5" in model and "flash" in model:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    resp = client.models.generate_content(
        model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
    text = _response_text(resp).strip()
    if not text:
        raise RuntimeError(f"Gemini returned an empty response: {_empty_reason(resp)}.")
    return text


def _translate_chunk(client, model: str, target: str, chunk: str, depth: int = 0) -> str:
    prompt = (
        f"Translate the following text into {target}. "
        "Preserve meaning, tone and line breaks. "
        "Return ONLY the translation, with no explanations or notes.\n\n"
        f"{chunk}"
    )
    try:
        return _generate(client, model, prompt)
    except RuntimeError:
        # Empty reply: retry once, then split the chunk in half and translate each part.
        if depth == 0:
            return _translate_chunk(client, model, target, chunk, depth=1)
        if len(chunk) > 500 and depth < 4:
            mid = chunk.rfind("\n", 0, len(chunk) // 2 + 200)
            mid = mid if mid > 0 else len(chunk) // 2
            left = _translate_chunk(client, model, target, chunk[:mid], depth + 1)
            right = _translate_chunk(client, model, target, chunk[mid:], depth + 1)
            return left + "\n" + right
        raise


def translate(text: str, target: str, api_key: str, model: str) -> str:
    client = genai.Client(api_key=api_key)
    chunks = chunk_text(text)
    candidates = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_err = None
    for m in candidates:
        try:
            return "\n\n".join(_translate_chunk(client, m, target, c) for c in chunks)
        except Exception as e:
            low = str(e).lower()
            if "404" in low or "not found" in low or "no longer available" in low:
                last_err = e   # model retired/unavailable: try the next one
                continue
            raise
    raise last_err


def synthesize(text: str, lang_code: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang=lang_code).write_to_fp(buf)
    return buf.getvalue()


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "403" in low or "permission" in low:
        return "Your API key looks invalid or lacks permission. Please check it in the sidebar."
    if "429" in low or "quota" in low or "resource_exhausted" in low:
        return "Rate limit or quota reached. Wait a moment and try again."
    if "404" in low or "not found" in low:
        return "The selected Gemini model was not found. Try another model name in the sidebar."
    return f"Something went wrong: {msg}"


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Google gTTS Project", page_icon="🌎", layout="centered")
st.title("🌎 Edureka Capstone Project")
st.caption("Translate text or documents with Google Gemini, then listen to or download the audio.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", value=get_api_key(), type="password",
                            help="Get a free key at aistudio.google.com/apikey")
    model = st.text_input("Gemini model", value=DEFAULT_MODEL)
    st.markdown("---")
    st.markdown(
        "**How to use**\n"
        "1. Paste your Google API key.\n"
        "2. Type text or upload a TXT, PDF, CSV or Excel file.\n"
        "3. Select the target language.\n"
        "4. Click **Translate** — then play or download the MP3."
    )

tab_text, tab_file = st.tabs(["✒️ Enter text", "📂 Upload file"])
source_text = ""
with tab_text:
    typed = st.text_area("Text to translate", height=180, placeholder="Type or paste text here…")
with tab_file:
    uploaded = st.file_uploader("Upload a file", type=SUPPORTED_FILES)
    if uploaded:
        try:
            file_text = extract_text(uploaded)
            st.success(f"Extracted {len(file_text):,} characters from {uploaded.name}.")
            with st.expander("Preview extracted text"):
                st.text(file_text[:2000] + ("…" if len(file_text) > 2000 else ""))
            source_text = file_text
        except Exception as e:
            st.error(f"Could not read the file: {e}")

# A typed message takes priority only if no file was successfully read.
source_text = source_text or typed

target_lang = st.selectbox("Translate into", list(LANGUAGES), index=list(LANGUAGES).index("Spanish"))

if st.button("Translate", type="secondary", use_container_width=True):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not source_text.strip():
        st.warning("Please enter some text or upload a file first.")
    elif len(source_text) > MAX_CHARS:
        st.error(f"Input is too long ({len(source_text):,} characters). Limit is {MAX_CHARS:,}.")
    else:
        try:
            with st.spinner("Translating with Gemini…"):
                translated = translate(source_text, target_lang, api_key, model)
            audio, audio_error = None, None
            try:
                with st.spinner("Generating speech…"):
                    audio = synthesize(translated, LANGUAGES[target_lang])
            except Exception as e:  # gTTS needs internet; keep the translation anyway
                audio_error = str(e)
            st.session_state["result"] = {
                "text": translated, "audio": audio, "lang": target_lang, "audio_error": audio_error,
            }
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(friendly_error(e))

# Results persist across reruns (e.g. after clicking a download button).
result = st.session_state.get("result")
if result:
    st.subheader(f"Translation ({result['lang']})")
    st.text_area("Translated text", result["text"], height=220, label_visibility="collapsed")
    st.download_button("📥 Download text", result["text"], file_name="translation.txt")
    if result["audio"]:
        st.audio(result["audio"], format="audio/mp3")
        st.download_button("📥 Download audio (MP3)", result["audio"],
                           file_name=f"translation_{LANGUAGES[result['lang']]}.mp3", mime="audio/mpeg")
    else:
        st.warning(f"Translation succeeded, but audio could not be generated: {result['audio_error']}")