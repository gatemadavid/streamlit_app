# Edureka — Generative AI and ML Capstone

A web application that uses Streamlit for the front end. The backend tranlates text, typed or uploaded documents, using **Google Gemini API**. Translated text is converted to speech using **gTTS**, and the allows the user to play or download the MP3.


## Features
- Text input or file upload (TXT, PDF, CSV, XLSX/XLS)
- 23 target languages via dropdown
- Context-aware translation through Gemini prompts
- Text-to-speech playback and MP3 download; translated text download
- Long inputs are split into chunks so they fit model limits
- Friendly error messages (missing key, bad key, quota, empty input, unreadable file)

## Setup
1. Python 3.10+ recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key at https://aistudio.google.com/apikey
4. Provide the key in one of three ways:
   - Paste it into the app sidebar (simplest)
   - Environment variable: `export GEMINI_API_KEY="your-key"`
   - `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your-key"`
5. Run:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter the API key (sidebar).
2. Type text in **Enter text**, or upload a file in **Upload file** (an uploaded file takes priority over typed text).
3. Select the target language.
4. Click **Translate**; read the result, play the audio, and download the MP3 or text.

## Architecture
| Component | Role |
|---|---|
| `extract_text()` | Reads TXT (multi-encoding), PDF (pypdf), CSV/Excel (pandas) |
| `chunk_text()` | Splits long text at line boundaries (~4,000 chars) |
| `translate()` | Sends a strict "return only the translation" prompt to Gemini per chunk |
| `synthesize()` | gTTS → in-memory MP3 bytes |
| `st.session_state` | Keeps results visible after download-button reruns |

## Deployment (Streamlit Community Cloud)
1. Push `app.py`, `requirements.txt` and this README to a GitHub repo.
2. At https://share.streamlit.io choose **New app**, select the repo and `app.py`.
3. Under **Advanced settings → Secrets**, add: `GEMINI_API_KEY = "your-key"`.
4. Deploy and share the URL. 

## Testing checklist
- Short sentence → several languages (including non-Latin scripts: Hindi, Arabic, Japanese)
- TXT, text-based PDF, CSV and multi-sheet Excel uploads
- Empty input, missing key, wrong key, oversized input
- Long document (multiple chunks) for consistent output
- Offline / gTTS failure → translation still shown with an audio warning

## Considerations and limitations
- **Model names change.** The default is `gemini-3.6-flash` (`gemini-2.5-flash` was shut down in June 2026). If a model is not found, the app automatically tries the fallbacks in `FALLBACK_MODELS`; you can also type any current model name in the sidebar. See https://ai.google.dev/gemini-api/docs/models for the live list.
- **Scanned PDFs** contain images, not text, so extraction fails (OCR would be a future enhancement).
- **gTTS** uses Google Translate's unofficial endpoint: it requires internet, has one voice per language, no voice choice, and can be rate-limited. Very long text produces long audio and may be slow.
- **Translation quality** varies by language and domain; LLMs can occasionally add, omit or soften content. Review important translations.
- **Tabular files** are flattened to lines of text, so table structure is not preserved.
- Input is capped at 50,000 characters to control cost and latency.

## Challenges faced
- The main challenge was errors from the Gemini model. I had to test multiple models to find one that works well.
- The current Gemini Model still throws an error when it receives too many requests.
- Time taken to translate and convert documents increases with the length of each document. 
- Editing code to reduce tokens consumed when translating long documents was a daunting task. 