# Multi-LLM Voice Agent with Retrieval-Augmented Generation

<p align="center">
  <img src="test/1.png" alt="Multi-LLM Voice Agent UI preview" width="85%">
</p>

> A production-ready multimodal AI copilot that blends Retrieval-Augmented Generation (RAG), NVIDIA NIM-hosted large language models, and real-time voice + vision understanding to answer questions across text, code, audio, and imagery.

---

## 🚀 Why this project?
- **Enterprise-grade RAG**: Combines NVIDIA `ai-embed-qa-4` embeddings with LangChain memory to ground answers in private knowledge bases.
- **Multimodal mastery**: Seamlessly routes prompts to Meta Llama 3 (text), IBM Granite (code), Microsoft Phi-3 Vision (image reasoning), and Whisper ASR (speech-to-text).
- **Voice-first UX**: Streamlit chat with live transcription, hallucination-safe streaming responses, and persistent conversation history.
- **Extensible toolchain**: Plug in new assistants, tools, or retrieval sources without rewriting the UI.

<p align="center">
  <img src="test/2.png" alt="Architecture overview" width="85%">
</p>

---

## ✨ Key Capabilities
- **Context-aware chat** powered by Meta Llama 3 70B.
- **Developer-first code pairer** with IBM Granite 34B Code Instruct.
- **Visual question answering & OCR-style reasoning** via Phi-3 Vision.
- **Studio-quality speech transcription** courtesy of Whisper/NVIDIA ASR.
- **Long-term memory** using LangChain `ConversationBufferMemory`.
- **RAG on your documents** with persistent `vectorstore.pkl` or custom embeddings.

---

## 🧱 Tech Stack
- **Interface**: Streamlit, PIL
- **Orchestration**: LangChain, custom Assistant Router
- **Models via NVIDIA NIM**: `meta/llama3-70b-instruct`, `ibm/granite-34b-code-instruct`, `microsoft/phi-3-vision-128k-instruct`, Whisper/NEMO ASR
- **Embeddings**: `ai-embed-qa-4` (passage + query dual encoders)
- **Persistence**: `vectorstore.pkl`, `ConversationBufferMemory`, local `.wav` captures

---

## ⚙️ Quickstart

### 1. Prerequisites
- Python ≥ 3.9
- NVIDIA API access (NIM endpoints) and OpenAI-compatible Whisper key
- `ffmpeg` installed for audio capture (optional but recommended)

### 2. Clone & install
```bash
git clone https://github.com/<your-org>/Multi-LLM-Voice-Agent-with-RAG.git
cd Multi-LLM-Voice-Agent-with-RAG
python -m venv .venv
.\.venv\Scripts\activate          # On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure secrets
Create a `.env` file in the project root:
```env
NVIDIA_API_KEY=nvapi_xxx
OPENAI_API_KEY=sk-xxx                 # Needed for Whisper fallback
WHISPER_MODEL=base
EMBEDDINGS_MODEL=ai-embed-qa-4
```

> 💡 Add any custom retriever credentials (S3, Elastic, etc.) in the same `.env`.

### 4. Launch
```bash
streamlit run app.py
```
Open the local URL (default `http://localhost:8501`) and start chatting, uploading vision files, or recording audio.

---

## 🗂️ Project Layout
```text
.
├── agent/                    # Future agent tools & notebooks
├── chains/                   # Assistant router, memory, and model wrappers
│   ├── models/               # Whisper / NeMo ASR interfaces
│   ├── language_assistant.py # Llama 3 text assistant
│   ├── code_assistant.py     # Granite code assistant
│   └── vision_assistant.py   # Phi-3 Vision assistant
├── utils/                    # Helper utilities (image, routing, etc.)
├── test/                     # Reference screenshots & assets
├── app.py                    # Streamlit entry point
├── requirements.txt          # Python dependencies
├── vectorstore.pkl           # Sample persisted embeddings
└── recorded_audio.wav        # Latest captured utterance
```

---

## 🧠 Retrieval-Augmented Flow
1. **Embed** uploaded docs with `ai-embed-qa-4` (see `chains/embedding_models.py`).
2. **Persist** vectors to disk (`vectorstore.pkl`, `.npy`, or your DB of choice).
3. **Route** each user turn through `AssistantRouter`, which inspects the input and attached media.
4. **Ground** the response using the central LangChain memory + nearest vector hits.
5. **Stream** answers back to the Streamlit UI with inline code blocks, citations, or structured JSON.

> Need enterprise storage? Swap `vectorstore.pkl` with Pinecone, Milvus, or pgvector by extending `EmbeddingModels`.

---

## 🗣️ Using the Agent
- **Text**: Type into the chat box and watch Llama 3 stream responses with traceable reasoning.
- **Code**: Paste snippets or debugging logs; Granite returns fixes, tests, and refactors with syntax-highlighted blocks.
- **Vision**: Upload PNG/JPG files; Phi-3 Vision handles captioning, OCR-style extraction, or multimodal reasoning.
- **Voice**: Click “Record and Transcribe Audio”; Whisper converts to text and routes to the best assistant automatically.
- **Memory**: Prior turns persist per-session via `ConversationBufferMemory`, so follow-ups stay contextual.

---

## 🧪 Local Development Tips
- Set `streamlit run app.py --server.headless true` for remote servers.
- Use `st.secrets` in Streamlit Cloud; locally prefer `.env`.
- Record troubleshooting logs by toggling `verbose=True` in LangChain chains.
- Regenerate embeddings anytime your knowledge base changes:
  ```python
  from chains.embedding_models import EmbeddingModels
  emb = EmbeddingModels()
  emb.save_embedding("my_doc", emb.embed_documents(["content here"]))
  ```

---

## 🛣️ Roadmap
- [ ] Plug-and-play tool executor (SQL, browser, automation hooks)
- [ ] GPU-accelerated on-device Whisper + NeMo fallback
- [ ] Conversation summarization + CRM handoff webhooks
- [ ] Native deployment template for Streamlit Community Cloud + NVIDIA Inference Microservices

---

## 🤝 Contributing
1. Fork the repo & create a feature branch.
2. Run `ruff` / `black` (or your formatter of choice) before committing.
3. Submit a PR describing the use case and any new environment variables.

---

## 📣 Community & Support
- Follow [Apollum](https://x.com/x_apollum) for release notes.
- Open a GitHub Issue for bugs or feature requests.
- Share demo videos by dropping them into `docs/media/` and linking them in this README’s demo section.

Let’s build safer, smarter multimodal copilots together. 🔊🖼️💻

