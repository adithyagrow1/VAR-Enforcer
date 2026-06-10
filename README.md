# ⚽ VAR Enforcer

**AI-Powered FIFA World Cup VAR Decision Explainer**

An intelligent system that explains Video Assistant Referee (VAR) decisions using IBM Granite LLM, Docling for PDF parsing, and ChromaDB for semantic search. Built with FastAPI and featuring a modern web interface.

![VAR Enforcer](https://img.shields.io/badge/AI-Powered-00ff88?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge)
![IBM Granite](https://img.shields.io/badge/IBM-Granite-052FAD?style=for-the-badge)

---

## 🎯 Features

- **🤖 AI-Powered Analysis**: Uses IBM Granite LLM for intelligent VAR decision explanations
- **📚 RAG Pipeline**: Retrieval-Augmented Generation with FIFA Laws of the Game
- **🔍 Semantic Search**: ChromaDB vector store for accurate rule retrieval
- **📊 Controversy Meter**: Rates decisions from 1-10 (clear vs controversial)
- **🌍 Multilingual**: Supports English, Spanish, French, and Portuguese
- **📈 Historical Context**: Integrates 49,000+ match records for consistency analysis
- **🎨 Modern UI**: Dark-themed, responsive interface with real-time updates
- **🔄 Demo Mode**: Works without API credentials for testing and development

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Backend│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│   RAG   │ │   Granite    │
│Pipeline │ │   Engine     │
└────┬────┘ └──────┬───────┘
     │             │
     ▼             ▼
┌─────────┐   ┌─────────┐
│ChromaDB │   │IBM      │
│Vector   │   │Watsonx  │
│Store    │   │AI       │
└─────────┘   └─────────┘
```

---

## 📋 Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **IBM Watsonx.ai account** (optional - demo mode available)
- **FIFA Laws of the Game PDF** (place in `app/data/fifa_laws.pdf`)
- **Match data CSV** (place in `app/data/results.csv`)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VAR-ENFORCER
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy and edit the `.env` file:

```bash
# IBM Watsonx.ai / Granite Configuration
IBM_WATSONX_API_KEY=your_api_key_here
IBM_WATSONX_PROJECT_ID=your_project_id_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Granite Model
GRANITE_MODEL_ID=ibm/granite-13b-chat-v2

# PDF Configuration
FIFA_PDF_URL=app/data/fifa_laws.pdf

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=fifa_laws

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Text Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API Configuration
API_HOST=0.0.0.0
API_PORT=8002

# Demo Mode (set to true to run without API credentials)
DEMO_MODE=false
```

### 4. Prepare Data Files

Place the following files in the `app/data/` directory:

- **`fifa_laws.pdf`**: FIFA Laws of the Game PDF
- **`results.csv`**: Historical match data (49,000+ records)

### 5. Run the Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002

# Or use the built-in runner
python -m app.main
```

### 6. Access the Application

Open your browser and navigate to:

- **Frontend**: http://localhost:8002
- **API Docs**: http://localhost:8002/docs
- **Alternative Docs**: http://localhost:8002/redoc

---

## 🎮 Usage

### Web Interface

1. **Open** http://localhost:8002 in your browser
2. **Describe** the VAR incident in the text area
3. **Add** optional match context (teams, tournament, etc.)
4. **Select** your preferred language
5. **Click** "Analyze VAR Decision"
6. **View** the comprehensive analysis including:
   - Technical explanation
   - FIFA rules cited
   - Controversy meter (1-10)
   - Historical consistency note
   - Plain language summary

### API Endpoints

#### POST `/api/explain` - Explain VAR Decision

```bash
curl -X POST "http://localhost:8002/api/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_description": "The defender made contact with the attacker in the penalty area",
    "match_context": "World Cup Final 2022",
    "language": "en",
    "top_k": 5
  }'
```

#### GET `/api/health` - System Health Check

```bash
curl "http://localhost:8002/api/health"
```

#### GET `/api/consistency/{team1}/{team2}` - Historical Matches

```bash
curl "http://localhost:8002/api/consistency/Argentina/France?limit=10"
```

#### POST `/api/initialize` - Rebuild Vector Store

```bash
curl -X POST "http://localhost:8002/api/initialize" \
  -H "Content-Type: application/json" \
  -d '{"force_reload": true}'
```

#### GET `/api/rules/search` - Search FIFA Rules

```bash
curl "http://localhost:8002/api/rules/search?query=handball&top_k=5"
```

#### GET `/api/stats` - System Statistics

```bash
curl "http://localhost:8002/api/stats"
```

---

## 🧪 Demo Mode

VAR Enforcer includes a **demo mode** that works without IBM Watsonx credentials:

1. Set `DEMO_MODE=true` in `.env` or don't configure API credentials
2. The system will use realistic mock responses
3. Perfect for testing, development, and demonstrations
4. All features work except live Granite API calls

---

## 📁 Project Structure

```
VAR-ENFORCER/
├── app/
│   ├── main.py              # FastAPI application
│   ├── rag_pipeline.py      # RAG pipeline with Docling & ChromaDB
│   ├── granite_engine.py    # IBM Granite LLM integration
│   ├── data/
│   │   ├── fifa_laws.pdf    # FIFA Laws of the Game
│   │   └── results.csv      # Historical match data
│   └── static/
│       └── index.html       # Frontend interface
├── chroma_db/               # ChromaDB persistence (auto-created)
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
└── README.md               # This file
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IBM_WATSONX_API_KEY` | IBM Watsonx API key | Required for live mode |
| `IBM_WATSONX_PROJECT_ID` | IBM Watsonx project ID | Required for live mode |
| `IBM_WATSONX_URL` | Watsonx endpoint URL | `https://us-south.ml.cloud.ibm.com` |
| `GRANITE_MODEL_ID` | Granite model identifier | `ibm/granite-13b-chat-v2` |
| `FIFA_PDF_URL` | Path to FIFA Laws PDF | `app/data/fifa_laws.pdf` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | Collection name | `fifa_laws` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size (chars) | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap (chars) | `200` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8002` |
| `DEMO_MODE` | Enable demo mode | `false` |

---

## 🎨 Frontend Features

- **Dark FIFA-themed design** with green accents
- **Real-time health status** indicator
- **Quick-fill examples** for common VAR scenarios
- **Controversy meter** with color-coded visualization
- **Responsive design** - works on mobile and desktop
- **Multilingual support** - 4 languages
- **Loading animations** and smooth transitions
- **Error handling** with user-friendly messages

---

## 🔍 How It Works

### 1. Document Processing (RAG Pipeline)

```python
# Parse FIFA Laws PDF with Docling
text = docling.parse_pdf("fifa_laws.pdf")

# Chunk text with overlap
chunks = chunk_text(text, size=1000, overlap=200)

# Generate embeddings
embeddings = sentence_transformers.encode(chunks)

# Store in ChromaDB
chromadb.add(chunks, embeddings)
```

### 2. Query Processing

```python
# User submits VAR incident
query = "Defender touched ball first but also hit attacker"

# Retrieve relevant FIFA rules
rules = chromadb.query(query, top_k=5)

# Generate explanation with Granite
explanation = granite.generate(
    prompt=build_prompt(query, rules),
    language="en"
)
```

### 3. Response Structure

```json
{
  "decision_explanation": "Technical analysis...",
  "rule_cited": ["Law 12 - Fouls", "VAR Protocol"],
  "controversy_score": 6,
  "consistency_note": "Aligns with recent decisions...",
  "plain_language_summary": "Simple explanation...",
  "language": "en"
}
```

---

## 🧩 Components

### RAG Pipeline (`app/rag_pipeline.py`)

- **Docling Integration**: Parses PDF with OCR support
- **Text Chunking**: Smart chunking with paragraph preservation
- **ChromaDB**: Persistent vector storage
- **Semantic Search**: Similarity-based retrieval

### Granite Engine (`app/granite_engine.py`)

- **IBM Watsonx Integration**: SDK and REST API support
- **Prompt Engineering**: Structured prompts for VAR analysis
- **Multilingual**: 4 language support
- **Controversy Scoring**: 1-10 scale with guidelines
- **Demo Mode**: Mock responses for testing

### FastAPI Backend (`app/main.py`)

- **Async Operations**: Non-blocking request handling
- **CORS Support**: Cross-origin requests enabled
- **Error Handling**: Global exception handler
- **Validation**: Pydantic models for type safety
- **Auto-initialization**: Loads data on startup

---

## 📊 Data Requirements

### FIFA Laws PDF

- Official FIFA Laws of the Game document
- PDF format (OCR-capable)
- Place in `app/data/fifa_laws.pdf`

### Match Data CSV

Expected columns:
- `date`: Match date
- `home_team`: Home team name
- `away_team`: Away team name
- Additional columns as needed

Place in `app/data/results.csv`

---

## 🐛 Troubleshooting

### Vector Store Not Initializing

```bash
# Manually rebuild vector store
curl -X POST "http://localhost:8002/api/initialize" \
  -H "Content-Type: application/json" \
  -d '{"force_reload": true}'
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Port Already in Use

```bash
# Change port in .env
API_PORT=8003

# Or specify when running
uvicorn app.main:app --port 8003
```

### Demo Mode Not Working

Ensure `DEMO_MODE=true` in `.env` or remove API credentials.

---

## 🚀 Deployment

### Production Considerations

1. **Set specific CORS origins** in `app/main.py`
2. **Use environment-specific `.env` files**
3. **Enable HTTPS** with reverse proxy (nginx/Apache)
4. **Set `DEBUG=false`** in production
5. **Use production ASGI server** (gunicorn + uvicorn workers)

### Example Production Command

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8002 \
  --timeout 120
```

---

## 📝 API Documentation

Full interactive API documentation is available at:

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc
- **OpenAPI Schema**: http://localhost:8002/openapi.json

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- **IBM Granite** - LLM for intelligent explanations
- **Docling** - PDF parsing and document processing
- **ChromaDB** - Vector database for semantic search
- **FastAPI** - Modern Python web framework
- **Sentence Transformers** - Embedding generation
- **FIFA/IFAB** - Laws of the Game

---

## 📧 Support

For issues, questions, or suggestions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review API documentation at `/docs`
3. Open an issue on GitHub

---

## 🎯 Roadmap

- [ ] Add more languages (German, Italian, Arabic)
- [ ] Real-time VAR incident tracking
- [ ] Video clip analysis integration
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Community voting on decisions
- [ ] Historical decision database

---

**Built with ❤️ for transparent and educational VAR analysis**

⚽ **VAR Enforcer** - Making football decisions clearer, one explanation at a time.