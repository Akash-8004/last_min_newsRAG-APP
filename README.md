# Live News Analyst

A real-time RAG (Retrieval-Augmented Generation) chatbot that ingests articles from a live news API and answers nuanced questions about breaking news stories. The system's understanding evolves continuously as new reports, updates, and corrections are published.

## Architecture Overview

The Live News Analyst implements a streaming RAG architecture using Pathway framework with the following components:

### Core Components
- **Real-time Data Ingestion**: Custom NewsAPI connector fetching live news articles
- **Document Processing Pipeline**: Text chunking with token-based splitting
- **Vector Embeddings**: Sentence transformers for semantic search
- **Vector Index**: KNN-based similarity search for relevant content retrieval
- **LLM Integration**: Groq/OpenAI integration for answer generation
- **HTTP API**: RESTful endpoint for query processing

### Data Flow
```
NewsAPI → Custom Connector → Document Chunking → Embedding → Vector Index
                                                                    ↓
User Query → Query Embedding → Similarity Search → Context Building → LLM → Response
```

### Technology Stack
- **Framework**: Pathway (streaming data processing)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Search**: KNN Index with cosine similarity
- **LLM**: Groq API (llama3-70b-8192) or OpenAI GPT-4
- **News Source**: NewsAPI.org
- **Text Processing**: tiktoken for token counting

## Setup and Installation

### Prerequisites
- Python 3.10+
- NewsAPI.org API key
- Groq API key (or OpenAI API key)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv pw_llm_env
source pw_llm_env/bin/activate  # On Windows: pw_llm_env\Scripts\activate

# Install dependencies
pip install requirments.txt
```

### 2. Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional alternative
NEWS_API_KEY=your_newsapi_key_here
```

### 3. API Keys Setup
- **NewsAPI**: Register at [newsapi.org](https://newsapi.org) for free API access
- **Groq**: Get API key from [console.groq.com](https://console.groq.com)
- **OpenAI**: Alternative LLM provider at [platform.openai.com](https://platform.openai.com)

## Execution Instructions

### Running the Application

1. **Start the Jupyter Notebook**:
```bash
jupyter notebook pathway.ipynb
```

2. **Execute cells sequentially** or **Run as Python script**:
```bash
# Convert notebook to Python script
jupyter nbconvert --to script pathway.ipynb
python pathway.py
```

3. **Access the API**:
The application starts an HTTP server on `http://localhost:8080`

### Making Queries

Send POST requests to the `/query` endpoint:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest technology news?"}'
```

**Example queries**:
- "What's happening with Trump's tariff policies?"
- "Tell me about recent sports news"
- "What are the latest developments in technology?"

## Real-time / Streaming Functionality

### Continuous Data Ingestion
- **Automatic Updates**: News connector fetches fresh articles every 15 minutes (900 seconds)
- **Real-time Processing**: New articles are immediately processed through the RAG pipeline
- **Dynamic Index**: Vector index updates automatically as new content arrives
- **Live Context**: Query responses reflect the most recent news developments

### Streaming Architecture Benefits
1. **Always Current**: Responses include breaking news and latest updates
2. **Incremental Updates**: Only new articles are processed, maintaining efficiency
3. **Scalable Processing**: Pathway's streaming engine handles continuous data flow
4. **Memory Efficient**: Processes data in chunks without loading entire datasets

### Data Freshness
- **News Refresh Rate**: Every 15 minutes
- **Processing Latency**: < 30 seconds for new articles to become searchable
- **Query Response Time**: Typically 2-5 seconds including LLM generation
- **Content Scope**: Top 20 headlines per fetch cycle

### Monitoring and Observability
The system provides real-time feedback:
- Article fetch counts and status codes
- Processing pipeline progress
- Query processing metrics
- Error handling and recovery

## Configuration Options

### Customizable Parameters
- **Chunk Size**: 400 tokens (adjustable for different content types)
- **Chunk Overlap**: 50 tokens (ensures context continuity)
- **Retrieval Count**: Top 5 relevant chunks per query
- **Update Frequency**: 15-minute intervals (configurable)
- **News Categories**: Technology focus (expandable to other topics)

### Performance Tuning
- Adjust embedding model for speed vs. accuracy trade-offs
- Modify chunk sizes based on content complexity
- Configure retrieval parameters for precision vs. recall optimization

## Project Structure
```
last_min_news/
├── pathway.ipynb          # Main application notebook
├── README.md             # This documentation
├── .env                  # Environment variables (create this)
└── .dist/               # Distribution files
```

## Features
- ✅ Real-time news ingestion from NewsAPI
- ✅ Semantic search with vector embeddings
- ✅ Context-aware answer generation
- ✅ RESTful API interface
- ✅ Streaming data processing
- ✅ Automatic content updates
- ✅ Error handling and recovery

## Limitations
- Requires active internet connection for news fetching
- NewsAPI free tier has rate limits (1000 requests/day)
- Processing latency depends on article volume and complexity
- LLM costs scale with query frequency

## Future Enhancements
- Multi-source news aggregation
- Advanced filtering and categorization
- User preference learning
- Historical news analysis
- Multi-language support
