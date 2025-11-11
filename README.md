# Simple AI Chatbot with RAG (Retrieval-Augmented Generation)

A conversational AI chatbot that answers questions based on content from any website URL. Built with LangChain, OpenAI, and FAISS vector store.

## Features

- **Website Knowledge Base**: Load content from any website URL
- **Conversational Memory**: Maintains chat history for follow-up questions
- **RAG Architecture**: Retrieval-Augmented Generation for accurate, context-based answers
- **Vector Search**: Uses FAISS for efficient semantic search
- **LangChain LCEL**: Modern LangChain Expression Language implementation

## Architecture

The chatbot uses a RAG (Retrieval-Augmented Generation) pipeline:

1. **Web Scraping**: Loads website content using `WebBaseLoader`
2. **Text Chunking**: Splits content into manageable chunks with overlap
3. **Embedding**: Converts chunks to vector embeddings using OpenAI's `text-embedding-ada-002`
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Finds relevant chunks based on user query
6. **Generation**: LLM generates answer using retrieved context and chat history

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Website   │────>│ Text Chunks  │────>│  Embeddings │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │<────│     LLM      │<────│    FAISS    │
└─────────────┘     └──────────────┘     └─────────────┘
                           ▲
                           │
                    ┌──────────────┐
                    │ User Query + │
                    │ Chat History │
                    └──────────────┘
```

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/khalilCodeX/SimpleAIChatbot.git
   cd SimpleAIChatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API Key**
   
   Option 1: Environment variable
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   Option 2: Create `.env` file
   ```bash
   echo "OPEN_AI_KEY=your-api-key-here" > .env
   ```
   
   Option 3: Enter interactively when prompted

## Required Packages

Create a `requirements.txt` file:
```txt
langchain-core
langchain-community
langchain-openai
openai
faiss-cpu
beautifulsoup4
```

## Usage

### Basic Usage

```python
from chat import chatbot

# Initialize chatbot with a URL
bot = chatbot("https://www.apple.com/iphone/")

# Ask a question
answer = bot.create_chain("What iPhones are available?")
print(answer)
```

### Interactive Chat

Run the main program:
```bash
python chat.py
```

Example conversation:
```
You: I want to purchase an iPhone
AI: There are iPhone 13, 14, and 15 available...

You: What is the price of the oldest one?
AI: The iPhone 13 starts at $599...

You: What colors does it come in?
AI: The iPhone 13 is available in Pink, Blue, Midnight, Starlight, and Red...
```

## Project Structure

```
SimpleAIChatbot/
│
├── chat.py              # Main chatbot class and entry point
├── WebLoader.py         # Website content loading and chunking
├── Vectordb.py          # Vector store creation and retrieval
├── management.py        # OpenAI API key management
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Module Breakdown

### `chat.py`
The main chatbot class that orchestrates the RAG pipeline.

**Key Methods:**
- `__init__(url)`: Initializes chatbot with website content
- `create_chain(user_prompt)`: Processes user query and returns answer
- `format_doc(documents)`: Formats retrieved documents into context string

### `WebLoader.py`
Handles loading and processing website content.

**Functions:**
- `load_website(url)`: Scrapes content from URL using BeautifulSoup
- `chunk_documents(documents)`: Splits text into 1000-character chunks with 200-character overlap

### `Vectordb.py`
Manages vector embeddings and similarity search.

**Functions:**
- `create_vectorstore(splits)`: Creates FAISS vector store from document chunks
- `query_vectorstore(retriever, query)`: Retrieves relevant documents

### `management.py`
Handles OpenAI API key configuration.

**Functions:**
- `initialize_openai_client()`: Sets up OpenAI client with API key

## How It Works

### Step-by-Step Process

1. **Initialization**
   ```python
   bot = chatbot("https://example.com")
   ```
   - Loads website content
   - Splits into chunks
   - Creates embeddings
   - Stores in FAISS vector database

2. **Query Processing**
   ```python
   answer = bot.create_chain("Your question here")
   ```
   - Embeds user query
   - Searches for relevant chunks (similarity search)
   - Retrieves top matching documents
   - Passes context + chat history to LLM
   - Generates and returns answer

3. **Chat History**
   - Stores previous Q&A pairs
   - Enables follow-up questions
   - LLM uses history for context

## Customization

### Change LLM Model

```python
self.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
```

### Adjust Chunk Size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Smaller chunks
    chunk_overlap=100    # Less overlap
)
```

### Filter Website Content

```python
bs4_strainer = bs4.SoupStrainer(class_=("content", "article"))
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs={"parse_only": bs4_strainer}
)
```

### Modify System Prompt

```python
system_prompt = "You are an expert assistant specializing in..."
```

## Example Use Cases

### E-commerce Product Assistant
```python
bot = chatbot("https://www.bestbuy.com/site/laptops")
answer = bot.create_chain("Which laptop has the best battery life?")
```

### Documentation Helper
```python
bot = chatbot("https://python.langchain.com/docs/")
answer = bot.create_chain("How do I create a retrieval chain?")
```

### Company FAQ Bot
```python
bot = chatbot("https://yourcompany.com/about")
answer = bot.create_chain("What are your business hours?")
```

## Limitations

- **Static Content Only**: Works best with static websites (JavaScript-heavy sites may need Selenium)
- **Token Limits**: Very large websites may exceed context limits
- **Rate Limits**: OpenAI API has rate limits on embeddings and completions
- **Cost**: Each query uses embeddings + LLM tokens (monitor usage)

## Troubleshooting

### "No documents were loaded"
- Check URL is accessible
- Website may block web scrapers
- Try removing BeautifulSoup filters

### "ModuleNotFoundError"
```bash
pip install langchain-core langchain-community langchain-openai faiss-cpu
```

### "API key not set"
```bash
export OPENAI_API_KEY='your-key-here'
```

### "0 chunks created"
- Remove `bs4_strainer` filter in `WebLoader.py`
- Check website HTML structure

## Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Guide](https://platform.openai.com/docs/)
- [FAISS Documentation](https://faiss.ai/)
- [RAG Explained](https://python.langchain.com/docs/tutorials/rag/)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this project for learning and commercial purposes.

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Vector search by [FAISS](https://faiss.ai/)

## Contact

For questions or suggestions, please open an issue on GitHub.

---
