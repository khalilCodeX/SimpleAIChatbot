# Tutorial: Building a RAG-based Chatbot from Scratch

This tutorial will guide you through understanding and building your own website-based AI chatbot using Retrieval-Augmented Generation (RAG).

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Code Walkthrough](#code-walkthrough)
4. [Building Your Own](#building-your-own)
5. [Advanced Topics](#advanced-topics)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** combines two powerful concepts:

- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to generate answers based on that information

### Why RAG?

Without RAG, LLMs have limitations:
- ❌ Only know information from their training data
- ❌ Can't access real-time or proprietary data
- ❌ May hallucinate facts

With RAG:
- ✅ Answers based on YOUR specific content
- ✅ Up-to-date information from websites
- ✅ Reduced hallucinations (grounded in retrieved facts)

---

## Understanding the Architecture

### The Pipeline

```
User Query → Embedding → Vector Search → Retrieve Context → LLM → Answer
                                              ↑
                                    Vector Database (FAISS)
                                              ↑
                           Website Content → Chunks → Embeddings
```

### Step-by-Step Flow

1. **Setup Phase** (runs once)
   - Load website content
   - Split into chunks
   - Create embeddings
   - Store in vector database

2. **Query Phase** (runs per question)
   - User asks a question
   - Question is embedded
   - Find similar chunks
   - Pass chunks + question to LLM
   - Return answer

---

## Code Walkthrough

Let's break down each module:

### 1. Management (`management.py`)

**Purpose**: Handle OpenAI API key securely

```python
def initialize_openai_client():
    api_key = os.getenv("OPEN_AI_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)
```

**What it does:**
1. Tries to get API key from environment variable
2. If not found, prompts user securely
3. Sets it for LangChain to use
4. Returns OpenAI client

**Key Learning**: Always handle API keys securely - never hardcode them!

---

### 2. Web Loader (`WebLoader.py`)

#### Loading Website Content

```python
def load_website(url: str):
    loader = WebBaseLoader(web_paths=(url,))
    documents = loader.load()
    return documents
```

**What it does:**
1. Uses BeautifulSoup under the hood to scrape HTML
2. Extracts text content
3. Returns as LangChain Document objects

**Pro Tip**: Add filters for cleaner content:
```python
bs4_strainer = bs4.SoupStrainer(class_=("content", "article"))
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs={"parse_only": bs4_strainer}
)
```

#### Chunking Documents

```python
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits
```

**Why chunk?**
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap ensures no context is lost at boundaries

**Parameters explained:**
- `chunk_size=1000`: Each chunk is ~1000 characters
- `chunk_overlap=200`: Last 200 chars of chunk N appear in chunk N+1

**Example:**
```
Original: "The iPhone 15 has USB-C. It comes in 5 colors..."

Chunk 1: "The iPhone 15 has USB-C. It comes..."
Chunk 2: "...It comes in 5 colors: Pink, Blue..."
         ^^^^^^^^^ (overlap)
```

---

### 3. Vector Database (`Vectordb.py`)

```python
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()
```

**What's happening:**

1. **Create embeddings**: Each chunk → numerical vector (1536 dimensions)
   ```
   "iPhone 15 has USB-C" → [0.123, -0.456, 0.789, ...]
   ```

2. **Store in FAISS**: Fast similarity search library
   - FAISS = Facebook AI Similarity Search
   - Optimized for finding nearest vectors

3. **Return retriever**: Interface for searching

**How retrieval works:**
```python
query = "What port does iPhone 15 have?"
# Query → embedding → find closest chunk vectors → return chunks
```

---

### 4. Main Chatbot (`chat.py`)

#### Initialization

```python
def __init__(self, url: str):
    self.client = initialize_openai_client()
    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    self.chat_history = []
    
    self.docs = load_website(url)
    self.splits = chunk_documents(self.docs)
    self.retriever = create_vectorstore(self.splits)
```

**What happens:**
1. Set up OpenAI client
2. Initialize LLM (low temperature = more focused)
3. Create empty chat history
4. Load, chunk, and index website content

#### The RAG Chain

```python
def create_chain(self, user_prompt: str):
    # 1. Define system prompt
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "{context}"
    )
    
    # 2. Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_prompt}")
    ])
    
    # 3. Retrieve relevant documents
    format_docs = self.retriever.invoke(user_prompt)
    context = self.format_doc(format_docs)
    
    # 4. Create chain using LCEL
    chain = prompt | self.llm | StrOutputParser()
    
    # 5. Invoke chain
    response = chain.invoke({
        "context": context,
        "user_prompt": user_prompt,
        "chat_history": self.chat_history
    })
    
    # 6. Update chat history
    self.chat_history.append(HumanMessage(content=user_prompt))
    self.chat_history.append(AIMessage(content=response))
    
    return response
```

**Step-by-step breakdown:**

**Step 1: System Prompt**
- Tells LLM its role
- Instructs it to use provided context
- `{context}` will be filled with retrieved chunks

**Step 2: Prompt Template**
- `("system", ...)`: System instructions
- `MessagesPlaceholder`: Inserts chat history
- `("human", ...)`: User's current question

**Step 3: Retrieval**
- Search vector DB for relevant chunks
- Format them into a single string

**Step 4: LCEL Chain**
```python
chain = prompt | self.llm | StrOutputParser()
```
- `|` is the pipe operator (like Unix pipes)
- Data flows: prompt → LLM → parse output

**Step 5: Invoke**
- Pass all variables to the chain
- LLM sees: system prompt + context + history + question
- Returns answer

**Step 6: Update History**
- Store user question
- Store AI answer
- Enables follow-up questions

---

## Building Your Own

### Exercise 1: Create a Simple RAG Bot

```python
from chat import chatbot

# Pick any topic/website
bot = chatbot("https://en.wikipedia.org/wiki/Python_(programming_language)")

# Ask questions
print(bot.create_chain("When was Python created?"))
print(bot.create_chain("Who created it?"))
print(bot.create_chain("What was the person's name again?"))  # Uses history!
```

### Exercise 2: Customize for Your Use Case

**Scenario**: Build a company FAQ bot

```python
# In chat.py, modify the system prompt
system_prompt = (
    "You are a customer service assistant for XYZ Company. "
    "Answer questions professionally and concisely. "
    "If information is not in the context, politely say you don't know. "
    "Always be friendly and helpful.\n\n"
    "{context}"
)
```

### Exercise 3: Add Features

**Add source citations:**

```python
def create_chain(self, user_prompt: str):
    # ... existing code ...
    
    # After retrieval
    docs_with_scores = self.retriever.invoke(user_prompt)
    
    # Modify response to include sources
    response += "\n\nSources:"
    for i, doc in enumerate(docs_with_scores[:3], 1):
        response += f"\n{i}. {doc.metadata.get('source', 'Unknown')}"
    
    return response
```

---

## Advanced Topics

### 1. Improving Retrieval Quality

**Technique 1: Adjust chunk size**
```python
# Smaller chunks for precise retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

**Technique 2: Hybrid search**
```python
# Combine semantic + keyword search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(splits)
ensemble_retriever = EnsembleRetriever(
    retrievers=[self.retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
```

**Technique 3: Re-ranking**
```python
# Use a cross-encoder to re-rank results
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=self.retriever
)
```

### 2. Handling Multiple Sources

```python
class MultiSourceChatbot:
    def __init__(self, urls: list):
        all_docs = []
        for url in urls:
            docs = load_website(url)
            all_docs.extend(docs)
        
        self.splits = chunk_documents(all_docs)
        self.retriever = create_vectorstore(self.splits)

# Usage
bot = MultiSourceChatbot([
    "https://example.com/products",
    "https://example.com/faq",
    "https://example.com/about"
])
```

### 3. Streaming Responses

```python
def create_chain_streaming(self, user_prompt: str):
    # ... setup code ...
    
    chain = prompt | self.llm | StrOutputParser()
    
    # Stream the response
    for chunk in chain.stream({
        "context": context,
        "user_prompt": user_prompt,
        "chat_history": self.chat_history
    }):
        print(chunk, end="", flush=True)
```

### 4. Adding Memory Management

```python
def manage_history(self, max_messages=10):
    """Keep only recent messages to avoid context overflow"""
    if len(self.chat_history) > max_messages:
        self.chat_history = self.chat_history[-max_messages:]
```

### 5. Caching for Performance

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def retrieve_cached(self, query: str):
    """Cache frequent queries"""
    return self.retriever.invoke(query)
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Empty Retrieval Results

**Problem**: No documents loaded or retrieved

**Solution**:
```python
# Add assertions
assert len(documents) > 0, "No documents loaded!"
assert len(splits) > 0, "No chunks created!"

# Debug retrieval
docs = self.retriever.invoke(query)
print(f"Retrieved {len(docs)} documents")
for doc in docs:
    print(f"Content: {doc.page_content[:100]}...")
```

### Pitfall 2: Poor Answer Quality

**Problem**: LLM gives generic answers or hallucinates

**Solutions**:
1. Increase retrieval count:
   ```python
   retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
   ```

2. Improve prompt:
   ```python
   system_prompt = (
       "ONLY use the provided context to answer. "
       "If the answer is not in the context, say 'I don't know based on the provided information.'"
   )
   ```

3. Lower temperature:
   ```python
   self.llm = ChatOpenAI(temperature=0.0)  # More deterministic
   ```

### Pitfall 3: High Costs

**Problem**: Too many API calls

**Solutions**:
1. Use cheaper models:
   ```python
   self.llm = ChatOpenAI(model="gpt-3.5-turbo")
   ```

2. Cache embeddings:
   ```python
   # Save vector store to disk
   vectorstore.save_local("my_vectorstore")
   
   # Load later
   vectorstore = FAISS.load_local("my_vectorstore", embeddings)
   ```

3. Batch operations:
   ```python
   # Process multiple queries at once
   ```

---

## Next Steps

1. **Experiment**: Try different websites, chunk sizes, models
2. **Extend**: Add file upload, PDF support, multiple sources
3. **Deploy**: Turn it into a web app (Streamlit, FastAPI)
4. **Optimize**: Add caching, improve retrieval, reduce costs
5. **Learn More**: Explore LangChain agents, tools, and advanced features

---

## Quiz Yourself

1. What's the difference between retrieval and generation in RAG?
2. Why do we chunk documents instead of using them whole?
3. What does the temperature parameter control?
4. How does chat history enable follow-up questions?
5. What's the role of embeddings in this system?

**Answers:**
1. Retrieval finds relevant info; generation creates the answer
2. Chunks fit in context windows and enable precise retrieval
3. Randomness/creativity of LLM outputs (0=deterministic, 1=creative)
4. LLM sees previous Q&A to understand context ("it" refers to previous topic)
5. Convert text to numbers for similarity comparison

---

## Resources for Deeper Learning

- **LangChain Tutorials**: https://python.langchain.com/docs/tutorials/
- **Vector Databases**: https://www.pinecone.io/learn/vector-database/
- **RAG Best Practices**: https://www.anthropic.com/research/rag
- **Embeddings Explained**: https://platform.openai.com/docs/guides/embeddings

---

**Happy Building! 🚀**

If you have questions or build something cool, share it on GitHub!
