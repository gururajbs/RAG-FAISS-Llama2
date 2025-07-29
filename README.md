# Llama-RAG: Retrieval-Augmented Generation Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot implementation using Llama 2, LangChain, and FAISS for intelligent document-based question answering.

## 🚀 Features

- **Document Processing**: Load and process PDF documents and web content
- **Intelligent Retrieval**: Use FAISS vector database for efficient similarity search
- **Advanced LLM**: Powered by Meta's Llama 2 7B Chat model
- **Conversational Memory**: Maintains chat history for contextual conversations
- **Web Scraping**: Built-in web content loader for dynamic data sources
- **GPU Support**: Optimized for GPU acceleration when available

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)
- Hugging Face account with access to Llama 2 models

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Llama-RAG.git
   cd Llama-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face access**
   - Visit [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
   - Request access to Llama 2 models
   - Accept the license agreement

## 📦 Dependencies

The project uses the following key libraries:

- `langchain` - Framework for building LLM applications
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for deep learning
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `beautifulsoup4` - Web scraping
- `pypdf` - PDF document processing

## 🏗️ Project Structure

```
Llama-RAG/
├── data/                          # Document storage
│   ├── annual-report-2022-2023.pdf
│   └── sample.pdf
├── vectorstore/                   # FAISS vector database
│   └── db_faiss/
│       ├── index.faiss
│       └── index.pkl
├── rag_chatbot_llama2_langchain_fiass.ipynb  # Main implementation
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🔧 Usage

### 1. Document Loading

The system supports multiple document sources:

```python
# Load PDF documents
from langchain.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader('data/')
documents = loader.load()

# Load web content
from langchain_community.document_loaders import WebBaseLoader
urls = ["https://example.com/article1", "https://example.com/article2"]
documents = []
for url in urls:
    loader = WebBaseLoader(url)
    documents.extend(loader.load())
```

### 2. Text Processing

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10,
    separators=["\n\n", "\n", ". "]
)
splits = text_splitter.split_documents(documents)
```

### 3. Vector Database Setup

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
faiss_db = FAISS.from_documents(splits, embeddings)
faiss_db.save_local('vectorstore/db_faiss')
```

### 4. LLM Setup

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = HuggingFacePipeline(
    pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer),
    model_kwargs={"temperature": 0.2, "max_length": 512}
)
```

### 5. RAG Chain Creation

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Custom prompt template
prompt_template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
<</SYS>>

CONTEXT:
{context}

Question: {question}
[/INST]"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=faiss_db.as_retriever(search_kwargs={"k": 6}),
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)
```

### 6. Interactive Chat

```python
chat_history = []

while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        break
    
    result = qa_chain({"question": query, "chat_history": chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
```

## 🎯 Key Components

### Document Processing
- **PyPDFDirectoryLoader**: Loads PDF documents from a directory
- **WebBaseLoader**: Scrapes content from web URLs
- **RecursiveCharacterTextSplitter**: Splits documents into manageable chunks

### Vector Database
- **FAISS**: High-performance similarity search and clustering
- **HuggingFaceEmbeddings**: Uses sentence-transformers for text embeddings
- **Similarity Search**: Retrieves relevant documents based on query similarity

### Language Model
- **Llama 2 7B Chat**: Meta's conversational AI model
- **HuggingFacePipeline**: Integrates the model with LangChain
- **Custom Prompts**: Tailored prompts for better response quality

### RAG Pipeline
- **ConversationalRetrievalChain**: Combines retrieval and generation
- **Chat History**: Maintains conversation context
- **Source Documents**: Returns relevant source information

## 🔍 Example Queries

The chatbot can answer questions about:
- Document content and summaries
- Specific information from loaded documents
- General knowledge questions (if not in documents, it will state so)
- Follow-up questions based on conversation history

## 🚀 Performance Optimization

### GPU Acceleration
```python
# Check GPU availability
import torch
print("CUDA Available:", torch.cuda.is_available())

# Use GPU for model inference
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',  # Automatically uses GPU if available
    trust_remote_code=True
)
```

### Quantization (Optional)
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
)
```

## 📊 Model Performance

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **LLM**: Llama 2 7B Chat (7 billion parameters)
- **Vector Database**: FAISS with IndexFlatIP for inner product similarity
- **Chunk Size**: 500 characters with 10 character overlap
- **Retrieval**: Top 6 most similar documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) for Llama 2 models
- [Hugging Face](https://huggingface.co/) for transformers library
- [LangChain](https://langchain.com/) for the RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/Llama-RAG/issues) page
2. Create a new issue with detailed information
3. Include your system specifications and error messages

## 🔄 Updates

Stay updated with the latest features and improvements by:
- Starring the repository
- Watching for updates
- Following the release notes

---

**Note**: This implementation requires appropriate access to Llama 2 models through Hugging Face. Make sure to comply with Meta's usage terms and conditions. 
