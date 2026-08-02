# Insurance Policy Intelligence System

AI-powered question answering using Retrieval-Augmented Generation (RAG) with contextual document retrieval
# Features

- Upload and analyze one or more PDF documents
- AI-powered question answering using Retrieval-Augmented Generation (RAG)
- Semantic search over insurance policy documents
- Internet search integration for public information
- Summarize YouTube videos and webpages
- AI Agent that automatically selects the appropriate tool
- AWS Bedrock integration for LLMs and embeddings
- Interactive Streamlit web interface

# Tech Stack

- **Programming Language:** Python
- **LLM:** Amazon Nova Lite (AWS Bedrock)
- **Embeddings:** Amazon Titan Embeddings v2
- **Framework:** LangChain
- **Vector Database:** Astra DB
- **Frontend:** Streamlit
- **PDF Processing:** PyPDFLoader
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Internet Search:** DuckDuckGo Search
- **Deployment Ready:** AWS BedrocK

  ## 🏗️ System Architecture

```text
                    +--------------------+
                    |       User         |
                    +--------------------+
                              |
                              v
                    +--------------------+
                    |   Streamlit UI     |
                    +--------------------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
        Uploaded PDF                 User Question
                |                           |
                +-------------+-------------+
                              |
                              v
                    +--------------------+
                    |   LangChain Agent   |
                    +--------------------+
                    |        |           |
          PDF Tool  |  Web Search | URL Tool
                    |        |           |
                    +--------+-----------+
                              |
                              v
                     Amazon Bedrock LLM
                              |
                              v
                      Final AI Response
```
