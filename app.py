#libraries
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_astradb import AstraDBVectorStore

from model import Model

load_dotenv()



# streamlit pag config with html code


st.set_page_config(
    page_title="PolicyMind AI",
    page_icon="📄",
    layout="wide"
)



# CUSTOM DESIGN


st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .app-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .app-subtitle {
        font-size: 16px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Make uploader smaller */
    [data-testid="stFileUploader"] {
        max-width: 260px;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 105px;
        padding: 10px;
        border-radius: 12px;
    }

    /* Rounded question field */
    [data-testid="stTextInput"] input {
        border-radius: 10px;
        min-height: 44px;
    }

    /* Rounded button */
    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# header


st.markdown(
    """
    <div class="app-title">PolicyMind AI</div>

    <div class="app-subtitle">
        Upload documents, summarize webpages and ask questions using AI.
    </div>
    """,
    unsafe_allow_html=True
)



# CONVERSATION MEMORY


if "messages" not in st.session_state:
    st.session_state.messages = []


# Optional clear-memory button
with st.sidebar:
    st.header("Conversation")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()



# input area


cl1, cl2, cl3 = st.columns(
    [1.4, 5, 1],
    gap="medium",
    vertical_alignment="bottom"
)

with cl1:
    uploaded_files = st.file_uploader(
        "Upload PDF",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload one or more PDF documents."
    )

with cl2:
    user_input = st.text_input(
        "Enter the question",
        placeholder="Ask about a PDF, paste a URL or search the web...",
        key="question_input"
    )

with cl3:
    ask_button = st.button(
        "Ask",
        use_container_width=True,
        type="primary"
    )



# display the previous coversation history


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# pull the model from class model


llm = Model()



# web search tool


@tool
def search_tool(query: str) -> str:
    """
    Search the public web for current or general information.

    Do not use this tool when the question is about an uploaded PDF.
    """

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=3
                )
            )

    except RatelimitException:
        return (
            "The web search service is temporarily rate-limited. "
            "Live search is currently unavailable."
        )

    except Exception as error:
        return f"Web search failed: {error}"

    if not results:
        return "No web search results were found."

    formatted_results = []

    for result in results:
        formatted_results.append(
            f"""
Title: {result.get("title", "No title")}
Description: {result.get("body", "No description")}
URL: {result.get("href", "No URL")}
"""
        )

    search_results = "\n\n".join(formatted_results)

    prompt = f"""
You are a helpful research assistant.

User question:
{query}

Search results:
{search_results}

Answer the question using only the search results.

Mention the relevant source URLs at the end of the answer.
"""

    return llm.create_response(prompt)



# url and youtube summary


@tool
def url_tool(url: str) -> str:
    """
    Load and summarize content from a YouTube URL or webpage URL.
    """

    try:
        if "youtube.com" in url or "youtu.be" in url:
            loader = YoutubeLoader.from_youtube_url(
                youtube_url=url
            )

        else:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
                    )
                }
            )

        documents = loader.load()

    except Exception as error:
        return f"Unable to load the URL: {error}"

    if not documents:
        return "No readable content was found at the provided URL."

    text = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prevent extremely large prompts
    text = text[:50000]

    prompt = f"""
You are an AI summarization assistant.

Summarize the following content clearly in approximately 300 words.

Content:
{text}
"""

    return llm.create_response(prompt)



# rag tool


@tool
def rag_chain_tool(query: str) -> str:
    """
    Search the PDF documents uploaded by the user and answer using
    only information retrieved from those PDFs.
    """

    if not uploaded_files:
        return "No PDF documents have been uploaded."

    all_documents = []

    try:
        for uploaded_file in uploaded_files:

            # Save each PDF temporarily and safely
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getbuffer()
                )

                temporary_path = temporary_file.name

            try:
                loader = PyPDFLoader(
                    file_path=temporary_path
                )

                pdf_documents = loader.load()

                # Add source filename to metadata
                for document in pdf_documents:
                    document.metadata["source_file"] = (
                        uploaded_file.name
                    )

                all_documents.extend(pdf_documents)

            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

    except Exception as error:
        return f"Unable to process the uploaded PDFs: {error}"

    if not all_documents:
        return "No readable text was found in the PDFs."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        all_documents
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )

    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="User_pdf",
        api_endpoint=os.getenv("API_ENDPOINT"),
        token=os.getenv("API_TOKEN")
    )

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_documents = retriever.invoke(query)

    if not retrieved_documents:
        return (
            "I could not find information related to that "
            "question in the uploaded documents."
        )

    context_parts = []

    for document in retrieved_documents:
        filename = document.metadata.get(
            "source_file",
            "Uploaded PDF"
        )

        page_number = document.metadata.get("page")

        if page_number is not None:
            source = f"{filename}, page {page_number + 1}"
        else:
            source = filename

        context_parts.append(
            f"""
Source: {source}

Content:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the retrieved PDF context.

Rules:

1. Do not use outside knowledge.
2. If the answer is absent, say:
   "I could not find this information in the uploaded documents."
3. Provide a concise and clear answer.
4. Mention the source document and page when available.

User question:
{query}

Retrieved context:
{context}
"""

    return llm.create_response(prompt)



# tools


tools = [
    search_tool,
    url_tool
]

if uploaded_files:
    tools.append(rag_chain_tool)

    pdf_instruction = """
One or more PDF documents have been uploaded.

Use rag_chain_tool whenever the question appears related to the
uploaded documents.

Do not use web search for questions that should be answered from
the uploaded PDFs.
"""

else:
    pdf_instruction = """
No PDF document has been uploaded.

Do not call rag_chain_tool.

Ask the user to upload a PDF when a question requires information
from a document.
"""



# AGENT MODEL


agent_model = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    max_tokens=512,
    temperature=0.3,
    top_p=0.9
)



# smart agent


agent = create_agent(
    model=agent_model,
    tools=tools,
    system_prompt=f"""
You are PolicyMind AI, a helpful AI assistant.

{pdf_instruction}

Tool-selection rules:

1. Use rag_chain_tool for questions about uploaded PDFs.
2. Use url_tool when the user provides a webpage or YouTube URL.
3. Use search_tool for recent or public web information.
4. Answer directly when no tool is required.
5. Use previous conversation messages to understand follow-up
   questions and references such as "it", "that document",
   "explain again", or "tell me more".
"""
)



# helper function to extract answer


def extract_answer(content) -> str:
    """Convert different Bedrock/LangChain response formats to text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []

        for block in content:
            if isinstance(block, dict):

                if block.get("type") == "text":
                    text_blocks.append(
                        block.get("text", "")
                    )

                elif "text" in block:
                    text_blocks.append(
                        block.get("text", "")
                    )

            elif isinstance(block, str):
                text_blocks.append(block)

        return "\n".join(text_blocks).strip()

    return str(content)



# agent with ask button


if ask_button:

    if not user_input.strip():
        st.warning("Please enter a question.")

    else:
        # Save and display current user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:
                    # Pass the entire previous conversation to the agent
                    result = agent.invoke(
                        {
                            "messages": st.session_state.messages
                        }
                    )

                    final_message = result["messages"][-1]

                    answer = extract_answer(
                        final_message.content
                    )

                    if not answer:
                        answer = (
                            "I was unable to generate a response. "
                            "Please try again."
                        )

                except Exception as error:
                    answer = f"An error occurred: {error}"

            st.markdown(answer)

        # Save assistant response to memory
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )




