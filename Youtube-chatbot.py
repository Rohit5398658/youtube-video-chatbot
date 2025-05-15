# main.py

from dotenv import load_dotenv
import os
load_dotenv()

# Load API keys from environment
api_key = os.getenv("OPENAI_API_KEY")
proxy_username = os.getenv("PROXY_USERNAME")
proxy_password = os.getenv("PROXY_PASSWORD")

# Install required packages (optional in actual scripts — not needed if using requirements.txt)
# !pip install -q youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Fetch YouTube transcript
ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username=proxy_username,
        proxy_password=proxy_password,
    )
)

video_id = "LPZh9BOjkQs"

try:
    transcript_list = ytt_api.fetch(video_id)
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions found for this video.")
    transcript = ""

# 2. Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# 3. Embed and store in vector DB
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

# 4. Setup LLM and prompt
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Only answer from the provided transcript context.
If the context is insufficient, just say "I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# 5. Run initial test question
question = "Is the topic of aliens discussed in this video? If yes, what is discussed?"
retrieved_documents = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_documents)
final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt)
print("Answer:", answer.content)

# 6. Define Runnable Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnablePassthrough()
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# 7. Run the main chain
response = main_chain.invoke("Can you summarize the video?")
print("\nSummary:\n", response)
