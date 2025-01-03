from langchain_openai import ChatOpenAI
import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st

# Title and description for Streamlit App
st.set_page_config(page_title="DMIS Chatbot", layout="centered")
st.title("Diagnostic Medical Images Segmentation (DMIS) Chatbot")
st.markdown(
    "A chatbot based on the survey paper: *A Comprehensive Survey of Semantic Segmentation Across Eight Diagnostic Medical Imaging Modalities*."
)
index_name = "survey-paper-index"  

pinecone_api_key = st.secrets["PINECONE_API_KEY"]


os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
llm_model = ChatOpenAI(model="gpt-4o-mini")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
num_chunks= 5






def semantic_search_rag(query):
    try:
        # retriever= vector_store.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

        retriever = MultiQueryRetriever.from_llm(llm=llm_model, 
                                         retriever=vector_store.as_retriever(search_kwargs={"k": num_chunks}))
        template = """Answer the question as accurately and clearly as possible based on the following context from the survey paper on semantic segmentation of diagnostic medical images:
                
                Context: {context}

                Question: {question}

        Ensure your response remains concise, professional, and aligned with the key findings or discussions from the paper."""
        prompt = ChatPromptTemplate.from_template(template)
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        output_parser= StrOutputParser()
        # chain = setup_and_retrieval | prompt | model | output_parser
        context=  setup_and_retrieval.invoke(query)
        prompt_answer= prompt.invoke({'context':context, 'question': query})
        model_answer= llm_model.invoke(prompt_answer)
        response= output_parser.invoke(model_answer)
        return response

    except Exception as e:
        raise Exception(f'Error: {e}')   


# Streamlit App Interface
st.markdown("<div style='text-align: center;'><h2>Ask a Query</h2></div>", unsafe_allow_html=True)
query = st.text_input("Enter your query about the survey paper:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Fetching the answer..."):
            try:
                response = semantic_search_rag(query)
                st.success("Answer fetched successfully!")
                st.markdown(f"**Answer:** {response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Additional UI Enhancements
st.markdown("---")
st.markdown(
    "### About this Chatbot\n"
    "This chatbot is built to provide quick answers from the survey paper titled *A Comprehensive Survey of Semantic Segmentation Across Eight Diagnostic Medical Imaging Modalities*.\n"
    "It supports queries related to methodologies, datasets, and imaging modalities discussed in the paper."
)
