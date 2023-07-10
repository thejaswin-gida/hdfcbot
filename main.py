from sentence_transformers import SentenceTransformer, util
import os
import sys
import streamlit as st
import requests
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain import OpenAI

st.title("HDFCAuto Bot")


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Create an instance of the OpenAI model
openai_model = OpenAI()

# Load a pre-trained semantic similarity model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

index = GPTSimpleVectorIndex.load_from_disk('./cars_bikes.json')
tools = [Tool(
    name="LlamaIndex",
    func=lambda q: str(index.query(q)),
    description="The input to this tool should be relevant to cars and bikes in Indian currency",
    return_direct=True
)]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history', k=5, return_messages=True)
llm = OpenAI(temperature=0.5, model_name="text-davinci-003")

agent_executor = initialize_agent(
    tools, llm, agent="conversational-react-description", memory=conversational_memory)


# def calculate_similarity(question1, question2):
#     # Encode the questions into embeddings
#     embeddings = semantic_model.encode(
#         [question1, question2], convert_to_tensor=True)

#     # Calculate cosine similarity between the embeddings
#     cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
#     print(cosine_scores)
#     return cosine_scores


# def find_similar_question(question):
#     highest_similarity = 0.92
#     similar_question = None

#     # Iterate over the conversational memory to find the most similar question
#     for stored_question in st.session_state.conversational_memory:
#         similarity = calculate_similarity(question, stored_question)
#         if similarity > highest_similarity:
#             highest_similarity = similarity
#             similar_question = stored_question

#     return similar_question


def greet(question):
    # # Check if a similar question exists in the conversational memory
    # similar_question = find_similar_question(question)

    # if similar_question:
    #     return st.session_state.conversational_memory[similar_question]

    # # Query the index with the question
    # response = index.query(question)

    # # Store the response in conversational memory
    # st.session_state.conversational_memory[question] = response.response

    # return st.session_state.conversational_memory[question].replace('\n', '')
    return agent_executor.run(input=question)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# if "conversational_memory" not in st.session_state:
#     st.session_state.conversational_memory = {}

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {greet(prompt)}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
