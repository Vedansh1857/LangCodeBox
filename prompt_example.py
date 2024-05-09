import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.sequential import SequentialChain
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title('Celebrity search results')
input_text=st.text_input("Search the topic u want")

# Prompt Template...
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celebrity {name}"
)

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OpenAI LLMs
llm = OpenAI(temperature=0.8)

# Now, W.R.T. to every promptTemplate we need a LLM Chain in order to execute those template

# Initializing a LLM Chain...
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, output_key='person', verbose=True)


second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)
# Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 3 major events happened around {dob} in the world"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)
parent_chain=SequentialChain(
    chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)