import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# groq_api_key=os.getenv("GROQ_API_KEY")
# hugging_api_key=os.getenv('HF_TOKEN')
# Arxiv and wikipedia Tools
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search=DuckDuckGoSearchRun(name="Search")

st.title(" 🔎 Langchain - chat with search")


## Sidebar for setting
st.sidebar.title("Settings")
api_key=st.sidebar.text_input('enter your Groq API key',type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
           "role":"assistant", "content": "Welcome to Langchain! I can help you with searching and answering questions.",
        }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
   st.session_state.messages.append({"role":"user","content":prompt})
   st.chat_message("user").write(prompt)

   llm=ChatGroq(groq_api_key=api_key,model="Llama3-8b-8192",streaming=True) 
   tools=[search,arxiv,wiki] 

   search_agents=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
   
   with st.chat_message("assistant"):
       st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
       response=search_agents.run(st.session_state.messages,callbacks=[st_cb])
       st.session_state.messages.append({'role':'assistant','content':response})
       st.write(response)
  


