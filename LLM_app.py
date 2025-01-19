# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
# from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
# from langchain.agents import initialize_agent,AgentType
# from langchain.callbacks import StreamlitCallbackHandler
# import os
# from dotenv import load_dotenv
# ## Code
# ####

# ## Arxiv and wikipedia Tools
# arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

# api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
# wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# search=DuckDuckGoSearchRun(name="Search")


# st.title("🔎 LangChain - Chat with search")
# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """


# ## Sidebar for setting
# st.sidebar.title("Settings")
# api_key=st.sidebar.text_input('enter your Groq API key',type="password")


# if "messages" not in st.session_state:
#     st.session_state["messages"]=[
#         {
#            "role":"assistant", "content": "Welcome to Langchain! I can help you with searching and answering questions.",
#         }
#     ]
# for msg in st.session_state.messages:
#     st.chat_message(msg['role']).write(msg['content'])

# if prompt:=st.chat_input(placeholder="what is machine learning?"):
#    st.session_state.messages.append({"role":"user","content":prompt})
#    st.chat_message("user").write(prompt)

#    llm=ChatGroq(groq_api_key=api_key,model="Llama3-8b-8192",streaming=True) 
#    tools=[search,arxiv,wiki] 

#    search_agents=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
   
#    with st.chat_message("assistant"):
#        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#        response=search_agents.run(st.session_state.messages,callbacks=[st_cb])
#        st.session_state.messages.append({'role':'assistant','content':response})
#        st.write(response)
  
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Ensure DuckDuckGo works
try:
    search = DuckDuckGoSearchRun(name="Search")
    test_result = search.run("Test query")
except Exception:
    st.warning("DuckDuckGo search may not be working.")

# Streamlit UI
st.title("🔎 LangChain - Chat with search")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

if not api_key:
    st.error("Please enter a valid Groq API key.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome to Langchain! I can help you with searching and answering questions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input(placeholder="What is deep learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agents = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    user_input = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
    
    response = "Sorry, an error occurred."  # Prevent UnboundLocalError

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agents.run(user_input, callbacks=[st_cb])
        except Exception as e:
            st.error(f"Error: {e}")

        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)




