import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
from langchain.utilities import GoogleSerperAPIWrapper

import streamlit as st

# API Keys
import streamlit as st

# Access secrets using st.secrets
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")



# Create tool for searching


def search(query):
  search = GoogleSerperAPIWrapper(serper_api_key=SERP_API_KEY)
  return search.run(query)


# Create tool for scraping relevant websites
def scrape_website(objective: str, url: str):
  headers = {
      'Cache-Control': 'no-cache',
      'Content-Type': 'application/json',
  }

  data = {
      'url': url
  }

  json_data = json.dumps(data)

  post_url = f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}"
  response = requests.post(post_url, headers=headers, data=json_data)

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser") # Use beutifulsoup to only extract the text from the html
    text = soup.get_text()

    if len(text) > 10000:
      output = summarize(objective, text)
      return output

    else:
      return text

  else:
    print("Failed to scrape website.")

def summarize(objective, content):
  llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613', openai_api_key=OPEN_AI_KEY)
  text_splitter = RecursiveCharacterTextSplitter(
      separators=["\n\n", "\n"],
      chunk_size=10000,
      chunk_overlap=500
  )
  docs = text_splitter.create_documents([content])
  map_prompt = """Write a summary of the following text for {objective}:
  "{text}"
  Summary:
  """

  map_prompt_template = PromptTemplate(
      template=map_prompt, input_variables=["text", "objective"])

  summary_chain = load_summarize_chain(
      llm=llm,
      chain_type="map_reduce",
      map_prompt=map_prompt_template,
      combine_prompt=map_prompt_template,
      verbose=True
  )

  output = summary_chain.run(input_documents=docs, objective=objective)

class ScrapeWebsiteInput(BaseModel):
  objective: str = Field(description="The objective and task the user gives to the agent")
  url: str = Field(description="The url of the website to be scraped ")

class ScrapeWebsiteTool(BaseTool):
  name = "scrape_website"
  description = "useful when you need to get data from a website url, passing both url and objective to the function"
  args_schema: Type[BaseModel] = ScrapeWebsiteInput

  def _run(self, objective: str, url: str):
    return scrape_website(objective, url)

  def _arun(self, url: str):
    raise NotImplementedError("SKET SIG")


# Implement the agent

tools = [
        Tool(
            name="Search",
            func=search,
            description="useful when you need answers to current events data. You should ask targeted questions",
    ),
    ScrapeWebsiteTool(),
    
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 4 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {"extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")], 
                "system_message": system_message,
                }

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", openai_api_key=OPEN_AI_KEY)
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs=agent_kwargs,
    memory=memory,
    )

# Set up streamlit
# def main():

#   st.set_page_config(page_title="Fred's AI Researcher", page_icon=":bird:")

#   st.header("Fred's AI Researcher :bird:")
#   query = st.text_input("Research goal")

#   if query:
#     st.write("Doing research for ", query)
#     result = agent({"input": query})
#     st.info(result['output'])

# if __name__== 'main':
#   main()



def main():
    st.set_page_config(page_title="Fred's AI Researcher", page_icon=":bird:")
    
    st.header("Fred's AI Researcher :bird:")

    query = st.text_input("Research goal")

    if query:
      st.write("Doing research for ", query)
      result = agent({"input": query})
      st.info(result['output'])

if __name__ == "__main__":
    main()






