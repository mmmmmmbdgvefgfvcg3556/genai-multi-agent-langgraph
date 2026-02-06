import os
from typing import TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langgraph.graph import StateGraph

# 🔑 Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# -------- Agent 1: Data Collector --------
data_prompt = ChatPromptTemplate.from_template("""
You are a Data Collector Agent.
Collect startup trend data including funding, sectors, and growth signals.
Use dummy data if needed.
""")

data_chain = data_prompt | llm | StrOutputParser()

# -------- Agent 2: Analyst --------
analyst_prompt = ChatPromptTemplate.from_template("""
You are an Analyst Agent.
Analyze the startup data below and provide:
- Insights
- Opportunities
- Risks

Startup Data:
{data}
""")

analyst_chain = analyst_prompt | llm | StrOutputParser()

# -------- Shared Memory --------
class AgentState(TypedDict):
    data: str
    analysis: str

# -------- Nodes --------
def data_collector(state: AgentState):
    result = data_chain.invoke({})
    return {"data": result}

def analyst(state: AgentState):
    result = analyst_chain.invoke({"data": state["data"]})
    return {"analysis": result}

# -------- LangGraph Orchestrator --------
graph = StateGraph(AgentState)
graph.add_node("collector", data_collector)
graph.add_node("analyst", analyst)

graph.set_entry_point("collector")
graph.add_edge("collector", "analyst")

app = graph.compile()

if __name__ == "__main__":
    output = app.invoke({})
    print("Startup Data:\n", output["data"])
    print("\nAnalysis:\n", output["analysis"])
