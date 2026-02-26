import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize FastAPI App
app = FastAPI(title="Honest25.Gen API", description="Jarvis-like Autonomous AI Agent")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace "*" with your actual frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Request Model
class ChatRequest(BaseModel):
    user_input: str

# 1. Setup Tools
def get_tools():
    # SERPAPI Tool
    search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
    search_tool = Tool(
        name="RealTimeSearch",
        func=search.run,
        description="Useful for when you need to answer questions about current events or real-time information."
    )
    
    # Wikipedia Tool
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    # Python REPL Tool (For calculations and code execution)
    python_tool = PythonREPLTool()
    
    return [search_tool, wiki_tool, python_tool]

# 2. Setup OpenRouter LLM & Agent
def setup_agent():
    # OpenRouter setup using LangChain's OpenAI wrapper
    llm = ChatOpenAI(
        model_name="nvidia/nemotron-3-nano-30b-a3b:free", # Uses OpenRouter's auto routing
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://honest25gen-render-app.com", # Update with your Render URL later
            "X-Title": "Honest25.Gen AI"
        }
    )

    tools = get_tools()

    # The System Prompt defining Honest25.Gen
    system_prompt = """
    You are Honest25.Gen – an advanced autonomous AI Personal Agent like Jarvis.
    
    Your Role:
    - Manage user's tasks
    - Perform real-time web search using SERPAPI
    - Retrieve knowledge from Wikipedia
    - Execute Python code when needed
    - Think step-by-step before answering
    - Act like a real executive assistant
    
    Rules:
    - Always decide which tool to use before answering.
    - If information is real-time → use RealTimeSearch tool.
    - If general knowledge → use Wikipedia tool.
    - If calculation, automation, or logic is needed → use Python tool.
    - Be confident, professional, and proactive.
    - Always address the user as "Sir".
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize Agent
agent_executor = setup_agent()

# 3. Create the API Endpoint
@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        # Run the agent with the user's input
        response = agent_executor.invoke({"input": request.user_input})
        return {"response": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Basic health check endpoint
@app.get("/")
def read_root():
    return {"status": "Honest25.Gen Systems Online. Awaiting your command, Sir."}
