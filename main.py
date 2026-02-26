import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent  # The new 2026 standard
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# Initialize FastAPI App
app = FastAPI(title="Honest25.Gen API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

def get_tools():
    search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
    search_tool = Tool(
        name="RealTimeSearch",
        func=search.run,
        description="Search the web for current events and real-time info."
    )
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    python_tool = PythonREPLTool()
    return [search_tool, wiki_tool, python_tool]

def setup_agent():
    # Configuring for OpenRouter
    llm = ChatOpenAI(
        model_name="anthropic/claude-3.5-sonnet", # Optimized for 2026 tool use
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://honest25gen.render.com",
            "X-Title": "Honest25.Gen"
        }
    )

    tools = get_tools()

    # The NEW 2026 Agent Factory
    # No more separate prompt objects or executors; it's all-in-one now.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are Honest25.Gen – an advanced autonomous AI Personal Agent like Jarvis. "
            "Manage tasks, search real-time info, use Wikipedia, and run Python code. "
            "Think step-by-step. Be professional and proactive. Always address the user as 'Sir'."
        )
    )
    return agent

# Global Agent Instance
honest_agent = setup_agent()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # In v1.x, we invoke with 'messages' to support modern chat history
        response = honest_agent.invoke({
            "messages": [{"role": "user", "content": request.user_input}]
        })
        
        # Extract content from the last message in the returned list
        return {"response": response["messages"][-1].content}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def status():
    return {"status": "Honest25.Gen Online, Sir."}
