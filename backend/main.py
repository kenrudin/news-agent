from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class NewsRequest(BaseModel):
    interests: Optional[List[str]] = None
    reading_style: Optional[str] = "balanced"  # quick, balanced, deep
    sources: Optional[List[str]] = None
    max_articles: Optional[int] = 10
    time_period: Optional[str] = "today"  # today, week, month


class NewsResponse(BaseModel):
    result: str
    articles: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# News Collection Tools
@tool
def fetch_breaking_news(sources: Optional[List[str]] = None) -> str:
    """Fetch latest breaking news from multiple sources."""
    source_str = ", ".join(sources) if sources else "major news outlets"
    return f"""Breaking News Summary from {source_str}:
    - Global Markets: Tech stocks rally amid AI breakthrough announcements
    - Politics: Congressional hearings continue on digital privacy legislation
    - Climate: New renewable energy targets announced by major economies
    - Technology: Major cloud provider reports record growth in AI services
    - Health: WHO releases updated guidelines for seasonal health preparedness
    - Business: Merger activity increases in the fintech sector
    - Science: Breakthrough in quantum computing research published in Nature"""


@tool
def search_topic_news(topic: str, time_period: str = "today") -> str:
    """Search for news articles on a specific topic within timeframe."""
    return f"""Recent news about '{topic}' ({time_period}):
    - Market Analysis: Sector shows resilience amid economic uncertainty
    - Innovation Updates: New developments in {topic} technology announced
    - Policy Changes: Regulatory frameworks evolving to address {topic} challenges
    - Industry Insights: Leading experts predict growth trends for {topic}
    - Global Impact: International perspective on {topic} developments
    - Research Findings: Academic studies reveal new insights about {topic}
    - Investment Trends: Funding patterns show increased interest in {topic}"""


@tool
def get_source_credibility(source: str) -> str:
    """Check credibility and bias information for a news source."""
    return f"""Source Analysis for {source}:
    - Credibility Score: High (verified through multiple fact-checking organizations)
    - Political Bias: Moderate/Center (based on content analysis)
    - Fact-Check Record: 95% accuracy rate over past year
    - Transparency: Clear editorial policies and correction procedures
    - Expertise: Strong track record in investigative journalism
    - International Coverage: Comprehensive global news network
    - Reader Trust: High trust rating from media literacy organizations"""


@tool
def analyze_sentiment(article_content: str) -> str:
    """Analyze sentiment and tone of news content."""
    return f"""Sentiment Analysis:
    - Overall Tone: Neutral-to-positive with factual presentation
    - Emotional Elements: Balanced reporting without sensationalism
    - Bias Indicators: Minimal partisan language detected
    - Objectivity Score: 8.5/10 (high objectivity)
    - Key Themes: Economic growth, technological innovation, policy stability
    - Expert Quotes: Multiple authoritative sources cited
    - Context Provided: Historical background and broader implications included"""


@tool
def fact_check_claims(claims: List[str]) -> str:
    """Verify factual claims in news content."""
    claims_str = "; ".join(claims[:3])  # Limit for demo
    return f"""Fact Check Results for: {claims_str}
    - Verification Status: Claims verified through primary sources
    - Cross-Reference: Confirmed by multiple independent outlets
    - Expert Validation: Subject matter experts consulted
    - Data Sources: Government databases and official statistics referenced
    - Historical Context: Claims align with established historical patterns
    - Confidence Level: High confidence in accuracy (90%+)
    - Additional Notes: All numerical data matches official records"""


@tool
def categorize_articles(articles: List[str]) -> str:
    """Categorize news articles by topic and importance."""
    article_count = len(articles) if articles else 5
    return f"""Article Categorization ({article_count} articles):
    - Politics & Policy: 2 articles (high importance)
    - Technology & Innovation: 2 articles (medium-high importance)  
    - Business & Economy: 1 article (medium importance)
    - Health & Science: 1 article (medium importance)
    - International Affairs: 1 article (high importance)
    - Climate & Environment: 1 article (medium importance)
    - Breaking News Priority: 3 articles flagged as urgent updates"""


@tool
def generate_summary(articles: List[str], reading_style: str = "balanced") -> str:
    """Generate personalized news summary based on reading style preference."""
    style_desc = {"quick": "brief bullet points", "balanced": "comprehensive overview", "deep": "detailed analysis"}
    return f"""Personalized News Summary ({style_desc.get(reading_style, 'balanced')}):
    
    Today's Key Headlines:
    - Technology sector shows strong growth with new AI partnerships announced
    - Climate policy developments gain momentum in major economies
    - Global markets respond positively to economic stability indicators
    - Healthcare innovations receive increased funding and regulatory support
    
    Market Implications:
    - Tech stocks likely to see continued interest from investors
    - Renewable energy sector positioning for expansion
    - Healthcare technology showing promise for long-term growth
    
    What This Means For You:
    - Economic indicators suggest stable consumer environment
    - Technology developments may impact daily digital services
    - Policy changes could affect long-term investment strategies"""


@tool
def personalize_content(user_interests: List[str], articles: List[str]) -> str:
    """Personalize content based on user interests and reading history."""
    interests_str = ", ".join(user_interests[:3]) if user_interests else "general news"
    return f"""Personalized Content for interests in {interests_str}:
    - Highly Relevant: 4 articles match your primary interests
    - Moderately Relevant: 3 articles relate to your secondary interests  
    - Trending in Your Areas: 2 emerging stories in your interest categories
    - Expert Recommendations: 3 articles suggested based on similar reader preferences
    - Deep Dive Opportunities: 2 complex stories with analysis available
    - Quick Updates: 5 brief summaries for staying informed
    - Priority Reading: 3 articles marked as essential for your interests"""


class NewsState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    news_request: Dict[str, Any]
    collection: Optional[str]
    personalization: Optional[str]
    analysis: Optional[str]
    delivery: Optional[str]
    final: Optional[str]
    articles: Annotated[List[Dict[str, Any]], operator.add]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def news_collection_agent(state: NewsState) -> NewsState:
    req = state["news_request"]
    interests = req.get("interests", [])
    sources = req.get("sources", [])
    time_period = req.get("time_period", "today")
    
    prompt_t = (
        "You are a news collection agent.\n"
        "Gather latest news from multiple sources for interests: {interests}.\n"
        "Time period: {time_period}. Use tools to fetch breaking news and search specific topics."
    )
    vars_ = {"interests": ", ".join(interests) if interests else "general news", "time_period": time_period}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [fetch_breaking_news, search_topic_news, get_source_credibility]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    articles: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    # Collect tool calls and execute them
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "collection", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content="Based on the collected news, provide a summary of the latest developments."))
        
        final_res = llm.invoke(messages)
        out = final_res.content
        
        # Mock articles for demo with more realistic data
        sample_articles = [
            {"title": "Congressional hearings continue on digital privacy legislation", "source": "Reuters", "category": "Politics", "time": "2 hours ago", "summary": "Lawmakers debate comprehensive data protection framework"},
            {"title": "Tech stocks rally amid AI breakthrough announcements", "source": "Bloomberg", "category": "Technology", "time": "1 hour ago", "summary": "Major cloud providers report record growth in AI services"},
            {"title": "Climate targets announced by major economies at summit", "source": "BBC", "category": "Climate", "time": "4 hours ago", "summary": "New renewable energy goals set for 2030"},
            {"title": "Healthcare innovation receives increased federal funding", "source": "AP", "category": "Health", "time": "3 hours ago", "summary": "Medical research grants focus on breakthrough therapies"},
            {"title": "Global markets respond to economic stability indicators", "source": "Financial Times", "category": "Business", "time": "5 hours ago", "summary": "Consumer confidence rises amid steady employment rates"},
            {"title": "Quantum computing research breakthrough published", "source": "Nature", "category": "Science", "time": "6 hours ago", "summary": "New algorithm shows promise for complex problem solving"},
            {"title": "Merger activity increases in fintech sector", "source": "Wall Street Journal", "category": "Business", "time": "8 hours ago", "summary": "Digital banking solutions drive consolidation"},
            {"title": "WHO releases seasonal health preparedness guidelines", "source": "CNN", "category": "Health", "time": "7 hours ago", "summary": "Updated protocols for respiratory illness prevention"}
        ]
        
        # Select articles based on user interests
        user_interests = req.get("interests", [])
        max_articles = req.get("max_articles", 10)
        if user_interests:
            # Filter articles that match user interests (simple keyword matching)
            filtered_articles = [
                article for article in sample_articles
                if any(interest.lower() in article["title"].lower() or 
                      interest.lower() in article["category"].lower() 
                      for interest in user_interests)
            ]
            if not filtered_articles:
                filtered_articles = sample_articles[:max_articles]
            articles = filtered_articles[:max_articles]
        else:
            articles = sample_articles[:max_articles]
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "collection": out, "tool_calls": calls, "articles": articles}


def personalization_agent(state: NewsState) -> NewsState:
    req = state["news_request"]
    interests = req.get("interests", [])
    reading_style = req.get("reading_style", "balanced")
    max_articles = req.get("max_articles", 10)
    
    prompt_t = (
        "You are a personalization agent.\n"
        "Customize news content for user interests: {interests}.\n"
        "Reading style: {reading_style}. Max articles: {max_articles}. Use tools to personalize content."
    )
    vars_ = {
        "interests": ", ".join(interests) if interests else "general topics", 
        "reading_style": reading_style,
        "max_articles": max_articles
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [personalize_content, categorize_articles]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "personalization", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content=f"Create a personalized news experience for {reading_style} reading style with focus on {', '.join(interests) if interests else 'general topics'}."))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "personalization": out, "tool_calls": calls}


def analysis_agent(state: NewsState) -> NewsState:
    req = state["news_request"]
    articles = state.get("articles", [])
    
    prompt_t = (
        "You are a news analysis agent.\n"
        "Analyze news content for sentiment, bias, and factual accuracy.\n"
        "Provide balanced perspectives and fact-checking for {article_count} articles."
    )
    vars_ = {"article_count": len(articles)}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [analyze_sentiment, fact_check_claims]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "analysis", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content="Provide comprehensive analysis including sentiment assessment and fact verification results."))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "analysis": out, "tool_calls": calls}


def delivery_agent(state: NewsState) -> NewsState:
    req = state["news_request"]
    reading_style = req.get("reading_style", "balanced")
    articles = state.get("articles", [])
    
    prompt_t = (
        "Create a personalized news briefing with {reading_style} style.\n\n"
        "Inputs:\nCollection: {collection}\nPersonalization: {personalization}\nAnalysis: {analysis}\n"
    )
    vars_ = {
        "reading_style": reading_style,
        "collection": (state.get("collection") or "")[:400],
        "personalization": (state.get("personalization") or "")[:400],
        "analysis": (state.get("analysis") or "")[:400],
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [generate_summary]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "delivery", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content=f"Create a final news briefing optimized for {reading_style} reading style."))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    
    return {"messages": [SystemMessage(content=out)], "delivery": out, "final": out, "tool_calls": calls}


def build_graph():
    g = StateGraph(NewsState)
    g.add_node("collection", news_collection_agent)
    g.add_node("personalization", personalization_agent)
    g.add_node("analysis", analysis_agent)
    g.add_node("delivery", delivery_agent)

    # Run collection, personalization, and analysis agents in parallel
    g.add_edge(START, "collection")
    g.add_edge(START, "personalization")
    g.add_edge(START, "analysis")
    
    # All three agents feed into the delivery agent
    g.add_edge("collection", "delivery")
    g.add_edge("personalization", "delivery")
    g.add_edge("analysis", "delivery")
    
    g.add_edge("delivery", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="Personalized News Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    from fastapi.responses import FileResponse
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(
            path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "frontend/index.html not found"}


@app.get("/test")
def serve_test_page():
    from fastapi.responses import FileResponse
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "test.html")
    if os.path.exists(path):
        return FileResponse(
            path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "test.html not found"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "personalized-news-agent"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="personalized-news-agent")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/get-news", response_model=NewsResponse)
def get_news(req: NewsRequest):
    graph = build_graph()
    # Only include necessary fields in initial state
    state = {
        "messages": [],
        "news_request": req.model_dump(),
        "articles": [],
        "tool_calls": [],
    }
    # No config needed without checkpointer
    out = graph.invoke(state)
    return NewsResponse(
        result=out.get("final", ""), 
        articles=out.get("articles", []),
        tool_calls=out.get("tool_calls", [])
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
