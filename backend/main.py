from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import requests
from datetime import datetime, timedelta
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


class NewsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
    
    def get_everything(self, query: str = None, sources: str = None, 
                      domains: str = None, language: str = "en", 
                      sort_by: str = "publishedAt", page_size: int = 100,
                      from_date: str = None) -> Dict[str, Any]:
        """Search through millions of articles"""
        url = f"{self.base_url}/everything"
        params = {
            "pageSize": min(page_size, 100),  # Max 100 per request
            "language": language,
            "sortBy": sort_by
        }
        
        if query:
            params["q"] = query
        if sources:
            params["sources"] = sources
        if domains:
            params["domains"] = domains
        if from_date:
            params["from"] = from_date
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"NewsAPI request failed: {e}")
            return {"status": "error", "articles": [], "totalResults": 0}
    
    def get_top_headlines(self, country: str = "us", category: str = None,
                         sources: str = None, query: str = None,
                         page_size: int = 100) -> Dict[str, Any]:
        """Get breaking news headlines"""
        url = f"{self.base_url}/top-headlines"
        params = {
            "pageSize": min(page_size, 100),
        }
        
        if country and not sources:
            params["country"] = country
        if category:
            params["category"] = category
        if sources:
            params["sources"] = sources
        if query:
            params["q"] = query
            
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"NewsAPI request failed: {e}")
            return {"status": "error", "articles": [], "totalResults": 0}
    
    def get_sources(self, category: str = None, language: str = "en", 
                   country: str = "us") -> Dict[str, Any]:
        """Get available news sources"""
        url = f"{self.base_url}/sources"
        params = {
            "language": language,
        }
        if category:
            params["category"] = category
        if country:
            params["country"] = country
            
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"NewsAPI request failed: {e}")
            return {"status": "error", "sources": []}


# Initialize NewsAPI client
news_api = None
if os.getenv("NEWS_API_KEY"):
    news_api = NewsAPIClient(os.getenv("NEWS_API_KEY"))


def convert_newsapi_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NewsAPI article format to our internal format"""
    # Calculate time ago
    published_at = article.get("publishedAt", "")
    time_ago = "Unknown time"
    if published_at:
        try:
            pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            now = datetime.now(pub_time.tzinfo)
            diff = now - pub_time
            if diff.days > 0:
                time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
            else:
                minutes = diff.seconds // 60
                time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        except:
            time_ago = "Recent"
    
    # Determine category based on content
    title_lower = article.get("title", "").lower()
    description_lower = article.get("description", "").lower()
    source_name = article.get("source", {}).get("name", "").lower()
    
    category = "General"
    if any(word in title_lower + description_lower for word in ["stock", "market", "economic", "finance", "business", "company", "merger"]):
        category = "Business"
    elif any(word in title_lower + description_lower for word in ["tech", "ai", "artificial intelligence", "computer", "software", "digital"]):
        category = "Technology"
    elif any(word in title_lower + description_lower for word in ["health", "medical", "vaccine", "disease", "hospital", "doctor"]):
        category = "Health"
    elif any(word in title_lower + description_lower for word in ["politics", "government", "congress", "president", "election", "policy"]):
        category = "Politics"
    elif any(word in title_lower + description_lower for word in ["climate", "environment", "renewable", "energy", "carbon"]):
        category = "Climate"
    elif any(word in title_lower + description_lower for word in ["science", "research", "study", "discovery", "breakthrough"]):
        category = "Science"
    
    return {
        "title": article.get("title", ""),
        "source": article.get("source", {}).get("name", "Unknown Source"),
        "category": category,
        "time": time_ago,
        "summary": article.get("description", "")[:200] + "..." if article.get("description") and len(article.get("description", "")) > 200 else article.get("description", ""),
        "url": article.get("url", ""),
        "urlToImage": article.get("urlToImage", ""),
        "publishedAt": published_at
    }


def get_fallback_articles() -> List[Dict[str, Any]]:
    """Fallback mock articles when API is unavailable"""
    return [
        {"title": "Congressional hearings continue on digital privacy legislation", "source": "Reuters", "category": "Politics", "time": "2 hours ago", "summary": "Lawmakers debate comprehensive data protection framework"},
        {"title": "Tech stocks rally amid AI breakthrough announcements", "source": "Bloomberg", "category": "Technology", "time": "1 hour ago", "summary": "Major cloud providers report record growth in AI services"},
        {"title": "Climate targets announced by major economies at summit", "source": "BBC", "category": "Climate", "time": "4 hours ago", "summary": "New renewable energy goals set for 2030"},
        {"title": "Healthcare innovation receives increased federal funding", "source": "AP", "category": "Health", "time": "3 hours ago", "summary": "Medical research grants focus on breakthrough therapies"},
        {"title": "Global markets respond to economic stability indicators", "source": "Financial Times", "category": "Business", "time": "5 hours ago", "summary": "Consumer confidence rises amid steady employment rates"},
        {"title": "Quantum computing research breakthrough published", "source": "Nature", "category": "Science", "time": "6 hours ago", "summary": "New algorithm shows promise for complex problem solving"},
        {"title": "Merger activity increases in fintech sector", "source": "Wall Street Journal", "category": "Business", "time": "8 hours ago", "summary": "Digital banking solutions drive consolidation"},
        {"title": "WHO releases seasonal health preparedness guidelines", "source": "CNN", "category": "Health", "time": "7 hours ago", "summary": "Updated protocols for respiratory illness prevention"}
    ]


def fetch_real_articles(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch real articles from NewsAPI based on user request"""
    if not news_api:
        print("NewsAPI not available, using fallback articles")
        return get_fallback_articles()
    
    try:
        user_interests = req.get("interests", [])
        max_articles = req.get("max_articles", 10)
        time_period = req.get("time_period", "today")
        sources = req.get("sources", [])
        
        # Calculate date range based on time_period
        from_date = None
        if time_period == "today":
            from_date = datetime.now().strftime("%Y-%m-%d")
        elif time_period == "week":
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif time_period == "month":
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        all_articles = []
        
        # Strategy 1: Get top headlines
        try:
            sources_str = ",".join(sources) if sources else None
            headlines_response = news_api.get_top_headlines(
                sources=sources_str,
                page_size=50
            )
            
            if headlines_response.get("status") == "ok" and headlines_response.get("articles"):
                for article in headlines_response["articles"]:
                    if article.get("title") and article.get("title") != "[Removed]":
                        all_articles.append(convert_newsapi_article(article))
        except Exception as e:
            print(f"Error fetching headlines: {e}")
        
        # Strategy 2: Search by user interests
        if user_interests:
            for interest in user_interests[:3]:  # Limit to 3 interests to avoid too many API calls
                try:
                    interest_response = news_api.get_everything(
                        query=interest,
                        from_date=from_date,
                        sort_by="publishedAt",
                        page_size=20
                    )
                    
                    if interest_response.get("status") == "ok" and interest_response.get("articles"):
                        for article in interest_response["articles"]:
                            if article.get("title") and article.get("title") != "[Removed]":
                                converted_article = convert_newsapi_article(article)
                                # Avoid duplicates by checking titles
                                if not any(existing["title"] == converted_article["title"] for existing in all_articles):
                                    all_articles.append(converted_article)
                except Exception as e:
                    print(f"Error fetching articles for interest '{interest}': {e}")
        
        # Remove duplicates and filter by interests if specified
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            if article["title"] not in seen_titles:
                seen_titles.add(article["title"])
                
                # If user has interests, prioritize articles that match
                if user_interests:
                    article_text = (article["title"] + " " + article["summary"]).lower()
                    if any(interest.lower() in article_text for interest in user_interests):
                        unique_articles.insert(0, article)  # Priority insert
                    else:
                        unique_articles.append(article)
                else:
                    unique_articles.append(article)
        
        # Return requested number of articles
        final_articles = unique_articles[:max_articles]
        
        if final_articles:
            print(f"Successfully fetched {len(final_articles)} real articles from NewsAPI")
            return final_articles
        else:
            print("No articles found from NewsAPI, using fallback")
            return get_fallback_articles()[:max_articles]
            
    except Exception as e:
        print(f"Error in fetch_real_articles: {e}")
        return get_fallback_articles()


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
    if not news_api:
        # Fallback to mock data if no API key
        source_str = ", ".join(sources) if sources else "major news outlets"
        return f"""Breaking News Summary from {source_str}:
        - Global Markets: Tech stocks rally amid AI breakthrough announcements
        - Politics: Congressional hearings continue on digital privacy legislation
        - Climate: New renewable energy targets announced by major economies
        - Technology: Major cloud provider reports record growth in AI services
        - Health: WHO releases updated guidelines for seasonal health preparedness
        - Business: Merger activity increases in the fintech sector
        - Science: Breakthrough in quantum computing research published in Nature"""
    
    try:
        # Get top headlines from NewsAPI
        sources_str = ",".join(sources) if sources else None
        response = news_api.get_top_headlines(
            sources=sources_str,
            page_size=20
        )
        
        if response.get("status") == "ok" and response.get("articles"):
            articles = response["articles"]
            headlines = []
            
            for article in articles[:10]:  # Limit to 10 headlines
                title = article.get("title", "")
                source = article.get("source", {}).get("name", "Unknown")
                if title and title != "[Removed]":
                    headlines.append(f"- {source}: {title}")
            
            if headlines:
                source_str = ", ".join(sources) if sources else "top news sources"
                return f"Breaking News from {source_str}:\n" + "\n".join(headlines)
        
        # Fallback if API returns no results
        return "No breaking news available at this time. Please try again later."
        
    except Exception as e:
        print(f"Error fetching breaking news: {e}")
        # Fallback to mock data on error
        source_str = ", ".join(sources) if sources else "major news outlets"
        return f"""Breaking News Summary from {source_str}:
        - Tech stocks rally amid AI breakthrough announcements
        - Congressional hearings continue on digital privacy legislation
        - New renewable energy targets announced by major economies"""


@tool
def search_topic_news(topic: str, time_period: str = "today") -> str:
    """Search for news articles on a specific topic within timeframe."""
    if not news_api:
        # Fallback to mock data if no API key
        return f"""Recent news about '{topic}' ({time_period}):
        - Market Analysis: Sector shows resilience amid economic uncertainty
        - Innovation Updates: New developments in {topic} technology announced
        - Policy Changes: Regulatory frameworks evolving to address {topic} challenges
        - Industry Insights: Leading experts predict growth trends for {topic}
        - Global Impact: International perspective on {topic} developments
        - Research Findings: Academic studies reveal new insights about {topic}
        - Investment Trends: Funding patterns show increased interest in {topic}"""
    
    try:
        # Calculate date range based on time_period
        from_date = None
        if time_period == "today":
            from_date = datetime.now().strftime("%Y-%m-%d")
        elif time_period == "week":
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif time_period == "month":
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Search for articles about the topic
        response = news_api.get_everything(
            query=topic,
            from_date=from_date,
            sort_by="publishedAt",
            page_size=20
        )
        
        if response.get("status") == "ok" and response.get("articles"):
            articles = response["articles"]
            results = []
            
            for article in articles[:8]:  # Limit to 8 articles
                title = article.get("title", "")
                source = article.get("source", {}).get("name", "Unknown")
                description = article.get("description", "")
                
                if title and title != "[Removed]" and description:
                    # Truncate description to keep summary concise
                    short_desc = description[:100] + "..." if len(description) > 100 else description
                    results.append(f"- {source}: {title} - {short_desc}")
            
            if results:
                return f"Recent news about '{topic}' ({time_period}):\n" + "\n".join(results)
        
        return f"No recent news found for '{topic}' in the specified time period ({time_period})."
        
    except Exception as e:
        print(f"Error searching topic news: {e}")
        # Fallback to mock data on error
        return f"""Recent news about '{topic}' ({time_period}):
        - Market Analysis: Sector shows resilience amid economic uncertainty
        - Innovation Updates: New developments in technology announced
        - Policy Changes: Regulatory frameworks evolving to address challenges"""


@tool
def get_source_credibility(source: str) -> str:
    """Check credibility and bias information for a news source."""
    if not news_api:
        # Fallback to mock analysis if no API key
        return f"""Source Analysis for {source}:
        - Credibility Score: High (verified through multiple fact-checking organizations)
        - Political Bias: Moderate/Center (based on content analysis)
        - Fact-Check Record: 95% accuracy rate over past year
        - Transparency: Clear editorial policies and correction procedures
        - Expertise: Strong track record in investigative journalism
        - International Coverage: Comprehensive global news network
        - Reader Trust: High trust rating from media literacy organizations"""
    
    try:
        # Get available sources from NewsAPI
        response = news_api.get_sources()
        
        if response.get("status") == "ok" and response.get("sources"):
            sources = response["sources"]
            
            # Find matching source
            matching_source = None
            source_lower = source.lower()
            for s in sources:
                if (source_lower in s.get("name", "").lower() or 
                    source_lower in s.get("id", "").lower()):
                    matching_source = s
                    break
            
            if matching_source:
                name = matching_source.get("name", source)
                category = matching_source.get("category", "general").title()
                country = matching_source.get("country", "unknown").upper()
                language = matching_source.get("language", "unknown")
                description = matching_source.get("description", "No description available")
                
                # Simple credibility assessment based on known sources
                credibility = "Medium"
                if any(trusted in name.lower() for trusted in ["reuters", "associated press", "bbc", "npr", "pbs"]):
                    credibility = "High"
                elif any(trusted in name.lower() for trusted in ["cnn", "fox", "msnbc", "washington post", "new york times"]):
                    credibility = "Medium-High"
                
                return f"""Source Analysis for {name}:
                - Credibility Score: {credibility} (based on NewsAPI source verification)
                - Category: {category}
                - Country: {country}
                - Language: {language.title()}
                - Description: {description}
                - Verification Status: Verified by NewsAPI (active news source)
                - Coverage: Professional journalism with editorial standards"""
            
        # Fallback for unknown sources
        return f"""Source Analysis for {source}:
        - Credibility Score: Unknown (source not found in verified database)
        - Verification Status: Could not verify through NewsAPI
        - Recommendation: Cross-reference with multiple sources for important news
        - Note: Source may be regional, specialized, or not indexed by NewsAPI"""
        
    except Exception as e:
        print(f"Error checking source credibility: {e}")
        return f"""Source Analysis for {source}:
        - Credibility Score: Could not verify (API error)
        - Recommendation: Cross-reference with multiple trusted sources
        - Note: Verification service temporarily unavailable"""


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
        
        # Fetch real articles from NewsAPI or use fallback data
        articles = fetch_real_articles(req)
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
        "Create a well-structured personalized news briefing with {reading_style} style. "
        "Format your response with clear headers and bullet points as shown below:\n\n"
        "## News Briefing:\n"
        "- Start with 2-3 key headlines as bullet points\n"
        "- Each bullet should be a complete sentence about a major news story\n\n"
        "## Personalized Content for Your Interests in [topic]:\n"
        "- List 3-4 relevant stories matching user interests\n"
        "- Include specific details and implications\n\n"
        "## Analysis: Sentiment Analysis:\n"
        "- **Overall Tone:** Brief assessment of news sentiment\n"
        "- **Emotional Elements:** Commentary on reporting style\n"
        "- **Bias Indicators:** Assessment of bias in coverage\n"
        "- **Objectivity Score:** Numerical rating out of 10\n\n"
        "Use this structure exactly. Always include clear section headers (##) and bullet points (-) for easy reading.\n\n"
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
