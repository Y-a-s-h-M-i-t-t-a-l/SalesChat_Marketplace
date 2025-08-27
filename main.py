import os
import operator
import json
from typing import List, Optional, TypedDict, Annotated, Dict
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set in the .env file")

llm = ChatOpenAI(model="gpt-5", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mobile-phones")
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if pinecone_index_name not in existing_indexes:
    print(f"Creating index '{pinecone_index_name}'...")
    pc.create_index(
        name=pinecone_index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created successfully.")

index = pc.Index(pinecone_index_name)


# --- 2. Pydantic Data Models ---
class Product(BaseModel):
    id: str = Field(..., alias="ID")
    company_name: str = Field(..., alias="Company Name")
    model_name: str = Field(..., alias="Model Name")
    max_price: int = Field(..., alias="Max Price")
    capacity: int = Field(..., alias="Capacity")
    ram: int = Field(..., alias="ram")
    back_camera: str = Field(..., alias="Back Camera")
    front_camera: str = Field(..., alias="Front Camera")
    processor: str = Field(..., alias="Processor")
    screen_size: str = Field(..., alias="Screen Size")
    battery: int = Field(..., alias="battery")
    description: str = Field(..., alias="Text")

class SpecialDeal(BaseModel):
    heading: str
    deal_price: float
    products_involved: List[Product]

class ChatRequest(BaseModel):
    user_message: str
    history: List[Dict[str, str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    text: str
    products: List[Product] = Field(default_factory=list)
    special_deal: Optional[SpecialDeal] = None

class FinalAnswer(BaseModel):
    text: str = Field(description="The conversational text to display to the user.")
    product_ids: List[str] = Field(default_factory=list)
    deal_heading: Optional[str] = None
    deal_price: Optional[float] = None
    deal_product_ids: Optional[List[str]] = Field(default_factory=list)


# --- 3. Tool Definitions ---
@tool
def find_product(query: str) -> List[dict]:
    "Gets product from Pinecone"
    # ... (This function is unchanged)
    print(f"--- TOOL: find_product(query='{query}') ---")
    query_embedding = embeddings_model.embed_query(query)
    results = index.query(vector=query_embedding, top_k=7, include_metadata=True, namespace="mobiles")
    products = []
    for match in results['matches']:
        metadata = match.get('metadata', {})
        product_data = {
            "ID": match.get('id'), "Company Name": metadata.get("Company Name"),
            "Model Name": metadata.get("Model Name"), "Max Price": metadata.get("Max Price"),
            "Capacity": metadata.get("Capacity"), "ram": metadata.get("ram"),
            "Back Camera": metadata.get("Back Camera"), "Front Camera": metadata.get("Front Camera"),
            "Processor": metadata.get("Processor"), "Screen Size": metadata.get("Screen Size"),
            "battery": metadata.get("battery"), "Text": metadata.get("Text")
        }
        if all(product_data.values()):
            products.append(product_data)
    return products

@tool
def get_deal(conversation_context: str, product_ids: List[str]) -> str:
    """
    Analyzes the conversation and a list of relevant products to generate a special deal,
    cross-sell, or upsell offer. Use this when the user shows buying intent, asks for discounts,
    or compares products.
    """
    # ... (This function is unchanged)
    print(f"--- TOOL: get_deal(context='{conversation_context}', product_ids={product_ids}) ---")
    deal_prompt = f" Make random deals of your choice, here is the input {conversation_context}"
    response = llm.invoke(deal_prompt)
    return response.content

# This LLM is for the main agent logic - it only knows about the REAL tools
tools = [find_product, get_deal]
llm_with_tools = llm.bind_tools(tools)

# --- 4. NEW LangGraph Agent Setup ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_products: Annotated[Dict[str, Product], operator.ior]
    product_context_ids: list[str]

def tool_using_agent_node(state: AgentState):
    """
    Node 1: The core agent. It decides whether to call a tool or respond to the user.
    """
    print("--- NODE: Tool Using Agent ---")
    
    # We now pass the available product context directly in the prompt
    product_context = state.get("product_context_ids", [])
    
    system_prompt = f"""
    You are an expert mobile salesperson. Follow these rules strictly.
    For normal Hi, Hello, or any information, you can reply yourself quickly without any tools.
    **Current Product Context:**
    The user is currently discussing the following product IDs: {product_context if product_context else "None"}.
    
    **Tool Usage Rules:**
    1.  Use `find_product` if the user asks for a new or different product. This will reset the context.
    2.  Use `get_deal` if the user asks for a discount or negotiates on a product in the Current Product Context. You MUST pass the product IDs from the context to the tool.
    3.  Respond without tools if presenting results or making small talk.

    You sell only mobiles. And yes, you can gift wrap if needed.  
    """

    response = llm_with_tools.invoke(
        [
            ("system", system_prompt),
        ]
        + state["messages"]
    )
    return {"messages": [response]}

def tool_node(state: AgentState):
    """
    Node 2: Executes the tools that the agent decided to call.
    """
    print("--- NODE: Tool Executor ---")
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    retrieved_products_update = {}
    
    # This will hold the IDs from the current tool call
    new_context_ids = []

    for tool_call in tool_calls:
        tool_output = globals()[tool_call['name']].invoke(tool_call['args'])
        if tool_call['name'] == 'find_product':
            for product_dict in tool_output:
                product_obj = Product(**product_dict)
                retrieved_products_update[product_obj.id] = product_obj
                # Add the found ID to our context list
                new_context_ids.append(product_obj.id)
            output_str = json.dumps(tool_output)
        else:
            output_str = str(tool_output)
        tool_messages.append(ToolMessage(content=output_str, tool_call_id=tool_call['id']))
        
    # The return dictionary now includes the context update
    return {
        "messages": tool_messages,
        "retrieved_products": retrieved_products_update,
        "product_context_ids": new_context_ids
    }

def final_answer_node(state: AgentState):
    """
    Node 3: The formatting node. Takes the full conversation and formats the final response.
    This is the ONLY node that knows about the FinalAnswer format.
    """
    print("--- NODE: Final Answer Formatter ---")
    # Use a dedicated LLM call that is forced to produce the FinalAnswer format
    formatter_llm = llm.with_structured_output(FinalAnswer)
    response = formatter_llm.invoke(
        [
            ("system", "You are a response formatting expert. Based on the entire conversation history, including all tool outputs, format the final response for the user using the `FinalAnswer` tool. The `product_ids` MUST be extracted from the `find_product` tool's output found in the conversation history."),
        ]
        + state["messages"]
    )
    return {"messages": [AIMessage(content="", tool_calls=[{"name": "FinalAnswer", "args": response.dict(), "id": "final"}])]}

def router(state: AgentState) -> str:
    """
    Conditional Edge: Decides the next step based on the agent's last message.
    """
    print("--- ROUTER ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # The agent has decided to use a tool.
        return "continue_with_tools"
    else:
        # The agent has responded conversationally, so it's time to format the final answer.
        return "generate_final_answer"

# Define the new graph structure
workflow = StateGraph(AgentState)
workflow.add_node("agent", tool_using_agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("final_answer_formatter", final_answer_node)

# Define the edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    router,
    {
        "continue_with_tools": "tools",
        "generate_final_answer": "final_answer_formatter",
    },
)
workflow.add_edge("tools", "agent")
workflow.add_edge("final_answer_formatter", END)

chatbot_graph = workflow.compile()


# --- 5. FastAPI Application ---
app = FastAPI(title="Mobile Salesperson Chatbot API", version="1.0.0")

# NEW: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.mount("/images", StaticFiles(directory="static/images"), name="images")

@app.on_event("startup")
def populate_pinecone_data():
    # ... (This function is unchanged)
    sample_data = [
        { "ID": "mobile_13", "Allowed Discount": 11490, "Back Camera": "200MP + 12MP", "Capacity": 256, "Company Name": "Samsung", "Front Camera": "12MP", "Max Price": 114900, "Model Name": "Galaxy S24 Ultra", "Processor": "Exynos 2400", "Screen Size": "6.8 inches", "Text": "Best for: Power users, professionals, creatives, and tech enthusiasts. Ideal use cases: Advanced photography, AI-powered productivity, intense gaming, and seamless multitasking. The ultimate flagship experience.", "battery": 5000, "ram": 12, "weight": 234 },
        { "ID": "mobile_12", "Allowed Discount": 10490, "Back Camera": "200MP + 12MP", "Capacity": 128, "Company Name": "Samsung", "Front Camera": "12MP", "Max Price": 104900, "Model Name": "Galaxy S24 Ultra", "Processor": "Exynos 2400", "Screen Size": "6.8 inches", "Text": "A great choice for users who want flagship features without needing maximum storage. Excellent for photography, productivity, and gaming.", "battery": 5000, "ram": 12, "weight": 234 }
    ]
    vectors_to_upsert = []
    for item in sample_data:
        embedding = embeddings_model.embed_query(item['Text'])
        metadata = {k: v for k, v in item.items() if k != 'ID' and k != 'Text'}
        metadata['Text'] = item['Text']
        vectors_to_upsert.append((item['ID'], embedding, metadata))
    if vectors_to_upsert:
        print("Upserting sample data to Pinecone...")
        index.upsert(vectors=vectors_to_upsert)
        print("Sample data successfully upserted.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # This endpoint logic is now correct and should work with the new graph
    history_messages = []
    for msg in request.history:
        if msg['role'] == 'user': history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant': history_messages.append(AIMessage(content=msg['content']))

    initial_state = { "messages": history_messages + [HumanMessage(content=request.user_message)], 
                      "retrieved_products": {},
                       "product_context_ids": []
                     }
    final_state = await chatbot_graph.ainvoke(initial_state)
    
    final_answer_call = final_state["messages"][-1].tool_calls[0]
    final_answer_args = final_answer_call['args']
    
    response_products = [
        final_state['retrieved_products'][pid]
        for pid in final_answer_args.get('product_ids', [])
        if pid in final_state['retrieved_products']
    ]
    response_deal = None
    if final_answer_args.get('deal_heading') and final_answer_args.get('deal_product_ids'):
        deal_products = [
            final_state['retrieved_products'][pid]
            for pid in final_answer_args['deal_product_ids']
            if pid in final_state['retrieved_products']
        ]
        response_deal = SpecialDeal(
            heading=final_answer_args['deal_heading'],
            deal_price=final_answer_args['deal_price'],
            products_involved=deal_products
        )
    return ChatResponse(
        text=final_answer_args['text'],
        products=response_products,
        special_deal=response_deal
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)