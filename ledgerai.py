from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Literal, Any
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from collections import Counter
import re  # regular expression =used for cleaning text

load_dotenv()
import json
import chromadb
import uuid  # to generate random ids

client = chromadb.PersistentClient(path="./memorydb")
collection = client.get_or_create_collection("storage")
from pypdf import PdfReader
# llama-3.1-8b-instant
# llama-3.3-70b-versatile
llm = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq"
)
def get_data():
    stored = collection.get(include=["metadatas"])
    return stored["metadatas"]
data = get_data()
class transactionpattern(BaseModel):
    description: str
    amount: float


class transactionlist(BaseModel):
    transactions: list[transactionpattern]


class rawcompany(BaseModel):
    rawmerchant_name: str = Field(
        ...,
        description="Merchant/company name only"
    )


class company(BaseModel):
    merchant_name: str = Field(
        ...,
        description="Merchant/company name only"
    )


class getcategory(BaseModel):
    category_name: Literal[
        "Food", "Transport", "Shopping", "Bills", "Entertainment", "Subscription", "Salary", "Transfer", "Investment", "Healthcare", "Education", "Other"] = Field(
        ...,
        description="You have to identify the category from the list provided",
    )


class billamount(BaseModel):
    amount: float = Field(
        ...,
        description="Amount given in the text"
    )


class choiceanalysis(BaseModel):
    typee: Literal["goal", "query", "hike"] = Field(
        ...,
        description="Classify the user intent"
    )


class budgetsetting(BaseModel):
    budget_goal: float = Field(
        ...,
        description="The amount of the goal user will provide"
    )


class questionfinding(BaseModel):
    ques_category: str | None = Field(
        ...,
        description="Extract the category from the user statement"
    )
    ques_merchant: str | None = Field(
        ...,
        description="Extract the merchant from the user statement"
    )


class State(TypedDict):
    input: str
    transactions: list[dict]
    transaction: str
    transaction_history: list[dict[
        str, Any]]  # a list where each item is a dictionary, and the dictionary has string keys and any type of values.”
    rawmerchant_name: str
    merchant_name: str
    category_name: str
    next: str
    category_freq: Any
    count_merchant: int
    popular_merchant: str
    analysis: str
    amount: float
    total_amount: dict[str, float]
    total_merchant: dict[str, float]
    total_percent: dict[str, float]
    max_spending: str
    total_amount_spend: float
    freq_merchant: str
    freq_merchant_amount: float
    average_transaction: float
    budget_goal: float
    question: str
    goal_analysis: str
    ques_analysis: str
    hike_analysis: str
    user_message: str
    user_preference: str
    typee: str


try:
    with open("merchant_memory.json", "r") as file:
        merchant_memory = json.load(file)
except:
    merchant_memory = {}


def extract_from_pdf(state: State):
    input = state["input"]
    classifyllm = llm.with_structured_output(transactionlist)
    message = [{
        "role": "system",
        "content": """You are a bank statement parser.
        Extract every transaction from the bank statement.

        Include:
        - Expenses
        - Income
        - Salary credits
        - Interest credits
        - Refunds

        Ignore only:
        - Opening Balance
        - Closing Balance
        - Headers
        - Footers

        Return all valid transactions.
            """

    }, {"role": "user", "content": input}
    ]
    result = classifyllm.invoke(message)
    return {
        "transactions": result.transactions
    }


def router(state: State):
    text = state["rawmerchant_name"]
    for key, value in merchant_memory.items():
        if key in text:
            return {"next": "no_need"}
    return {"next": "need"}


def clean_transaction(text: str) -> str:
    text = text.upper()
    text = text.replace('.', "")
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b(UPI|NEFT|IMPS|DR|CR|COM|INDIA)\b', '', text)
    text = re.sub(r'[/\-_:]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_transaction(state: State):
    text = clean_transaction(state["transaction"])
    noise_words = [
        "PRIVATE LIMITED",
        "PVT LTD",
        "LIMITED",
        "LTD",
        "PRIVATE",
        "PVT",
        "SERVICES",
        "PAYMENTS",
        "PAYMENT",
        "PG",
        "TECHNOLOGIES",
        "TECH"
    ]
    for i in noise_words:
        text = text.replace(i, "")
        # s means any whitespace and + means one or more  and '.' means Replace all those multiple spaces with a SINGLE space.
        text = re.sub(r'\s+', ' ', text)  # re.sub(pattern, replacement, text)
    return {"rawmerchant_name": text.strip()}  # removes spaces from:beginning and end


def get_name_from_memory(state: State):
    text = state["rawmerchant_name"]
    for key, value in merchant_memory.items():
        if key in text:
            return {"merchant_name": value}
    return {"merchant_name": text}


def getmerchantname(state: State):
    last_message = state["rawmerchant_name"]
    classifyllm = llm.with_structured_output(company)
    result = classifyllm.invoke(
        [
            {
                "role": "system",
                "content": """You are a financial transaction cleaner.
                Your task is to extract the main merchant or company name from messy bank transaction text.
                Rules:
                - Return ONLY the merchant name.
                - Remove numbers, transaction IDs, symbols, and payment prefixes.
                - Keep output short and clean.
                - Convert to uppercase.
                - Do not explain anything.
                Examples:
                Input: UPI-SWIGGY-82738
                Output: SWIGGY
                Input: NETFLIX.COM 8899
                Output: NETFLIX
                Input: AMAZON PAY INDIA
                Output: AMAZON
                Input: UBER TRIP BLR
                Output: UBER"""
            }, {"role": "user", "content": last_message}
        ]
    )
    return {"merchant_name": result.merchant_name}


def amountsnatch(state: State):
    last_message = state["transaction"]
    numbers = re.findall(r'\d+(?:\.\d+)?', last_message)
    if not numbers:
        amount = 0.0
    else:
        amount = float(numbers[-1])
    return {"amount": amount}


def getcategoryname(state: State):
    last_message = f"""
        Transaction: {state["transaction"]}
        Merchant: {state["merchant_name"]}
        """
    classifyllm = llm.with_structured_output(getcategory)
    result = classifyllm.invoke(
        [
            {
                "role": "system",
                "content": """You are a financial transaction classifier.
            Your task is to classify the merchant into ONE category only.
            Possible categories:
            - Food
            - Transport
            - Shopping
            - Bills
            - Entertainment
            - Subscription
            - Salary
            - Transfer
            - Investment
            - Healthcare
            - Education
            - Other
            Rules:
            - Return ONLY one category name.
            - Do not explain.
            - Choose the closest category.
            - If uncertain, return "Other".
            Examples:
            Merchant: SWIGGY
            Category: Food

            Merchant: UBER
            Category: Transport

            Merchant: NETFLIX
            Category: Subscription

            Merchant: AMAZON
            Category: Shopping

            Merchant: SPOTIFY
            Category: Entertainment"""
            }, {"role": "user", "content": last_message}
        ]
    )
    record = {
        "transaction": state["transaction"],
        "merchant_name": state["merchant_name"],
        "category_name": result.category_name,
        "amount": state["amount"]
    }
    state["transaction_history"].append(record)
    collection.add(
        documents=[f"""
            Merchant: {state['merchant_name']}
            Category: {result.category_name}
            Amount: {state['amount']}
            Transaction: {state['transaction']}
        """],
        ids=[str(uuid.uuid4())],
        metadatas=[{"merchant_name": state["merchant_name"],
                    "category_name": result.category_name,
                    "amount": state["amount"]}]
    )
    return {"category_name": result.category_name}


# to add data that we have just added
def get_data():
    stored = collection.get(include=["metadatas"])
    return stored["metadatas"]

def count(state: State):
    data = get_data()
    total = data + state["transaction_history"]
    totalamount = {}
    totalmerchant = {}
    for i in total:
        category = i["category_name"]
        amount = i["amount"]
        if category.lower() == "salary":
            continue
        if category in totalamount:
            totalamount[category] += amount
        else:
            totalamount[category] = amount
    for i in total:
        merchant = i["merchant_name"]
        amount = i["amount"]
        if merchant in totalmerchant:
            totalmerchant[merchant] += amount
        else:
            totalmerchant[merchant] = amount
    return {"total_amount": totalamount,
            "total_merchant": totalmerchant}


def gettingmaximum(state: State):
    data = get_data()
    maximum = data + state["transaction_history"]
    total = sum(state["total_amount"].values())
    percentaget = {}
    for key, value in state["total_amount"].items():
        percentaget[key] = value / total * 100
    maxcat = Counter(i["category_name"] for i in maximum)
    maxmer = Counter(i["merchant_name"] for i in maximum)
    max_key = max(state["total_amount"], key=state["total_amount"].get)
    freq_mer = max(state["total_merchant"], key=state["total_merchant"].get)
    freq_mer_amount = state["total_merchant"][freq_mer]
    freqcat = maxcat.most_common(1)[0]
    freqmer, cntmer = maxmer.most_common(1)[0]
    average = total / len(maximum)
    return {
        "category_freq": freqcat,
        "count_merchant": cntmer,
        "popular_merchant": freqmer,
        "total_percent": percentaget,
        "max_spending": max_key,
        "total_amount_spend": total,
        "freq_merchant": freq_mer,
        "freq_merchant_amount": freq_mer_amount,
        "average_transaction": average
    }


def spending(state: State):
    statistics = f"""
            category_freq:{state["category_freq"]},
            merchant_freq:{state["popular_merchant"]},
            percentage:{state["total_percent"]},
            "count_frequency":{state["count_merchant"]},
            "max_spending":{state["max_spending"]},
            "total_amount_spend":{state["total_amount_spend"]},
            "freq_merchant":{state["freq_merchant"]},
            "freq_merchant_amount":{state["freq_merchant_amount"]},
            "average_transaction":{state["average_transaction"]}
    """
    message = [{
        "role": "system",
        "content": f"""You are a personal finance analyst.
        You will receive already-calculated financial statistics.
        IMPORTANT:
        - Do NOT perform calculations.
        - Do NOT change any numbers.
        - Use only the data provided.
        Your tasks:
        1. Identify the user's major spending habits.
        2. Explain which categories dominate spending.
        3. Compare frequent spending vs high-value spending if relevant.
        4. Give 2-3 practical saving suggestions.
        5. Keep the explanation concise and easy to understand.

        Statistics:{statistics}"""
    }, {"role": "user", "content": statistics}]
    result = llm.invoke(message)
    return {"analysis": result.content}


def questionidentifier(state: State):
    text = state["user_message"]
    classfyllm = llm.with_structured_output(choiceanalysis)
    message = [
        {
            "role": "system",
            "content": """Classify the user's request into exactly one of these labels:
                goal:
                - user wants to save money
                - user wants to buy something
                - user asks for budget advice
                Examples:
                "I want to buy a laptop"
                "How can I save 5000"
                "Help me plan for a bike purchase"
                query:
                - user asks about past transactions, merchants, categories, expenses, spending
                Examples:
                "Show my transport expenses"
                "What are my subscriptions?"
                "How much did I spend on food?"
                hike:
                - user asks about subscription price increase / recurring amount increase
                Examples:
                "Which subscription increased?"
                "Did Netflix price go up?"
                "Show me subscription hikes"
                Return only one label: goal, query, or hike."""
        }, {"role": "user", "content": text}
    ]
    result = classfyllm.invoke(message)
    return {"typee": result.typee}


def routertochoose(state: State):
    text = state["typee"]
    if text == "goal":
        return {"user_preference": "goal"}
    elif text == "query":
        return {"user_preference": "query"}
    else:
        return {"user_preference": "hike"}


def goalanalizer(state: State):
    budget = state.get("budget_goal", 0)
    goal = f"""
            total_amount={state["total_amount"]},
            goal={budget},
            ques={state["user_message"]},
            freq_merchant:{state["freq_merchant"]},
            freq_merchant_amount:{state["freq_merchant_amount"]},
            total_amount_spend={state["total_amount_spend"]},
            total_merchant={state["total_merchant"]},
            """
    message = [
        {
            "role": "system",
            "content": f"""You are a personal finance coach.

        You will receive:
        1. The user's savings goal.
        2. Spending statistics that have already been calculated.

        IMPORTANT:
        - Do NOT perform calculations unless necessary for explanation.
        - Use only the provided statistics.
        - Focus on practical and realistic advice.
        - Do not suggest reducing essential expenses excessively.
        - Prioritize categories where the user spends the most.

        IMPORTANT:
        -Never estimate.
        -Never infer missing values.
        -Never calculate any value.
        -Only discuss numbers explicitly provided.
        -If information is unavailable, say so.

        Your tasks:
        1. Explain whether the savings goal appears realistic.
        2. Identify the categories that should be reduced first.
        3. Suggest 3 specific actions the user can take.
        4. Keep the advice concise and encouraging.

        goal:{goal}"""
        }, {"role": "user", "content": goal}
    ]
    result = llm.invoke(message)
    return {"goal_analysis": result.content}


def question_analizer(state: State):
    question = state["user_message"]
    n_amount = state.get("n_amount", 5)
    result2 = collection.query(
        query_texts=[question],
        n_results=n_amount
    )
    retrieved_docs = result2["documents"][0]
    retrieved_metadata = result2["metadatas"][0]
    total = sum(item.get("amount", 0) for item in retrieved_metadata)
    result = llm.invoke(
        [
            {"role": "system",
             "content": f"""You are a personal finance assistant.
            The user asked a question about their financial records.
            You are provided with transactions retrieved from a vector database (ChromaDB).
            IMPORTANT RULES:
            - Answer ONLY using the retrieved transactions.
            - Do NOT invent transactions.
            - Do NOT assume missing information.
            - If the retrieved transactions do not contain enough information, clearly say so.
            - Mention transaction amounts when relevant.
            - Summarize patterns if visible.
            - Calculate totals only from the retrieved transactions.
            - Keep the answer concise and easy to understand.
            Retrieved Transactions:
            {retrieved_docs}
            Retrieved Metadata:
            {retrieved_metadata}
            total_sum:{total}"""
             }, {"role": "user", "content": question}
        ]
    )
    return {"ques_analysis": result.content}


def hikedetector(state: State):
    merchant_group = {}
    increase = 0
    prev = 0
    curr = 0
    name = None
    subscriptions = [
        "NETFLIX",
        "YOUTUBE",
        "SPOTIFY",
        "CHATGPT PLUS",
        "HOTSTAR",
        "GEMINI"
    ]
    data = get_data()
    for i in data:
        merchant = i["merchant_name"]
        if merchant not in subscriptions:
            continue
        amount = i["amount"]
        if merchant not in merchant_group:
            merchant_group[merchant] = []
        merchant_group[merchant].append(amount)
    for merchant, amount in merchant_group.items():
        if len(amount) < 2:
            continue
        if len(amount) >= 2:
            prev = amount[-2]
            curr = amount[-1]
            if curr > prev:
                increase = curr - prev
                name = merchant
    if name is None:
        return {"hike_analysis": "NO price hikes"}
    else:
        return {"hike_analysis":
                    f"{name} subscription increased from {prev} to {curr}"
                    f"\nincreased by :{increase}"}


graph_starter = StateGraph(State)
graph_builder = StateGraph(State)
graph1_builder = StateGraph(State)
graph_starter.add_node("extract_from_pdf", extract_from_pdf)
graph_builder.add_node("getmerchantname", getmerchantname)
graph_builder.add_node("getcategoryname", getcategoryname)
graph_builder.add_node("getname_from_memory", get_name_from_memory)
graph_builder.add_node("normalize_transaction", normalize_transaction)
graph_builder.add_node("amountss", amountsnatch)
graph_builder.add_node("router", router)
graph_builder.add_node("count", count)
graph_builder.add_node("spending", spending)
graph_builder.add_node("gettingmaximum", gettingmaximum)
graph1_builder.add_node("goalanalizer", goalanalizer)
graph1_builder.add_node("routertochoose", routertochoose)
graph1_builder.add_node("questionidentifier", questionidentifier)
graph_starter.add_edge(START, "extract_from_pdf")
graph_starter.add_edge("extract_from_pdf", END)

graph_builder.add_edge(START, "normalize_transaction")
graph_builder.add_edge("normalize_transaction", "router")
graph_builder.add_conditional_edges("router",
                                    lambda state: state["next"],
                                    {"no_need": "getname_from_memory", "need": "getmerchantname"})
graph_builder.add_edge("getmerchantname", "amountss")
graph_builder.add_edge("getname_from_memory", "amountss")
graph_builder.add_edge("amountss", "getcategoryname")
graph_builder.add_edge("getcategoryname", "count")
graph_builder.add_edge("count", "gettingmaximum")
graph_builder.add_edge("gettingmaximum", "spending")
graph_builder.add_edge("spending", END)

graph1_builder.add_edge(START, "questionidentifier")
graph1_builder.add_edge("questionidentifier", "routertochoose")
graph1_builder.add_conditional_edges("routertochoose", lambda state: state["user_preference"],
                                     {"goal": "goalanalizer", "query": "question_analizer",
                                      "hike": "hikedetector"})
graph1_builder.add_edge("goalanalizer", END)

graph1_builder.add_node("question_analizer", question_analizer)
graph1_builder.add_edge("question_analizer", END)

graph1_builder.add_node("hikedetector", hikedetector)
graph1_builder.add_edge("hikedetector", END)

graph = graph_starter.compile()
graph1 = graph_builder.compile()
graph2 = graph1_builder.compile()
