# Ledger-Agent : Local First AI Wealth & Budgeting Assistant

Ledger-Agent is an AI-powered financial assistant that analyzes bank statements, categorizes transactions, remembers spending habits, and provides personalized financial insights.

The project is designed using LangGraph workflows and Retrieval-Augmented Generation (RAG) while keeping all user data local for privacy.

---

## Features

- Bank Statement PDF Processing
- Transaction Extraction using LLM
- Merchant Name Extraction
- Automatic Transaction Categorization
- Historical Merchant Memory
- Spending Analysis
- Personalized Saving Suggestions
- Local Memory Storage
- Retrieval-Augmented Generation (RAG)
- Streamlit Dashboard (Upcoming)

---

## Workflow

```
              PDF Upload
                   |
                   ▼
        PDF Text Extraction
             (PyPDF)
                   |
                   ▼
       Transaction Extraction
               (LLM)
                   |
                   ▼
       Transaction Processing
                   |
    --------------------------------
    |              |               |
    ▼              ▼               ▼
Normalize      Merchant      Amount
Transaction    Extraction    Extraction
                   |
                   ▼
        Merchant Memory Check
                   |
         -------------------
         |                 |
         ▼                 ▼
 Existing Merchant     New Merchant
         |                 |
         ▼                 ▼
 Category Lookup     LLM Categorization
         |
         ▼
     Save Memory
         |
         ▼
 Spending Analysis
         |
         ▼
 AI Financial Suggestions
```

---

## Technologies

- Python
- LangGraph
- LangChain
- Groq API
- PyPDF
- JSON Memory
- Vector Database (RAG)
- Streamlit (Upcoming)

---

## AI Workflow

The workflow is built using LangGraph where each node performs a dedicated task.

Example nodes include:

- PDF Parser
- Transaction Extractor
- Transaction Normalizer
- Merchant Extractor
- Category Classifier
- Memory Retrieval
- Spending Analyzer
- Recommendation Generator

---

## Memory System

The project maintains two memory files:

### merchant_memory.json

Stores

```
Merchant Name
Category
```

Example

```
SWIGGY -> Food
UBER -> Transport
NETFLIX -> Subscription
```

---

### memory.json

Stores

- Previous Transactions
- Spending History
- Historical Categories

This allows the assistant to make consistent future predictions.

---

## Categories

- Food
- Transport
- Shopping
- Bills
- Entertainment
- Subscription
- Salary
- Healthcare
- Education
- Investment
- Transfer
- Others

---

## Current Capabilities

- Analyze PDF bank statements
- Categorize transactions
- Build merchant memory
- Generate spending summaries
- Suggest saving opportunities
- Maintain persistent local memory

---

## Upcoming Features

- Streamlit Dashboard
- Monthly Spending Charts
- Budget Goals
- Goal Tracking
- Multi-Bank Support
- Voice Assistant
- Automatic Report Generation

---

## Privacy

Ledger-Agent is designed as a local-first application.

User financial data remains on the local machine and is not stored on external servers.

---

## Learning Outcomes

This project helped me understand:

- Agentic AI Workflows
- LangGraph State Machines
- Retrieval-Augmented Generation
- Prompt Engineering
- Memory Management
- AI Pipeline Design
- LLM Integration
- Financial Data Processing
