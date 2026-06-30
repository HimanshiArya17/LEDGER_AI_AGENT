import streamlit as st
from pypdf import PdfReader
import io

# import your graphs
from ledgerai import graph,graph1,graph2


st.set_page_config(page_title="Finance AI Agent", layout="wide")

st.title("💰 Finance AI Agent")

# ---------------------------
# SESSION STATE
# ---------------------------
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None

if "pdf_result" not in st.session_state:
    st.session_state.pdf_result = None

# ---------------------------
# STEP 1: UPLOAD PDF
# ---------------------------
st.header("📄 Step 1: Upload Bank Statement")

uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

if uploaded_file:

    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.session_state.pdf_text = text

    state = {"input": text}

    with st.spinner("Extracting transactions..."):
        result = graph.invoke(state)

    st.session_state.pdf_result = result

    st.success("PDF processed successfully!")

    # ---------------------------
    # SHOW EXTRACTED DATA
    # ---------------------------
    st.subheader("📊 Extracted Transactions")

    st.write(result["transactions"])


# ---------------------------
# STEP 2: ANALYSIS (AUTO AFTER PDF)
# ---------------------------
if st.session_state.pdf_result:

    st.header("📈 Step 2: Auto Analysis")

    state = {
        "transaction_history": st.session_state.pdf_result.get("transactions", [])
    }

    with st.spinner("Analyzing spending patterns..."):
        analysis_result = graph1.invoke(state)

    st.subheader("💡 Spending Analysis")
    st.write(analysis_result["analysis"])


# ---------------------------
# STEP 3: OPTIONS
# ---------------------------
if st.session_state.pdf_result:

    st.header("🧠 Step 3: Choose What You Want Next")

    option = st.radio(
        "Select Feature",
        ["Budget Analysis", "Query Assistant", "Hike Detector"]
    )

    # ---------------------------
    # BUDGET ANALYSIS
    # ---------------------------
    if option == "Budget Analysis":
        if st.button("Run Budget Analysis"):

            state = {
                "transaction_history": st.session_state.pdf_result.get("transactions", [])
            }

            result = graph2.invoke({
                **state,
                "user_preference": "goal",
                "user_message": "budget analysis"
            })

            st.subheader("🎯 Budget Analysis Result")
            st.write(result.get("goal_analysis"))


    # ---------------------------
    # QUERY ASSISTANT
    # ---------------------------
    elif option == "Query Assistant":
        query = st.text_input("Ask about your spending")

        if st.button("Ask"):
            state = {
                "user_message": query
            }

            result = graph2.invoke({
                **state,
                "user_preference": "query"
            })

            st.subheader("🤖 Answer")
            st.write(result.get("ques_analysis"))


    # ---------------------------
    # HIKE DETECTOR
    # ---------------------------
    elif option == "Hike Detector":

        if st.button("Check Subscription Hikes"):

            state = {
                "transaction_history": st.session_state.pdf_result.get("transactions", [])
            }

            result = graph2.invoke({
                **state,
                "user_preference": "hike"
            })

            st.subheader("📈 Subscription Hikes")
            st.write(result.get("hike_analysis"))
