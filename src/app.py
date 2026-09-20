import streamlit as st
from graph import app

#Home page display image
st.image("src/assets/image.jpg")

#Home page title
st.title("🔎 Company Research Assistant")

#Description
st.write("""
Research public companies using AI-powered company insights, financial
performance, and recent news, all combined into one concise research report.
""")

#Chatbox
question = st.chat_input("Enter a company for eg : Microsoft")


if question:

    with st.spinner("🔎 Analyzing company, financials, news & generating report..."):

        # Generate final response
        result = app.invoke({"query": question})

    st.header(f"📊 {question} Research Report")

    st.subheader("🏢 Company Overview")
    with st.container(border=True):

        #Display about company section
        st.write(result["about_company_data"])

    st.subheader("💰 Financial Performance")
    with st.container(border=True):

        #Display finacial section
        st.write(result["financial_data"])

    st.subheader("📰 Recent News")
    with st.container(border=True):

        #Display recent news section
        st.write(result["recent_news_data"])

    st.subheader("📋 Key Takeaways")
    with st.container(border=True):

        #Display key takeaways section
        st.write(result["final_report_data"])