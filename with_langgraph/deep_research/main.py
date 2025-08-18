import streamlit as st
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from planner import get_planned_queries
from research import get_search_responses
from writer import get_final_resport




st.set_page_config(page_title="AI Deep Search")

st.title("AI Deep Search")

text_input = st.text_input(label="enter your query for search")

if text_input:
    with st.spinner('Searching.....'):
        searched_results = []
        searched_queries = get_planned_queries(search_query=text_input)
        for query in searched_queries:
            searched_results.append(get_search_responses(research_topic=query))

        initial_research = "\n".join(searched_queries)

        final_report = get_final_resport(research=initial_research)

        st.markdown(final_report)