import os
import streamlit as st
from langchain import PromptTemplate
from langchain_groq import ChatGroq


# Secure API key loading (server-side only)
groq_api_key = (
    st.secrets.get("GROQ_API_KEY")
    or os.getenv("GROQ_API_KEY")
)


template = """\
For the following text, extract the following \
information:

sentiment: Is the customer happy with the product? 
Answer Positive if yes, Negative if \
not, Neutral if either of them, or Unknown if unknown.

delivery_days: How many days did it take \
for the product to arrive? If this \
information is not found, output No information about this.

price_perception: How does it feel the customer about the price? 
Answer Expensive if the customer feels the product is expensive, 
Cheap if the customer feels the product is cheap,
not, Neutral if either of them, or Unknown if unknown.

Format the output as bullet-points text with the \
following keys:
- Sentiment
- How long took it to deliver?
- How was the price perceived?

Input example:
This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap

text: {review}
"""


# PromptTemplate variables definition
prompt = PromptTemplate(
    input_variables=["review"],
    template=template,
)


# ✅ LLM loader (OpenAI → Groq)
def load_LLM(groq_api_key):
    llm = ChatGroq(
        temperature=0,
        model="llama-3.1-8b-instant",
        groq_api_key=groq_api_key
    )
    return llm


# Page title and header 
st.set_page_config(page_title="Extract Key Information from Product Reviews")
st.header("Extract Key Information from Product Reviews")


# Intro section 
col1, col2 = st.columns(2)

with col1:
    st.markdown("Extract key information from a product review.")
    st.markdown("""
        - Sentiment
        - How long took it to deliver?
        - How was its price perceived?
        """)

with col2:
    st.write("Contact with riyadhgenai@gmail.com to build your AI Projects")


# Input section 
st.markdown("## Enter the product review")

def get_review():
    review_text = st.text_area(
        label="Product Review",
        label_visibility='collapsed',
        placeholder="Your Product Review...",
        key="review_input"
    )
    return review_text

review_input = get_review()

if len(review_input.split(" ")) > 700:
    st.write("Please enter a shorter product review. The maximum length is 700 words.")
    st.stop()


# Output section
st.markdown("### Key Data Extracted:")

if review_input:
    if not groq_api_key:
        st.error("Groq API key is not configured on the server.")
        st.stop()

    llm = load_LLM(groq_api_key=groq_api_key)

    prompt_with_review = prompt.format(
        review=review_input
    )

    #  Groq-compatible invocation
    key_data_extraction = llm.invoke(prompt_with_review)

    st.write(key_data_extraction.content)
