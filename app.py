import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- 1. SETUP THE CORE LANGCHAIN LOGIC (MODIFIED FOR CONVERSATION) ---

# Load environment variables (for the GROQ_API_KEY)
load_dotenv()

# Get the API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the model. Llama3-70b is a great choice for this.
# The 'model' variable is now our LLM for conversation.
llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

# The new prompt template for conversational Q&A
# This is the same powerful prompt we used before.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert multilingual assistant. Your task is to understand a question provided in an {input_language} and generate a comprehensive, detailed answer written exclusively in the {output_language}. Do not simply translate the question; you must ANSWER it in the target language.",
        ),
        ("human", "{question}"),
    ]
)

# The output parser remains the same
parser = StrOutputParser()

# The final chain for conversation
chain = prompt | llm | parser


# --- 2. BUILD THE STREAMLIT USER INTERFACE (UPDATED FOR CONVERSATION) ---

st.title("Multilingual Conversational Assistant 💬")
st.markdown("Powered by Llama3-70B on Groq for incredible speed!")

# Create two columns for the language selection dropdowns
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("Language of Your Question", ["English", "Hindi", "Marathi"])
with col2:
    output_lang = st.selectbox("Language for the Answer", ["Hindi", "English", "Marathi"])

# Add a text area for the user's question
question_text = st.text_area("Enter your question here:", height=100)

# Add a button to generate the answer
if st.button("Generate Answer"):
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
    elif question_text:
        # Show a spinner while waiting for the fast API response
        with st.spinner("Llama 3 is thinking..."):
            # Prepare the inputs for the new chain
            inputs = {
                "input_language": input_lang,
                "output_language": output_lang,
                "question": question_text,
            }
            # Invoke the chain to get the response
            answer = chain.invoke(inputs)

            # Display the result
            st.write("### Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question first.")
