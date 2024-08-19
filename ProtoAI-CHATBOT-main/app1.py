import os
from langchain.llms import huggingface_hub 
import streamlit as st

os.environ['HUGGINGFACEHUB_API_TOKEN']= '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'
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")




from langchain import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="ProtoAI , YOUR CHATBOT PAL🤖", layout="wide")
st.title("ProtoAI , YOUR CHATBOT PAL🤖")
st.markdown("<h2 style='color:  #000000;'>Ask me anything!</h2>", unsafe_allow_html=True)
input_text = st.text_input("How can I help you today?", key="main_input")


       
# Add a footer
st.markdown("<footer style='text-align: center; color: gray;'>ask anything! ❤️ </footer>", unsafe_allow_html=True)

# Add a sidebar for additional options

st.sidebar.title("Options")

st.sidebar.markdown("### Settings")
st.sidebar.markdown("Adjust the temperature and max length for the model.")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_length = st.sidebar.slider("Max Length", 1, 256, 64)
st.sidebar.markdown("### Prompt Templates")
prompt_options = ["General", "Specific", ]
selected_prompt_type = st.sidebar.selectbox("Choose a prompt template", prompt_options)

st.sidebar.markdown("### Model")
model_options = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "google/flan-t5-large"]
selected_model = st.sidebar.selectbox("Choose a model", model_options)

from langchain  import HuggingFaceHub
if selected_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
    llm1 = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={'temperature': temperature, 'max_length': max_length},
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    st.sidebar.write("Using Mistral AI model for responses.")
elif selected_model == "google/flan-t5-large":
    llm1 = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Changed to flan-t5-base for FLAN model
        model_kwargs={'temperature': temperature, 'max_length': max_length},
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    st.sidebar.write("Using FLAN-T5 model for responses.")

general_prompt = PromptTemplate(input_variables=['question'], template='Can you tell me about {question}?')
specific_prompt = PromptTemplate(input_variables=['question'], template='What are the notable achievements of {question}?')

general_chain = LLMChain(llm=llm1, prompt=general_prompt, output_key="general_info")
specific_chain = LLMChain(llm=llm1, prompt=specific_prompt, output_key="specific_info")



if selected_prompt_type == "General":
    selected_prompt = general_prompt
elif selected_prompt_type == "Specific":
    selected_prompt = specific_prompt

# Run the chatbot logic
if input_text:
    try:
        # Prepare the question based on the selected prompt
        if selected_prompt_type == "General":
            person_info = general_chain.run(question=input_text)
        elif selected_prompt_type == "Specific":
            person_info = specific_chain.run(question=input_text)
       
        
        else:
                st.warning("Please provide context for your question.")

        # Display the general information
        st.write(f"Information about your question: {person_info}")
        
        # Optional: Ask for user input for the follow-up question
        follow_up_input = st.text_input("Follow-up Question:", placeholder="Type your follow-up question here...")
        
        if follow_up_input:
            follow_up_response = general_chain.run(question=follow_up_input)  # You can adjust this for follow-up logic
            st.write(f"Follow-up Response: {follow_up_response}")

    except Exception as e:
        st.error("Oops! Something went wrong. Please try again later.")
        st.error(str(e))



# Optionally update the LLM parameters based on user input
llm1.model_kwargs['temperature'] = temperature
llm1.model_kwargs['max_length'] = max_length



