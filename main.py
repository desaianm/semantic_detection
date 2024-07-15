import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances, manhattan_distances
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
import streamlit as st
import matplotlib.pyplot as plt


model = SentenceTransformer("all-mpnet-base-v2")

def process_text(text):
    # remove duplicates and NaN
    sentences = [word for word in list(set(text)) if type(word) is str]
    result = ' '.join(sentences)
    return result


# Generate text using open ai api model
def generate_llm_response(message,passage):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.5)

    template = """
    try to find answer for  below request from data given in the database.if not available still answer the request
    Below is request 
    {message}

    data:
    {best_responses}

    """

    prompt = PromptTemplate(
        input_variables=["message", "best_responses"],
        template=template
    )

    chain =  prompt | llm
    response = chain.invoke({"message": message, "best_responses": passage})
    return response

# Simalarity Check
def sim_check(distance):
    if distance < 0.4:
        return "entailment"
    elif distance > 0.7:
        return  "containment"
    elif distance >= 0.4 and distance <= 0.7:
        return "neutral"


def main():
    st.set_page_config(page_title="Semantic Detection System ", page_icon=":bird:")
    st.header("Semantic Detection System 👾")
    
    passages = st.text_area("Enter Single or Multiple Passages for llm Passages (separate each passage with a new line)").split('\n')
    query = st.text_input("Enter query for llm output:")
    Click = st.button("Run",disabled=False)

    if Click:
        if query and passages is not None:
            processed_text = process_text(passages)
            #retreive info
            response = generate_llm_response(query,processed_text)
            print(processed_text)
            llm_response = response['text']
            emb_pass = model.encode(passages)
            emb_resp = model.encode(llm_response)

            # Calculating distance between llm output and ground truth passage
            dist_llm = cosine_similarity(emb_pass,emb_resp.reshape(1,-1)).flatten()[0]
            dist_eucl = euclidean_distances(emb_pass,emb_resp.reshape(1,-1)).flatten()[0]
            
            embeddings_2d = np.vstack((emb_pass, emb_resp))

            # Visulaize Semantic Understanding
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot each passage embedding
            for i, passage in enumerate(passages):
                ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], label=f'Passage {i+1}', s=100)
                ax.annotate(f' {i+1}', (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
            ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], label='Output', s=100, c='red')
            ax.set_xlim(-0.1, 0.1)
            ax.set_ylim(-0.1, 0.1)
            ax.set_title('Similarity between Passage and LLM Output')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)


            # Displaying Distance Metrics for Semantic Detection
            st.write(" Cosine Similarality (Measured between 0 for unlikely match and 1 for perfect match):"+str(dist_llm))
            st.write(" Euclidean Distance (More Distance more irrelvant): "+str(dist_eucl))
            st.info("Semantic Check : "+sim_check(dist_llm))
            with st.expander("LLM Output",False):
                st.write(llm_response)
        else:
            if passages or query is None:
                st.warning("Please enter Text to get started",icon="⚠️")
        

if __name__ == '__main__':
    main()