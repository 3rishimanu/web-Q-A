#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit newspaper3k sentence-transformers faiss-cpu


# In[2]:


import streamlit as st  # UI banane ke liye
from newspaper import Article  # Webpage se text extract karne ke liye
from sentence_transformers import SentenceTransformer  # Text to vector
import faiss  # Fast vector search
import numpy as np  # Numerical operations


# In[3]:


model = SentenceTransformer('all-MiniLM-L6-v2')


# In[4]:


st.title(" Web Q&A Tool (From URL Content)")


# In[5]:


url = st.text_input(" Enter a webpage URL:")


# In[6]:


if st.button(" Get Content"):
    article = Article(url)         # URL object banate hain
    article.download()             # Page ko download karte hain
    article.parse()                # Text content nikalte hain
    content = article.text         # Sirf text extract karte hain

    st.session_state.content = content  # Memory me save karte hain
    st.success("Content Loaded Successfully!")


# In[7]:


question = st.text_input(" Ask your question:")


# In[8]:


if question and 'content' in st.session_state:
    paragraphs = st.session_state.content.split('\n')  # Text ko todte hain lines me
    paragraphs = [p for p in paragraphs if len(p.strip()) > 20]  # Choti lines hata dete hain

    para_embeddings = model.encode(paragraphs)  # Har paragraph ka embedding
    q_embedding = model.encode([question])      # Question ka embedding

    index = faiss.IndexFlatL2(para_embeddings.shape[1])  # FAISS index
    index.add(np.array(para_embeddings))                # Embeddings add karte hain

    D, I = index.search(np.array(q_embedding), k=1)     # Search similar para
    answer = paragraphs[I[0][0]]                        # Best match

    st.markdown("###  Answer from content:")
    st.write(answer)







