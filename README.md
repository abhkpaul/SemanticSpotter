# Semantic Spotter Application

This Python application enables you to load a PDF and ask questions about it in natural language. Utilizing a large language model (LLM), the application generates responses specifically related to the content of your PDF, ensuring that unrelated questions are not addressed.

## How it works

The application processes the PDF by dividing the text into smaller chunks suitable for input into a large language model (LLM). It employs OpenAI embeddings to create vector representations of these chunks. When a user poses a question, the application identifies the chunks that are semantically similar to the query and uses these chunks to generate a response from the LLM.

The graphical user interface (GUI) is built with Streamlit, while Langchain is used to manage interactions with the LLM

## Installation


```
pip install -r requirements.txt
```

## Usage

```
streamlit run app.py
```


## Creator
```
Abhishek Paul
```



