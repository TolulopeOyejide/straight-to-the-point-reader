# -*- coding: utf-8 -*-
"""pdf_processor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12dJsd1rCjLMlqncO3kp2N4TAk_Ll4Tbh
"""

import streamlit as st
from transformers import pipeline
from pypdf import PdfReader

model_pipeline = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

def chunk_text(text, max_length=512):
    words = text.split()
    chunks = [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
    return chunks

def get_answer_from_book(question, pdf_text, max_chunk_size=512):
    chunks = chunk_text(pdf_text, max_chunk_size)
    answers = []
    for chunk in chunks:
        try:
            result = model_pipeline({"question": question, "context": chunk})
            answers.append({"answer": result["answer"], "score": result["score"], "chunk": chunk})
        except Exception as e:
            print(f"Error processing chunk: {e}")

    if answers:
        best_answer = max(answers, key=lambda x: x["score"])
        return best_answer["answer"], best_answer["score"]
    else:
        return "No answer found", 0