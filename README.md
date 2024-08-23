Note: this project was made for learning purposes.
# Project Name: **QA System with Elasticsearch and Transformers**

## Overview

This project is a Question Answering (QA) system that uses Elasticsearch for retrieving relevant contexts from a dataset and a fine-tuned transformer model for generating precise answers. The QA system allows for querying large datasets efficiently, returning answers based on the most relevant context.

## Features

- **Context Retrieval**: Uses Elasticsearch to fetch relevant contexts based on the user's query.
- **Question Answering**: Utilizes a transformer model, specifically fine-tuned on QA datasets, to generate answers from the retrieved contexts.
- **Gradio Interface**: Provides an interactive interface using Gradio for easy querying and answer retrieval.
- **Custom Model Training**: Supports training custom models using the `squad` dataset for fine-tuning QA models.

## Preperation

1. **Setup Elasticsearch:**

   Ensure you have Elasticsearch running locally or on a server. Modify the connection settings in the code if necessary.

2. **Download and Prepare the Model:**

   download a pre-trained transformer model compatible with Hugging Face's `transformers` library.

3. **fine-tune the model:**
   using the `squad` dataset for qa finetuning.

5. **Index the Dataset:**

   index the `squad` dataset into Elasticsearch.

### Question Answering Pipeline

```python
from transformers import pipeline

question_answerer = pipeline("question-answering", model="models/checkpoint-7971")
```

### Querying and Answer Generation

```python
question = "What is the capital of USA"
answer = answer_question(question)
print(answer)
```

## Custom Model Training

### Dataset Preparation

```python
from datasets import load_dataset

squad = load_dataset("squad", split="train[:50000]").shuffle(seed=1234)
squad = squad.train_test_split(test_size=0.15)
```
