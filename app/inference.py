from elasticsearch import Elasticsearch
from transformers import pipeline

index_name = "qa_dataset"
es = Elasticsearch(['http://localhost:9200'], basic_auth=('elastic', 'elastic'))
question_answerer = pipeline("question-answering", model="models\checkpoint-7971")

def context_gen_from_elasticsearch(question):
  contexts = []
  body = {
    "query": {
      "multi_match": {
        "query": question,
        "fields": [
          "title^3",
          "question^2",
          "context"
        ],
        "fuzziness": "AUTO"
      }
    }
  }

  

  response = es.search(index=index_name, body=body)
  top_hits = response["hits"]["hits"]
  for hit in top_hits:
    contexts.append(hit["_source"]["context"])
  return contexts

def answer_question(question, k=3):
  contexts = context_gen_from_elasticsearch(question)
  unique_answers = set()
  results = []
  for c in contexts:
    model_out = question_answerer(question=question, context=c)
    result = {
      "answer": model_out["answer"],
      "score": model_out["score"],
      "context": c
    }
    
    if result["answer"] not in unique_answers:
      results.append(result)
      unique_answers.add(result["answer"])
  results.sort(key=lambda x: x['score'], reverse=True)
  prompt_answer = ""
  for r in results[:k]:
    prompt_answer += r['answer'].replace('.','') + ", "
  prompt_answer = prompt_answer[:-2] + "."
  return prompt_answer