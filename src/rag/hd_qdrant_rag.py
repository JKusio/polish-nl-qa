from rag.rag import RAG
from retrievers.retriever import Retriever
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

class HDQdrantRAG(RAG):
    def __init__(self, retriever: Retriever, model_name: str):
        super().__init__(retriever)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)


    def generate(self, query: str):
        data = self.retriever.get_relevant_passages(query)
        context = " ".join(data)
        return self.pipeline(question=query, context=context)