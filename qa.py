from groq import Groq

class QASystem:
    def __init__(self, api_key, retriever):
        self.client = Groq(api_key=api_key)
        self.retriever = retriever

    def answer_question(self, question):
        results = self.retriever.hybrid_search(question)

        context = "\n\n".join([doc for doc, _ in results])

        prompt = f"""
Answer using ONLY this context:

{context}

Question: {question}
"""

        stream = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
