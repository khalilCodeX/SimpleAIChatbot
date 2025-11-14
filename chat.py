from langchain_openai import ChatOpenAI
from management import initialize_openai_client
from WebLoader import load_website, chunk_documents
from Vectordb import create_vectorstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class chatbot:

    def __init__(self, url: str):
        self.client = initialize_openai_client()
        self.url = url
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        self.chat_history = []
        
        self.docs = load_website(self.url)
        self.splits = chunk_documents(self.docs)
        self.retriever = create_vectorstore(self.splits)

    def format_doc(self, documents):
        formatted = "\n\n".join([doc.page_content for doc in documents])
        return formatted

    def create_chain(self, user_prompt: str):
        system_prompt = ("You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer based on the context, say that you don't know. "
            "Keep the answer concise.\n\n"
            "{context}")
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{user_prompt}")
            ]
        )
        
        similar_docs = self.retriever.invoke(user_prompt)
        context = self.format_doc(similar_docs)

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
            "context": context,
            "user_prompt": user_prompt,
            "chat_history": self.chat_history
            }
        )

        self.chat_history.append(HumanMessage(content=user_prompt))
        self.chat_history.append(AIMessage(content=response))

        return response
        

def main():
    bot = chatbot("https://www.apple.com/iphone/")
    
    while True:
        question = input("Enter your question (or q to quit): ")
        if question.lower() == 'q':
            break
        answer = bot.create_chain(question)
        print(f"AI: {answer}\n")

if __name__ == "__main__":
    main()