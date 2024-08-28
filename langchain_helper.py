# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType

# load_dotenv()

# def generate_pet_name(animal_type, pet_color):
#     llm = OpenAI(temperature=0.7)
#     prompt_templates_name = PromptTemplate(
#         input_variables=['animal_type', 'pet_color'],
#         template = "I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five coll names for my pet"
#     )
#     name_chain = LLMChain(llm=llm, prompt=prompt_templates_name, output_key="pet_name")
#     response = name_chain({'animal_type': animal_type, 'pet_color':pet_color})
    
#     return response


# def langchain_agent():
#     llm = OpenAI(temperature=0.5)
    
#     tools = load_tools(['wikipedia','llm-math'], llm = llm)
#     agent = initialize_agent(
#         tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
#     )
    
#     result = agent.run(
#         'What is the average age of a dog? Multiplt the age by 3'
#     )
#     print(result)

# if __name__ == "__main__":
#     langchain_agent()
#     print(generate_pet_name("cow","black"))


from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# video_url = 'https://youtu.be/-Osca2Zax4Y?si=iyOiePxzUy_bUayO'

def create_vector_db_from_youtube_url (video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = OpenAI(model='davinci-002')
    
    prompt = PromptTemplate(
        input_variables = ['question'],
        template = """
         You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response