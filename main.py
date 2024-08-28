# import langchain_helper as lch
# import streamlit as st

# st.title("Pets name generator")

# user_animal_type = st.sidebar.selectbox("What is your pet?" , ("Cat","Dog","Cow","Hamster"))

# if user_animal_type == "Cat":
#     pet_color = st.sidebar.text_area("What color is your Cat?",max_chars=15)
    
# if user_animal_type == "Dog":
#     pet_color = st.sidebar.text_area("What color is your Dog?",max_chars=15)
    
# if user_animal_type == "Cow":
#     pet_color = st.sidebar.text_area("What color is your Cow?",max_chars=15)
    
# if user_animal_type == "Hamster":
#     pet_color = st.sidebar.text_area("What color is your Hamster?",max_chars=15)
    
# if pet_color:
#     response = lch.generate_pet_name(user_animal_type,pet_color)
#     st.text(response['pet_name'])


import langchain_helper as lch
import streamlit as st
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=50,
        )
        query = st.sidebar.text_area(
            label="Ask me about the video!",
            max_chars=50,
            key='query',
        )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    response = lch.get_response_from_query(db, query)  # Adjusted: assuming it only returns the response
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
