import streamlit as st
from langchain.llms import GooglePalm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import time



llm = GooglePalm(google_api_key=st.secrets["api_key"], temperature=0.5)

st.write(':red[If this page returns an error, it is probably because the Google generative ai API key is unavailable where the app is running.]')
st.title('Vacation City and Sightseeing Recommender')
st.image('cover.jpg')
st.subheader('First, select which month and continent you wanna travel,\
             then click on the Recommend button, to see the city and places to visit!')

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox(label='##### select the month', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
with col2:
    continent = st.selectbox(label='##### select the continent', options=['Asia','Europe','Africa','North America','South America','Oceania'])


col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    button = st.button('Recommend!')


def main():

    prompt_template_city = PromptTemplate(
    input_variables=['month','continent'],
    template = 'Please suggest to me a city to go for a vacation in {continent} in {month}, please justify your suggestion, why you tell me to visit this city in {month}?'
    )   

    city_chain = LLMChain(llm = llm, 
                      prompt = prompt_template_city,
                      output_key='city')


    prompt_template_sight = PromptTemplate(
        input_variables=['city'],
        template = 'please tell me the three major sightseeings of {city} city and their address in the {city} and make them bold and in bulletpoint'
    )

    sight_chain = LLMChain(llm = llm,
                        prompt = prompt_template_sight,
                        output_key='major sightseeings')

    chain = SequentialChain(chains = [city_chain,sight_chain],
                        input_variables=['month','continent'],
                        output_variables=['city','major sightseeings'])

    answer = chain({'month':month,'continent':continent})

    if button:
        with st.spinner('Please wait a second...!'):
          time.sleep(8)
        st.subheader(f'The most recommended city to visit in {answer["continent"]} in {answer["month"]} is {answer["city"]}')     


if __name__ == '__main__':
    main()
