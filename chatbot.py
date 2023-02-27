import streamlit as st
import openai


# load API key
openai.api_key = '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'

def chatResponse(user_input):
    prompt = f"User: {user_input}\Bias Buster:"
    response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5,
                n=1,
                stop=None,
                frequency_penalty=0,
                presence_penalty=0
            )
    chatbot_response = response.choices[0].text.strip()
    st.text_area("Bias Buster", value=chatbot_response, height=200, max_chars=None, key=None)

    

def detectBias(df):
    df = df.dropna(subset='contact_reason')
    fc_num = st.selectbox('Choose A Field Case Interaction Number', df['fc_num'].unique())
    textSource = df[df['fc_num'] == fc_num].iloc[0]['contact_reason']
    st.text_area("Field Interaction Description", value=textSource, height=300, max_chars=None, key=None)
    prompt = f'What potential bias can you identify in this police interaction? Here is information on the interaction: {textSource}'
    chatResponse(prompt)

