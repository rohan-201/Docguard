import tensorflow as tf
import numpy as np
import streamlit as st
import cohere


# define skin lesions detection function through the CNN model
def get_prediction(img, Model):
  class_names = [
    'Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'
  ]
  img = tf.keras.utils.load_img(img, target_size=(180, 180))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  prediction = Model.predict(img_array)
  score = tf.nn.softmax(prediction)

  return [class_names[np.argmax(score)], 100 * np.max(score)]


# get answers form Cohere LLM models
def get_response(prompt):
  try:
    response = co.generate(
      model='command',
      prompt=
      'act as a professional medical specialist and answer this quesion {}'.
      format(prompt),
      max_tokens=1024,
      temperature=0.750)
    paragraph = response.generations[0].text
    return paragraph
  except Exception as e:
    print(e)
  finally:
    print('LLM ansered successfully')


# streamlit app

st.set_page_config(page_title='Docguard', page_icon='')
st.title('Docguard')
st.subheader('Skin Lesion detection')

# Caching the model for faster loading
@st.cache_data()

# laod the model
def load_model(model_path):
  model = tf.keras.models.load_model(model_path)
  return model
model = load_model('SkinNet-23M.h5')

API_KEY = "9pVkNHDRcEFXNkYq29iIG6oo4kKBgtwEhrgFkil1"

# connect to cohere API
co = cohere.Client(API_KEY)

image = st.file_uploader("Upload image",
                         type=['.jpg', '.jpeg', '.png'],
                         )
case = ""
case_paragraph = 'Upload an image for the region of lesion'

if image is not None:
  try:
    case = get_prediction(image, model)
    case_paragraph = 'the case in this image is {} with {:.2f}% confidence.'.format(case[0], case[1])
    st.image(image, caption='skin lesion image', width=200)
    st.subheader(case_paragraph)
    info = get_response('what is {}'.format(case[0]))
    st.write(info)
  except Exception as e:
    print(e)
  finally:
    print('image has been uploadded successfully')

prompt_input = st.chat_input('Ask The AI specialist')
  
if prompt_input:
  if prompt_input == '':
    st.error('what is your quesion?')
  else:
    st.subheader('The AI agent answer is:')
    answer = get_response(prompt_input)
    st.write(answer)