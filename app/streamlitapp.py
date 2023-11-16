# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char,load_video
from modelutil import load_model
from moviepy.editor import VideoFileClip
from PIL import Image

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://blog.relecura.com/wp-content/uploads/2019/08/800_390.png')
    st.title('LipReading')

    # st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Lip Reading') 
# Generating a list of options or videos 
# options = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Choose video', options)

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov","mpg"])


# Generate two columns 
col1, col2 = st.columns(2)

if uploaded_file: 

    # Rendering the video 
    # with col1: 
    #     st.info('Input Video')
    #     file_path = os.path.join('..','data','s1', selected_video)
    #     # os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video_1.mp4 -y')

    #     clip = VideoFileClip(file_path)
    #     clip.write_videofile('test_video.mp4', codec="libx264", audio_codec="aac")

    #     # Rendering inside of the app
    #     video = open('test_video.mp4', 'rb') 
    #     video_bytes = video.read() 
    #     st.video(video_bytes)

    with col1:
        st.info('Input Video')
        # Save the uploaded file to a temporary location
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        clip = VideoFileClip("uploaded_video.mp4")
        clip.write_videofile('test_video.mp4', codec="libx264", audio_codec="aac")

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2: 
        # st.info('This is all the machine learning model sees when making a prediction')
        # video, annotations = load_data(tf.convert_to_tensor(file_path))
        video= load_video('uploaded_video.mp4')
        # imageio.mimsave('animation.gif', video, fps=8)
        # st.image('animation.gif', width=400) 

        st.info('Prediction')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        # st.text(decoder)

        # Convert prediction to text
        # st.info('Here is the predicted sentences')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
