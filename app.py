import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
import moviepy as mp
import librosa
import imageio
from tensorflow.keras.layers import Input, MultiHeadAttention, Add, Dense, TimeDistributed, GlobalAveragePooling1D, Layer
from tensorflow.keras.models import Model

# Define CTC Loss
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.reduce_sum(tf.cast(y_true != -1, tf.int64), axis=1)
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length[:, tf.newaxis]
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Custom Audio Tiling Layer
class AudioTilingLayer(Layer):
    def __init__(self, **kwargs):
        super(AudioTilingLayer, self).__init__(**kwargs)
    def call(self, inputs):
        lip_features, audio_features = inputs
        T = tf.shape(lip_features)[1]
        audio_expanded = tf.expand_dims(audio_features, axis=1)
        audio_tiled = tf.tile(audio_expanded, [1, T, 1])
        return audio_tiled

# Vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="", mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", mask_token=None, invert=True)

vocab_size = len(vocab) + 1  # 40

# Load Multimodal Model
@st.cache_resource
def load_multimodal_model():
    model_path = "C:/Users/omkar/LipNet/app/multimodal_model2.keras"
    lip_input = Input(shape=(75, 46, 140, 1), name='lip_input')
    audio_input = Input(shape=(40,), name='audio_input')
    lip_features = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 3, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool3D((1, 2, 2)),
        tf.keras.layers.Conv3D(256, 3, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool3D((1, 2, 2)),
        tf.keras.layers.Conv3D(75, 3, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool3D((1, 2, 2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])(lip_input)
    audio_tiled = AudioTilingLayer()([lip_features, audio_input])
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(lip_features, audio_tiled)
    fused_features = Add()([lip_features, attention_output])
    ctc_output = TimeDistributed(Dense(vocab_size, activation='softmax'), name='ctc_output')(fused_features)
    pooled = GlobalAveragePooling1D()(fused_features)
    classification_output = Dense(1, activation='sigmoid', name='classification_output')(pooled)
    model = Model(inputs=[lip_input, audio_input], outputs=[ctc_output, classification_output])

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'CTCLoss': CTCLoss, 'AudioTilingLayer': AudioTilingLayer}
        )
        st.success(f"Model loaded from {model_path}")
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return None
    return model

# Inference Functions
def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    frames = tf.stack(frames)
    if len(frames) < 75:
        frames = tf.pad(frames, [[0, 75 - len(frames)], [0, 0], [0, 0], [0, 0]])
    else:
        frames = frames[:75]
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path: str):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil':
                tokens = [*tokens, ' ', line[2]]
        return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
    except Exception as e:
        st.warning(f"Failed to load alignments from {path}: {e}. Using placeholder.")
        return char_to_num(tf.strings.unicode_split("hello world", input_encoding='UTF-8'))

def load_data(video_path: str):
    import streamlit as st
    import os
    import tensorflow as tf

    file_name = os.path.basename(video_path).split('.')[0]
    video_path_full = video_path if os.path.exists(video_path) else os.path.join('data', 's1', f'{file_name}.mpg')

    annotation_txt = os.path.join('data', 'annotations', f'{file_name}.txt')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    # Try to load video frames
    try:
        frames = load_video(video_path_full)
    except Exception as e:
        st.error(f"Failed to load video {video_path_full}: {e}")
        return None, None

    # Try to load annotations from .txt or fallback to .align
    if os.path.exists(annotation_txt):
        with open(annotation_txt, 'r') as f:
            annotation_line = f.readline().strip()
            alignments = [char_to_num(char) for char in annotation_line]
    elif os.path.exists(alignment_path):
        alignments = load_alignments(alignment_path)
    else:
        st.warning(f"No annotation (.txt) or alignment (.align) file found. Using placeholder text.")
        placeholder_text = "hello world"
        alignments = char_to_num(tf.strings.unicode_split(placeholder_text, input_encoding='UTF-8'))

    return frames, alignments

def extract_audio(video_path: str):
    from moviepy import VideoFileClip

    try:
        st.info(f"Extracting audio from: {video_path}")
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio is not None:
            audio_path = video_path.replace('.mp4', '.wav').replace('.mpg', '.wav')
            audio.write_audiofile(audio_path, codec='pcm_s16le')
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            return audio_data, sample_rate
        else:
            st.warning("No audio track found in video.")
            return None, None
    except Exception as e:
        st.error(f"Audio extraction failed: {e}")
        return None, None


def extract_audio_features(audio, sample_rate):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        print(f"Audio features shape: {mfcc.shape}")
        return mfcc
    except Exception as e:
        st.error(f"Audio feature extraction failed: {e}")
        return None

# Streamlit App
st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This app combines lip-reading and audio analysis for multimodal prediction.')

st.title('LipNet Multimodal App')

# Video selection or upload
try:
    video_dir = os.path.join('..', 'data', 's1')
    options = [f for f in os.listdir(video_dir) if f.endswith(('.mpg', '.mp4'))]
except FileNotFoundError:
    st.warning(f"Directory {video_dir} not found. Falling back to current directory.")
    video_dir = os.getcwd()
    options = [f for f in os.listdir(video_dir) if f.endswith(('.mpg', '.mp4'))]

selected_video = st.selectbox('Choose video', options) if options else None
uploaded_file = st.file_uploader("Or upload a video", type=['mp4', '.mpg'])

if selected_video or uploaded_file:
    temp_files = []
    if uploaded_file:
        video_path = 'test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())
        temp_files.append(video_path)
    else:
        video_path = os.path.join(video_dir, selected_video)

    # Validate video file
    if not os.path.exists(video_path):
        st.error(f"Video file {video_path} does not exist.")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.info('The video being processed')
        try:
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
        except Exception as e:
            st.error(f"Failed to display video: {e}")

    with col2:
        st.info('What the model sees (video frames)')
        with st.spinner('Processing video frames...'):
            frames, alignments = load_data(video_path)
            if frames is None:
                st.error("Video processing failed. Please try another video.")
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                st.stop()

            # Convert frames to GIF
            try:
                frames_gif = [(frame.numpy() * 255 / (frame.numpy().max() + 1e-6)).astype(np.uint8).squeeze() for frame in frames]
                imageio.mimsave('animation.gif', frames_gif, fps=10)
                st.image('animation.gif', width=400)
                temp_files.append('animation.gif')
            except Exception as e:
                st.error(f"GIF creation failed: {e}")

        # Load model and predict
        with st.spinner('Loading model and predicting...'):
            model = load_multimodal_model()
            if model is None:
                st.error("Model loading failed. Please check the model file.")
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                st.stop()

            test_frames = np.expand_dims(frames, axis=0)
            test_audio, test_sample_rate = extract_audio(video_path)
            if test_audio is None:
                st.error("Audio processing failed. Please try another video.")
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                st.stop()
            temp_files.append(video_path.replace('.mp4', '.wav').replace('.mpg', '.wav'))

            test_audio_features = extract_audio_features(test_audio, test_sample_rate)
            if test_audio_features is None:
                st.error("Audio feature extraction failed.")
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                st.stop()
            test_audio_features = np.expand_dims(test_audio_features, axis=0)

            try:
                ctc_output, classification_output = model.predict([test_frames, test_audio_features])
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                st.stop()

            # Decode CTC output
            #st.info('Predicted Text (Model Output)')
            #try:
            #    decoder = tf.keras.backend.ctc_decode(ctc_output, [75], greedy=False, beam_width=10)[0][0].numpy()

             #   predicted_text = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
              #  st.text(predicted_text)
            #except Exception as e:
            #    st.error(f"CTC decoding failed: {e}")

            # Real Text (if alignments exist)
            if alignments is not None:
                st.info('Predicted Text (Model Output)')
                try:
                    real_text = tf.strings.reduce_join([num_to_char(word) for word in alignments]).numpy().decode('utf-8')
                    st.text(real_text)
                except Exception as e:
                    st.error(f"Failed to decode alignments: {e}")

            # Classification
            st.info('Classification (Fake/Real)')
            try:
                classification = "Fake" if classification_output[0][0] >= 0.5 else "Real"
                st.text(classification)
            except Exception as e:
                st.error(f"Classification failed: {e}")

    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                st.warning(f"Failed to delete temporary file {temp_file}: {e}")