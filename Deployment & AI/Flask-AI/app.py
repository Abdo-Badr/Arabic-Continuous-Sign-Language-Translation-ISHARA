import os
import pickle
import cv2
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm.auto import tqdm
import numpy as np
import mediapipe as mp
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv1D , Dense , Embedding , Dropout , LayerNormalization , MultiHeadAttention
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam  
import subprocess

# Define the target frame rate and resolution
target_fps = 24
target_resolution = (680, 480)



def convert_video_format(input_path, output_path):
    # Create a VideoFileClip object
    clip = VideoFileClip(input_path)
    # Convert and save as MP4
    print(input_path)
    print(output_path)
    clip.write_videofile(output_path.split('.')[0]+'.mp4', codec='libx264')
    
def resize(video_path):
    output_path = None
    try:
        print(f"Attempting to load video: {video_path}")
        clip = VideoFileClip(video_path)
        print(f"Video duration: {clip.duration}")

        # Check if the video duration is available
        if clip.duration is None:
            raise ValueError("Unable to read video duration")

        # Resize the video
        clip_resized = clip.resize(newsize=target_resolution)
        print(f"Resizing video to resolution: {target_resolution}")

        # Set the frame rate
        clip_resized = clip_resized.set_fps(target_fps)
        print(f"Setting frame rate to: {target_fps}")

        # Define the output path
        output_path = os.path.join('uploads', 'tmp' + os.path.basename(video_path))
        print(f"Output path: {output_path}")

        # Write the video to the output path
        clip_resized.write_videofile(output_path, codec='libx264', audio=False, logger=None)
        print(f"Video saved to: {output_path}")

        # Close the clip to release resources
        clip.close()
        clip_resized.close()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

        # Attempt to convert video format and try again
        converted_path = os.path.join('uploads', 'converted_tmp.mp4')
        print(f"Attempting to convert video format: {video_path}")
        convert_result = convert_video_format(video_path, converted_path)
        
        if convert_result:
            try:
                print(f"Loading converted video: {convert_result}")
                clip = VideoFileClip(convert_result)
                print(f"Video duration: {clip.duration}")

                # Resize the video
                clip_resized = clip.resize(newsize=target_resolution)
                print(f"Resizing video to resolution: {target_resolution}")

                # Set the frame rate
                clip_resized = clip_resized.set_fps(target_fps)
                print(f"Setting frame rate to: {target_fps}")

                # Write the video to the output path
                clip_resized.write_videofile(output_path, codec='libx264', audio=False, logger=None)
                print(f"Video saved to: {output_path}")

                # Close the clip to release resources
                clip.close()
                clip_resized.close()
                output_path = convert_result
            except Exception as e:
                print(f"Error processing converted video {convert_result}: {e}")
                output_path = None

    return output_path



mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detections(img, model):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img,results



def extract_keypoints(results):
    nose = sorted([168,6,197,195,5,4,1,48,278,75,305,42,62,2])
    left_brow = [70,63,105,66,107]
    right_brow = [336,296,334,293,300]
    left_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
    right_eye =[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
    eyes = sorted(left_eye + right_eye + left_brow + right_brow)#42
    
    lips_external = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
    lips_inner = [78,95,88,178,87,14,317,402,318,324,306,415,310,311,312,13,82,81,80,191,78]
    lips = sorted(lips_external + lips_inner)#41
    
    external_contour = sorted([21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162])
    #36
    
    face_indices = sorted(nose+eyes+lips+external_contour)#133
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark])if results.pose_landmarks else np.zeros((33,4))#132
    if results.face_landmarks :
        face = []
        for idx in face_indices :
            landmark = results.face_landmarks.landmark[idx]
            x, y, z = landmark.x, landmark.y, landmark.z
            face.append((x, y, z))
        face = np.array(face)
    else :
        face = np.zeros((133,3))
    # face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((133,3))#339
    left_hand = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))#63
    right_hand = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))#63
    # return pose,face,left_hand,right_hand
    return pose.flatten(),face.flatten(),left_hand.flatten(),right_hand.flatten()


def ISHARA_video_preprocessing(path):
    path = resize(path)
    cap = cv2.VideoCapture(path)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.50,
                                        min_tracking_confidence=0.50,
                                        refine_face_landmarks=True,
                                        model_complexity=0) as holistic:
        video_keypoints = []
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print(f"{path} Finished reading the video.")
                break
    
            # Process frame
            frame, results = mediapipe_detections(frame, holistic)
            pose, face, left_hand, right_hand = extract_keypoints(results)
    
            # Store keypoints for the current frame
            keypoints = pose.tolist() + face.tolist() + left_hand.tolist() + right_hand.tolist()
            video_keypoints.append(keypoints)
            
        num_files = len(video_keypoints)
                
        # Check if the number of JSON files is less than 406
        if num_files < 406:
            for i in range(num_files, 406):
                # Create new JSON file with zero-filled content
                video_keypoints.append([0] * 657)
        # Store keypoints for the current video
        # all_video_keypoints.append(video_keypoints)
    
        # Release the video capture
        cap.release()

    return np.array(video_keypoints)

def QATAR_video_preprocessing(path):
    path = resize(path)
    cap = cv2.VideoCapture(path)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.50,
                                        min_tracking_confidence=0.50,
                                        refine_face_landmarks=True,
                                        model_complexity=0) as holistic:
        video_keypoints = []
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
#                 print(f"{path} Finished reading the video.")
                break
    
            # Process frame
            frame, results = mediapipe_detections(frame, holistic)
            pose, face, left_hand, right_hand = extract_keypoints(results)
    
            # Store keypoints for the current frame
            keypoints = pose.tolist() + face.tolist() + left_hand.tolist() + right_hand.tolist()
            video_keypoints.append(keypoints)
            
        num_files = len(video_keypoints)
                
        # Check if the number of JSON files is less than 406
        if num_files < 650:
            for i in range(num_files, 650):
                # Create new JSON file with zero-filled content
                video_keypoints.append([0] * 657)
        # Store keypoints for the current video
        # all_video_keypoints.append(video_keypoints)
    
        # Release the video capture
        cap.release()

    return np.array(video_keypoints)


#load tokenizer using pickle
with open("ISHARAtokenizer_data.pkl", 'rb') as f:
    ishara_data = pickle.load(f)
    ishara_tokenizer = ishara_data['tokenizer']
    ishara_num_words = ishara_data['num_words']
    ishara_maxlen = ishara_data['maxlen']
    
    
    
#load tokenizer using pickle
with open("QATARtokenizer_data.pkl", 'rb') as f:
    qatar_data = pickle.load(f)
    qatar_tokenizer = qatar_data['tokenizer']
    qatar_num_words = qatar_data['num_words']
    qatar_maxlen = qatar_data['maxlen']
    
    
class LandmarkEmbedding(tf.keras.layers.Layer) : 
    def __init__(self , embedding_dim) : 
        super(LandmarkEmbedding , self).__init__() 
        
        self.conv1 = Conv1D(
            embedding_dim , 11 , strides = 2 , padding = 'same' , activation = 'relu'
        )
        
        self.conv2 = Conv1D(
            embedding_dim , 11 , strides = 2 , padding = 'same' , activation = 'relu'
        )
        
        self.conv3 = Conv1D(
            embedding_dim , 11 , strides = 2 , padding = 'same' , activation = 'relu'
        )
        self.conv4 = Conv1D(
            embedding_dim , 11 , strides = 2 , padding = 'same' , activation = 'relu'
        )
        
        self.conv5 = Conv1D(
            embedding_dim , 11 , strides = 2 , padding = 'same' , activation = 'relu'
        )
        
        
    def call(self , x) : 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # print(x.shape)
        return x
    

class EncoderBlock(tf.keras.layers.Layer) : 
    def __init__(self , embedding_dim , num_heads , fc_dim , dropout_rate = 0.1) : 
        super(EncoderBlock , self).__init__() 
        
        self.MHA = MultiHeadAttention(num_heads=num_heads , key_dim=embedding_dim) 
        
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6) 
        
        self.dropout1 = Dropout(dropout_rate) 
        self.dropout2 = Dropout(dropout_rate) 
        
        self.fc = tf.keras.Sequential([
            Dense(fc_dim , activation = 'relu') , 
            Dense(embedding_dim)
        ]) 
        
    def call(self , x) : 
        
        attn_out = self.MHA(x , x) 
        attn_out = self.dropout1(attn_out) 
        out1 = self.norm1(x + attn_out) 
        
        fc_out = self.dropout2(self.fc(out1)) 
        
        enc_out = self.norm2(out1 + fc_out) 
        
        return enc_out
    
    

class Encoder(tf.keras.layers.Layer) : 
    def __init__(
        self , 
        num_layers , 
        embedding_dim , 
        num_heads , 
        fc_dim ,  
        dropout_rate = 0.1
    ) : 
        super(Encoder , self).__init__() 
        
        self.num_layers = num_layers 
        
        self.enc_input = LandmarkEmbedding(embedding_dim)
        
        self.enc_layers = [EncoderBlock(embedding_dim , num_heads , fc_dim , dropout_rate)
                          for _ in range(num_layers)] 
        
        self.dropout = Dropout(dropout_rate)
        
    def call(self , x) : 
        
        x = self.enc_input(x)
        
        for i in range(self.num_layers) : 
            x = self.enc_layers[i](x) 
            
        return x # (batch_size , seqlen , embedding_dim)
    
    
    
class DecoderBlock(tf.keras.layers.Layer) : 
    def __init__(self , embedding_dim , num_heads , fc_dim , dropout_rate = 0.1) : 
        super(DecoderBlock , self).__init__() 
        
        self.MHA1 = MultiHeadAttention(num_heads=num_heads , key_dim=embedding_dim)
        self.MHA2 = MultiHeadAttention(num_heads=num_heads , key_dim=embedding_dim)
        
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6) 
        self.norm3 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate) 
        self.dropout2 = Dropout(dropout_rate) 
        self.dropout3 = Dropout(dropout_rate)
        
        self.fc = tf.keras.Sequential([
            Dense(fc_dim , activation = 'relu') , 
            Dense(embedding_dim)
        ])
        
    def look_ahead_mask(self , trg) : 
        batch_size = tf.shape(trg)[0] 
        seqlen = tf.shape(trg)[1] 
        
        i = tf.range(seqlen)[:, None]
        j = tf.range(seqlen)
        m = i >= j - seqlen + seqlen
        mask = tf.cast(m, tf.bool)
        mask = tf.reshape(mask, [1, seqlen, seqlen])
        mult = tf.concat(
            [batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0
        )
        
        return tf.tile(mask, mult)
        
    def call(self , x , enc_output) : 
        mask = self.look_ahead_mask(x) 
        
        attn1 = self.MHA1(x , x , attention_mask = mask) 
        attn1 = self.dropout1(attn1) 
        out1 = self.norm1(attn1 + x) 
        
        attn2 = self.MHA2(out1 , enc_output) 
        attn2 = self.dropout2(attn2) 
        out2 = self.norm2(attn2 + out1) 
        
        fc_out = self.dropout3(self.fc(out2)) 
        
        dec_out = self.norm3(fc_out + out2) 
        
        return dec_out
    
    
class Decoder(tf.keras.layers.Layer) : 
    def __init__(
        self , 
        num_layers ,
        embedding_dim , 
        num_heads , 
        fc_dim , 
        trg_vocab_size , 
        max_length , 
        dropout_rate = 0.1
    ) : 
        super(Decoder , self).__init__() 
        
        self.num_layers = num_layers 
        
        self.embedding = Embedding(trg_vocab_size , embedding_dim) 
        self.pos_encoding = Embedding(max_length , embedding_dim) 
        
        self.dec_layers = [DecoderBlock(embedding_dim , num_heads , fc_dim , dropout_rate)
                          for _ in range(num_layers)] 
        
        self.dropout = Dropout(dropout_rate)
        
    def call(self , trg , enc_output) : 
        batch_size = tf.shape(trg)[0] 
        seqlen = tf.shape(trg)[1]
        
        positions = tf.range(start=0, limit=seqlen, delta=1) 
        positions = tf.expand_dims(positions , axis = 0) 
        positions = tf.tile(positions , [batch_size , 1])
        
        x = self.dropout((self.embedding(trg) + self.pos_encoding(positions)))  
        
        for i in range(self.num_layers) : 
            x = self.dec_layers[i](x , enc_output)
        
        return x # (batch_size , seqlen , embedding_dim)
    
    
    
class Transformer(Model) : 
    def __init__(
        self , 
        enc_num_layers ,
        dec_num_layers,
        embedding_dim , 
        num_heads , 
        fc_dim , 
        trg_vocab_size , 
        trg_max_length , 
        dropout_rate = 0.1 
    ) : 
        super(Transformer , self).__init__() 
        
        self.encoder = Encoder(
            enc_num_layers , 
            embedding_dim , 
            num_heads , 
            fc_dim ,  
            dropout_rate 
        )
        
        self.decoder = Decoder(
            dec_num_layers , 
            embedding_dim , 
            num_heads , 
            fc_dim , 
            trg_vocab_size , 
            trg_max_length , 
            dropout_rate
        ) 
        
        self.fc_out = Dense(trg_vocab_size) 
        
    def call(self , src , trg) : 
        
        enc_output = self.encoder(src) 
        
        dec_output = self.decoder(trg , enc_output) 
        
        out = self.fc_out(dec_output) 
        
        return out



# set hyperparameters
EPOCHS = 120
EMBEDDING_DIM = 200 
FC_DIM = 400
enc_num_layers = 2
dec_num_layers = 1
NUM_HEADS = 4 
TRG_VOCAB_SIZE = len(ishara_tokenizer.word_index)
TRG_MAXLEN = ishara_maxlen
LR = 0.0001 
DROPOUT_RATE = 0.1 

ishara_model = Transformer(
    enc_num_layers ,
    dec_num_layers,
    EMBEDDING_DIM , 
    NUM_HEADS , 
    FC_DIM , 
    TRG_VOCAB_SIZE ,  
    TRG_MAXLEN , 
    DROPOUT_RATE)


temp_ishara_trg_out = ishara_model(np.zeros((1,406, 657)),np.zeros((1,ishara_maxlen),dtype=int))
ishara_model.summary() 


# set hyperparameters
EPOCHS = 120
EMBEDDING_DIM = 200 
FC_DIM = 400
enc_num_layers = 2
dec_num_layers = 1
NUM_HEADS = 4 
TRG_VOCAB_SIZE = len(qatar_tokenizer.word_index)
TRG_MAXLEN = qatar_maxlen
LR = 0.0001 
DROPOUT_RATE = 0.1 

qatar_model = Transformer(
    enc_num_layers ,
    dec_num_layers,
    EMBEDDING_DIM , 
    NUM_HEADS , 
    FC_DIM , 
    TRG_VOCAB_SIZE ,  
    TRG_MAXLEN , 
    DROPOUT_RATE)


temp_qatar_trg_out = qatar_model(np.zeros((1,650, 657)),np.zeros((1,qatar_maxlen),dtype=int))
qatar_model.summary() 


ishara_model.load_weights("ISHARA_Augmanted Model BATCH 32_weights.h5")
qatar_model.load_weights(r"QATAR_Augmanted Model BATCH 32_weights.h5")


def reverse_sentence(sentence):
    # Split the sentence into words
    words = sentence.split()
    # Reverse the order of words
    reversed_words = reversed(words)
    # Join the reversed words back into a sentence
    reversed_sentence = ' '.join(reversed_words)
    return reversed_sentence


def ishara_evaluate(path): 
    vid_keypoints = ISHARA_video_preprocessing(path)
    encoder_input = tf.expand_dims(vid_keypoints, 0)
    encoder_input = tf.cast(encoder_input, tf.float32)  # Ensure correct data type
    decoder_input = ishara_tokenizer.texts_to_sequences(['sos'])
    decoder_input = tf.convert_to_tensor(np.array(decoder_input), dtype=tf.int64) 
    for i in tqdm(range(ishara_maxlen)): 
        preds = ishara_model(encoder_input, decoder_input) 

        preds = preds[:, -1:, :] # (batch_size, 1, vocab_size) 
        predicted_id = tf.cast(tf.argmax(preds, axis=-1), tf.int64) 
        
        if predicted_id == ishara_tokenizer.word_index['eos']: 
            result = tf.squeeze(decoder_input, axis=0)
            pred_sent = ' '.join([ishara_tokenizer.index_word[idx] for idx in result.numpy() if idx != 0 and idx != 2 and idx != 3])
            pred_sent = reverse_sentence(pred_sent)
            return pred_sent
        decoder_input = tf.concat([decoder_input, predicted_id], axis=1)

    result = tf.squeeze(decoder_input, axis=0)
    pred_sent = ' '.join([ishara_tokenizer.index_word[idx] for idx in result.numpy() if idx != 0 and idx != 2 and idx != 3])
    pred_sent = reverse_sentence(pred_sent) 
    return pred_sent


def qatar_evaluate(path): 
    vid_keypoints = QATAR_video_preprocessing(path)
    encoder_input = tf.expand_dims(vid_keypoints, 0)
    encoder_input = tf.cast(encoder_input, tf.float32)  # Ensure correct data type
    decoder_input = qatar_tokenizer.texts_to_sequences(['sos'])
    decoder_input = tf.convert_to_tensor(np.array(decoder_input), dtype=tf.int64) 
    for i in tqdm(range(qatar_maxlen)): 
        preds = qatar_model(encoder_input, decoder_input) 

        preds = preds[:, -1:, :] # (batch_size, 1, vocab_size) 
        predicted_id = tf.cast(tf.argmax(preds, axis=-1), tf.int64) 
        
        if predicted_id == qatar_tokenizer.word_index['eos']: 
            result = tf.squeeze(decoder_input, axis=0)
            pred_sent = ' '.join([qatar_tokenizer.index_word[idx] for idx in result.numpy() if idx != 0 and idx != 2 and idx != 3])
            pred_sent = reverse_sentence(pred_sent)
            return pred_sent
        decoder_input = tf.concat([decoder_input, predicted_id], axis=1)

    result = tf.squeeze(decoder_input, axis=0)
    pred_sent = ' '.join([qatar_tokenizer.index_word[idx] for idx in result.numpy() if idx != 0 and idx != 2 and idx != 3])
    pred_sent = reverse_sentence(pred_sent) 
    return pred_sent




from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  #  routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/flask-api', methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    lang = request.form.get('model')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Save the file to the specified location
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Process the file
        if lang == 'Qatar':
            result = qatar_evaluate(file_path)
        else:
            result = ishara_evaluate(file_path)

        if result is None:
            raise ValueError("Error processing video file")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return jsonify({'error': str(e)}), 500
    
    # Remove the file after processing (optional)
    # os.remove(file_path)

    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True, port=5001)