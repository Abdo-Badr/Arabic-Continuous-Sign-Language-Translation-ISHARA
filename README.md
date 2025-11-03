## 🖐️ Ishara: Real-Time Egyptian Sign Language Translation & Chat System

**Ishara** is Egypt’s **first real-time Egyptian Sign Language translation and chat system**, designed to break communication barriers for the **deaf and hard-of-hearing community**.  
It offers two core applications — a **Translation App** and a **Chat App** — that work together to provide accessible, inclusive communication.

---

## 🚀 Overview

Ishara combines **Machine Learning**, **Computer Vision**, and **Natural Language Processing (NLP)** to interpret Egyptian Sign Language and facilitate communication through both **translation** and **chatting** features.

It enables:
- Real-time sign language to text translation  
- Two-way text communication  
- Seamless integration between signers and non-signers

---

## 🧩 System Components

### 🧏‍♀️ 1. Translation App
A camera-based application that detects and translates **continuous Egyptian Sign Language gestures** into text using deep learning models.

**Features:**
- Live video feed for gesture recognition  
- Real-time translation from signs to text  
- Option for text-to-sign language conversion (animated or symbolic)  
- Built with **Python**, **TensorFlow**, **Keras**, and **OpenCV**


### 💬 2. Chat App
A web-based or desktop chat system that connects signers and non-signers.  
It integrates directly with the translation system — enabling a signer to communicate naturally using gestures while the non-signer sees translated text.

**Features:**
- Real-time text communication  
- Automatic translation of sign inputs  
- Secure and simple interface built using **Flask**, **Streamlit**, or **FastAPI** with **WebSocket/Socket.IO**

---

## 🧠 Key Technologies

| Category | Tools & Frameworks |
|-----------|-------------------|
| **Programming Language** | Python |
| **Machine Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV |
| **Natural Language Processing** | spaCy, NLTK |
| **Web/Chat Framework** | Flask / Streamlit / FastAPI |
| **Database (optional)** | SQLite / Firebase / MongoDB |
| **Libraries** | NumPy, pandas, scikit-learn |

---

## 🎥 Data Collection & Training

The Ishara team created a **custom dataset** of continuous Egyptian Sign Language gestures.  
- Data was collected through recorded videos of native signers.  
- Each video frame was labeled and processed for ML training.  
- Models were trained and validated for gesture accuracy and context recognition.

---

## 🧪 How It Works

1. **Translation App** captures video frames using a webcam.  
2. The trained model identifies gestures and converts them into text.  
3. **Chat App** sends translated messages to the other user.  
4. Messages appear in real time — as text or signs — depending on the user’s mode.

---

## 👥 Team

Developed by the **Ishara Team** — students at the *Faculty of Computers & Information Technology, Benha University.*  

📩 Contact: [**Abdulrahman Badr**](https://www.linkedin.com/in/abdulrahman--badr/)
