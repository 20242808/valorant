import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

def load_model():
    return tf.keras.models.load_model('./valorant_model.h5')

model = load_model()

# 모델 로드
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('valorant_model.h5')

model = load_model()

# 클래스 이름 (캐릭터 이름)
CLASS_NAMES = [
    "Astra", "Breach", "Brimstone", "Chamber", "Cypher",
    "Fade", "Harbor", "Jett", "KAY/O", "Killjoy", 
    "Neon", "Omen", "Phoenix", "Raze", "Reyna", 
    "Sage", "Skye", "Sova", "Viper", "Yoru", "Deadlock", "Gekko"
]

# Streamlit UI
st.title("발로란트 캐릭터 분류기")
st.write("캐릭터 사진을 업로드하면 어떤 캐릭터인지 알려드립니다!")

uploaded_file = st.file_uploader("캐릭터 사진을 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # 이미지 전처리
    img_array = np.array(image.resize((224, 224))) / 255.0  # 모델 입력 크기 맞춤
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    # 결과 출력
    st.write(f"### 이 캐릭터는: {predicted_class} 입니다!")
