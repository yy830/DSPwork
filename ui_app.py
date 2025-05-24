#ui_app.py
try:
    import librosa
except ModuleNotFoundError:
    st.error("librosa 模块未安装，请检查 requirements.txt")
import streamlit as st
import librosa
import numpy as np
from preprocessing import load_audio, extract_mfcc
from utils import plot_waveform, plot_mfcc,plot_fft,plot_fft_3d_interactive,evaluate_model
from model import train_model_from_file, predict_digit
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

st.title("🎙️ 语音信号识别系统")

# 加载分类模型
st.sidebar.title("模型加载")
model = train_model_from_file()
st.sidebar.success("模型已加载")

uploaded_file = st.file_uploader("📤 上传一个 .wav 文件", type="wav")

if uploaded_file is not None:
    # 1. 加载音频
    y, sr = load_audio(uploaded_file)
    st.write("采样率:", sr)
    #2. 波形图
    st.subheader("⏱️ 原始波形（时域）")
    plot_waveform(y, sr)

    mfcc = extract_mfcc(y, sr)
    st.markdown("### 🧠 MFCC 特征图（感知域）")
    plot_mfcc(y, sr)


    st.markdown("### 📊 FFT 频谱图（频域）")
    plot_fft(y, sr)

    plot_fft_3d_interactive(y, sr, N=512)

    try:
        prediction = predict_digit(model, mfcc)
        st.subheader("🌟 识别结果")
        st.write(f"模型预测的数字是：**{prediction}**")
    except Exception as e:
        st.error(f"模型预测失败：{e}")

    # 从数据文件重新划分一次测试集
    data = np.load("digit_mfcc_dataset.npz")
    X_all, y_all = data["X"], data["y"]
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # 侧边栏控制按钮
    if st.sidebar.button("📊 显示模型评价"):
        evaluate_model(model, X_test_eval, y_test_eval)
#cd C:\Users\daiyan\Desktop\Stu\DSP_design
#streamlit run ui_app.py
