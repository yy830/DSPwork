import matplotlib.pyplot as plt
import librosa.display
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)


def plot_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC")
    st.pyplot(fig)


def plot_fft(y, sr):
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(Y), 1/sr)
    magnitude = np.abs(Y)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[:len(freqs) // 2], magnitude[:len(Y) // 2])
    ax.set_title("FFT - Frequency Domain")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    st.pyplot(fig)

def plot_fft_3d_interactive(y, sr, N=512):
    st.markdown("### 🎛️ 交互式 3D FFT 频谱图")

    hop_length = N // 2
    stft_result = librosa.stft(y, n_fft=N, hop_length=hop_length)
    magnitude = np.abs(stft_result)
    magnitude = magnitude[:N//2, :]  # 只取正频率部分

    freq = np.linspace(0, sr / 2, N // 2)
    time = np.arange(magnitude.shape[1])

    T, F = np.meshgrid(time, freq)
    Z = magnitude

    surface = go.Surface(x=T, y=F, z=Z, colorscale='Viridis')

    layout = go.Layout(
        title="FFT 时频结构（交互式）",
        scene=dict(
            xaxis=dict(title='Time Frame'),
            yaxis=dict(title='Frequency (Hz)'),
            zaxis=dict(title='Magnitude'),
        ),
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )

    fig = go.Figure(data=[surface], layout=layout)
    st.plotly_chart(fig)


#评价系统
def evaluate_model(model, X_test, y_test):
    st.markdown("### 📈 模型评价结果")

    y_pred = model.predict(X_test)

    # 分类报告（表格形式）
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    st.subheader("📋 分类指标表格")
    st.dataframe(report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    }))

    # 混淆矩阵
    st.subheader("🔀 混淆矩阵")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
     ax.set_title("SVM model confusion matrix")
    ax.set_xlabel("Tag Estimation")
    ax.set_ylabel("True label")

    st.pyplot(fig)
