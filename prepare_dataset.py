import os
import librosa
import numpy as np

# 设置你的语音文件夹路径（替换为你实际路径）
DATASET_PATH = r"\mount\src\dspwork\free-spoken-digit-dataset-master\recordings"

X, y = [], []

print("开始提取特征...")

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".wav"):
        label = filename[0]  # 文件名以 0_、1_ 开头
        file_path = os.path.join(DATASET_PATH, filename)
        y_audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X.append(mfcc_mean)
        y.append(int(label))

X = np.array(X)
y = np.array(y)

np.savez("digit_mfcc_dataset.npz", X=X, y=y)
print("✅ 提取完成，已保存到 digit_mfcc_dataset.npz")
