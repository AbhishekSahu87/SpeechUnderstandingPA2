
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import seaborn as sns
from scipy.stats import skew, kurtosis
import librosa
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Question 2 - Part A
# Set parameters
dataset_path = "/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset"
# Slected Malayalam, Tamil and Urdu
languages = ['Malayalam', 'Tamil', 'Urdu']
#sample_rate = 22050
sample_rate = 16000
# No. of MFCC coefficients
n_mfcc = 13
# FFT window size
n_fft = 2048
# No. of samples between successive frames
hop_length = 512

# MFCC Extract
def mfcc_ext(audio_path):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs

# MFCC Spectrogram Plot
def mfcc_spectrogram(mfccs, language, sample_idx, audio_file):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCC Spectrogram - {language.capitalize()} (Sample {sample_idx}) {audio_file}')
    plt.tight_layout()
    plt.show()

# Analyze and visualize samples for a given language
def analyze_samples(language, num_samples=3):
    language_path = os.path.join(dataset_path, language)
    audio_files = [f for f in os.listdir(language_path) if f.endswith('.mp3')][:num_samples]
    print(f"\nAnalyzing {len(audio_files)} samples for {language}:")
    mfcc_list = []
    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(language_path, audio_file)
        mfccs = mfcc_ext(audio_path)

        if mfccs is not None:
            mfcc_list.append(mfccs)
            mfcc_spectrogram(mfccs, language, i+1, audio_file)
            print(f"Mean:{np.mean(mfccs, axis=1)}")
            print(f"Variance:{np.var(mfccs, axis=1)}")
            print(f"Maximum:{np.max(mfccs, axis=1)}")
            print(f"Minimum:{np.min(mfccs, axis=1)}")
            print(f"Standard Deviation:{np.std(mfccs, axis=1)}")
            print(f"Skewness:{skew(mfccs, axis=1)}")

    return mfcc_list

# Statistical analysis on MFCC data
def stats_analysis(mfcc_data):
    stats = {}
    for lang, mfcc_list in mfcc_data.items():
        # Stack all MFCC frames for the language
        all_mfcc = np.hstack(mfcc_list)

        # Calculate statistics for each coefficient
        means = np.mean(all_mfcc, axis=1)
        variances = np.var(all_mfcc, axis=1)
        std = np.std(all_mfcc, axis=1)
        skewness = skew(all_mfcc, axis=1)
        maxval = np.max(all_mfcc, axis=1)
        minval = np.min(all_mfcc, axis=1)
        stats[lang] = {
            'mean': means,
            'variance': variances,
            'std': std,
            'skewness': skewness,
            'maxval': maxval,
            'minval': minval
        }

    return stats

#Plot statistical comparison
def stats_plot(stats):
    # Prepare data for plotting
    coeffs = range(1, n_mfcc+1)
    languages = list(stats.keys())

    # Create subplots
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(24, 15))

    # Plot means
    for lang in languages:
        ax1.plot(coeffs, stats[lang]['mean'], label=lang.capitalize())
    ax1.set_title('Mean MFCC Coefficients by Language')
    ax1.set_xlabel('MFCC Coefficient')
    ax1.set_ylabel('Mean Value')
    ax1.legend()
    ax1.grid(True)

    # Plot variances
    for lang in languages:
        ax2.plot(coeffs, stats[lang]['variance'], label=lang.capitalize())
    ax2.set_title('Variance of MFCC Coefficients by Language')
    ax2.set_xlabel('MFCC Coefficient')
    ax2.set_ylabel('Variance')
    ax2.legend()
    ax2.grid(True)

    # Plot Standard Deviation
    for lang in languages:
        ax3.plot(coeffs, stats[lang]['std'], label=lang.capitalize())
    ax3.set_title('Standard Deviation of MFCC Coefficients by Language')
    ax3.set_xlabel('MFCC Coefficient')
    ax3.set_ylabel('Standard Deviation')
    ax3.legend()
    ax3.grid(True)

    # Plot Skewness
    for lang in languages:
        ax4.plot(coeffs, stats[lang]['skewness'], label=lang.capitalize())
    ax4.set_title('Skewness of MFCC Coefficients by Language')
    ax4.set_xlabel('MFCC Coefficient')
    ax4.set_ylabel('Skewness')
    ax4.legend()
    ax4.grid(True)

    # Plot maximum
    for lang in languages:
        ax5.plot(coeffs, stats[lang]['maxval'], label=lang.capitalize())
    ax5.set_title('Maximum of MFCC Coefficients by Language')
    ax5.set_xlabel('MFCC Coefficient')
    ax5.set_ylabel('Maximum')
    ax5.legend()
    ax5.grid(True)

    # Plot Minimum
    for lang in languages:
        ax6.plot(coeffs, stats[lang]['minval'], label=lang.capitalize())
    ax6.set_title('Minimum of MFCC Coefficients by Language')
    ax6.set_xlabel('MFCC Coefficient')
    ax6.set_ylabel('Minimum')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()


# Analyze samples for each language
mfcc_data = {}
for lang in languages:
    mfcc_data[lang] = analyze_samples(lang, 5)

# Statistical analysis
stats = stats_analysis(mfcc_data)
stats_plot(stats)

# Question 2 - Part B
# Function to extract MFCC features from  audio file
def mfcc_ext_mean(audio_path, n_mfcc=13, sr=22050, max_pad_len=200):
    try:

        audio, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=512, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            padding = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        scaled_feature = np.mean(mfcc.T,axis=0)
        return scaled_feature
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Function to flatten the MFCC features
def flatten_mfcc(mfcc):
    return mfcc.flatten()

# extract MFCCs and corresponding labels
def prepare_dataset(dataset_path, languages):
    try:
        features = []
        labels = []
        for language in languages:
            language_path = os.path.join(dataset_path, language)
            all_files = os.listdir(language_path)

            for audio_file in all_files:
                audio_path = os.path.join(language_path, audio_file)
                mfcc = mfcc_ext_mean(audio_path)
                if mfcc is not None:
                    mfcc_flat = flatten_mfcc(mfcc)
                    features.append(mfcc_flat)
                    labels.append(language)

        return np.array(features), np.array(labels)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Set dataset path and languages
dataset_path = '/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset'
languages = ['Hindi', 'Tamil', 'Bengali', 'Gujarati','Kannada', 'Malayalam', 'Marathi', 'Punjabi', 'Telugu', 'Urdu']

# Prepare the dataset
try:
    X, y = prepare_dataset(dataset_path, languages)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    print(f"Error processing: {e}")

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=languages)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=languages, yticklabels=languages)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\nDataset Sizes:")
print(f"Total samples: {len(X)}")
print(f"Training set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"Number of features: {X_train.shape[1]}")