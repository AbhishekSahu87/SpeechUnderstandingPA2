
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

# Question 1 - II
import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from torch import nn
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained WavLM Base Plus model with huggingface
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name).to(device)
model.eval()

# Function Load audio files
def load_audio(file_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
    return waveform.squeeze(0)  # Remove channel dimension if mono

# Function to extract embeddings
def ext_embedding(audio_path):
    waveform = load_audio(audio_path)
    # Process audio with feature extractor
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values)
        # Use the mean of the last hidden state as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Cosine similarity function
cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Paths
voxceleb_dir = "/kaggle/input/voxceleb/vox1_test_wav/wav"
trial_file = "/kaggle/input/voxceleb/vox1_cleaned_trials.txt"

# Load trial pairs
trials = []
with open(trial_file, "r") as f:
    for line in f:
        label, file1, file2 = line.strip().split()
        trials.append((int(label), file1, file2))

# Dictionary to cache embeddings
embedding_cache = {}

# Compute similarity scores
scores = []
labels = []
for label, file1, file2 in tqdm(trials[:1000]):
    file1_path = os.path.join(voxceleb_dir, file1)
    file2_path = os.path.join(voxceleb_dir, file2)

    # Verify file existence
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print(f"Skipping missing file: {file1_path} or {file2_path}")
        continue

    # Get embeddings (cache to avoid recomputation)
    if file1_path not in embedding_cache:
        embedding_cache[file1_path] = ext_embedding(file1_path)
    if file2_path not in embedding_cache:
        embedding_cache[file2_path] = ext_embedding(file2_path)

    emb1 = torch.from_numpy(embedding_cache[file1_path]).to(device)
    emb2 = torch.from_numpy(embedding_cache[file2_path]).to(device)

    score = cosine_similarity(emb1, emb2).item()
    scores.append(score)
    labels.append(label)

#Calculate EER
def calc_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer = interp1d(fpr, fnr)(eer_threshold)
    return eer * 100

eer = calc_eer(labels, scores)
print(f"Equal Error Rate (EER): {eer:.2f}%")

# Metric TAR@1%FAR
def calc_tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    tar_at_far = interp1d(fpr, tpr)(target_far)
    return tar_at_far * 100

tar_at_1far = calc_tar_at_far(labels, scores, target_far=0.01)
print(f"TAR@1%FAR: {tar_at_1far:.2f}%")

# Metric Speaker Identification Accuracy
def calc_identification_accuracy(labels, scores, threshold=0.5):
    predictions = [1 if score >= threshold else 0 for score in scores]
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    accuracy = correct / len(labels) * 100  # Convert to percentage
    return accuracy

# Use EER threshold for identification 
fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
id_accuracy = calc_identification_accuracy(labels, scores, threshold=eer_threshold)
print(f"Speaker Identification Accuracy: {id_accuracy:.2f}%")


from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ArcFace Loss Implementation
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * (theta + self.m) + (1.0 - one_hot) * theta).cos() * self.s
        return F.cross_entropy(output, labels)

# Custom Dataset with padding/truncation
class VoxCeleb2Dataset(Dataset):
    def __init__(self, files, max_length=48000):  # 3 seconds at 16kHz
        self.files = files
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, speaker_id = self.files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        waveform = waveform.squeeze(0)


        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.size(0) < self.max_length:
            padding = torch.zeros(self.max_length - waveform.size(0))
            waveform = torch.cat([waveform, padding])

        return waveform, speaker_id

# Load pre-trained model and feature extractor
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name).to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=32,  # Increased rank
    lora_alpha=32,
    target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.train()

# Paths
voxceleb2_dir = "/kaggle/input/voxceleb/vox2_test_aac/aac"
voxceleb1_trial_file = "/kaggle/input/voxceleb/vox1_cleaned_trials.txt"
voxceleb1_dir = "/kaggle/input/voxceleb/vox1_test_wav/wav"

# Load VoxCeleb2 identities
all_ids = sorted([d for d in os.listdir(voxceleb2_dir) if os.path.isdir(os.path.join(voxceleb2_dir, d))])[:118]
train_ids = all_ids[:100]
test_ids = all_ids[100:]

# Prepare training data
train_files = []
for speaker_id in train_ids:
    speaker_path = os.path.join(voxceleb2_dir, speaker_id)
    for session in os.listdir(speaker_path):
        session_path = os.path.join(speaker_path, session)
        files = [f for f in os.listdir(session_path) if f.endswith((".wav", ".m4a"))]
        for file in files:
            train_files.append((os.path.join(session_path, file), speaker_id))

print(f"Collected {len(train_files)} training files from {len(train_ids)} speakers.")

# Fine-tuning Model setup
train_dataset = VoxCeleb2Dataset(train_files[:10000])  # Larger subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower learning rate
arcface_loss = ArcFaceLoss(in_features=768, out_features=len(train_ids)).to(device)
id_to_idx = {id: idx for idx, id in enumerate(train_ids)}

# Training loop (increase epochs and apply early stopping)
early_stopping_patience = 5  # Number of epochs to wait before stopping if no improvement

best_loss = float('inf')

patience_counter = 0

for epoch in range(30):
    total_loss = 0
    for waveforms, speaker_ids in tqdm(train_loader):

        inputs = feature_extractor(waveforms.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)

        optimizer.zero_grad()
        outputs = model(input_values).last_hidden_state.mean(dim=1)
        labels = torch.tensor([id_to_idx[sid] for sid in speaker_ids], dtype=torch.long).to(device)
        loss = arcface_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), '/kaggle/working/finetuned_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

model.eval()

# Evaluation function
def ext_embedding(audio_path, model):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = waveform.squeeze(0)
    inputs = feature_extractor(waveform.tolist(), sampling_rate=16e3, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Load VoxCeleb1 trial pairs
trials = []
with open(voxceleb1_trial_file, "r") as f:
    for line in f:
        label, file1, file2 = line.strip().split()
        trials.append((int(label), file1, file2))

# Evaluate pre-trained and fine-tuned models
def evaluate_model(model, name):
    embedding_cache = {}
    scores = []
    labels = []
    trial_subset = trials[:1000]
    for label, file1, file2 in tqdm(trial_subset):
        file1_path = os.path.join(voxceleb1_dir, file1)
        file2_path = os.path.join(voxceleb1_dir, file2)

        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Skipping missing file: {file1_path} or {file2_path}")
            continue

        if file1_path not in embedding_cache:
            embedding_cache[file1_path] = ext_embedding(file1_path, model)
        if file2_path not in embedding_cache:
            embedding_cache[file2_path] = ext_embedding(file1_path, model)

        emb1 = torch.from_numpy(embedding_cache[file1_path]).to(device)
        emb2 = torch.from_numpy(embedding_cache[file2_path]).to(device)
        score = cosine_similarity(emb1, emb2).item()
        scores.append(score)
        labels.append(label)

    if not labels:
        print(f"No valid trial pairs processed for {name}. Check voxceleb1_dir and trial file paths.")
        return None, None, None

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
    eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    tar_at_1far = interp1d(fpr, tpr)(0.01) * 100
    predictions = [1 if score >= eer_threshold else 0 for score in scores]
    id_accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels) * 100

    print(f"{name} - EER: {eer:.2f}%, TAR@1%FAR: {tar_at_1far:.2f}%, Speaker ID Accuracy: {id_accuracy:.2f}%")
    return eer, tar_at_1far, id_accuracy

# Load pre-trained model for comparison
pretrained_model = WavLMModel.from_pretrained(model_name).to(device)
pretrained_model.eval()

# Evaluate both models
pretrained_metrics = evaluate_model(pretrained_model, "Pre-trained")
finetuned_metrics = evaluate_model(model, "Fine-tuned")

#!pip install speechbrain
#!pip install pesq
#!pip install pystoi

# Q. III Create the Multi-Speaker Dataset
import os
import torch
import torchaudio
import numpy as np
import random
from tqdm import tqdm

# Folder Paths
voxceleb2_dir = "/kaggle/input/voxceleb/vox2_test_aac/aac"
output_train_dir = "/kaggle/working/mix/train"
output_test_dir = "/kaggle/working/mix/test"
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Load VoxCeleb2 identities (sorted ascending)
all_ids = sorted([d for d in os.listdir(voxceleb2_dir) if os.path.isdir(os.path.join(voxceleb2_dir, d))])
train_ids = all_ids[:50]  # First 50 for training
test_ids = all_ids[50:100]  # Next 50 for testing

# Function to load and resample audio
def load_audio(file_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
    return waveform.squeeze(0)
# Function to mix two utterances
def mix_utterances(file1, file2, max_length=48000):  # 3 seconds
    wav1 = load_audio(file1)
    wav2 = load_audio(file2)

    # Truncate or pad to max_length
    if wav1.size(0) > max_length:
        wav1 = wav1[:max_length]
    elif wav1.size(0) < max_length:
        wav1 = torch.cat([wav1, torch.zeros(max_length - wav1.size(0))])

    if wav2.size(0) > max_length:
        wav2 = wav2[:max_length]
    elif wav2.size(0) < max_length:
        wav2 = torch.cat([wav2, torch.zeros(max_length - wav2.size(0))])

    # Mix with random gain between 0.5 and 1.0
    gain1, gain2 = random.uniform(0.5, 1.0), random.uniform(0.5, 1.0)
    mixture = gain1 * wav1 + gain2 * wav2
    mixture = mixture / torch.max(torch.abs(mixture))

    return mixture, wav1, wav2

# Collect files for each identity
def collect_files(ids, root_dir):
    files_dict = {}
    for speaker_id in ids:
        speaker_path = os.path.join(root_dir, speaker_id)
        files = []
        for session in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session)
            files.extend([os.path.join(session_path, f) for f in os.listdir(session_path) if f.endswith(".m4a")])
        files_dict[speaker_id] = files
    return files_dict

# Create mixtures
def create_mixtures(ids, files_dict, output_dir, num_mixtures=100):
    for i in tqdm(range(num_mixtures)):
        # Randomly select two different speakers
        spk1, spk2 = random.sample(ids, 2)
        file1 = random.choice(files_dict[spk1])
        file2 = random.choice(files_dict[spk2])

        mixture, wav1, wav2 = mix_utterances(file1, file2)

        # Save mixture and original sources
        torchaudio.save(os.path.join(output_dir, f"mix_{i}.wav"), mixture.unsqueeze(0), 16000)
        torchaudio.save(os.path.join(output_dir, f"src1_{i}.wav"), wav1.unsqueeze(0), 16000)
        torchaudio.save(os.path.join(output_dir, f"src2_{i}.wav"), wav2.unsqueeze(0), 16000)

# Generate datasets
train_files = collect_files(train_ids, voxceleb2_dir)
test_files = collect_files(test_ids, voxceleb2_dir)
create_mixtures(train_ids, train_files, output_train_dir, num_mixtures=50)  # 50 training mixtures
create_mixtures(test_ids, test_files, output_test_dir, num_mixtures=50)    # 50 testing mixtures


# Q. III A  Speaker Separation with SepFormer
import os
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import SepformerSeparation
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

# Load pre-trained SepFormer model
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

# Evaluation metrics functions
def calc_sdr(ref, est):
    s_target = ref
    e_noise = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_noise**2) + 1e-8))

def calc_sir(ref, est, interferer):
    s_target = ref
    e_interf = interferer
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_interf**2) + 1e-8))

def calc_sar(ref, est):
    s_target = ref
    e_artifacts = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_artifacts**2) + 1e-8))

# Paths
test_dir = "/kaggle/working/mix/test"

# Evaluate on test set
results = {"SIR": [], "SAR": [], "SDR": [], "PESQ": []}
for i in tqdm(range(50)):  # 50 test mixtures
    mix_path = os.path.join(test_dir, f"mix_{i}.wav")
    src1_path = os.path.join(test_dir, f"src1_{i}.wav")
    src2_path = os.path.join(test_dir, f"src2_{i}.wav")

    # Load mixture and references
    mixture, sr = torchaudio.load(mix_path)
    ref1, _ = torchaudio.load(src1_path)
    ref2, _ = torchaudio.load(src2_path)
    mixture = mixture.squeeze(0).numpy()
    ref1 = ref1.squeeze(0).numpy()
    ref2 = ref2.squeeze(0).numpy()

    print(f"Mixture length: {mixture.shape[0]} samples ({mixture.shape[0]/16000:.2f}s)")

    # Perform separation
    est_sources = model.separate_file(mix_path)
    est_sources = est_sources.squeeze(0).detach().cpu().numpy()
    print(f"Est sources shape: {est_sources.shape}")

    # Validate shape
    if len(est_sources.shape) != 2 or est_sources.shape[1] != 2:
        raise ValueError(f"Expected [samples, 2], got {est_sources.shape}")

    est1, est2 = est_sources[:, 0], est_sources[:, 1]
    print(f"Est1 shape: {est1.shape}, Est2 shape: {est2.shape}")

    # Adjust lengths to match estimated sources
    min_len = min(est1.shape[0], ref1.shape[0])
    est1, est2 = est1[:min_len], est2[:min_len]
    ref1, ref2 = ref1[:min_len], ref2[:min_len]
    print(f"Adjusted lengths to {min_len} samples ({min_len/16000:.2f}s)")

    # Compute metrics
    sir1 = calc_sir(ref1, est1, ref2)
    sir2 = calc_sir(ref2, est2, ref1)
    sar1 = calc_sar(ref1, est1)
    sar2 = calc_sar(ref2, est2)
    sdr1 = calc_sdr(ref1, est1)
    sdr2 = calc_sdr(ref2, est2)
    pesq1 = pesq(16000, ref1, est1, "wb")
    pesq2 = pesq(16000, ref2, est2, "wb")

    # Store results
    results["SIR"].extend([sir1, sir2])
    results["SAR"].extend([sar1, sar2])
    results["SDR"].extend([sdr1, sdr2])
    results["PESQ"].extend([pesq1, pesq2])

# Compute averages
for metric in results:
    avg = np.mean(results[metric])
    print(f"Average {metric}: {avg:.2f}")

# Q. III B
#!pip install speechbrain

import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from speechbrain.inference import SepformerSeparation
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained WavLM and feature extractor
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
pretrained_model = WavLMModel.from_pretrained(model_name).to(device)
pretrained_model.eval()

# Load fine-tuned WavLM (assuming saved from first task)
finetuned_model = WavLMModel.from_pretrained(model_name).to(device)
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],
    lora_dropout=0.1
)

finetuned_model = get_peft_model(finetuned_model, lora_config)
# Load fine-tuned weights (update path to your saved model)
finetuned_model.load_state_dict(torch.load("/kaggle/working/finetuned_model.pth", weights_only=True))
finetuned_model.eval()

# Load SepFormer model
sep_model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

# Paths
test_dir = "/kaggle/working/mix/test"
voxceleb2_dir = "/kaggle/input/voxceleb/vox2_test_aac/aac"

# Test identities (50-99)
all_ids = sorted([d for d in os.listdir(voxceleb2_dir) if os.path.isdir(os.path.join(voxceleb2_dir, d))])
test_ids = all_ids[50:100]
id_to_idx = {id: idx for idx, id in enumerate(test_ids)}

# Function to extract embedding
def ext_embedding(waveform, model):
    inputs = feature_extractor(waveform.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Cosine similarity
cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Collect reference embeddings for test identities
ref_embeddings_pretrained = {}
ref_embeddings_finetuned = {}
for speaker_id in test_ids:
    speaker_path = os.path.join(voxceleb2_dir, speaker_id, os.listdir(os.path.join(voxceleb2_dir, speaker_id))[0])
    file = os.path.join(speaker_path, os.listdir(speaker_path)[0])
    waveform, sr = torchaudio.load(file)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.squeeze(0).numpy()

    ref_embeddings_pretrained[speaker_id] = ext_embedding(waveform, pretrained_model)
    ref_embeddings_finetuned[speaker_id] = ext_embedding(waveform, finetuned_model)

# Evaluate on separated test set
correct_pretrained = 0
correct_finetuned = 0
total = 0

for i in tqdm(range(50)):  # 50 test mixtures
    mix_path = os.path.join(test_dir, f"mix_{i}.wav")
    src1_path = os.path.join(test_dir, f"src1_{i}.wav")
    src2_path = os.path.join(test_dir, f"src2_{i}.wav")

    # Ground truth speaker IDs
    src1_file = os.path.basename(src1_path).split("_")[1]
    src2_file = os.path.basename(src2_path).split("_")[1]
    true_id1 = os.path.basename(os.path.dirname(os.path.dirname(src1_path)))
    true_id2 = os.path.basename(os.path.dirname(os.path.dirname(src2_path)))

    # Separation
    est_sources = sep_model.separate_file(mix_path).squeeze(0).detach().cpu().numpy()
    est1, est2 = est_sources[:, 0], est_sources[:, 1]

    # Extract embeddings from separated sources
    emb1_pretrained = ext_embedding(est1, pretrained_model)
    emb2_pretrained = ext_embedding(est2, pretrained_model)
    emb1_finetuned = ext_embedding(est1, finetuned_model)
    emb2_finetuned = ext_embedding(est2, finetuned_model)

    # Compute similarities and predict speakers
    pretrained_scores = {}
    finetuned_scores = {}
    for speaker_id in test_ids:
        ref_pre = torch.from_numpy(ref_embeddings_pretrained[speaker_id]).to(device)
        ref_fin = torch.from_numpy(ref_embeddings_finetuned[speaker_id]).to(device)
        pretrained_scores[speaker_id] = [
            cosine_similarity(torch.from_numpy(emb1_pretrained).to(device), ref_pre).item(),
            cosine_similarity(torch.from_numpy(emb2_pretrained).to(device), ref_pre).item()
        ]
        finetuned_scores[speaker_id] = [
            cosine_similarity(torch.from_numpy(emb1_finetuned).to(device), ref_fin).item(),
            cosine_similarity(torch.from_numpy(emb2_finetuned).to(device), ref_fin).item()
        ]

    # Rank-1 prediction
    pred_id1_pre = max(pretrained_scores, key=lambda k: pretrained_scores[k][0])
    pred_id2_pre = max(pretrained_scores, key=lambda k: pretrained_scores[k][1])
    pred_id1_fin = max(finetuned_scores, key=lambda k: finetuned_scores[k][0])
    pred_id2_fin = max(finetuned_scores, key=lambda k: finetuned_scores[k][1])

    # Check correctness (permutation invariant)
    pre_correct = (pred_id1_pre == true_id1 and pred_id2_pre == true_id2) or \
                  (pred_id1_pre == true_id2 and pred_id2_pre == true_id1)
    fin_correct = (pred_id1_fin == true_id1 and pred_id2_fin == true_id2) or \
                  (pred_id1_fin == true_id2 and pred_id2_fin == true_id1)

    if pre_correct == True:
        correct_pretrained += 1
    if fin_correct == True:
        correct_finetuned += 1
    total += 1

# Compute Rank-1 accuracy
rank1_acc_pretrained = correct_pretrained / total * 100
rank1_acc_finetuned = correct_finetuned / total * 100

print(f"Pre-trained WavLM Rank-1 Accuracy: {rank1_acc_pretrained:.2f}%")
print(f"Fine-tuned WavLM Rank-1 Accuracy: {rank1_acc_finetuned:.2f}%")


# Q. IV A,B
import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from speechbrain.pretrained import SepformerSeparation
from tqdm import tqdm
from pesq import pesq
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder Paths
voxceleb2_dir = "/kaggle/input/voxceleb/vox2_test_aac/aac"
train_dir = "/kaggle/working/mix/train"
test_dir = "/kaggle/working/mix/test"


# Load models
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
pretrained_wavlm = WavLMModel.from_pretrained(model_name).to(device)
pretrained_wavlm.eval()

# Fine-tuned WavLM with LoRA
finetuned_wavlm = WavLMModel.from_pretrained(model_name).to(device)
lora_config = LoraConfig(r=32, lora_alpha=32, target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"], lora_dropout=0.1)
finetuned_wavlm = get_peft_model(finetuned_wavlm, lora_config)

# SepFormer
sepformer = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="pretrained_models/sepformer-wsj02mix").to(device)

# Dataset
class MultiSpeakerDataset(Dataset):
    def __init__(self, data_dir, max_length=48000):
        self.data_dir = data_dir
        self.max_length = max_length
        self.files = [f for f in os.listdir(data_dir) if f.startswith("mix_") and f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mix_path = os.path.join(self.data_dir, self.files[idx])
        src1_path = os.path.join(self.data_dir, f"src1_{idx}.wav")
        src2_path = os.path.join(self.data_dir, f"src2_{idx}.wav")

        mix, sr = torchaudio.load(mix_path)
        src1, _ = torchaudio.load(src1_path)
        src2, _ = torchaudio.load(src2_path)

        if sr != 16000:
            mix = torchaudio.transforms.Resample(sr, 16000)(mix)
            src1 = torchaudio.transforms.Resample(sr, 16000)(src1)
            src2 = torchaudio.transforms.Resample(sr, 16000)(src2)

        mix, src1, src2 = mix.squeeze(0), src1.squeeze(0), src2.squeeze(0)
        if mix.size(0) > self.max_length:
            mix, src1, src2 = mix[:self.max_length], src1[:self.max_length], src2[:self.max_length]
        elif mix.size(0) < self.max_length:
            padding = torch.zeros(self.max_length - mix.size(0))
            mix = torch.cat([mix, padding])
            src1 = torch.cat([src1, padding])
            src2 = torch.cat([src2, padding])
        if len(mix.shape) == 1:
            mix = mix.unsqueeze(0)
        # Extract IDs from filenames (assuming format src1_idXXXXX_idx.wav)
        id1 = src1_path.split("src1_")[1].split(".")[0] if "src1_" in src1_path else os.path.basename(os.path.dirname(os.path.dirname(src1_path)))
        id2 = src2_path.split("src2_")[1].split(".")[0] if "src2_" in src2_path else os.path.basename(os.path.dirname(os.path.dirname(src2_path)))

        return mix, src1, src2, id1, id2

# Load datasets
train_dataset = MultiSpeakerDataset(train_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = MultiSpeakerDataset(test_dir)

# Identification loss
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * (theta + self.m) + (1.0 - one_hot) * theta).cos() * self.s
        return F.cross_entropy(output, labels)

# Training setup
train_ids = sorted([d for d in os.listdir(voxceleb2_dir) if os.path.isdir(os.path.join(voxceleb2_dir, d))])[:50]
test_ids = sorted([d for d in os.listdir(voxceleb2_dir) if os.path.isdir(os.path.join(voxceleb2_dir, d))])[50:100]

#id_to_idx = {id: idx for idx, id in enumerate(train_ids)}
id_to_idx = {str(idx): id for idx, id in enumerate(train_ids)}

optimizer = torch.optim.Adam(list(sepformer.parameters()) + list(finetuned_wavlm.parameters()), lr=1e-4)
arcface_loss = ArcFaceLoss(in_features=768, out_features=len(train_ids)).to(device)
cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Fine-tuning loop
def train_pipeline():
    sepformer.train()
    finetuned_wavlm.train()

    # Create speaker ID mapping
    all_speaker_ids = set()
    for _, _, _, id1, id2 in train_loader:
        all_speaker_ids.update(id1)
        all_speaker_ids.update(id2)
    id_to_idx = {id: idx for idx, id in enumerate(sorted(all_speaker_ids))}
    num_speakers = len(id_to_idx)

    # Reinitialize loss with correct number of speakers
    arcface_loss = ArcFaceLoss(in_features=768, out_features=num_speakers).to(device)

    for epoch in range(5):
        total_loss = 0
        for mix, src1, src2, id1, id2 in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Ensure all tensors are on the correct device
            mix = mix.to(device)
            src1 = src1.to(device)
            src2 = src2.to(device)

            # Convert speaker IDs to indices
            labels = []
            for i in range(len(id1)):  # For each sample in batch
                labels.append(id_to_idx[id1[i]])
                labels.append(id_to_idx[id2[i]])
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            # Reshape input properly for SepFormer
            # Current shape: [batch_size, 1, 48000] -> need [batch_size, 48000]
            if len(mix.shape) == 3:
                mix = mix.squeeze(1)  # Remove channel dim -> [batch, samples]

            # Verify final shape
            if len(mix.shape) != 2:
                raise ValueError(f"Final mix shape should be [batch, samples], got {mix.shape}")

            # Process through SepFormer - ensure input is on same device as model
            try:
                # First try with [batch, samples] shape
                est_sources = sepformer(mix)
            except RuntimeError as e:
                print(f"SepFormer failed with shape {mix.shape}: {e}")
                print("Trying alternative input shapes...")
                try:
                    # Try adding channel dimension [batch, 1, samples]
                    est_sources = sepformer(mix.unsqueeze(1))
                except RuntimeError as e2:
                    print(f"SepFormer failed with all shape options: {e2}")
                    continue

            # Get separated sources - shape should be [batch, samples, num_sources]
            if len(est_sources.shape) != 3:
                print(f"Unexpected SepFormer output shape: {est_sources.shape}")
                continue

            est1 = est_sources[..., 0]  # First source [batch, samples]
            est2 = est_sources[..., 1]  # Second source [batch, samples]

            # Extract embeddings
            try:
                # Convert to list of numpy arrays for feature extractor
                est1_list = [e.detach().cpu().numpy() for e in est1]
                est2_list = [e.detach().cpu().numpy() for e in est2]

                inputs1 = feature_extractor(est1_list, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs2 = feature_extractor(est2_list, sampling_rate=16000, return_tensors="pt", padding=True)

                # Move inputs to correct device
                emb1 = finetuned_wavlm(inputs1["input_values"].to(device)).last_hidden_state.mean(dim=1)
                emb2 = finetuned_wavlm(inputs2["input_values"].to(device)).last_hidden_state.mean(dim=1)
                embeddings = torch.cat([emb1, emb2], dim=0)
            except Exception as e:
                print(f"Embedding extraction failed: {e}")
                continue

            # Calculate losses
            try:
                batch_size = mix.size(0)
                sdr_values = []
                for i in range(batch_size):
                    sdr1 = calc_sdr(src1[i].cpu().numpy(), est1[i].cpu().numpy())
                    sdr2 = calc_sdr(src2[i].cpu().numpy(), est2[i].cpu().numpy())
                    sdr_values.append(sdr1 + sdr2)

                sep_loss = -torch.mean(torch.tensor(sdr_values, device=device))
                id_loss = arcface_loss(embeddings, labels)
                loss = sep_loss + 0.1 * id_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            except Exception as e:
                print(f"Loss calculation failed: {e}")
                continue

        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")

# Metric functions
def calc_sdr(ref, est):
    s_target = ref
    e_noise = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_noise**2) + 1e-8))

def calc_sir(ref, est, interferer):
    s_target = ref
    e_interf = interferer
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_interf**2) + 1e-8))

def calc_sar(ref, est):
    s_target = ref
    e_artifacts = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_artifacts**2) + 1e-8))

# Evaluation
def evaluate_pipeline():
    sepformer.eval()
    pretrained_wavlm.eval()
    finetuned_wavlm.eval()
    results = {"SIR": [], "SAR": [], "SDR": [], "PESQ": []}
    correct_pre, correct_fin, total = 0, 0, 0

    ref_emb_pre, ref_emb_fin = {}, {}
    for speaker_id in test_ids:
        speaker_path = os.path.join(voxceleb2_dir, speaker_id, os.listdir(os.path.join(voxceleb2_dir, speaker_id))[0])
        file = os.path.join(speaker_path, os.listdir(speaker_path)[0])
        waveform, sr = torchaudio.load(file)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze(0).numpy()
        ref_emb_pre[speaker_id] = ext_embedding(waveform, pretrained_wavlm)
        ref_emb_fin[speaker_id] = ext_embedding(waveform, finetuned_wavlm)

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            mix, src1, src2, id1, id2 = test_dataset[i]
            mix = mix.unsqueeze(0).to(device)
            src1, src2 = src1.numpy(), src2.numpy()
            mix = mix.squeeze()
            #est_sources = sepformer(mix.unsqueeze(1)).squeeze(0).cpu().numpy()
            est_sources = sepformer(mix.unsqueeze(1))
            est1, est2 = est_sources[:, 0], est_sources[:, 1]

            min_len = min(est1.shape[0], src1.shape[0])
            est1, est2 = est1[:min_len], est2[:min_len]
            src1, src2 = src1[:min_len], src2[:min_len]

            results["SIR"].extend([calc_sir(src1, est1, src2), calc_sir(src2, est2, src1)])
            results["SAR"].extend([calc_sar(src1, est1), calc_sar(src2, est2)])
            results["SDR"].extend([calc_sdr(src1, est1), calc_sdr(src2, est2)])
            results["PESQ"].extend([pesq(16000, src1, est1, "wb"), pesq(16000, src2, est2, "wb")])

            emb1_pre = ext_embedding(est1, pretrained_wavlm)
            emb2_pre = ext_embedding(est2, pretrained_wavlm)
            emb1_fin = ext_embedding(est1, finetuned_wavlm)
            emb2_fin = ext_embedding(est2, finetuned_wavlm)

            pre_scores, fin_scores = {}, {}
            for sid in test_ids:
                ref_pre = torch.from_numpy(ref_emb_pre[sid]).to(device)
                ref_fin = torch.from_numpy(ref_emb_fin[sid]).to(device)
                pre_scores[sid] = [cosine_similarity(torch.from_numpy(emb1_pre).to(device), ref_pre).item(),
                                   cosine_similarity(torch.from_numpy(emb2_pre).to(device), ref_pre).item()]
                fin_scores[sid] = [cosine_similarity(torch.from_numpy(emb1_fin).to(device), ref_fin).item(),
                                   cosine_similarity(torch.from_numpy(emb2_fin).to(device), ref_fin).item()]

            pred_id1_pre = max(pre_scores, key=lambda k: pre_scores[k][0])
            pred_id2_pre = max(pre_scores, key=lambda k: pre_scores[k][1])
            pred_id1_fin = max(fin_scores, key=lambda k: fin_scores[k][0])
            pred_id2_fin = max(fin_scores, key=lambda k: fin_scores[k][1])

            pre_correct = (pred_id1_pre == id1 and pred_id2_pre == id2) or (pred_id1_pre == id2 and pred_id2_pre == id1)
            fin_correct = (pred_id1_fin == id1 and pred_id2_fin == id2) or (pred_id1_fin == id2 and pred_id2_fin == id1)
            correct_pre += pre_correct
            correct_fin += fin_correct
            total += 1

    for metric in results:
        avg = np.mean(results[metric])
        print(f"Average {metric}: {avg:.2f}")
    rank1_pre = correct_pre / total * 100
    rank1_fin = correct_fin / total * 100
    print(f"Pre-trained WavLM Rank-1 Accuracy: {rank1_pre:.2f}%")
    print(f"Fine-tuned WavLM Rank-1 Accuracy: {rank1_fin:.2f}%")

# Extract embedding
def ext_embedding(waveform, model):
    inputs = feature_extractor(waveform.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

train_pipeline()
print("\nResult on Test Set")
evaluate_pipeline()