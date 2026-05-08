# =====================================================
# PPG-DaLiA FAST AGENT ANALYSIS (ALL SUBJECTS, 100 WINDOWS)
# =====================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import glob
import requests
from tqdm import tqdm

# ====================== CONFIG ======================
DATA_DIR = r"/Users/anandmohan/Downloads/PPG_FieldStudy"
SUBJECTS = [f"S{i}" for i in range(1, 16)]

WINDOW_SEC = 10
FS_PPG = 64
MAX_WINDOWS = 100

OUTPUT_FILE = "/Users/anandmohan/Downloads/PPG_FieldStudyPPG_Subject_Reports.csv"

# ===================================================


# ====================== LLM ======================
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# ====================== LOAD ======================
def load_subject(subject_id):
    files = glob.glob(os.path.join(DATA_DIR, subject_id, "*.pkl"))
    if not files:
        files = glob.glob(os.path.join(DATA_DIR, subject_id, f"{subject_id}.pkl"))
    if not files:
        print(f"❌ Missing: {subject_id}")
        return None

    with open(files[0], 'rb') as f:
        return pickle.load(f, encoding='latin1')


# ====================== FILTER ======================
def bandpass_filter(sig, fs, low=0.5, high=5.0):
    nyq = fs / 2
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)


# ====================== HR ======================
def compute_hr(ppg, fs=64):
    ppg_filt = bandpass_filter(ppg, fs)
    peaks, _ = find_peaks(ppg_filt, distance=fs*0.4)

    if len(peaks) < 2:
        return np.nan

    rr = np.diff(peaks) / fs
    return 60 / np.mean(rr)


# ====================== MAIN ======================
results = []

for subject in tqdm(SUBJECTS, desc="Processing Subjects"):

    data = load_subject(subject)
    if data is None:
        continue

    bvp = data['signal']['wrist']['BVP'].flatten()
    acc = data['signal']['wrist']['ACC']

    window = WINDOW_SEC * FS_PPG

    hrs = []
    motions = []

    print(f"\nProcessing {subject}...")

    count = 0

    for start in range(0, len(bvp) - window + 1, window):

        if count >= MAX_WINDOWS:
            break

        end = start + window

        ppg_win = bvp[start:end]

        acc_win = acc[int(start*32/64):int(end*32/64)]
        acc_mag = np.sqrt(np.sum(acc_win**2, axis=1))

        hr = compute_hr(ppg_win)
        motion = np.sqrt(np.mean(acc_mag**2))

        hrs.append(hr)
        motions.append(motion)

        count += 1

    hrs = np.array(hrs)
    motions = np.array(motions)

    # ================= SUMMARY =================
    mean_hr = np.nanmean(hrs)
    std_hr = np.nanstd(hrs)
    mean_motion = np.nanmean(motions)

    print(f"{subject} → HR: {mean_hr:.2f}, Motion: {mean_motion:.2f}")

    # ================= LLM REPORT =================
    prompt = f"""
You are a biomedical assistant.

Subject: {subject}

Mean Heart Rate: {mean_hr:.2f} bpm
HR Variability (STD): {std_hr:.2f}
Motion Level: {mean_motion:.2f}

Give a short health report:
- cardiovascular condition
- activity level
- signal reliability
"""

    report = ask_llm(prompt)

    print(f"\n{subject} Report:\n{report}\n")

    results.append({
        "subject": subject,
        "mean_hr": mean_hr,
        "std_hr": std_hr,
        "motion": mean_motion,
        "report": report
    })


# ================= SAVE =================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ DONE! Reports saved to: {OUTPUT_FILE}")