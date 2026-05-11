# =====================================================
# FINAL CLEAN CHAT-BASED PPG MULTI-SUBJECT SYSTEM
# =====================================================

import os
import pickle
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import glob
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ================= CONFIG =================
DATA_DIR = r"/Users/anandmohan/Downloads/PPG_FieldStudy"
SUBJECTS = [f"S{i}" for i in range(1, 16)]

FS = 64
WINDOW_SEC = 10
MAX_WINDOWS = 100

llm = Ollama(model="phi")

# ==========================================


# ================= SIGNAL FUNCTIONS =================

def load_subject(subject):
    files = glob.glob(os.path.join(DATA_DIR, subject, "*.pkl"))
    if not files:
        return None
    with open(files[0], 'rb') as f:
        return pickle.load(f, encoding='latin1')


def bandpass(sig):
    b, a = butter(3, [0.5/(FS/2), 5/(FS/2)], btype='band')
    return filtfilt(b, a, sig)


def compute_hr(ppg):
    ppg = bandpass(ppg)
    peaks, _ = find_peaks(ppg, distance=FS * 0.4)
    if len(peaks) < 2:
        return np.nan
    return 60 / np.mean(np.diff(peaks) / FS)


# ================= PROMPTS =================

report_prompt = PromptTemplate.from_template("""
You are a biomedical signal analysis assistant.

STRICT RULES:
- Do NOT say "sorry"
- Do NOT apologize
- Do NOT mention limitations
- Do NOT invent anything
- Only use given numerical data
- Be direct, clinical, and concise

DATA:
Mean HR: {mean_hr}
STD HR: {std_hr}
Motion Level: {motion}

STYLE: {style}

OUTPUT FORMAT:
Condition:
Activity:
Reliability:

Suggestions:
- ...
- ...
""")

chat_prompt_template = """
You are a biomedical assistant.

STRICT RULES:
- Do NOT say "sorry"
- Do NOT apologize
- Do NOT invent information
- Answer only using given data
- Be direct and precise

DATA:
Mean HR: {mean_hr}
STD HR: {std_hr}
Motion: {motion}

User Question:
{question}
"""


# ================= PROCESS SUBJECT =================

def process_subject(subject):
    data = load_subject(subject)
    if data is None:
        print("❌ Missing:", subject)
        return None

    bvp = data['signal']['wrist']['BVP'].flatten()
    acc = data['signal']['wrist']['ACC']

    window = WINDOW_SEC * FS

    hrs, motions = [], []

    for i, start in enumerate(range(0, len(bvp) - window, window)):
        if i >= MAX_WINDOWS:
            break

        end = start + window
        ppg = bvp[start:end]

        acc_win = acc[int(start * 32 / 64):int(end * 32 / 64)]
        acc_mag = np.sqrt(np.sum(acc_win**2, axis=1))

        hrs.append(compute_hr(ppg))
        motions.append(np.sqrt(np.mean(acc_mag**2)))

    return np.nanmean(hrs), np.nanstd(hrs), np.nanmean(motions)


# ================= MAIN SYSTEM =================

def run_system():

    print("\n" + "="*60)
    print("🧠 PPG MULTI-SUBJECT ANALYSIS SYSTEM")
    print("="*60)

    for subject in SUBJECTS:

        result = process_subject(subject)
        if result is None:
            continue

        mean_hr, std_hr, motion = result

        print("\n" + "-"*60)
        print(f"📊 SUBJECT: {subject}")
        print(f"HR: {mean_hr:.2f} | STD: {std_hr:.2f} | Motion: {motion:.2f}")
        print("-"*60)

        # -------- REPORT TYPE --------
        style = input("Select report type (short / long): ").lower()
        if style not in ["short", "long"]:
            style = "short"

        # -------- GENERATE REPORT --------
        report = llm.invoke(
            report_prompt.format(
                mean_hr=round(mean_hr, 2),
                std_hr=round(std_hr, 2),
                motion=round(motion, 2),
                style=style
            )
        )

        print("\n🧾 REPORT:\n")
        print(report)

        # -------- DOUBT LOOP --------
        while True:
            doubt = input("\n❓ Any doubts? (yes/no): ").lower()

            if doubt == "no":
                print("➡️ Moving to next subject...\n")
                break

            elif doubt == "yes":
                print("\n💬 Chat mode activated (type 'next' to continue)\n")

                while True:
                    question = input("Ask: ")

                    if question.lower() == "next":
                        print("➡️ Exiting chat...\n")
                        break

                    chat_prompt = chat_prompt_template.format(
                        mean_hr=mean_hr,
                        std_hr=std_hr,
                        motion=motion,
                        question=question
                    )

                    answer = llm.invoke(chat_prompt)

                    print("\n🧠 Answer:\n", answer)

            else:
                print("⚠️ Please type 'yes' or 'no'")

    print("\n✅ All subjects completed.")


# ================= RUN =================

if __name__ == "__main__":
    run_system()