"""
PPG Multi-Subject Agentic Analysis System
Improvements: unified prompt builder, cleaner loop, type hints, graceful errors
"""

import os, glob, pickle
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ── CONFIG ──────────────────────────────────────────
DATA_DIR  = r"/Users/anandmohan/Downloads/PPG_FieldStudy"
SUBJECTS  = [f"S{i}" for i in range(1, 16)]
FS, WIN_SEC, MAX_WIN = 64, 10, 100
llm = Ollama(model="phi")

# ── SIGNAL ──────────────────────────────────────────
def load_subject(subj: str) -> dict | None:
    files = glob.glob(os.path.join(DATA_DIR, subj, "*.pkl"))
    if not files:
        return None
    with open(files[0], "rb") as f:
        return pickle.load(f, encoding="latin1")

def bandpass(sig: np.ndarray) -> np.ndarray:
    b, a = butter(3, [0.5 / (FS / 2), 5 / (FS / 2)], btype="band")
    return filtfilt(b, a, sig)

def compute_hr(ppg: np.ndarray) -> float:
    peaks, _ = find_peaks(bandpass(ppg), distance=FS * 0.4)
    return 60 / np.mean(np.diff(peaks) / FS) if len(peaks) >= 2 else np.nan

def process_subject(subj: str) -> tuple | None:
    data = load_subject(subj)
    if data is None:
        print(f"❌ Missing: {subj}"); return None

    bvp = data["signal"]["wrist"]["BVP"].flatten()
    acc = data["signal"]["wrist"]["ACC"]
    win = WIN_SEC * FS
    hrs, motions = [], []

    for i, s in enumerate(range(0, len(bvp) - win, win)):
        if i >= MAX_WIN: break
        acc_mag = np.sqrt(np.sum(acc[s // 2:(s + win) // 2] ** 2, axis=1))
        hrs.append(compute_hr(bvp[s:s + win]))
        motions.append(np.sqrt(np.mean(acc_mag ** 2)))

    return np.nanmean(hrs), np.nanstd(hrs), np.nanmean(motions)

# ── PROMPTS ──────────────────────────────────────────
RULES = (
    "RULES: Never apologise. Never invent data. "
    "Use only the numbers provided. Be direct and clinical.\n\n"
)

REPORT_TMPL = PromptTemplate.from_template(
    RULES +
    "Mean HR: {mean_hr} | STD HR: {std_hr} | Motion: {motion} | Style: {style}\n\n"
    "OUTPUT:\nCondition:\nActivity:\nReliability:\n\nSuggestions:\n- ...\n- ..."
)

def chat_prompt(mean_hr, std_hr, motion, question):
    return (
        f"{RULES}"
        f"Mean HR: {mean_hr:.2f} | STD HR: {std_hr:.2f} | Motion: {motion:.2f}\n\n"
        f"Question: {question}"
    )

# ── AGENT LOOP ───────────────────────────────────────
def run_agent():
    print("\n" + "=" * 60)
    print("🧠  PPG MULTI-SUBJECT AGENTIC ANALYSIS")
    print("=" * 60)

    for subj in SUBJECTS:
        result = process_subject(subj)
        if result is None: continue

        mean_hr, std_hr, motion = result
        print(f"\n{'─'*60}\n📊 {subj}  |  HR: {mean_hr:.2f}  STD: {std_hr:.2f}  Motion: {motion:.2f}")

        style = input("Report style (short/long): ").strip().lower() or "short"
        report = llm.invoke(REPORT_TMPL.format(
            mean_hr=round(mean_hr, 2), std_hr=round(std_hr, 2),
            motion=round(motion, 2), style=style
        ))
        print(f"\n🧾 REPORT:\n{report}")

        while (doubt := input("\n❓ Any doubts? (yes/no): ").strip().lower()) != "no":
            if doubt != "yes":
                print("⚠️  Type 'yes' or 'no'"); continue
            print("💬 Chat mode — type 'next' to exit\n")
            while (q := input("Ask: ").strip().lower()) != "next":
                print("\n🧠", llm.invoke(chat_prompt(mean_hr, std_hr, motion, q)))

        print("➡️  Next subject…")

    print("\n✅ All subjects completed.")

if __name__ == "__main__":
    run_agent()
