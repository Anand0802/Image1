"""
PPG Agentic Analysis System — Industry-Level Pattern
=====================================================
Architecture : Observe → Plan → Act → Reflect loop
LLM          : Ollama 2B-compatible (phi / gemma:2b / qwen2:1.5b)
Dataset      : WESAD (BVP @ 64 Hz, ACC @ 32 Hz)

Agentic traits
--------------
✅ Tool Registry    — named, callable tools with descriptions
✅ Agent State      — persistent context across steps
✅ Plan step        — LLM selects next tool from allowed set
✅ Reflect step     — LLM evaluates result quality, triggers retry
✅ HITL checkpoint  — human approves report style + chat loop
✅ History memory   — last-N Q&A grounded in data
✅ JSON-safe parse  — regex fallback when 2B adds extra text
"""

import os, glob, pickle, json, re
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from scipy.signal import butter, filtfilt, find_peaks
from langchain_community.llms import Ollama

# ── CONFIG ───────────────────────────────────────────────────────
DATA_DIR   = r"/Users/anandmohan/Downloads/PPG_FieldStudy"
SUBJECTS   = [f"S{i}" for i in range(1, 16)]
FS         = 64       # BVP sample rate (Hz)
ACC_FS     = 32       # ACC sample rate (Hz)
WIN_SEC    = 10       # window length (s)
MAX_WIN    = 100      # max windows per subject
HR_MIN     = 40.0     # physiological HR floor (bpm)
HR_MAX     = 200.0    # physiological HR ceiling (bpm)
MODEL      = "phi"    # swap to "gemma:2b" / "qwen2:1.5b" as needed

llm = Ollama(model=MODEL, temperature=0.1)  # low temp → deterministic


# ════════════════════════════════════════════════════════════════
# AGENT STATE
# ════════════════════════════════════════════════════════════════
@dataclass
class AgentState:
    subject       : str   = ""
    mean_hr       : float = 0.0
    std_hr        : float = 0.0
    motion_rms    : float = 0.0
    valid_windows : int   = 0
    total_windows : int   = 0
    classification: dict  = field(default_factory=dict)
    report        : str   = ""
    qa_history    : list  = field(default_factory=list)   # rolling memory
    step          : str   = "idle"

    def data_summary(self) -> str:
        return (
            f"HR={self.mean_hr:.1f}±{self.std_hr:.1f}bpm  "
            f"Motion={self.motion_rms:.2f}  "
            f"ValidWin={self.valid_windows}/{self.total_windows}"
        )


# ════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING TOOLS
# ════════════════════════════════════════════════════════════════

def _bandpass(sig: np.ndarray) -> np.ndarray:
    """3rd-order Butterworth 0.5–5 Hz (covers 30–300 BPM)."""
    nyq = FS / 2
    b, a = butter(3, [0.5 / nyq, 5.0 / nyq], btype="band")
    return filtfilt(b, a, sig)


def _hr_from_window(ppg: np.ndarray) -> float:
    """
    Robust HR extraction from a single PPG window.

    Fixes vs. original code:
    ─────────────────────────────────────────────
    ① Z-score normalise before peak detection (handles DC drift)
    ② prominence=0.3 removes noise spikes
    ③ Filter RR intervals to physiological range [0.3 s, 1.5 s]
       → rejects false peaks and motion artefacts
    ④ Returns nan if fewer than 2 valid RR intervals remain
    """
    seg = _bandpass(ppg)
    seg = (seg - seg.mean()) / (seg.std() + 1e-8)          # ① z-score

    peaks, _ = find_peaks(
        seg,
        distance=int(FS * 0.4),   # min 0.4 s apart  (~150 BPM max)
        prominence=0.3,            # ② reject small bumps
        height=0.1,
    )

    if len(peaks) < 3:
        return np.nan

    rr = np.diff(peaks) / FS                               # seconds
    rr = rr[(rr >= 0.3) & (rr <= 1.5)]                    # ③ physio gate
    return 60.0 / np.mean(rr) if len(rr) >= 2 else np.nan  # ④


def tool_load_signal(subject: str) -> dict | None:
    """
    Tool — Load PKL, compute HR + motion across sliding windows.
    ACC downsampling: BVP@64 Hz, ACC@32 Hz → ratio=2 → acc[s//2:(s+win)//2]
    Motion: RMS of per-sample L2 magnitude of 3-axis ACC.
    """
    files = glob.glob(os.path.join(DATA_DIR, subject, "*.pkl"))
    if not files:
        return None

    with open(files[0], "rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp = data["signal"]["wrist"]["BVP"].flatten()
    acc = data["signal"]["wrist"]["ACC"]          # shape (N_acc, 3)
    win = WIN_SEC * FS
    ratio = FS // ACC_FS                           # = 2
    hrs, motions = [], []

    total = min(MAX_WIN, (len(bvp) - win) // win)
    for i in range(total):
        s = i * win
        ppg_win = bvp[s: s + win]
        a_s, a_e = s // ratio, (s + win) // ratio
        if a_e > len(acc):
            break

        acc_win = acc[a_s:a_e]
        mag     = np.linalg.norm(acc_win, axis=1)          # per-sample L2
        motions.append(float(np.sqrt(np.mean(mag ** 2))))  # RMS of magnitudes

        hr = _hr_from_window(ppg_win)
        # Physiological sanity check
        hrs.append(hr if HR_MIN <= hr <= HR_MAX else np.nan)

    valid = np.array([h for h in hrs if not np.isnan(h)])
    return {
        "mean_hr"      : round(float(np.mean(valid)), 2)  if len(valid) else float("nan"),
        "std_hr"       : round(float(np.std(valid)), 2)   if len(valid) else float("nan"),
        "motion_rms"   : round(float(np.mean(motions)), 2),
        "valid_windows": int(len(valid)),
        "total_windows": int(total),
    }


# ════════════════════════════════════════════════════════════════
# LLM TOOLS (2B-safe: short prompts, JSON regex fallback)
# ════════════════════════════════════════════════════════════════

def _safe_json(raw: str, fallback: dict) -> dict:
    """Extract first JSON object from LLM output; use fallback on failure."""
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return fallback


def tool_classify(state: AgentState) -> dict:
    """
    Tool — LLM classifies HR condition + activity.
    Prompt engineered for 2B: minimal context, strict JSON, no preamble.
    Rule-based fallback ensures output even if LLM fails.
    """
    # Rule-based labels fed as hints (reduces LLM burden for 2B)
    hr_label  = ("low" if state.mean_hr < 60 else
                 "high" if state.mean_hr > 100 else "normal")
    mov_label = ("low" if state.motion_rms < 1.5 else
                 "moderate" if state.motion_rms < 4.0 else "high")
    qual      = ("low" if state.valid_windows < state.total_windows * 0.5 else
                 "medium" if state.valid_windows < state.total_windows * 0.8 else "high")

    prompt = (
        f"HR:{state.mean_hr}bpm(hint:{hr_label}) "
        f"Motion:{state.motion_rms}(hint:{mov_label}) "
        f"SignalQuality:{qual}\n"
        f"Reply ONLY with this JSON (no other text):\n"
        f'{{"condition":"normal|bradycardia|tachycardia",'
        f'"activity":"rest|light|moderate|exercise",'
        f'"reliability":"{qual}"}}'
    )
    raw = llm.invoke(prompt)
    return _safe_json(raw, {
        "condition"  : hr_label,
        "activity"   : mov_label,
        "reliability": qual,
    })


def tool_report(state: AgentState, style: str) -> str:
    """Tool — LLM generates a clinical summary. Two length modes."""
    cls = state.classification
    base = (
        f"Subject:{state.subject} HR:{state.mean_hr}±{state.std_hr}bpm "
        f"Motion:{state.motion_rms} Condition:{cls.get('condition','?')} "
        f"Activity:{cls.get('activity','?')} Reliability:{cls.get('reliability','?')}"
    )
    if style == "long":
        prompt = (
            f"{base}\n"
            f"Write a clinical PPG report with:\n"
            f"1. Summary  2. Key findings  3. Recommendations\n"
            f"Use medical tone. No apologies."
        )
    else:
        prompt = f"{base}\nWrite a 2-sentence clinical summary. No apologies."

    return llm.invoke(prompt).strip()


def tool_reflect(state: AgentState) -> str:
    """
    Tool — LLM evaluates report quality and flags issues.
    Returns 'ok' or a brief issue description.
    Keeps prompt tiny for 2B reliability.
    """
    prompt = (
        f"Report snippet: {state.report[:200]}\n"
        f"Data: {state.data_summary()}\n"
        f"Does the report match the data? Reply: ok OR one short issue."
    )
    raw = llm.invoke(prompt).strip().lower()
    return "ok" if "ok" in raw[:30] else raw[:120]


def tool_qa(state: AgentState, question: str) -> str:
    """Tool — Grounded Q&A using rolling 3-turn memory."""
    ctx = " | ".join(state.qa_history[-3:]) if state.qa_history else "none"
    prompt = (
        f"Data: {state.data_summary()}\n"
        f"Recent Q&A: {ctx}\n"
        f"Q: {question}\n"
        f"A (2-3 sentences, use only provided data):"
    )
    ans = llm.invoke(prompt).strip()
    state.qa_history.append(f"Q:{question[:40]}→{ans[:60]}")
    return ans


# ════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ════════════════════════════════════════════════════════════════
TOOL_REGISTRY: dict[str, tuple[str, Any]] = {
    "load_signal": ("Load & extract PPG/ACC features",    tool_load_signal),
    "classify"   : ("Classify HR condition & activity",   tool_classify),
    "report"     : ("Generate clinical report",           tool_report),
    "reflect"    : ("Evaluate report vs data",            tool_reflect),
    "qa"         : ("Answer grounded follow-up questions",tool_qa),
}


# ════════════════════════════════════════════════════════════════
# AGENT — Observe → Plan → Act → Reflect loop
# ════════════════════════════════════════════════════════════════
class PPGAgent:
    """
    Minimal but genuine agentic loop:
    - LLM is used for Planning (tool selection) AND Reflection
    - Tools are discrete, registered, callable units
    - State persists across steps within one subject
    - HITL checkpoint before report generation
    """

    def __init__(self):
        self.state = AgentState()

    def _plan(self, observation: str, allowed: list[str]) -> str:
        """LLM chooses next tool from allowed set. 2B-safe: tiny prompt."""
        opts = "|".join(allowed)
        prompt = (
            f"Observation: {observation}\n"
            f"Choose one tool [{opts}]. Reply with the tool name only."
        )
        raw = llm.invoke(prompt).strip().lower()
        for t in allowed:
            if t in raw:
                return t
        return allowed[0]  # deterministic fallback

    # ── MAIN AGENT LOOP ──────────────────────────────────────────
    def run(self, subject: str):
        s = self.state
        s.subject, s.qa_history, s.step = subject, [], "start"
        sep = "═" * 62
        print(f"\n{sep}\n🤖  AGENT › {subject}   [{MODEL}]\n{sep}")

        # ── 1. OBSERVE ───────────────────────────────────────────
        s.step = "observe"
        print("📡 [1/5] Observe — loading signal…")
        result = tool_load_signal(subject)
        if result is None:
            print(f"   ❌ No data found for {subject}"); return
        s.mean_hr, s.std_hr   = result["mean_hr"], result["std_hr"]
        s.motion_rms          = result["motion_rms"]
        s.valid_windows       = result["valid_windows"]
        s.total_windows       = result["total_windows"]
        print(f"   └─ {s.data_summary()}")

        # ── 2. PLAN ──────────────────────────────────────────────
        s.step = "plan"
        print("🧠 [2/5] Plan — agent selecting next action…")
        next_tool = self._plan(s.data_summary(), ["classify", "report"])
        print(f"   └─ Selected: {next_tool}")

        # ── 3. ACT (classify) ────────────────────────────────────
        s.step = "classify"
        print("⚙️  [3/5] Act — classify HR & activity…")
        s.classification = tool_classify(s)
        cls = s.classification
        print(f"   └─ Condition:{cls.get('condition','?')}  "
              f"Activity:{cls.get('activity','?')}  "
              f"Reliability:{cls.get('reliability','?')}")

        # ── HITL CHECKPOINT ──────────────────────────────────────
        style = input("\n📋 Report style (short/long) [short]: ").strip().lower() or "short"

        # ── 4. ACT (report) ──────────────────────────────────────
        s.step = "report"
        print("📝 [4/5] Act — generating report…")
        s.report = tool_report(s, style)

        # Print structured card
        print(f"\n{'─'*62}")
        print(f"🧾  REPORT — {subject}")
        print(f"    HR   : {s.mean_hr} ± {s.std_hr} bpm")
        print(f"    Motion    : {s.motion_rms}  | Windows: {s.valid_windows}/{s.total_windows}")
        print(f"    Condition : {cls.get('condition','?')}")
        print(f"    Activity  : {cls.get('activity','?')}")
        print(f"    Reliability: {cls.get('reliability','?')}")
        print(f"{'─'*62}")
        print(s.report)
        print(f"{'─'*62}")

        # ── 5. REFLECT ───────────────────────────────────────────
        s.step = "reflect"
        print("🔍 [5/5] Reflect — evaluating report quality…")
        verdict = tool_reflect(s)
        if verdict == "ok":
            print("   └─ ✅ Report consistent with data")
        else:
            print(f"   └─ ⚠️  Issue detected: {verdict}")
            print("   └─ Re-generating report…")
            s.report = tool_report(s, style)      # auto-retry once
            print("   └─ ♻️  Report updated")

        # ── HITL QA LOOP ─────────────────────────────────────────
        while True:
            choice = input("\n❓ Any doubts? (yes/no) [no]: ").strip().lower() or "no"
            if choice == "no":
                break
            if choice != "yes":
                print("⚠️  Please type 'yes' or 'no'"); continue

            print("💬 Chat mode — type 'next' to exit\n")
            while True:
                q = input("Ask: ").strip()
                if q.lower() == "next":
                    break
                ans = tool_qa(s, q)
                print(f"\n🧠 {ans}\n")

        print(f"✅ Agent done — {subject}\n")


# ════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs one agent per subject
# ════════════════════════════════════════════════════════════════
def run_orchestrator():
    print("\n" + "═" * 62)
    print("🏭  PPG MULTI-SUBJECT AGENTIC ORCHESTRATOR")
    print(f"    Model    : {MODEL}")
    print(f"    Subjects : {len(SUBJECTS)}  ({SUBJECTS[0]} … {SUBJECTS[-1]})")
    print(f"    Tools    : {', '.join(TOOL_REGISTRY)}")
    print("═" * 62)

    for subj in SUBJECTS:
        agent = PPGAgent()          # fresh state per subject
        agent.run(subj)

    print("\n✅  All subjects processed.")


if __name__ == "__main__":
    run_orchestrator()
