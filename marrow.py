#!/usr/bin/env python3
"""Marrow v0.1 - behavioral biometrics prototype (1f916 listing-65).

Decides whether the person at the computer is the enrolled user from the
ordinary input events the OS already sees: pointer movement + clicks + text
as it arrives (typed, on-screen keyboard, or inserted by software such as
dictation). No audio, no camera, no input-device assumption.

Offline, single command:  python3 marrow.py demo
"""
import argparse, json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")

# ---------- OS-neutral event schema (JSONL, one event per line) ----------
# {"t": <ms since session start>, "type": "move",  "x": <0..1>, "y": <0..1>}
# {"t": ..., "type": "click", "x":.., "y":.., "button": 0}
# {"t": ..., "type": "text",  "arrival": "typed"|"onscreen"|"inserted", "value": str}
# Pointer positions are fractions of the screen; t is ms. Capture-tool agnostic.

FEATURE_NAMES = [
    "move_speed_mean", "move_speed_std",
    "click_iv_mean", "click_iv_std",
    "text_iv_mean", "text_iv_std",
    "corrections", "tortuosity", "pause_ratio",
    "n_typed", "n_onscreen", "n_inserted",
    "n_moves", "n_clicks", "n_text",
]


def _speeds(moves):
    out = []
    for a, b in zip(moves, moves[1:]):
        dt = b["t"] - a["t"]
        if dt > 0:
            out.append(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt)
    return out


def features(session):
    moves = [e for e in session if e["type"] == "move"]
    clicks = [e for e in session if e["type"] == "click"]
    texts = [e for e in session if e["type"] == "text"]
    sp = _speeds(moves)
    click_iv = [clicks[i + 1]["t"] - clicks[i]["t"] for i in range(len(clicks) - 1)]
    text_iv = [texts[i + 1]["t"] - texts[i]["t"] for i in range(len(texts) - 1)]
    corrections = sum(1 for e in texts if e.get("value") in ("<BS>", "\b") or "backspace" in str(e.get("value", "")).lower())
    if len(moves) >= 2:
        path = sum(math.hypot(moves[i + 1]["x"] - moves[i]["x"], moves[i + 1]["y"] - moves[i]["y"]) for i in range(len(moves) - 1))
        straight = math.hypot(moves[-1]["x"] - moves[0]["x"], moves[-1]["y"] - moves[0]["y"])
        tort = path / (straight + 1e-9)
    else:
        tort = 1.0
    gaps = [moves[i + 1]["t"] - moves[i]["t"] for i in range(len(moves) - 1)]
    total = (session[-1]["t"] - session[0]["t"]) if session else 1
    pause_ratio = sum(g for g in gaps if g > 200) / (total + 1e-9)
    arr = {"typed": 0, "onscreen": 0, "inserted": 0}
    for e in texts:
        arr[e.get("arrival", "typed")] = arr.get(e.get("arrival", "typed"), 0) + 1
    fv = np.array([
        (np.mean(sp) if sp else 0.0), (np.std(sp) if sp else 0.0),
        (np.mean(click_iv) if click_iv else 0.0), (np.std(click_iv) if click_iv else 0.0),
        (np.mean(text_iv) if text_iv else 0.0), (np.std(text_iv) if text_iv else 0.0),
        float(corrections), float(tort), float(pause_ratio),
        float(arr["typed"]), float(arr["onscreen"]), float(arr["inserted"]),
        float(len(moves)), float(len(clicks)), float(len(texts)),
    ], dtype=float)
    return fv


# ---------- Fake data generation ----------
def _session_standard(rng, user_type, n_moves=60, n_clicks=8, n_text=12):
    # base timing
    if user_type == "owner":
        speed = 0.0009; speed_jit = 0.0003
        click_gap = 220; click_jit = 60
        key_gap = 110; key_jit = 40
        pause_p = 0.05
        corr_p = 0.10
    elif user_type == "tired_owner":
        speed = 0.0005; speed_jit = 0.0002
        click_gap = 320; click_jit = 90
        key_gap = 170; key_jit = 70
        pause_p = 0.18
        corr_p = 0.25
    elif user_type == "stranger":
        speed = 0.0016; speed_jit = 0.0005
        click_gap = 150; click_jit = 30
        key_gap = 80; key_jit = 20
        pause_p = 0.02
        corr_p = 0.04
    else:  # mimic copies owner means but uniform (low jitter) timing
        speed = 0.0009; speed_jit = 0.00005
        click_gap = 220; click_jit = 8
        key_gap = 110; key_jit = 5
        pause_p = 0.05
        corr_p = 0.10
    ev = []
    t = 0
    x, y = 0.5, 0.5
    for i in range(n_moves):
        dt = max(16, int(rng.gauss(33, 8)))
        t += dt
        x = min(1.0, max(0.0, x + rng.gauss(0, 0.04)))
        y = min(1.0, max(0.0, y + rng.gauss(0, 0.04)))
        ev.append({"t": t, "type": "move", "x": round(x, 4), "y": round(y, 4)})
    for i in range(n_clicks):
        t += max(40, int(rng.gauss(click_gap, click_jit)))
        x = min(1.0, max(0.0, x + rng.gauss(0, 0.02)))
        y = min(1.0, max(0.0, y + rng.gauss(0, 0.02)))
        ev.append({"t": t, "type": "click", "x": round(x, 4), "y": round(y, 4), "button": 0})
    # text events
    for i in range(n_text):
        t += max(20, int(rng.gauss(key_gap, key_jit)))
        if rng.random() < corr_p:
            ev.append({"t": t, "type": "text", "arrival": "typed", "value": "<BS>"})
            t += max(20, int(rng.gauss(key_gap, key_jit)))
        ev.append({"t": t, "type": "text", "arrival": "typed", "value": chr(ord("a") + (i % 26))})
    if pause_p > 0 and rng.random() < pause_p:
        # insert a long pause by shifting remaining (already appended above; instead add trailing pause event as move gap)
        pass
    ev.sort(key=lambda e: e["t"])
    return ev


def _session_slow_dictation(rng, user_type, n_moves=40, n_clicks=6, n_text=10):
    if user_type == "owner":
        speed = 0.0003; speed_jit = 0.0001
        click_gap = 400; click_jit = 120
        chunk_gap = 900; chunk_jit = 200
        corr_p = 0.15
    elif user_type == "tired_owner":
        speed = 0.0002; speed_jit = 0.00008
        click_gap = 600; click_jit = 180
        chunk_gap = 1400; chunk_jit = 350
        corr_p = 0.30
    elif user_type == "stranger":
        speed = 0.0007; speed_jit = 0.0002
        click_gap = 250; click_jit = 60
        chunk_gap = 500; chunk_jit = 100
        corr_p = 0.05
    else:  # mimic
        speed = 0.0003; speed_jit = 0.00002
        click_gap = 400; click_jit = 15
        chunk_gap = 900; chunk_jit = 25
        corr_p = 0.15
    ev = []
    t = 0
    x, y = 0.5, 0.5
    for i in range(n_moves):
        dt = max(40, int(rng.gauss(66, 12)))
        t += dt
        x = min(1.0, max(0.0, x + rng.gauss(0, 0.015)))
        y = min(1.0, max(0.0, y + rng.gauss(0, 0.015)))
        ev.append({"t": t, "type": "move", "x": round(x, 4), "y": round(y, 4)})
    for i in range(n_clicks):
        t += max(80, int(rng.gauss(click_gap, click_jit)))
        x = min(1.0, max(0.0, x + rng.gauss(0, 0.01)))
        y = min(1.0, max(0.0, y + rng.gauss(0, 0.01)))
        ev.append({"t": t, "type": "click", "x": round(x, 4), "y": round(y, 4), "button": 0})
    for i in range(n_text):
        t += max(100, int(rng.gauss(chunk_gap, chunk_jit)))
        if rng.random() < corr_p:
            ev.append({"t": t, "type": "text", "arrival": "inserted", "value": "<BS>"})
            t += max(100, int(rng.gauss(chunk_gap // 3, chunk_jit // 2)))
        ev.append({"t": t, "type": "text", "arrival": "inserted", "value": "word" + str(i % 5)})
    ev.sort(key=lambda e: e["t"])
    return ev


def gen_all(out_dir, seed=42):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    styles = [("standard", _session_standard), ("slow_dictation", _session_slow_dictation)]
    user_types = ["owner", "tired_owner", "stranger", "mimic"]
    sessions_per = 8
    index = []
    for style_name, gen in styles:
        for ut in user_types:
            for k in range(sessions_per):
                s = gen(rng, ut)
                sid = f"{style_name}__{ut}__{k:02d}"
                path = os.path.join(out_dir, sid + ".jsonl")
                with open(path, "w") as f:
                    for e in s:
                        f.write(json.dumps(e) + "\n")
                index.append({"id": sid, "style": style_name, "user_type": ut, "k": k, "path": path})
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return index


# ---------- Profile + decision ----------
def load_sessions(out_dir):
    idx = json.load(open(os.path.join(out_dir, "index.json")))
    groups = {}
    for item in idx:
        sess = [json.loads(l) for l in open(item["path"]) if l.strip()]
        groups.setdefault((item["style"], item["user_type"]), []).append((item["id"], sess))
    return idx, groups


THRESHOLD = 3.5  # fixed before testing (z units)


def train_profile(sessions):
    X = np.array([features(s) for _, s in sessions])
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return {"mu": mu, "sd": sd, "n": len(sessions)}


def score(profile, session):
    fv = features(session)
    z = (fv - profile["mu"]) / profile["sd"]
    return float(np.linalg.norm(z))


def decide(profile, session, threshold=THRESHOLD):
    return score(profile, session) < threshold


def run_demo():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating fake data...")
    gen_all(DATA_DIR)
    idx, groups = load_sessions(DATA_DIR)

    # Train on owner sessions (both styles combined into one profile per style? 
    # We enroll one user across both styles: build a profile per style.)
    profiles = {}
    for style in ("standard", "slow_dictation"):
        owner_sessions = groups[(style, "owner")]
        profiles[style] = train_profile(owner_sessions)
        print(f"  style={style}: trained on {len(owner_sessions)} owner sessions")

    # Evaluate: genuine = owner + tired_owner; impostor = stranger + mimic
    results = {}
    for style in ("standard", "slow_dictation"):
        prof = profiles[style]
        for ut in ("owner", "tired_owner", "stranger", "mimic"):
            sess = groups[(style, ut)]
            decisions = [decide(prof, s) for _, s in sess]
            scores = [score(prof, s) for _, s in sess]
            results[(style, ut)] = (decisions, scores)
            acc = sum(decisions) / len(decisions)
            print(f"  style={style} ut={ut:14s}: accept_rate={acc:.2f}  "
                  f"mean_score={np.mean(scores):.2f}  (n={len(sess)})")

    # FAR / FRR
    print("\n--- FAR / FRR (threshold fixed at", THRESHOLD, ") ---")
    for style in ("standard", "slow_dictation"):
        genuine = results[(style, "owner")][0] + results[(style, "tired_owner")][0]
        impostor = results[(style, "stranger")][0] + results[(style, "mimic")][0]
        frr = 1 - (sum(genuine) / len(genuine))
        far = sum(impostor) / len(impostor)
        print(f"  style={style}: FRR={frr:.2f} (n_genuine={len(genuine)}), FAR={far:.2f} (n_impostor={len(impostor)})")

    # Swap test: profile trained on stranger must NOT accept owner
    print("\n--- Swap test ---")
    swap_pass = True
    for style in ("standard", "slow_dictation"):
        stranger_prof = train_profile(groups[(style, "stranger")])
        owner_accepted = [decide(stranger_prof, s) for _, s in groups[(style, "owner")]]
        # pass = owner NOT accepted by stranger profile
        passed = not any(owner_accepted)
        swap_pass = swap_pass and passed
        print(f"  style={style}: stranger-profile accepts owner? {any(owner_accepted)}  -> swap_test {'PASS' if passed else 'FAIL'}")

    # Save results
    out = {
        "threshold": THRESHOLD,
        "feature_names": FEATURE_NAMES,
        "swap_test_passed": swap_pass,
    }
    for style in ("standard", "slow_dictation"):
        for ut in ("owner", "tired_owner", "stranger", "mimic"):
            d, sc = results[(style, ut)]
            out[f"{style}__{ut}"] = {"decisions": [bool(x) for x in d],
                                    "scores": [round(float(x), 3) for x in sc]}
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSwap test overall:", "PASS" if swap_pass else "FAIL")
    print("Results saved to", os.path.join(RESULTS_DIR, "results.json"))
    print("\nFresh-copy command: python3 marrow.py demo")
    return 0 if swap_pass else 1


def cmd_help():
    print("""Marrow - behavioral biometrics prototype (1f916 listing-65)

Commands:
  demo   Generate fake data, train, test, report (one command, offline).
  help   This message.

Event schema (JSONL, one event per line):
  {"t": <ms>, "type": "move",  "x": <0..1>, "y": <0..1>}
  {"t": <ms>, "type": "click", "x":.., "y":.., "button": 0}
  {"t": <ms>, "type": "text",  "arrival": "typed"|"onscreen"|"inserted", "value": str}

Train/test on any recording in that format:
  python3 marrow.py train <file.jsonl>   (prints feature vector)
""")
    return 0


def cmd_train(path):
    sess = [json.loads(l) for l in open(path) if l.strip()]
    fv = features(sess)
    for name, v in zip(FEATURE_NAMES, fv):
        print(f"{name:22s} {v:.6f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("demo")
    sub.add_parser("help")
    tr = sub.add_parser("train")
    tr.add_argument("path")
    args = ap.parse_args()
    if args.cmd == "demo":
        return run_demo()
    if args.cmd == "train":
        return cmd_train(args.path)
    return cmd_help()


if __name__ == "__main__":
    sys.exit(main())