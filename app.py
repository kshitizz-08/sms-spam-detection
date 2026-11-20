"""SMS Spam Detection System.
Simple, clean UI using native Streamlit components.
100% Local Execution (Zero External APIs).
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import database
import auth
import preprocessing
import model_pipeline

importlib.reload(preprocessing)
importlib.reload(model_pipeline)

from preprocessing import (preprocess_with_steps, detect_script,
                            highlight_message_html)
from model_pipeline import load_1m_bundle, train_5k_bundle, predict_message
from report_generator import create_pdf_report

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Minimal CSS — only colour overrides ─────────────────────────
st.markdown("""
<style>
/* App background */
.stApp { background: #ffffff; }

/* All buttons → green */
div.stButton > button {
    background: #16a34a;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.4rem 1rem;
}
div.stButton > button:hover { background: #15803d; color: #fff; }

/* Primary buttons */
div.stButton > button[kind="primary"] { background: #15803d; }

/* Sidebar */
[data-testid="stSidebar"] { background: #f0fdf4; }

/* Download button */
div.stDownloadButton > button {
    background: #15803d; color: #fff;
    border: none; border-radius: 8px; font-weight: 600;
}
div.stDownloadButton > button:hover { background: #166534; }

/* Headings */
h1, h2, h3 { color: #14532d; }
</style>
""", unsafe_allow_html=True)

# ── DB ───────────────────────────────────────────────────────────
database.init_db()

@st.cache_resource(show_spinner=False)
def get_bundle(mode):
    if mode == "1m":
        try:
            return load_1m_bundle()
        except Exception:
            return train_5k_bundle()
    return train_5k_bundle()

# ── Auth ─────────────────────────────────────────────────────────
if not auth.login_ui():
    st.stop()

username = st.session_state["username"]

# ── Session defaults ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "check"
if "engine" not in st.session_state:
    st.session_state["engine"] = "1m"

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🛡️ SMS Spam Shield")
    st.caption("AI-Powered Scam & Fraud Protection")
    st.divider()

    st.write(f"👤 **{username}**")
    st.divider()

    # ── Dataset selector ────────────────────────────────────────
    st.subheader("Dataset")
    if st.session_state["engine"] == "1m":
        st.success("⚡ Using: 1,000,000 Messages")
    else:
        st.warning("🧪 Using: 5,572 Messages (UCI)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 1 Million", width="stretch"):
            st.session_state["engine"] = "1m"
            st.rerun()
    with c2:
        if st.button("🧪 5K UCI", width="stretch"):
            st.session_state["engine"] = "5k"
            st.rerun()

    st.divider()

    # ── Page navigation ─────────────────────────────────────────
    st.subheader("Pages")

    pages = [
        ("check",   "🔍 Check a Message"),
        ("module1", "📊 Module 1 — Dataset"),
        ("module2", "🧹 Module 2 — Cleaning"),
        ("module3", "🧮 Module 3 — AI Features"),
        ("module4", "🏆 Module 4 — Scoreboard"),
        ("history", "📜 History & Report"),
    ]

    for key, label in pages:
        if st.session_state["page"] == key:
            st.info(f"**▶ {label}**")          # active page — highlighted box
        else:
            if st.button(label, width="stretch", key=f"nav_{key}"):
                st.session_state["page"] = key
                st.rerun()

    st.divider()
    if st.button("🚪 Log Out", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Load models ──────────────────────────────────────────────────
with st.spinner("Loading AI models..."):
    try:
        bundle = get_bundle(st.session_state["engine"])
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

df      = bundle["df"]
metrics = bundle["metrics"]
page    = st.session_state["page"]


# ════════════════════════════════════════════════════════════════
#  PAGE 1 — CHECK A MESSAGE
# ════════════════════════════════════════════════════════════════
if page == "check":
    st.title("🔍 Check a Message")
    st.write("Paste any SMS below — 5 AI models will instantly tell you if it is **safe** or **spam**.")
    st.divider()

    tab_single, tab_bulk, tab_words = st.tabs(["Single Message", "Bulk Upload", "Scam Words"])

    # ── Single Message ──────────────────────────────────────────
    with tab_single:
        st.write("**Try a quick example:**")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            if st.button("🚨 Bank Scam", width="stretch"):
                st.session_state["_msg"] = (
                    "URGENT: Your bank account is locked. "
                    "Verify now at http://bank-secure-login.link or call 0800123456."
                )
                st.rerun()
        with e2:
            if st.button("🎁 Fake Prize", width="stretch"):
                st.session_state["_msg"] = (
                    "Congratulations! You won a £1,000 Gift Card. "
                    "Claim by texting CLAIM to 88291!"
                )
                st.rerun()
        with e3:
            if st.button("📦 Fake Parcel", width="stretch"):
                st.session_state["_msg"] = (
                    "Your package cannot be delivered. "
                    "Pay £1.99 at http://track-parcel-now.com"
                )
                st.rerun()
        with e4:
            if st.button("☕ Normal SMS", width="stretch"):
                st.session_state["_msg"] = (
                    "Hey! Are you free this evening to meet for coffee?"
                )
                st.rerun()

        st.write("")
        user_text = st.text_area(
            "Paste or type the message here:",
            value=st.session_state.get("_msg", ""),
            height=120,
            placeholder="Type or paste an SMS message here..."
        )

        col_a, col_b, _ = st.columns([1.2, 1, 5])
        with col_a:
            check = st.button("🔍 Check Now", type="primary", width="stretch")
        with col_b:
            if st.button("🗑️ Clear", width="stretch"):
                st.session_state["_msg"] = ""
                st.rerun()

        # ── Result ───────────────────────────────────────────
        if check:
            msg = user_text.strip()
            if not msg:
                st.warning("Please enter a message first.")
            else:
                # ── Detect language / script ───────────────
                script = detect_script(msg)

                st.divider()

                # ── Language badge ─────────────────────────
                lang_labels = {
                    "hindi":    ("🇮🇳 Hindi (Devanagari)", "#FEF3C7", "#92400E"),
                    "marathi":  ("🇮🇳 Marathi (Devanagari)", "#FEF3C7", "#92400E"),
                    "hinglish": ("🇮🇳 Hinglish (Roman Hindi)", "#ECFDF5", "#065F46"),
                    "english":  ("🇬🇧 English", "#EFF6FF", "#1E40AF"),
                }
                lang_name, lang_bg, lang_fg = lang_labels.get(script, ("🇬🇧 English", "#EFF6FF", "#1E40AF"))
                st.markdown(
                    f'<span style="background:{lang_bg};color:{lang_fg};'
                    f'border-radius:6px;padding:4px 12px;font-size:.85rem;font-weight:700;">'
                    f'🌐 Language Detected: {lang_name}</span>',
                    unsafe_allow_html=True
                )
                st.write("")

                # ── Unified Multilingual 5-AI Inference ───
                with st.spinner("Analyzing message across all 5 AI models..."):
                    result = predict_message(bundle, msg)

                # Store prediction in local database
                database.add_prediction(username, result)

                is_spam  = result["Consensus prediction"] == "SPAM"
                prob     = float(result["Consensus spam probability"])
                votes    = int(str(result["Spam votes"]).split("/")[0])

                # ── Main Verdict Banner ────────────────────
                if is_spam:
                    st.error(f"## ⚠️ SPAM / SCAM DETECTED — {prob:.0%} Scam Risk")
                    st.write(f"🗳️ **{votes} out of 5 AI models** classified this message as spam.")
                    st.write("🚫 Do **NOT** click any links, share OTPs, or reply.")
                else:
                    st.success(f"## ✅ SAFE MESSAGE — {1 - prob:.0%} Safety Score")
                    st.write(f"🗳️ **{5 - votes} out of 5 AI models** classified this as safe.")
                    st.write("No suspicious scam patterns or fraudulent triggers detected.")

                # ── Sentence-level Highlighting ────────────
                st.write("")
                st.write("**🔍 Highlighted Message** — scam indicators marked:")
                full_spam_vocab = {w for w, _ in bundle["spam_words"]}
                highlighted_html = highlight_message_html(msg, full_spam_vocab)
                st.markdown(
                    f'<div style="background:#F9FAFB;border:1.5px solid #E5E7EB;'
                    f'border-radius:10px;padding:14px 18px;font-size:1rem;line-height:1.9;">'
                    f'{highlighted_html}</div>',
                    unsafe_allow_html=True
                )
                # Legend
                st.markdown(
                    '<div style="font-size:.78rem;color:#6B7280;margin-top:6px;">'
                    '<span style="background:#FECACA;padding:1px 6px;border-radius:3px;">🚨 Scam keyword</span> &nbsp;'
                    '<span style="background:#FEE2E2;padding:1px 6px;border-radius:3px;">🔗 Suspicious URL</span> &nbsp;'
                    '<span style="background:#FEF3C7;padding:1px 6px;border-radius:3px;">📱💰 Phone / Money</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # ── All 5 Model Breakdown (For Viva) ───────
                with st.expander("📊 See all 5 model results (for Viva)"):
                    rows = {
                        "Model":   ["Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Bi-LSTM"],
                        "Verdict": [result["Naive Bayes prediction"],
                                    result["Logistic Regression prediction"],
                                    result["Support Vector Machine prediction"],
                                    result["Random Forest prediction"],
                                    result["LSTM prediction"]],
                        "Spam Risk": [f"{result['Naive Bayes spam probability']:.1%}",
                                      f"{result['Logistic Regression spam probability']:.1%}",
                                      f"{result['Support Vector Machine spam probability']:.1%}",
                                      f"{result['Random Forest spam probability']:.1%}",
                                      f"{result['LSTM spam probability']:.1%}"],
                    }
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── Bulk Upload ─────────────────────────────────────────────
    with tab_bulk:
        st.subheader("Bulk Upload")
        st.write("Upload a **CSV** (needs a `message` column) or a **TXT** (one message per line).")
        up = st.file_uploader("Choose file", type=["csv", "txt"])

        if up:
            if up.name.endswith(".csv"):
                tmp = pd.read_csv(up)
                if "message" not in tmp.columns:
                    st.error("CSV must have a column called `message`.")
                    st.stop()
                msgs = tmp["message"].fillna("").astype(str).tolist()
            else:
                msgs = up.getvalue().decode("utf-8", errors="replace").splitlines()

            st.info(f"**{len(msgs):,} messages** found.")
            if st.button("🚀 Scan All Messages", type="primary"):
                rows, prog = [], st.progress(0)
                for i, m in enumerate(msgs):
                    if m.strip():
                        r = predict_message(bundle, m)
                        rows.append({
                            "Message":      m,
                            "Result":       "SPAM" if r["Consensus prediction"] == "SPAM" else "SAFE",
                            "Risk":         f"{r['Consensus spam probability']:.1%}",
                            "Votes":        r["Spam votes"],
                        })
                        database.add_prediction(username, r)
                    prog.progress((i + 1) / max(len(msgs), 1))

                res = pd.DataFrame(rows)
                st.success(f"Done! Scanned {len(res):,} messages.")
                st.dataframe(res, width="stretch")
                st.download_button("📥 Download Results (CSV)",
                                   res.to_csv(index=False).encode(),
                                   "results.csv", "text/csv")

    # ── Scam Words ──────────────────────────────────────────────
    with tab_words:
        st.subheader("Common Scam Words")
        wc1, wc2 = st.columns(2)
        with wc1:
            st.write("🚨 **Top Scam Keywords**")
            st.dataframe(pd.DataFrame(bundle["spam_words"][:20],
                         columns=["Word", "Score"]), width="stretch")
        with wc2:
            st.write("✅ **Top Safe Words**")
            st.dataframe(pd.DataFrame(bundle["ham_words"][:20],
                         columns=["Word", "Score"]), width="stretch")

        fig, ax = plt.subplots(figsize=(7, 4))
        top = bundle["spam_words"][:12]
        ax.barh([w for w,_ in top][::-1], [s for _,s in top][::-1], color="#16a34a")
        ax.set_title("Top Scam Words"); ax.set_xlabel("Score")
        st.pyplot(fig); plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  MODULE 1 — DATASET & STATS
# ════════════════════════════════════════════════════════════════
elif page == "module1":
    st.title("📊 Module 1 — Dataset & Statistics")
    st.write("How much data was used to train the AI, and what does it look like?")
    st.divider()

    total = bundle.get("total_messages", len(df))
    ham   = bundle.get("ham_count",  int((df["label"] == "ham").sum()))
    spam  = bundle.get("spam_count", int((df["label"] == "spam").sum()))
    train = bundle.get("training_samples", 0)
    test  = bundle.get("testing_samples",  0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Messages",   f"{total:,}")
    m2.metric("Safe (Ham)",       f"{ham:,}",  f"{ham/total:.0%}")
    m3.metric("Spam",             f"{spam:,}", f"{spam/total:.0%}")
    m4.metric("Train / Test",     f"{train:,} / {test:,}")

    st.divider()
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Safe vs Spam")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie([ham, spam], labels=["Safe", "Spam"], autopct="%1.1f%%",
               colors=["#22c55e", "#ef4444"], startangle=140, explode=(0, 0.06))
        st.pyplot(fig); plt.close(fig)

    with g2:
        st.subheader("Message Length Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(data=df, x="char_length", hue="label", bins=40, kde=True,
                     palette={"ham": "#22c55e", "spam": "#ef4444"}, alpha=0.6, ax=ax)
        ax.set_xlim(0, 320); ax.set_xlabel("Characters per message")
        st.pyplot(fig); plt.close(fig)

    st.info("💡 **Insight:** Spam messages are usually longer — they contain links, phone numbers, and urgent phrases. With 1,000,000+ messages the AI learns thousands of scam variations.")


# ════════════════════════════════════════════════════════════════
#  MODULE 2 — TEXT CLEANING DEMO
# ════════════════════════════════════════════════════════════════
elif page == "module2":
    st.title("🧹 Module 2 — Text Cleaning Pipeline")
    st.write("See exactly how raw SMS text is cleaned before the AI processes it.")
    st.divider()

    st.info("**Why clean text?** Scammers use CAPS, symbols ($£€), random phone numbers and short URLs to dodge filters. Cleaning converts all of these into standard tags so the AI can still catch them.")

    examples = {
        "🚨 Prize Scam":    "WINNER! Claim a $1,000 Gift Card today! Visit http://free-card.org or call 0800123456 immediately.",
        "🏦 Phishing Bank": "URGENT: Your account is locked! Verify at https://secure-bank.link with code 88291 or call 07712345678.",
        "☕ Normal Chat":   "Hey, are we still meeting at 4 PM tomorrow in the library to work on the project?",
    }

    chosen = st.selectbox("Pick an example:", list(examples.keys()))
    steps  = preprocess_with_steps(examples[chosen])

    b, a = st.columns(2)
    with b:
        st.write("**Before (Raw text):**")
        st.code(steps["original"], language="text")
    with a:
        st.write("**After (Cleaned tokens):**")
        st.code(steps["step4_stopwords_removed"] or "(empty)", language="text")

    st.divider()
    st.write("**Step-by-step breakdown:**")
    with st.expander("Show all 4 steps"):
        st.write("**Step 1 — Lowercasing:**")
        st.code(steps["step1_lowercased"])
        st.caption("Converts ALL CAPS to lowercase.")

        st.write("**Step 2 — Replace special patterns:**")
        st.code(steps["step2_entities_replaced"])
        st.caption("URLs → urltoken | Phone numbers → numtoken | $£€ → moneytoken")

        st.write("**Step 3 — Remove punctuation:**")
        st.code(steps["step3_punct_removed"])
        st.caption("Removes !, ?, #, @ and other special characters.")

        st.write("**Step 4 — Remove common words (stopwords):**")
        st.code(steps["step4_stopwords_removed"])
        st.caption(f"Removes filler words like 'the', 'is', 'a'. Remaining tokens: {steps['clean_token_count']}")


# ════════════════════════════════════════════════════════════════
#  MODULE 3 — HOW AI READS TEXT
# ════════════════════════════════════════════════════════════════
elif page == "module3":
    st.title("🧮 Module 3 — How AI Reads Text")
    st.write("Computers don't understand words — they understand numbers. Here's how we convert text to numbers.")
    st.divider()

    t1, t2 = st.columns(2)

    with t1:
        st.subheader("Method 1: TF-IDF")
        st.write("Used by: **Naive Bayes, Logistic Regression, SVM, Random Forest**")
        st.write("""
        Each word gets a score based on:
        - How often it appears in **spam** messages
        - How rarely it appears in **normal** messages

        Words like *claim*, *winner*, *free*, *urgent* get **high spam scores**.
        """)
        st.metric("Vocabulary Size", f"{len(bundle['feature_names']):,} words")
        st.write("**Sample words learned:**")
        st.code(", ".join(bundle["feature_names"][:12]))

    with t2:
        st.subheader("Method 2: Word Embeddings (LSTM)")
        st.write("Used by: **Bidirectional LSTM Neural Network**")
        st.write("""
        Instead of scoring individual words, the LSTM:
        - Reads the **whole sentence** forwards and backwards
        - Understands **word context** (e.g. "free gift" vs "free time")
        - Catches new scam patterns that word-scores might miss
        """)
        st.metric("Sequence Length", f"{bundle['max_length']} words")
        st.metric("Embedding Size", "32 dimensions per word")

    st.divider()
    st.info("💡 **Why use both?** TF-IDF is fast and great for known spam patterns. LSTM catches new and evolving scam styles. Together they give a 5-model ensemble that is very hard to fool.")


# ════════════════════════════════════════════════════════════════
#  MODULE 4 — MODEL SCOREBOARD
# ════════════════════════════════════════════════════════════════
elif page == "module4":
    st.title("🏆 Module 4 — Model Performance Scoreboard")
    st.write("All 5 AI algorithms tested on the same dataset — see which performs best.")
    st.divider()

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Algorithm":  name,
            "Accuracy":   m["accuracy"],
            "Precision":  m["precision"],
            "Recall":     m["recall"],
            "F1 Score":   m["f1"],
            "ROC-AUC":    m.get("roc_auc", 0.0),
            "Train Time": f"{m['training_time']:.2f}s",
        })
    df_m = pd.DataFrame(rows).set_index("Algorithm")

    st.subheader("Results Table")
    st.dataframe(
        df_m.style
            .format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]})
            .highlight_max(subset=["Accuracy","Precision","Recall","F1 Score"], color="#bbf7d0"),
        width="stretch"
    )
    st.caption("🟢 Green = best score in that column")

    st.divider()
    tc, tr = st.tabs(["Confusion Matrix", "ROC Curves"])

    with tc:
        sel = st.selectbox("Select model:", list(metrics.keys()))
        cm  = metrics[sel]["confusion_matrix"]
        cc, ic = st.columns(2)
        with cc:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["Safe","Spam"], yticklabels=["Safe","Spam"],
                        cbar=False, ax=ax, annot_kws={"size":13,"weight":"bold"})
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(sel, fontweight="bold")
            st.pyplot(fig); plt.close(fig)
        with ic:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            st.write(f"**{sel} — What happened:**")
            st.metric("✅ Correct Safe detections", f"{tn:,}")
            st.metric("✅ Correct Spam caught",     f"{tp:,}")
            st.metric("⚠️ Safe wrongly flagged",    f"{fp:,}")
            st.metric("⚠️ Spam missed",             f"{fn:,}")

    with tr:
        st.write("ROC Curve — higher curve = better at separating spam from safe:")
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, m in metrics.items():
            if "fpr" in m:
                ax.plot(m["fpr"], m["tpr"], lw=2,
                        label=f"{name} (AUC={m['roc_auc']:.4f})")
        ax.plot([0,1],[0,1],"k--",alpha=.3,label="Random guess")
        ax.set_xlabel("False Alarm Rate"); ax.set_ylabel("Spam Catch Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("ROC Curves — All 5 Models", fontweight="bold")
        st.pyplot(fig); plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  HISTORY & REPORT
# ════════════════════════════════════════════════════════════════
elif page == "history":
    st.title("📜 History & Report")
    st.write("View all messages you have checked, or download a PDF report.")
    st.divider()

    records = database.get_predictions(username)

    st.subheader("Messages Checked")
    if not records:
        st.info("No messages checked yet. Go to 🔍 Check a Message to get started.")
    else:
        df_h = pd.DataFrame(records)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Checked", len(df_h))
        c2.metric("🚨 Spam Found", int((df_h["consensus_prediction"] == "SPAM").sum()))
        c3.metric("✅ Safe",       int((df_h["consensus_prediction"] == "HAM").sum()))

        filt = st.selectbox("Filter:", ["All", "SPAM only", "SAFE only"])
        if filt == "SPAM only":
            df_h = df_h[df_h["consensus_prediction"] == "SPAM"]
        elif filt == "SAFE only":
            df_h = df_h[df_h["consensus_prediction"] == "HAM"]

        df_display = df_h[["message","consensus_prediction","consensus_spam_probability","created_at"]].copy()
        df_display["consensus_prediction"] = df_display["consensus_prediction"].map(
            lambda x: "🚨 SPAM" if str(x).upper() == "SPAM" else "✅ SAFE"
        )
        df_display["consensus_spam_probability"] = df_display["consensus_spam_probability"].map(
            lambda p: f"{float(p):.1%}" if pd.notnull(p) else "0.0%"
        )

        st.dataframe(
            df_display.rename(columns={
                "message":                    "Message",
                "consensus_prediction":       "Result",
                "consensus_spam_probability": "Scam Risk",
                "created_at":                 "Time",
            }),
            width="stretch"
        )

        col_dl, col_clr = st.columns([2, 1])
        with col_dl:
            st.download_button("📥 Download as CSV",
                               df_h.to_csv(index=False).encode(),
                               "history.csv", "text/csv")
        with col_clr:
            if st.button("🗑️ Clear Scan History"):
                database.clear_predictions(username)
                st.rerun()

    st.divider()
    st.subheader("Download PDF Report")
    st.write("Generate a full academic PDF with dataset info, model scores, and confusion matrices.")
    if st.button("📄 Generate PDF Report", type="primary"):
        try:
            pdf = create_pdf_report(bundle, records)
            st.download_button("📥 Save PDF",
                               pdf, "SpamShield_Report.pdf", "application/pdf")
            st.success("PDF ready! Click the button above to download.")
        except Exception as e:
            st.error(f"PDF error: {e}")
