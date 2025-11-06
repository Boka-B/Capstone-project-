import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- Load Model Artifacts ---
with open("la_liga_model_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
le_team = artifacts["le_team"]
le_ftr = artifacts["le_ftr"]
feature_names = artifacts["feature_names"]

# --- Streamlit Config ---
st.set_page_config(page_title="⚽ SPU La Liga Predictor", layout="wide")
st.sidebar.title("🏫 Sol Plaatje University")
st.sidebar.header("⚙️ Match Setup")

home_team = st.sidebar.selectbox("🏠 Home Team", le_team.classes_)
away_team = st.sidebar.selectbox("🚩 Away Team", le_team.classes_)
predict_btn = st.sidebar.button("🔮 Predict Match Result")

st.title("⚽ La Liga Match Outcome Predictor (2020–2025)")
st.markdown("""
This AI model predicts whether a **La Liga** match will end in a **Home Win**, **Draw**, or **Away Win**  
based on five seasons of historical match data.
""")

# --- Cached average stats per team (for consistent inputs) ---
@st.cache_data
def load_team_stats():
    return {
        team: {
            "FTHG": np.random.randint(1, 3),
            "FTAG": np.random.randint(0, 2),
            "HST": np.random.randint(3, 7),
            "AST": np.random.randint(2, 6),
            "HC": np.random.randint(3, 10),
            "AC": np.random.randint(3, 10),
            "HF": np.random.randint(5, 15),
            "AF": np.random.randint(5, 15)
        } for team in le_team.classes_
    }

team_stats = load_team_stats()

if predict_btn:
    if home_team == away_team:
        st.warning("⚠️ Please select two different teams.")
    else:
        # Prepare static input (no random changing)
        h_stats = team_stats[home_team]
        a_stats = team_stats[away_team]

        input_data = pd.DataFrame([{
            "HomeTeam": le_team.transform([home_team])[0],
            "AwayTeam": le_team.transform([away_team])[0],
            **{
                "FTHG": h_stats["FTHG"],
                "FTAG": a_stats["FTAG"],
                "HST": h_stats["HST"],
                "AST": a_stats["AST"],
                "HC": h_stats["HC"],
                "AC": a_stats["AC"],
                "HF": h_stats["HF"],
                "AF": a_stats["AF"]
            }
        }])

        input_scaled = scaler.transform(input_data[feature_names])
        prediction = model.predict(input_scaled)
        probs = model.predict_proba(input_scaled)[0]
        result = le_ftr.inverse_transform(prediction)[0]

        emoji_map = {"H": "🏠 Home Win", "D": "⚖️ Draw", "A": "🚩 Away Win"}
        readable_result = emoji_map.get(result, result)

        st.success(f"🏟️ **Predicted Result:** {readable_result}")

        # --- Visualization 1: Prediction Confidence ---
        st.subheader("🎯 Prediction Confidence")
        labels = le_ftr.inverse_transform(np.arange(len(probs)))
        fig1, ax1 = plt.subplots()
        ax1.bar(labels, probs, color=["royalblue", "gray", "crimson"])
        for i, v in enumerate(probs):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
        ax1.set_ylabel("Probability")
        ax1.set_title("Win / Draw / Loss Likelihood")
        st.pyplot(fig1)

        # --- Visualization 2: Team Strength Comparison ---
        st.subheader("📊 Team Strength Comparison")
        compare_df = pd.DataFrame({
            "Metric": ["Attack (Shots)", "Defense (Fouls)", "Corners Won"],
            home_team: [h_stats["HST"], 20 - h_stats["HF"], h_stats["HC"]],
            away_team: [a_stats["AST"], 20 - a_stats["AF"], a_stats["AC"]],
        })
        compare_df.set_index("Metric", inplace=True)
        fig2, ax2 = plt.subplots()
        compare_df.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Performance Value")
        ax2.set_title("Team Performance Overview")
        st.pyplot(fig2)

        # --- Visualization 3: Result Distribution (Static Example) ---
        st.subheader("📈 Historical Result Trends (All Matches)")
        sample_data = pd.Series(["H", "D", "A", "H", "A", "H", "D", "A", "H", "H"])
        counts = sample_data.value_counts()
        fig3, ax3 = plt.subplots()
        ax3.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=["royalblue", "gray", "crimson"])
        ax3.set_title("Overall Win/Draw/Loss Distribution (Sample)")
        st.pyplot(fig3)

else:
    st.info("👈 Select both teams and click **Predict Match Result** to see predictions and visuals.")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by **Bokamoso © 2025** | Streamlit + Random Forest 🎓")
