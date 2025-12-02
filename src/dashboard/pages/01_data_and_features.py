import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.data_manager import get_processed_data

st.set_page_config(page_title="Data & Features", page_icon="📊", layout="wide")
sidebar_navigation()

st.title("📊 Data & Feature Engineering")

st.markdown("""
### 1. The Dataset
We are using historical ATP Tennis match data. Each row represents a single match with:
*   **Player 1 & Player 2**: The competitors.
*   **Winner**: Who won.
*   **Score**: The set scores.
*   **Stats**: Aces, Double Faults, Break Points (if available).
*   **Odds**: Betting odds from various bookmakers.
""")

df = get_processed_data()

if not df.empty:
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Showing first 10 rows of {len(df):,} matches.")

st.markdown("---")

st.header("2. Feature Engineering")
st.markdown("""
Raw data isn't enough. To predict the future, we need to understand the **state** of the players *before* the match starts.
We engineered several advanced features:
""")

tab1, tab2, tab3 = st.tabs(["🏆 Elo Ratings", "🔥 Recent Form", "🌱 Surface Performance"])

with tab1:
    st.subheader("The Elo Rating System")
    st.markdown("""
    Originally designed for Chess, Elo is a method for calculating the relative skill levels of players in zero-sum games.

    #### The Math
    The probability of Player A winning against Player B is:
    """)

    st.latex(r"P(A) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}")

    st.markdown("""
    Where:
    *   $R_A$ is Player A's current rating.
    *   $R_B$ is Player B's current rating.
    *   400 is a scaling factor.

    **Update Rule:**
    After a match, ratings are updated:
    """)

    st.latex(r"R_A' = R_A + K \cdot (S_A - P(A))")

    st.markdown("""
    *   $K$ is the K-factor (how fast ratings change).
    *   $S_A$ is the actual score (1 for win, 0 for loss).
    *   $P(A)$ was the expected probability.

    **Intuition:**
    *   If you beat a Grand Slam champion (High $R_B$), your rating goes up a lot.
    *   If you lose to a beginner (Low $R_B$), your rating drops significantly.
    """)

    # Interactive Elo Plot
    st.subheader("📈 Visualize Elo History")
    players = sorted(list(set(df['Player_1'].unique()) | set(df['Player_2'].unique())))
    default_idx = players.index('Federer R.') if 'Federer R.' in players else 0
    selected_player = st.selectbox("Select Player", players, index=default_idx)

    # Filter matches for this player
    p_matches = df[(df['Player_1'] == selected_player) | (df['Player_2'] == selected_player)].sort_values('Date')

    if not p_matches.empty:
        # Extract Elo history
        dates = []
        elos = []

        for _, row in p_matches.iterrows():
            dates.append(row['Date'])
            if row['Player_1'] == selected_player:
                elos.append(row['elo_p1'])
            else:
                elos.append(row['elo_p2'])

        hist_df = pd.DataFrame({'Date': dates, 'Elo': elos})

        fig = px.line(hist_df, x='Date', y='Elo', title=f"Elo Rating History: {selected_player}")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Momentum & Form")
    st.markdown("""
    Tennis is a mental game. A player on a winning streak is dangerous.

    **Feature: `win_rate_last_10`**
    *   We look at the last 10 matches *before* the current one.
    *   Calculate the percentage of wins.
    *   This captures short-term momentum.
    """)

    st.info("💡 **Data Leakage Alert**: We must be careful to shift the window! The form for a match on Sunday must ONLY use matches from Saturday and before. It cannot include Sunday's result.")

with tab3:
    st.subheader("Surface Specialists")
    st.markdown("""
    Nadal is the "King of Clay". Federer loved Grass.

    We calculate **Surface-Specific Elo** and **Surface Win Rates**.
    *   A player might have a Global Elo of 2000, but a Clay Elo of 2200.
    *   Our models use both to make the best prediction.
    """)

    if not df.empty:
        # Show surface distribution
        surf_counts = df['Surface'].value_counts().reset_index()
        surf_counts.columns = ['Surface', 'Matches']
        fig_surf = px.pie(surf_counts, values='Matches', names='Surface', title='Matches by Surface')
        st.plotly_chart(fig_surf, use_container_width=True)
