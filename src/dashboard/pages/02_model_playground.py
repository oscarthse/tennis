import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from src.dashboard.config import BASE_DIR
from src.dashboard.components.data_manager import get_player_stats, load_data
from src.dashboard.components.model_manager import load_model, get_model_features, discover_models
from src.dashboard.components.plotting import plot_gauge
from src.dashboard.components.navigation import sidebar_navigation

# Page Config
st.set_page_config(page_title="Model Playground", page_icon="🎮", layout="wide")
sidebar_navigation()

st.title('🎮 Match Prediction Playground')
st.markdown("Test our trained models on real ATP Tennis data. Tweak the inputs and see how the probability changes!")

# Load Data & Models
df = load_data()
available_models = discover_models()

@st.cache_data
def load_player_state():
    """Load the player state (Elo, Form) from JSON."""
    path = os.path.join(BASE_DIR, 'models', 'player_state.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

if not available_models:
    st.warning("⚠️ No models found. Please train a model first using `train_v2.py`.")
    st.stop()

# Model Selection
col_m1, col_m2 = st.columns([1, 3])
with col_m1:
    selected_model_name = st.selectbox('Select Model', list(available_models.keys()))

model_info = available_models[selected_model_name]
model = load_model(model_info['path'])
is_v2 = model_info['metadata'].get('version') == 'v2'

player_state = load_player_state() if is_v2 else {}

# Player Selection
col1, col2 = st.columns(2)

all_players = sorted(list(set(df['Player_1'].dropna().unique()) | set(df['Player_2'].dropna().unique())))

with col1:
    st.subheader('👤 Player 1')
    # Default to Nadal if available
    default_p1 = all_players.index('Nadal R.') if 'Nadal R.' in all_players else 0
    p1 = st.selectbox('Select Player 1', all_players, index=default_p1, key='p1')

    # Auto-fill stats
    p1_stats = get_player_stats(p1, df)

    # V2 Stats
    p1_elo = 1500
    p1_form = 0.0
    p1_surf_elo = 1500

    if is_v2 and p1 in player_state:
        p1_elo = player_state[p1]['elo']
        p1_form = player_state[p1]['win_rate_last_10']

    c1_1, c1_2, c1_3 = st.columns(3)
    p1_rank = c1_1.number_input('Rank', 1, 2000, p1_stats['rank'], key='p1_rank')
    p1_pts = c1_2.number_input('Points', 0, 20000, p1_stats['pts'], key='p1_pts')
    p1_odds = c1_3.number_input('Odds', 1.0, 100.0, 1.5, 0.1, key='p1_odds')

    if is_v2:
        st.caption(f"Elo: {p1_elo:.0f} | Form (L10): {p1_form:.1%}")

with col2:
    st.subheader('👤 Player 2')
    # Default to Federer if available
    default_p2 = all_players.index('Federer R.') if 'Federer R.' in all_players else 1
    p2 = st.selectbox('Select Player 2', all_players, index=default_p2, key='p2')

    # Auto-fill stats
    p2_stats = get_player_stats(p2, df)

    # V2 Stats
    p2_elo = 1500
    p2_form = 0.0

    if is_v2 and p2 in player_state:
        p2_elo = player_state[p2]['elo']
        p2_form = player_state[p2]['win_rate_last_10']

    c2_1, c2_2, c2_3 = st.columns(3)
    p2_rank = c2_1.number_input('Rank', 1, 2000, p2_stats['rank'], key='p2_rank')
    p2_pts = c2_2.number_input('Points', 0, 20000, p2_stats['pts'], key='p2_pts')
    p2_odds = c2_3.number_input('Odds', 1.0, 100.0, 2.5, 0.1, key='p2_odds')

    if is_v2:
        st.caption(f"Elo: {p2_elo:.0f} | Form (L10): {p2_form:.1%}")

st.markdown('---')

# Match Context
st.subheader('🏟️ Match Context')
col3, col4, col5, col6 = st.columns(4)

with col3:
    series = st.selectbox('Series', sorted(df['Series'].dropna().unique()))
with col4:
    court = st.selectbox('Court', sorted(df['Court'].dropna().unique()))
with col5:
    surface = st.selectbox('Surface', sorted(df['Surface'].dropna().unique()))
with col6:
    round_val = st.selectbox('Round', sorted(df['Round'].dropna().unique()))

best_of = st.radio('Best of', [3, 5], horizontal=True)

# Update Surface Elo based on selection
if is_v2:
    if p1 in player_state:
        p1_surf_elo = player_state[p1]['elo_surfaces'].get(surface, p1_elo)
        p1_surf_win_rate = player_state[p1]['win_rate_surfaces'].get(surface, 0.0)
        p1_career_win_rate = player_state[p1]['win_rate_career']
    else:
        p1_surf_elo = 1500
        p1_surf_win_rate = 0.0
        p1_career_win_rate = 0.0

    if p2 in player_state:
        p2_surf_elo = player_state[p2]['elo_surfaces'].get(surface, p2_elo)
        p2_surf_win_rate = player_state[p2]['win_rate_surfaces'].get(surface, 0.0)
        p2_career_win_rate = player_state[p2]['win_rate_career']
    else:
        p2_surf_elo = 1500
        p2_surf_win_rate = 0.0
        p2_career_win_rate = 0.0

# Prediction
if st.button('🚀 Predict Winner', type='primary', use_container_width=True):
    # Construct input dataframe
    input_data = {
        'Rank_1': p1_rank, 'Rank_2': p2_rank,
        'Pts_1': p1_pts, 'Pts_2': p2_pts,
        'Odd_1': p1_odds, 'Odd_2': p2_odds,
        'Best of': best_of,
        'Series': series, 'Court': court, 'Surface': surface, 'Round': round_val
    }

    if is_v2:
        # Add V2 features
        # Calculate Elo Prob
        elo_prob_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
        elo_prob_p2 = 1 - elo_prob_p1

        v2_features = {
            'elo_p1': p1_elo, 'elo_p2': p2_elo,
            'elo_prob_p1': elo_prob_p1, 'elo_prob_p2': elo_prob_p2,
            'elo_surf_p1': p1_surf_elo, 'elo_surf_p2': p2_surf_elo,
            'p1_win_rate_career': p1_career_win_rate,
            'p1_win_rate_last_10': p1_form,
            'p1_win_rate_surface': p1_surf_win_rate,
            'p2_win_rate_career': p2_career_win_rate,
            'p2_win_rate_last_10': p2_form,
            'p2_win_rate_surface': p2_surf_win_rate
        }
        input_data.update(v2_features)

    input_df = pd.DataFrame([input_data])

    try:
        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]

        # Display Results
        st.markdown('### 🏆 Prediction Result')

        res_col1, res_col2 = st.columns(2)

        winner_name = p1 if pred == 1 else p2
        win_prob = prob[1] if pred == 1 else prob[0]

        with res_col1:
            if pred == 1:
                st.success(f"**Winner: {p1}**")
            else:
                st.error(f"**Loser: {p1}**")

            fig_p1 = plot_gauge(prob[1], p1, pred == 1)
            st.plotly_chart(fig_p1, use_container_width=True)

        with res_col2:
            if pred == 0:
                st.success(f"**Winner: {p2}**")
            else:
                st.error(f"**Loser: {p2}**")

            fig_p2 = plot_gauge(prob[0], p2, pred == 0)
            st.plotly_chart(fig_p2, use_container_width=True)

        # Feature Explanation
        st.markdown('---')
        st.subheader('🔍 Feature Inputs')
        st.json(input_data)

        features = get_model_features(model)
        if features:
            with st.expander("See Engineered Features (from Pipeline)"):
                st.write(features)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
