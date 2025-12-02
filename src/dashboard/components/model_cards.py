import streamlit as st

def render_model_card(title, description, pros, cons, playground_link=True):
    with st.container():
        st.subheader(f"📌 TL;DR: {title}")
        st.markdown(description)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Pros")
            for pro in pros:
                st.markdown(f"- {pro}")

        with col2:
            st.markdown("#### ⚠️ Cons")
            for con in cons:
                st.markdown(f"- {con}")

        if playground_link:
            st.markdown("---")
            st.page_link("pages/02_model_playground.py", label=f"Try {title} in Playground →", icon="🎮")
