import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(page_title="Lotka-Volterra Predator-Prey Simulator", layout="wide")

st.title("Predator-Prey Dynamics in Marine Ecosystems")

st.markdown(r"""
**Lotka-Volterra Equations**

$$
\begin{cases}
\frac{dx}{dt} = \alpha x - \beta x y \\
\frac{dy}{dt} = \delta x y - \gamma y
\end{cases}
$$

- $x$: Prey population  
- $y$: Predator population  
- $\alpha$: Prey birth rate  
- $\beta$: Predation rate  
- $\gamma$: Predator death rate  
- $\delta$: Energy conversion rate  
""")

DEFAULTS = {
    "alpha": 1.0,
    "beta": 0.1,
    "gamma": 1.5,
    "delta": 0.075,
    "prey0": 40,
    "predator0": 9,
}

# 初始化 session_state，避免控件无值
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_params():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

with st.sidebar:
    st.header("Model Parameters")
    st.slider(
        "Prey Birth Rate (α)", min_value=0.1, max_value=2.0,
        value=st.session_state["alpha"], step=0.1, key="alpha"
    )
    st.slider(
        "Predation Rate (β)", min_value=0.001, max_value=1.0,
        value=st.session_state["beta"], step=0.001, format="%.3f", key="beta"
    )
    st.slider(
        "Predator Death Rate (γ)", min_value=0.1, max_value=2.0,
        value=st.session_state["gamma"], step=0.1, key="gamma"
    )
    st.slider(
        "Energy Conversion Rate (δ)", min_value=0.001, max_value=1.0,
        value=st.session_state["delta"], step=0.001, format="%.3f", key="delta"
    )
    st.number_input(
        "Initial Prey Population", min_value=1, value=st.session_state["prey0"], step=1, key="prey0"
    )
    st.number_input(
        "Initial Predator Population", min_value=1, value=st.session_state["predator0"], step=1, key="predator0"
    )

    st.button("Reset to Default", on_click=reset_params)

def lotka_volterra(prey0, predator0, alpha, beta, delta, gamma, t):
    dt = t[1] - t[0]
    prey = np.zeros_like(t)
    predator = np.zeros_like(t)
    prey[0] = prey0
    predator[0] = predator0
    for i in range(1, len(t)):
        prey[i] = prey[i-1] + (alpha * prey[i-1] - beta * prey[i-1] * predator[i-1]) * dt
        predator[i] = predator[i-1] + (delta * prey[i-1] * predator[i-1] - gamma * predator[i-1]) * dt
        prey[i] = max(prey[i], 0)
        predator[i] = max(predator[i], 0)
    return prey, predator

t = np.linspace(0, 50, 1000)
prey, predator = lotka_volterra(
    st.session_state["prey0"],
    st.session_state["predator0"],
    st.session_state["alpha"],
    st.session_state["beta"],
    st.session_state["delta"],
    st.session_state["gamma"],
    t
)
ratio = predator / prey
ratio[np.isnan(ratio) | np.isinf(ratio)] = 0

col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=prey, name='Prey'))
    fig1.add_trace(go.Scatter(x=t, y=predator, name='Predator',line=dict(color="red")))
    fig1.update_layout(title='Population Over Time', xaxis_title='Time', yaxis_title='Population')
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Prey outpaces predator → prey overshoots → predator rebounds with delay.")

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prey, y=predator, mode='lines', name='Phase'))
    fig2.update_layout(title='Phase Space: Predator vs. Prey', xaxis_title='Prey Population', yaxis_title='Predator Population')
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("This phase plot shows the cyclical predator-prey relationship.")

st.markdown("### Predator to Prey Ratio Over Time")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=ratio, name='Predator/Prey Ratio'))
fig3.update_layout(title='Predator to Prey Ratio Over Time', xaxis_title='Time', yaxis_title='Predator/Prey Ratio')
st.plotly_chart(fig3, use_container_width=True)
st.caption("This plot shows how the predator/prey ratio evolves over time.")

df = pd.DataFrame({'Time': t, 'Prey': prey, 'Predator': predator, 'Ratio': ratio})

st.download_button(
    label="Download Data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='lotka_volterra_data.csv',
    mime='text/csv'
)

st.markdown("""
---
#### How to use
- Adjust parameters in the sidebar.
- Download simulation data using the button above.
- All plots are interactive and can be zoomed or saved as images.
""")
