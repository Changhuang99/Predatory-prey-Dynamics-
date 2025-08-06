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

with st.sidebar:
    st.header("Model Parameters")
    alpha = st.slider("Prey Birth Rate (α)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    beta = st.slider("Predation Rate (β)", min_value=0.001, max_value=1.0, value=0.1, step=0.001, format="%.3f")
    gamma = st.slider("Predator Death Rate (γ)", min_value=0.1, max_value=2.0, value=1.5, step=0.1)
    delta = st.slider("Energy Conversion Rate (δ)", min_value=0.001, max_value=1.0, value=0.075, step=0.001, format="%.3f")
    prey0 = st.number_input("Initial Prey Population", min_value=1, value=40, step=1)
    predator0 = st.number_input("Initial Predator Population", min_value=1, value=9, step=1)

    if st.button("Reset to Default"):
        st.experimental_rerun()

def lotka_volterra(prey0, predator0, alpha, beta, delta, gamma, t):
    dt = t[1] - t[0]
    prey = np.zeros_like(t)
    predator = np.zeros_like(t)
    prey[0] = prey0
    predator[0] = predator0
    for i in range(1, len(t)):
        prey[i] = prey[i-1] + (alpha * prey[i-1] - beta * prey[i-1] * predator[i-1]) * dt
        predator[i] = predator[i-1] + (delta * prey[i-1] * predator[i-1] - gamma * predator[i-1]) * dt
    return prey, predator

t = np.linspace(0, 50, 1000)
prey, predator = lotka_volterra(prey0, predator0, alpha, beta, delta, gamma, t)
ratio = predator / prey
ratio[np.isnan(ratio) | np.isinf(ratio)] = 0

col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=prey, name='Prey'))
    fig1.add_trace(go.Scatter(x=t, y=predator, name='Predator'))
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
