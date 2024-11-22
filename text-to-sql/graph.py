import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [5, 4, 3, 2, 1]
})

# Create a Seaborn plot
fig, ax = plt.subplots()
sns.scatterplot(data=data, x='x', y='y', ax=ax)

# Display the plot in Streamlit
st.pyplot(fig)
