import plotly.express as px

# Data from the provided JSON
categories = ["Needs", "Wants", "Savings & Inv"]
amounts = [50000, 30000, 20000]

# Create pie chart
fig = px.pie(values=amounts, 
             names=categories,
             title="50/30/20 Budget (₹100k Income)")

# Update layout for pie chart specifications  
fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')

# Save the chart
fig.write_image("budget_pie_chart.png")