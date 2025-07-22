import pandas as pd
# Sample 50-30-20 budget breakdown assuming net income 100000 INR per month
budget = pd.DataFrame({
    'Category':['Needs','Wants','Savings'],
    'Amount':[50000,30000,20000]
})
budget