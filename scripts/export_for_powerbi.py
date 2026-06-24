import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from data_ingestion import df

## master Analytical table

master = df.copy()

# Add all engineered columns

master['Season'] = master['Month'].map({
    12:'Winter',1:'Winter',2:'Winter',
    3:'Summer',4:'Summer',5:'Summer',
    6:'Monsoon',7:'Monsoon',8:'Monsoon',
    9:'Post-Monsoon',10:'Post-Monsoon',11:'Post-Monsoon'})

master['Target_Achievement_%'] = (
    master['Actual_Units_Sold'] / master['Sales_Target'] * 100).round(2)
master['Inventory_Turnover'] = (
    master['Actual_Units_Sold'] / master['Warehouse_Quantity']).round(4)
master['Revenue_Per_Unit'] = (
    master['Total_Revenue'] / master['Actual_Units_Sold']).round(2)
master['Cost_Per_Unit'] = (
    master['Total_Cost'] / master['Actual_Units_Sold']).round(2)
master['Is_Profitable'] = master['Profit_Margin_%'] > 0
master['KPI_Margin_Status'] = pd.cut(
    master['Profit_Margin_%'],
    bins=[-100, -10, 0, 5, 15, 100],
    labels=['Critical Loss','Loss','Marginal','Healthy','Excellent'])
master['Date_Formatted'] = pd.to_datetime(
    master['Year'].astype(str) + '-' + master['Month'].astype(str).str.zfill(2) + '-01')


# Save master table
master.to_csv('outputs/master_analytics_table.csv', index=False)
print(f"✅ Saved: master_analytics_table.csv — {master.shape}")

# KPI Summary Table

kpi_summary = master.groupby('Year').agg(
    Total_Revenue=('Total_Revenue','sum'),
    Total_Cost=('Total_Cost','sum'),
    Avg_Profit_Margin=('Profit_Margin_%','mean'),
    Avg_Days_to_Sell=('Days_to_Sell_Inventory','mean'),
    Avg_Farmer_Sentiment=('Farmer_Sentiment_Score','mean'),
    Avg_Target_Achievement=('Target_Achievement_%','mean'),
    Avg_Inventory_Turnover=('Inventory_Turnover','mean')
).round(2)
kpi_summary['Revenue_Crores'] = (kpi_summary['Total_Revenue'] / 1e7).round(2)
kpi_summary.to_csv('outputs/kpi_summary_by_year.csv')
print("✅ Saved: kpi_summary_by_year.csv")