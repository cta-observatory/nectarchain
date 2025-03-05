import ast
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_list(value):
    return ast.literal_eval(value)


df = pd.read_csv("deadtime_data.txt", delimiter="\t")


temps = [5, 0, -5]


temp_5 = [3702, 3701, 3700]
temp_0 = [3765, 3766, 3767]
temp_M5 = [3799, 3800, 3801]

Source2 = [3702, 3701, 3765, 3766, 3799, 3800]
Source1 = [3700, 3767, 3801]
Source0 = []


def categorize_run1(run):
    if run in Source2:
        return "LASER"
    if run in Source1:
        return "NSB"
    if run in Source0:
        return "FF"


def categorize_run2(run):
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


df["Source_Label"] = df["Run"].apply(categorize_run1)
df["Temperature"] = df["Run"].apply(categorize_run2)
df.columns = (
    df.columns.str.strip()
)  # This removes any leading/trailing whitespace from column names

print(df)

# Plot Mean_RMS_Value vs Temperature, separated by NSB_Label
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="Collected_trigger_rate",
    y="Measured_trigger_rate",
    hue="Temperature",
    marker="o",
)

# Add plot labels and title
plt.xlabel("Collected trigger rate (khz)")
plt.ylabel("Collected trigger rate")
plt.legend(title="Temperature (°C)")
plt.grid(True)

# Show the plot
plt.savefig("deadtime_plot1.png")
plt.show()
plt.close()


# Plot Mean_RMS_Value vs Temperature, separated by NSB_Label
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="Collected_trigger_rate",
    y="Deadtime_value",
    hue="Temperature",
    marker="o",
)

# Add plot labels and title
plt.xlabel("Collected trigger rate (khz)")
plt.ylabel("Deadtime value")
plt.legend(title="Temperature (°C)")
plt.grid(True)

# Show the plot
plt.savefig("deadtime_plot2.png")
plt.show()
plt.close()
