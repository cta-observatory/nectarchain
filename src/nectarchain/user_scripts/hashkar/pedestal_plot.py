import ast
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_list(value):
    return ast.literal_eval(value)


df = pd.read_csv(
    "pedestal_data.txt",
    delimiter="\t",
    converters={
        "Mean_ped_Per_Pix": parse_list,
        "Mean_RMS_Per_Pix": parse_list,
    },  # Use the parse_list function to convert the string to list
)
print(df["Run"])

temps = [5, 0, -5]
NSB_INT = [0]
NSB_INT = np.array(NSB_INT)

temp_5 = [3696, 3697, 3698, 3699]
temp_0 = [3751, 3752, 3753, 3754]
temp_M5 = [3785, 3786, 3787, 3788]

NSB0 = [3696, 3751, 3785]
NSB30 = [3697, 3752, 3786]
NSB40 = [3698, 3753, 3787]
NSB70 = [3699, 3754, 3788]


def categorize_run1(run):
    if run in NSB0:
        return "No source"
    if run in NSB30:
        return "18 mA"
    if run in NSB40:
        return "35 mA"
    if run in NSB70:
        return "77 mA"


def categorize_run2(run):
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


df["NSB_Label"] = df["Run"].apply(categorize_run1)
df["Temperature"] = df["Run"].apply(categorize_run2)

print(df)


# Plot Mean_RMS_Value vs Temperature, separated by NSB_Label
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Temperature", y="Mean_RMS", hue="NSB_Label", marker="o")

# Add plot labels and title
plt.xlabel("Temperature (°C)")
plt.ylabel("Mean pedestal RMS (p.e.)")
plt.title("Mean Pedestal RMS Value as a Function of Temperature")
plt.legend(title="NSB source intensity")
plt.grid(True)

# Show the plot
plt.savefig("mean_rms_pedestal_all.png")
plt.show()
plt.close()


# Plot Mean_RMS_Value vs Temperature, separated by NSB_Label
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Temperature", y="Mean_ped", hue="NSB_Label", marker="o")

# Add plot labels and title
plt.xlabel("Temperature (°C)")
plt.ylabel("Mean pedestal value (p.e.)")
plt.title("Mean Pedestal Value as a Function of Temperature")
plt.legend(title="NSB source intensity")
plt.grid(True)

# Show the plot
plt.savefig("mean_mean_pedestal_all.png")
plt.show()
plt.close()


filtered_df1 = df[df["NSB_Label"] == "No source"]
# Loop through unique temperatures and plot each subset with a different color
for temperature in filtered_df1["Temperature"].unique():
    subset = filtered_df1[filtered_df1["Temperature"] == temperature]

    # Plot Mean_ped_Per_Pix as y, with x going from 1 to the length of Mean_ped_Per_Pix for each row
    for i, mean_values in enumerate(subset["Mean_ped_Per_Pix"]):
        x_values = range(
            1, len(mean_values) + 1
        )  # x-axis values from 1 to length of the list
        plt.plot(
            x_values, mean_values, label=f"{temperature}°C" if i == 0 else "", alpha=0.7
        )

# Add labels and legend
plt.xlabel("Pixel Index")
plt.ylabel("Mean pedestal per pixel")
plt.title("Mean pedestal per pixel for dark pedestals at different temperatures")
plt.legend(title="Temperature")
plt.savefig("mean_pedestal_per_pixel.png")
plt.show()
plt.close()


filtered_df2 = df[df["NSB_Label"] == "No source"]
# Loop through unique temperatures and plot each subset with a different color
for temperature in filtered_df2["Temperature"].unique():
    subset = filtered_df2[filtered_df2["Temperature"] == temperature]

    # Plot Mean_ped_Per_Pix as y, with x going from 1 to the length of Mean_ped_Per_Pix for each row
    for i, mean_values in enumerate(subset["Mean_RMS_Per_Pix"]):
        x_values = range(
            1, len(mean_values) + 1
        )  # x-axis values from 1 to length of the list
        plt.plot(
            x_values, mean_values, label=f"{temperature}°C" if i == 0 else "", alpha=0.7
        )

# Add labels and legend
plt.xlabel("Pixel Index")
plt.ylabel("Mean pedestal RMS per pixel")
plt.title("Mean pedestal RMS per pixel for dark pedestals at different temperatures")
plt.legend(title="Temperature")
plt.savefig("mean_pedestal_RMS_per_pixel.png")
plt.show()
plt.close()

# NEW plot mean value or RMS for each pixel (on the x axis there wil be pixel number)... for each temperature for 1 type of run
