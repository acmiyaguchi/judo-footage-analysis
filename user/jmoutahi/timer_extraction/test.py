import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Specify the folder path containing frames
frames_root_folder = "/cs-share/pradalier/tmp/judo/frames/"

timer_json = "combined.json"

# Load the json in a pandas dataframe
df = pd.read_json(timer_json)

# Print the first 5 rows of the dataframe
print(df.head())

# Sort by the filename
df = df.sort_values(by="filename")

df["mat_number"] = df["filename"].apply(lambda x: x[-19:-13])
df["number"] = df["filename"].apply(lambda x: x[-12:-4])
print(df.head())

# For each mat, sort the frames by the frame number in ascending order
df = df.groupby("mat_number").apply(lambda x: x.sort_values("number")).reset_index(drop=True)

# Create a new column to store the time in seconds
df["time_seconds"] = df["minutes"] * 60 + df["seconds"]
print(df.head())

# Use interpolation to fill the NaN values in the time_seconds column only if the "hole" is less than 10 frames otherwise fill with the next value that is not NaN
# df["time_seconds"] = df.groupby("mat_number")["time_seconds"].apply(lambda x: x.interpolate(limit=10))
df["time_seconds"] = df["time_seconds"].fillna(method="bfill")
print(df.head())

# Compute the time difference between consecutive frames for each mat
df["time_diff"] = df.groupby("mat_number")["time_seconds"].diff()
# Fill the NaN values with 0 and the values other than -1 and 1 with 0
df["time_diff"] = df["time_diff"].fillna(0)
df["time_diff"] = df["time_diff"].apply(lambda x: 0 if abs(x) > 1 else x)
print(df.head())

# Apply a big convolution kernel to smooth the time_seconds column
df["time_seconds_rolling"] = df.groupby("mat_number")["time_seconds"].rolling(window=100, center=True).mean().reset_index(0, drop=True)
print(df.head())

# Apply a small convolution kernel to smooth the time_seconds column
df["time_seconds_rolling_small"] = df.groupby("mat_number")["time_seconds"].rolling(window=5, center=True).mean().reset_index(0, drop=True)
print(df.head())

# Compute the time difference between consecutive frames for each mat
df["time_diff_rolling"] = df.groupby("mat_number")["time_seconds_rolling"].diff()

# Compute the time difference between consecutive frames for each mat using the small rolling mean
df["time_diff_rolling_small"] = df.groupby("mat_number")["time_seconds_rolling_small"].diff()

# Add a new column to store the match/non-match status : 1 if the rolling time difference is negative, 0 otherwise
df["match"] = df["time_diff_rolling"].apply(lambda x: 1 if x < 0 else 0)
print(df.head())

# Add a new column to store the timer_paused status : 1 if the match status is 1 and the time difference is 0, 0 otherwise
df["timer_paused"] = df["match"] * (df["time_diff_rolling_small"] == 0)

# Create a rolling window to count consecutive ones in timer_paused
rolling_ones = df.groupby("mat_number")["timer_paused"].rolling(window=10, center=True).sum().reset_index(0, drop=True)

# Remove the timer_paused status if the pause is less than 10 consecutive frames
df["timer_paused"] = df["timer_paused"] * (rolling_ones >= 10)

# Plot the time in seconds for each mat and the diff_time with different colors and scales
for mat_number in df["mat_number"].unique():
    plt.figure(figsize=(10, 5))
    
    # Plotting time_seconds on the left y-axis
    plt.plot(df[df["mat_number"] == mat_number]["time_seconds"], label="Time in seconds", color="b")
    #plt.plot(df[df["mat_number"] == mat_number]["time_seconds_rolling"], label="Time in seconds (rolling mean)", color="g")
    plt.xlabel("Frame number")
    plt.ylabel("Time (s)")
    
    # Creating a second y-axis for time_diff
    ax2 = plt.gca().twinx()
    ax2.plot(df[df["mat_number"] == mat_number]["time_diff_rolling_small"], label="Time difference", color="r")
    # ax2.set_ylabel("Time difference")
    ax2.plot(df[df["mat_number"] == mat_number]["time_diff_rolling"], label="Time difference (rolling mean)", color="y")

    # Make the plot background green match and red non-match
    plt.fill_between(df[df["mat_number"] == mat_number].index, 0, 100, where=df[df["mat_number"] == mat_number]["match"] == 1, color="g", alpha=0.3)
    plt.fill_between(df[df["mat_number"] == mat_number].index, 0, 100, where=df[df["mat_number"] == mat_number]["match"] == 0, color="r", alpha=0.3)

    # Make the plot background yellow for timer_paused
    plt.fill_between(df[df["mat_number"] == mat_number].index, 0, 100, where=df[df["mat_number"] == mat_number]["timer_paused"] == 1, color="y", alpha=0.6)

    plt.title(f"Time in seconds and time difference for mat {mat_number}")
    # plt.legend()
    plt.savefig(f"time_seconds_and_diff_mat_{mat_number}.png")
    plt.show()