"""
Oura Recovery Trends and Correlations Analyzer
CS50P Final Project by Elliot Lee
Github Username: ehlee29
EdX Username: ehl33
From West Chester, PA in United States of America (but currently in NYC)
Date Recorded: July 5th, 2026

Reads an Oura Ring data export (CSV) and produces a summary report: average metrics, correlations between metrics, best days, and flagged anomalies (e.g. unusually short sleep).

Usage:
    python project.py oura_data.csv
"""

import argparse
import sys
import csv
import math
import matplotlib.pyplot as plt

NUMERIC_COLUMNS = [
    "Sleep Score", "Readiness Score", "Activity Score", "Average HRV",
    "Lowest Resting Heart Rate", "Average Resting Heart Rate", "Respiratory Rate",
    "Total Sleep Duration", "Deep Sleep Duration", "REM Sleep Duration",
    "Sleep Efficiency", "Steps"
]

def main():
    # Flow:
    #   1. Parse the CSV filename from the command line.
    #   2. Read the data with read_data().
    #   3. Print a report using the helper functions below.
    parser = argparse.ArgumentParser(description="Read an Oura CSV and extract key points and trends.")
    parser.add_argument("filename", help="name of Oura CSV file in [filename].csv format")

    args = parser.parse_args()

    rows = read_data(args.filename)

    general_summary(rows)

    target = input("Category to examine correlations with: ")
    plot_correlation(rows, target, NUMERIC_COLUMNS)

def general_summary(rows):
    """
    Prints the general summary data.
    """
    print(f"Days of data: {len(rows)}")
    print("")

    print("Average Sleep Metrics Summary")
    print(f"Average Readiness Score: {average_metric(rows, 'Readiness Score'):.1f}")
    print(f"Average Sleep Score: {average_metric(rows, 'Sleep Score'):.1f}")
    print(f"Average sleep: {seconds_to_hours(average_metric(rows, 'Total Sleep Duration'))} hr(s)")
    print(f"Average sleep effiency: {average_metric(rows, 'Sleep Efficiency'):.1f}")

    print("")

    print(f"Average sleep latency: {seconds_to_hours(average_metric(rows, 'Sleep Latency'))} hr(s)")
    print(f"Average awake time: {seconds_to_hours(average_metric(rows, 'Awake Time'))} hr(s)")
    print(f"Average respiratory rate: {average_metric(rows, 'Respiratory Rate'):.1f}")
    print(f"Average temperature trend: {average_metric(rows, 'Temperature Trend Deviation'):.1f}")
    print("")


    print("Average Activity Metrics Summary")
    print(f"Average steps/day: {average_metric(rows, 'Steps'):.0f} steps")
    print(f"Average Activity Score: {average_metric(rows, 'Activity Score'):.1f}")
    print("")

    print(f"Best Readiness Score day: {best_day(rows, 'Readiness Score')}")
    print(f"Best Sleep Score day: {best_day(rows, 'Sleep Score')}")
    print(f"Best Steps day: {best_day(rows, 'Steps')}")
    print("")

    short_nights = flag_short_sleep(rows, min_hours=6)
    if short_nights:
        print(f"Short-sleep days ({len(short_nights)}):", ", ".join(short_nights))

# general_summary funcs

def read_data(filename):
    """
    Reads the Oura CSV into a list of dictionaries (one dict per row).

    Each dict maps column name -> raw str value, e.g.
        {"date": "2026-06-07", "Activity Score": "89",
         "Average HRV": "25", "Total Sleep Duration": "28500", ...}

    Returns data in str.
    """
    data = list()
    try:
        with open(filename, encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                sys.exit("File is empty!")
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                data.append({
                        "date" : row["date"],
                        "Sleep Latency" : row["Sleep Latency"],
                        "Respiratory Rate" : row["Respiratory Rate"],
                        "Awake Time" : row["Awake Time"],
                        "Average Resting Heart Rate" : row["Average Resting Heart Rate"],
                        "REM Sleep Duration" : row["REM Sleep Duration"],
                        "Deep Sleep Duration" : row["Deep Sleep Duration"],
                        "Light Sleep Duration" : row["Light Sleep Duration"],
                        "Sleep Efficiency" : row["Sleep Efficiency"],
                        "Temperature Trend Deviation" : row["Temperature Trend Deviation"],
                        "Temperature Deviation (°C)" : row["Temperature Deviation (°C)"],
                        "Lowest Resting Heart Rate" : row["Lowest Resting Heart Rate"],
                        "Activity Score" : row["Activity Score"],
                        "Total Sleep Duration" : row["Total Sleep Duration"],
                        "Total Bedtime" : row["Total Bedtime"],
                        "Average HRV" : row["Average HRV"],
                        "Sleep Score" : row["Sleep Score"],
                        "Readiness Score" : row["Readiness Score"],
                        "Steps" : row["Steps"]
                        })
    except FileNotFoundError:
        sys.exit(f"Could not find a file named {filename}")

    return data

def seconds_to_hours(seconds):
    """
    Converts a sleep duration in seconds to hours, rounded to 2 decimals.
    """
    return round(seconds/3600, 2)

def none_check(value):
    if value in ("None", ""):
        return None
    return float(value)

def average_metric(rows, column):
    """
    Returns the mean (float) of one column across all rows.

    `rows` is the list of dicts from read_data(); `column` is a key like
    "Steps" or "Readiness Score".
    """
    values = [none_check(row[column]) for row in rows]
    values = [v for v in values if v is not None]
    return sum(values) / len(values)

def flag_short_sleep(rows, min_hours):
    """
    Returns a list of dates (strings) where total sleep was below
    `min_hours`. Catches partial/missing nights.
    """
    dates = [
        row["date"] for row in rows if (none_check(row["Total Sleep Duration"]) is not None) and (seconds_to_hours(float(row["Total Sleep Duration"])) < min_hours)
    ]

    return dates

def best_day(rows, column):
    """
    Returns the date (string) of the row with the highest value in
    `column`.
    """
    best_date = ""
    highest_value = float("-inf")
    for row in rows:
        row_value = none_check(row[column])
        if row_value is None:
            continue
        if highest_value < row_value:
            highest_value = row_value
            best_date = row["date"]

    return best_date

# correlation funcs

def correlation(rows, col_a, col_b):
    """
    Returns the Pearson correlation coefficient (float, from -1.0 to 1.0) between two columns of col_a and col_b.
    The Pearson correlation coefficient is a statistical measure that quantifies the strength and direction of the linear relationship between two variables.
    Positive indicates a direct relationship, negative indicates an inverse relationship. As magnitude approaches 1, the relationship becomes stronger.

    Pearson r:
        r = sum((x - x_mean)(y - y_mean))
            / sqrt(sum((x - x_mean)**2) * sum((y - y_mean)**2))
    """
    xs = list() # --> stores first col_a
    ys = list() # --> stores second col_b

    for row in rows:
        if none_check(row[col_a]) != None and none_check(row[col_b]) != None:
            xs.append(float(row[col_a]))
            ys.append(float(row[col_b]))
        else:
            continue

    # now we want to calculate r, so we need avgs and individual data points

    xs_avg = sum(xs) / len(xs)
    ys_avg = sum(ys) / len(ys)
    num_sum = 0
    denom_sum_x = 0
    denom_sum_y = 0

    for i in range(len(xs)):
        num_sum += (xs[i] - xs_avg) * (ys[i] - ys_avg)
        denom_sum_x += ((xs[i] - xs_avg) ** 2)
        denom_sum_y += ((ys[i] - ys_avg) ** 2)

    return num_sum / math.sqrt(denom_sum_x * denom_sum_y)

def correlate_against(rows, target, columns):
    """
    Returns a list of (column, r) tuples, sorted ascending by r.
    """

    pc_values = [
        (comparison_column, correlation(rows, comparison_column, target)) for comparison_column in columns if comparison_column != target
    ]

    pc_values.sort(key=lambda pair: pair[1])
    return pc_values

def plot_correlation(rows, target, columns):
    """
    Draws a horizontal bar chart of how every column correlates with
    `target`.
    """

    if target not in columns:
        sys.exit("Invalid target column.")

    pairs = correlate_against(rows, target, columns)
    column_names = [pair[0] for pair in pairs]
    pc_values = [pair[1] for pair in pairs]
    colors = ["b" if r >= 0 else "r" for r in pc_values]

    fig, ax = plt.subplots(figsize=(9,6))
    ax.barh(column_names, pc_values, color=colors)
    ax.set_xlim(-1.0,1.0)
    ax.axvline(color="k", linewidth=1)
    ax.bar_label(ax.barh(column_names, pc_values, color=colors), fmt="%.2f")
    ax.set_title(f"Correlations with {target}")
    fig.tight_layout()
    fig.savefig(f"correlations_{target}.png")

if __name__ == "__main__":
    main()

