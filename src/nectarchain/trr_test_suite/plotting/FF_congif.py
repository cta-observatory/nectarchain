# config_runs.py

# ----------------- Run lists by temperature ----------------- #
temp_25 = [7020, 7029, 7038, 7047, 7056]
temp_20 = [7077, 7086, 7095, 7104, 7113]
temp_14 = [6954, 6963, 6972, 6981, 6990]
temp_10 = [7144, 7153, 7162, 7171, 7180]
temp_5 = [6543, 6552, 6561, 6570, 6579]
temp_0 = [6672, 6681, 6690, 6699, 6708]
temp_M5 = [6729, 6738, 6747, 6756, 6765]

# ----------------- NSB categories ----------------- #
NSB0 = [7020, 7077, 6954, 7144, 6543, 6672, 6729]
NSB10 = [7029, 7086, 6963, 7153, 6552, 6681, 6738]
NSB20 = [7038, 7095, 6972, 7162, 6561, 6690, 6747]
NSB40 = [7047, 7104, 6981, 7171, 6570, 6699, 6756]
NSB70 = [7056, 7113, 6990, 7180, 6579, 6708, 6765]


# ----------------- Functions to categorize runs ----------------- #
def categorize_run1(run):
    if run in NSB0:
        return "No source"
    if run in NSB10:
        return "10.6 mA"
    if run in NSB20:
        return "20.4 mA"
    if run in NSB40:
        return "39.8 mA"
    if run in NSB70:
        return "78.8 mA"


def categorize_run2(run):
    if run in temp_25:
        return 25
    if run in temp_20:
        return 20
    if run in temp_14:
        return 14
    if run in temp_10:
        return 10
    if run in temp_5:
        return 5
    if run in temp_0:
        return 0
    if run in temp_M5:
        return -5


# ----------------- Runs and paths ----------------- #
Runs = [
    7056,
    7113,
    6990,
    6579,
    6708,
    6765,
    7047,
    7104,
    6981,
    7171,
    7171,
    6570,
    6699,
    6756,
    7038,
    7095,
    6972,
    7162,
    6561,
    6690,
    6747,
    7029,
    7088,
    6963,
    7153,
    6552,
    6681,
    6738,
    7020,
    7077,
    6954,
    7144,
    6543,
    6672,
    6729,
]

dirname = "/Users/hashkar/Desktop/20221108/FlatFieldTests"
