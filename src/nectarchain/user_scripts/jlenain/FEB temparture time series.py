# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (nectar-dev)
#     language: python
#     name: nectar-dev
# ---

# %%
# %matplotlib inline

import os
import sqlite3

import numpy as np
import pandas as pd

# from astropy.utils.introspection import minversion
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.timeseries import LombScargle, TimeSeries
from astropy.visualization import time_support
from DIRAC.Interfaces.API.Dirac import Dirac
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.signal import find_peaks

# %%
# Date
date = Time("2022-11-28", format="iso")

dirac = Dirac()
sqlite_file = (
    f"nectarcam_monitoring_db_{date.to_datetime().strftime('%Y-%m-%d')}.sqlite"
)
lfns = [
    f"/vo.cta.in2p3.fr/nectarcam/{date.to_datetime().strftime('%Y')}/{date.to_datetime().strftime('%Y%m%d')}/{sqlite_file}"
]

tmpdir = f"/tmp/{os.environ['USER']}/scratch"
if not os.path.isdir(tmpdir):
    print(f"{tmpdir} does not exist yet, I will create it for you")
    os.makedirs(tmpdir)
if not os.path.isfile(f"{tmpdir}/{sqlite_file}"):
    dirac.getFile(lfn=lfns, destDir=tmpdir, printOutput=True)

# %%
conn = sqlite3.connect(f"{tmpdir}/{sqlite_file}")

# %%
sql_query = "SELECT name FROM sqlite_master WHERE type='table';"
data = pd.read_sql(sql_query, conn, parse_dates=["time"])

# %%
data

# %%
sql_query = (
    """select time, tfeb1, tfeb2 from monitoring_drawer_temperatures order by time"""
)
data = pd.read_sql(sql_query, conn, parse_dates=["time"])

# %%
data

# %%
df = pd.DataFrame()
df["temp1_feb"] = data["tfeb1"]
df["temp2_feb"] = data["tfeb2"]
df.set_index(pd.DatetimeIndex(data.time), inplace=True)

ts = TimeSeries.from_pandas(df)
ts["temp1_feb"] *= u.Celsius
ts["temp2_feb"] *= u.Celsius

# %%
# Plot the FEB temperatures

# %%
with time_support(format="iso", scale="utc", simplify=True):
    fig, ax = plt.subplots()
    ax.plot(ts.time.iso, ts["temp1_feb"], alpha=0.2)
    ax.plot(ts.time.iso, ts["temp2_feb"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Temperature ({ts['temp1_feb'].unit})")
    plt.xticks(rotation=45)
    plt.title(f"FEB temperatures")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

# %%
