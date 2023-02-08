---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%matplotlib inline

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.signal import find_peaks

from astropy.timeseries import TimeSeries, LombScargle
from astropy.time import TimeDelta, Time
from astropy.visualization import time_support
# from astropy.utils.introspection import minversion
from astropy import units as u

from DIRAC.Interfaces.API.Dirac import Dirac
```

Retrieve monitoring file from DIRAC

```python
# Date
date = Time('2022-11-28', format='iso')
```

```python
dirac = Dirac()
sqlite_file = f"nectarcam_monitoring_db_{date.to_datetime().strftime('%Y-%m-%d')}.sqlite"
lfns = [f"/vo.cta.in2p3.fr/nectarcam/{date.to_datetime().strftime('%Y')}/{date.to_datetime().strftime('%Y%m%d')}/{sqlite_file}"]

tmpdir = f"/tmp/{os.environ['USER']}/scratch"
if not os.path.isdir(tmpdir):
    print(f'{tmpdir} does not exist yet, I will create it for you')
    os.makedirs(tmpdir)
if not os.path.isfile(f'{tmpdir}/{sqlite_file}'):
    dirac.getFile(lfn=lfns,destDir=tmpdir,printOutput=True)
```

Open SQLite file as Pandas DataFrame

```python
temp_sensor = 4

conn = sqlite3.connect(f'{tmpdir}/{sqlite_file}')
sql_query = f"""select time, temp_{temp_sensor:02} from monitoring_darkroom order by time"""
data = pd.read_sql(sql_query, conn, parse_dates=['time'])
```

Convert Pandas DataFrame into an astropy time series

```python
df = pd.DataFrame()
df['temp'] = data[f'temp_{temp_sensor:02}']
df.set_index(pd.DatetimeIndex(data.time), inplace=True)
```

```python
ts = TimeSeries.from_pandas(df)
ts['temp'] *= u.Celsius
```

Plot the darkroom temperature

```python
with time_support(format='iso', scale='utc', simplify=True):
    fig, ax = plt.subplots()
    ax.plot(ts.time.iso,
            ts['temp'])
    ax.set_xlabel('Time')
    ax.set_ylabel(f"Temperature ({ts['temp'].unit})")
    plt.xticks(rotation=45)
    plt.title(f'Darkroom temperature, sensor {temp_sensor:02}')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
```

Lomb-Scargle periodogram

```python
frequency, power = LombScargle(ts['time'], ts['temp']).autopower()
```

```python
duration = ts.time.max() - ts.time.min()
mask = frequency < 100 * 1./(duration)
frequency = frequency[mask]
power = power[mask]
```

Find peaks of periodogram

```python
peaks, _ = find_peaks(power, height=0.2*np.max(power.value))
```

```python
fig, ax = plt.subplots()
ax.plot(frequency, power)
ax.plot(frequency[peaks], power[peaks], "x")
ax.set_xlabel(f"Frequency ({frequency.unit})")
```

```python
idx_max = np.argmax(power)
print(f'Peak time simply from max(power): {(1./frequency[idx_max]).to(u.min):.3f}')
print(f'Peak time from find_peaks: {(1./frequency[peaks]).to(u.min)}')
```

```python

```
