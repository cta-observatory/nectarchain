import sqlite3

import numpy as np
from astropy import time as astropytime
from ctapipe.io import EventSource


def main():
    sql_file_name = (
        "../../../../../nectar_cam_data/runs/nectarcam_monitoring_db_2025-07-29.sqlite"
    )
    con = sqlite3.connect(sql_file_name)
    cursor = con.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # tables = cursor.fetchall()
    # print("Tables in the database:")
    # for table in tables:
    #     print(table)

    cursor.execute("SELECT * FROM monitoring_drawer_temperatures;")
    drawer_temp = cursor.fetchall()
    cursor.close()

    con.close()

    chosen_path = "../../../../../nectar_cam_data/runs/NectarCAM.Run7213.0001.fits.fz"
    reader = EventSource(input_url=chosen_path, max_events=10000)

    trigger_times, trigger_ids = [], []

    for event in reader:
        trigger_times.append(event.trigger.time.value)
        trigger_ids.append(event.index.event_id)

    trigger_times, trigger_ids = np.array(trigger_times), np.array(trigger_ids)

    run_start = trigger_times[trigger_ids == np.min(trigger_ids)] - 100
    run_end = np.max(trigger_times) + 100

    print(run_start)
    print(run_end)

    drawer_temp = np.array(drawer_temp)
    drawer_times = np.array(drawer_temp[:, 3])

    for ii in range(len(drawer_times)):
        drawer_times[ii] = astropytime.Time(drawer_times[ii], format="iso").unix

    drawer_temp_11 = drawer_temp[:, 4][drawer_times > run_start]
    drawer_temp_21 = drawer_temp[:, 5][drawer_times > run_start]
    drawer_num_1 = drawer_temp[:, 2][drawer_times > run_start]

    drawer_times_new = drawer_times[drawer_times > run_start]

    drawer_temp_12 = drawer_temp_11[drawer_times_new < run_end]
    drawer_temp_22 = drawer_temp_21[drawer_times_new < run_end]
    drawer_num_2 = drawer_num_1[drawer_times_new < run_end]

    total_drawers = np.max(drawer_num_2)

    drawer_temp1_mean, drawer_temp2_mean = [], []
    drawer_temp_std = []

    for ii in range(total_drawers + 1):
        mask_drawer = drawer_num_2 == ii
        if len(mask_drawer) == len(drawer_temp_12):
            for _ in range(7):
                drawer_temp1_mean.append(np.mean(drawer_temp_12[mask_drawer]))
                drawer_temp2_mean.append(np.mean(drawer_temp_22[mask_drawer]))
                drawer_temp_std.append(
                    np.std([drawer_temp_12[mask_drawer], drawer_temp_22[mask_drawer]])
                )

    drawer_temp1_mean = np.array(drawer_temp1_mean)
    drawer_temp2_mean = np.array(drawer_temp2_mean)

    drawer_temp_mean = (drawer_temp1_mean + drawer_temp2_mean) / 2
    drawer_temp_std = np.array(drawer_temp_std)

    print(drawer_temp_mean)
    print(drawer_temp_std)


if __name__ == "__main__":
    main()
