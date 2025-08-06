import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os


ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")


attendance_dir = "Attendance"

attendance_file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")


if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)
    st.info(f"Created directory: {attendance_dir}")


if not os.path.exists(attendance_file_path):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv(attendance_file_path, index=False)
    st.info(f"Created new attendance file for today: {attendance_file_path}")
else:
    
    df = pd.read_csv(attendance_file_path)


st.title("Attendance Dashboard")

from streamlit_autorefresh import st_autorefresh


count = st_autorefresh(interval=2000, key="fizzbuzzcounter")

if count == 0:
    st.write("Refreshing...")

st.subheader("Today's Attendance")
st.dataframe(df.style.highlight_max(axis=0))