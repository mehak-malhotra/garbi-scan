import os
import time
import pandas as pd
import threading
import cv2
import json
# import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from plyer import notification
import pygame
from ultralytics import YOLO  
import csv
# from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import plotly.express as px
import plotly.io as pio
import pandas as pd

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO(r"runs\detect\train6\weights\best.pt")
model.fuse()

# Store last alert time for cooldown
last_alert_time = {}
garbage_time_tracker = {}  # {(lat, lon): last_time_above_threshold}
cleaning_cooldown_tracker = {}  # {(lat, lon): last_cleaning_time}
pygame.mixer.init()
ALARM_SOUND_PATH = "static/alarm.mp3"
# Define locations with specific video files
locations = [
    {"name": "Main Gate", "lat": 30.515, "lon": 76.659, "video": "static/testvideo/test.mp4"},
    {"name": "Hostel Block", "lat": 30.517, "lon": 76.661, "video": "static/testvideo/hostel.mp4"},
    {"name": "Cafeteria", "lat": 30.519, "lon": 76.660, "video": "static/testvideo/maingate.mp4"}  # Default test video
]

def load_data():
    df = pd.read_csv('notifications.csv')
    df['Garbage Level (%)'] = df['Garbage Level (%)'].str.rstrip('%').astype(float)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def create_graph1(df):
    fig = px.line(df, x='Time', y='Garbage Level (%)', color='Location', title='Garbage Level Over Time')
    return pio.to_json(fig)  # Correct way to convert Plotly figure to JSON

def create_graph2(df):
    avg_garbage = df.groupby('Location')['Garbage Level (%)'].mean().reset_index()
    fig = px.bar(avg_garbage, x='Location', y='Garbage Level (%)', title='Average Garbage Level by Location')
    return pio.to_json(fig)  # Correct way to convert Plotly figure to JSON

def create_graph3(df):
    fig = px.scatter(df, x='Longitude', y='Latitude', size='Garbage Level (%)', color='Location', title='Garbage Levels by Location')
    return pio.to_json(fig)  # Correct way to convert Plotly figure to JSON

@app.route('/')
def dashboard():
    df = load_data()
    graph1_json = create_graph1(df)
    graph2_json = create_graph2(df)
    graph3_json = create_graph3(df)
    return render_template('index.html', graph1_json=graph1_json, graph2_json=graph2_json, graph3_json=graph3_json)

@app.route('/get_graphs')
def get_graphs():
    df = load_data()

    graph1_json = create_graph1(df)
    graph2_json = create_graph2(df)
    graph3_json = create_graph3(df)

    return jsonify({
        "graph1": graph1_json,
        "graph2": graph2_json,
        "graph3": graph3_json
    })

def analyze_garbage(lat, lon, video_path, return_coordinates=False):
    """
    Processes the video to detect garbage.

    - If return_coordinates=True: Returns a list of detected garbage bounding boxes [(lat, lon, x1, y1, x2, y2), ...].
    - Otherwise: Returns (selected_total_area, selected_garbage_area, garbage_percentage).
    """

    # print(f"🔍 Analyzing garbage for ({lat}, {lon}) from video: {video_path}")  # Debugging

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # print(f"❌ Error: Could not open video for ({lat}, {lon})")
        return [] if return_coordinates else (0, 0, 0)

    total_garbage_area = 0
    selected_garbage_area = 0
    detected_objects = []  # Store garbage bounding boxes

    selected_area = selected_areas.get((lat, lon), None)  # Fetch selected area
    sx1, sy1, sx2, sy2 = selected_area if selected_area else (None, None, None, None)
    frame_area = None  # To compute total frame size once
    frame_count = 0  # Track number of frames processed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment frame count

        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]  # Calculate once

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)

        for result in results:
            for det in result.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                garbage_area = (x2 - x1) * (y2 - y1)
                total_garbage_area += garbage_area  # Total detected garbage

                detected_objects.append((lat, lon, int(x1), int(y1), int(x2), int(y2)))  # Store full data

                # Debugging: Print detected garbage box
                # print(f"🗑️ Garbage detected at ({lat}, {lon}) -> x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}")

                if selected_area:
                    # Compute Intersection (Common Area)
                    inter_x1 = max(x1, sx1)
                    inter_y1 = max(y1, sy1)
                    inter_x2 = min(x2, sx2)
                    inter_y2 = min(y2, sy2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:  
                        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        selected_garbage_area += intersection_area  

    cap.release()  # Release the video capture after processing

    # print(f"✅ Finished analyzing ({lat}, {lon}), processed {frame_count} frames.")
    # print(f"📊 Total garbage detections: {len(detected_objects)}")

    # If return_coordinates=True, return detected bounding boxes
    if return_coordinates:
        return detected_objects  

    # Otherwise, return calculated garbage percentages
    selected_total_area = (sx2 - sx1) * (sy2 - sy1) if selected_area else frame_area
    garbage_percentage = (selected_garbage_area / selected_total_area) * 100 if selected_total_area > 0 else 0

    return selected_total_area, selected_garbage_area, garbage_percentage  # Return all three values

def send_notification(location_name, lat, lon, garbage_percentage, threshold):
    """Sends a desktop notification and saves it to notifications.csv."""
    print("Notification sent")
    message = f"🚨 Garbage Alert: {garbage_percentage:.2f}% (Threshold: {threshold}%)"
    
    notification.notify(
        title=f"Garbi Scan Alert - {location_name}",
        message=message,
        timeout=10  
    )

    # Save to 
    # .csv
    csv_filename = "notifications.csv"
    alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_alert = [location_name, lat, lon, f"{garbage_percentage:.2f}%", f"{threshold}%", alert_time]

    # Check if the file exists
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Location", "Latitude", "Longitude", "Garbage Level (%)", "Threshold (%)", "Time"])  
            writer.writerow(new_alert)
    else:
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_alert)

    # print(f"✅ Alert saved: {new_alert}")


def get_location_name(lat, lon):
    """Returns the name of a location based on latitude and longitude."""
    for loc in locations:
        if float(loc["lat"]) == float(lat) and float(loc["lon"]) == float(lon):
            return loc["name"]
    return "Unknown Location"  # Default if not found

def get_video_path(lat, lon):
    """Find the video path for a given latitude and longitude."""
    for loc in locations:
        if float(loc["lat"]) == float(lat) and float(loc["lon"]) == float(lon):
            return loc["video"]
    return None

# Store selected areas in a dictionary (lat, lon as key)
selected_areas = {}

def background_detection():
    """Continuously processes all location videos and updates the CSV instantly with only the latest detection."""
    while True:
        # print("🔄 Running background detection for all locations...")  # Debugging

        df = pd.read_csv("selected_areas.csv")  # Load selected areas

        for _, row in df.iterrows():
            lat, lon = row["Latitude"], row["Longitude"]
            video_path = get_video_path(lat, lon)  # Get video path for location

            if video_path and os.path.exists(video_path):
                # Fetch real-time garbage coordinates from analyze_garbage
                detected_objects = analyze_garbage(lat, lon, video_path, return_coordinates=True)

                if detected_objects:
                    # print(f"🔍 {len(detected_objects)} objects detected at ({lat}, {lon})")  # Debugging
                    
                    # Take only the latest detection (last entry) and ensure correct unpacking
                    latest_detection = detected_objects[-1]  

                    if len(latest_detection) >= 4:  # Ensure it has at least 4 coordinates
                        x1, y1, x2, y2 = latest_detection[:4]  # Extract first 4 values

                        # Create DataFrame with latest detection and overwrite CSV
                        latest_data = pd.DataFrame([[lat, lon, x1, y1, x2, y2]], 
                                                   columns=["Latitude", "Longitude", "x1", "y1", "x2", "y2"])
                        latest_data.to_csv("realtime_garbage_data.csv", index=False)  # Overwrite file

                        # print(f"✅ Latest garbage detection updated: {latest_data.values}")

                else:
                    print(f"⚠️ No garbage detected in ({lat}, {lon})")

            else:
                print(f"⚠️ Video not found for ({lat}, {lon})")  # Debugging

        time.sleep(60)  # Wait 1 minute before the next update

@app.route('/notifications')
def notifications():
    """Reads garbage active data from CSV and displays it."""
    alerts = []

    csv_filename = "garbage_active.csv"
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Read header (if exists)
            alerts = list(reader)  # Read all active garbage entries
    else:
        print("⚠️ No active garbage detected.")  # Debugging message

    return render_template("notifications.html", alerts=alerts, headers=["Latitude", "Longitude", "Garbage%", "Threshold%"])

@app.route("/save_selected_area", methods=["POST"])
def save_selected_area():
    try:
        data = request.get_json()
        # print("📥 Received Data:", data)  # Debugging log

        # Validate received data
        required_keys = ["x1", "y1", "x2", "y2", "lat", "lon"]
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required parameters"}), 400

        x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
        lat, lon = data["lat"], data["lon"]

        file_path = "selected_areas.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Remove existing entries for the same lat/lon
            df = df[~(((df["Latitude"].astype(float)) == float(lat)) & ((df["Longitude"].astype(float) )== float(lon)))]
            # print("existing removed")
            # print(df)
        else:
            df = pd.DataFrame(columns=["Latitude", "Longitude", "x1", "y1", "x2", "y2"])
            # print("directly added not removed")

        # Append new selected area
        new_entry = pd.DataFrame([{"Latitude": lat, "Longitude": lon, "x1": x1, "y1": y1, "x2": x2, "y2": y2}])
        df = pd.concat([df, new_entry], ignore_index=True)
        # print("nyawala",df)
        # Delete the existing file before writing new data

        if os.path.exists(file_path):
            os.remove(file_path)  # Remove the old file

        # Save updated CSV
        df.to_csv(file_path, index=False)

        # print("✅ Selected area updated successfully:", new_entry)
        return jsonify({"message": "Selected area saved successfully"}), 200

    except Exception as e:
        # print("❌ Error:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

def generate_frames(video_path, lat, lon):
    global last_alert_time

    cap = cv2.VideoCapture(video_path)
    cooldown_time = 1800  # 30 minutes

    # Ensure lat & lon are floats
    lat, lon = float(lat), float(lon)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)

        # Load selected area from CSV and ensure correct data types
        selected_areas_df = pd.read_csv("selected_areas.csv")
        selected_areas_df["Latitude"] = selected_areas_df["Latitude"].astype(float)
        selected_areas_df["Longitude"] = selected_areas_df["Longitude"].astype(float)

        selected_area = None
        selected_row = selected_areas_df[
            (selected_areas_df["Latitude"] == lat) & (selected_areas_df["Longitude"] == lon)
        ]

        if not selected_row.empty:
            selected_area = selected_row.iloc[0][["x1", "y1", "x2", "y2"]].tolist()

        detected_garbage_data = []  # Store garbage bounding boxes

        for result in results:
            for det in result.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                detected_garbage_data.append([lat, lon, x1, y1, x2, y2])

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Garbage {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save real-time garbage detections to CSV (overwriting with latest values)
        garbage_df = pd.DataFrame(detected_garbage_data, columns=["Latitude", "Longitude", "x1", "y1", "x2", "y2"])
        garbage_df.to_csv("realtime_garbage_data.csv", index=False)

        # 📊 Calculate garbage percentage using CSV-based function
        garbage_percentage = calculate_garbage_percentage(lat, lon)  # Call the function we made earlier

        threshold_value = get_threshold_for_location(lat, lon)
        current_time = time.time()
        last_alert = last_alert_time.get((lat, lon), 0)
        location_name= get_location_name(lat,lon)
        # 🚨 Trigger an alert if threshold is exceeded
        if garbage_percentage > threshold_value and (current_time - last_alert) > cooldown_time:
            alert_message = f"🚨 Alert: Garbage level {garbage_percentage:.2f}% exceeded threshold {threshold_value}%!"
            print(alert_message)
            play_alarm()
            send_notification(location_name, lat, lon, garbage_percentage, threshold_value)
            last_alert_time[(lat, lon)] = current_time

        cv2.putText(frame, f"Garbage in selected area: {garbage_percentage:.2f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/cctv')
def cctv():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    try:
        lat = float(lat)  # Convert to float
        lon = float(lon)  # Convert to float
    except (TypeError, ValueError):
        return "Error: Invalid latitude or longitude format", 400

    # Find location name & video
    location_name = None
    video_file = "static/testvideo/test1.mp4"  # Default video

    for loc in locations:
        if loc["lat"] == lat and loc["lon"] == lon:
            location_name = loc["name"]
            video_file = loc["video"]
            break

    if not location_name:
        return "Error: Location not found", 400  

    return render_template("cctv.html", lat=lat, lon=lon, location_name=location_name, locations=locations, video_file=video_file)

@app.route('/video_feed')
def video_feed():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    # print(f"Request received for lat: {lat}, lon: {lon}")  # Debugging

    video_file = "static/testvideo/test1.mp4"  # Default video
    for loc in locations:
        if str(loc["lat"]) == lat and str(loc["lon"]) == lon:
            video_file = loc["video"]
            break

    # print(f"Loading video: {video_file}")  # Debugging
    if not os.path.exists(video_file):
        # print("ERROR: Video file not found!")  # Debugging
        return "Error: Video file not found", 404

    # ✅ FIX: Lat & Lon bhi pass kar
    return Response(generate_frames(video_file, lat, lon), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_threshold', methods=['POST'])
def save_threshold():
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    threshold = request.form.get("threshold")

    csv_filename = "thresholds.csv"

    # Read existing data
    rows = []
    try:
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Read header
            rows = list(reader)  # Read remaining data
    except FileNotFoundError:
        headers = ["Latitude", "Longitude", "Threshold"]  # Default header

    # Check if the entry already exists (matching Latitude & Longitude)
    updated = False
    for row in rows:
        if row[0] == lat and row[1] == lon:  # Compare Latitude & Longitude
            row[2] = threshold  # Update threshold value
            updated = True
            break

    # If no match found, add a new row
    if not updated:
        rows.append([lat, lon, threshold])

    # Write back to CSV with correct format
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude", "Threshold"])  # Ensure header
        writer.writerows(rows)

    return redirect(url_for('map'))  # Redirect to map page

# Function to get threshold value for a given location
def get_threshold_for_location(lat, lon):
    df = pd.read_csv("thresholds.csv")  # Load CSV file

    for _, row in df.iterrows():
        if str(row["Latitude"]) == str(lat) and str(row["Longitude"]) == str(lon):  
            return row["Threshold"]

    return 30  # Default threshold if not found

def monitor_garbage_threshold():
    """Continuously checks garbage percentage, compares with threshold, and updates garbage_active.csv."""
    
    file_selected_areas = "selected_areas.csv"
    file_garbage_active = "garbage_active.csv"
    file_garbage_analysis = "garbageanalysisdata.csv"

    while True:
        # print("🔄 Checking garbage levels against thresholds...")

        # Load selected areas
        if not os.path.exists(file_selected_areas):
            # print("⚠️ selected_areas.csv not found! Retrying in 1 min...")
            time.sleep(60)
            continue

        df_selected = pd.read_csv(file_selected_areas)
        active_garbage_data = []  # Stores locations where garbage % exceeds threshold
        analysis_data = []
        for _, row in df_selected.iterrows():
            lat, lon = float(row["Latitude"]), float(row["Longitude"])

            # Get garbage percentage using your function
            garbage_percentage = calculate_garbage_percentage(lat, lon)  # ✅ Using your function!

            # Get threshold for this location
            threshold_value = get_threshold_for_location(lat, lon)

            print(f"📍 Location ({lat}, {lon}) - Garbage: {garbage_percentage:.2f}% | Threshold: {threshold_value}%")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis_data.append([lat, lon, garbage_percentage, threshold_value, timestamp])

            if garbage_percentage > threshold_value:
                # print(f"🚨 Alert! Garbage level exceeded at ({lat}, {lon})")
                active_garbage_data.append([lat, lon, garbage_percentage, threshold_value])

        # Update garbage_active.csv
        if active_garbage_data:
            df_active = pd.DataFrame(active_garbage_data, columns=["Latitude", "Longitude", "Garbage%", "Threshold%"])
            df_active.to_csv(file_garbage_active, index=False)
            # print(f"✅ garbage_active.csv updated with {len(active_garbage_data)} locations.")
        else:
            if os.path.exists(file_garbage_active):
                os.remove(file_garbage_active)  # Remove file if no active garbage
                # print("🗑️ No active garbage locations. garbage_active.csv removed.")
        if analysis_data:
            df_analysis = pd.DataFrame(analysis_data, columns=["Latitude", "Longitude", "Garbage Percentage", "Threshold", "Timestamp"])
            file_exists = os.path.exists(file_garbage_analysis)
            df_analysis.to_csv(file_garbage_analysis, mode='a', header=not file_exists, index=False)
            # print(f"📊 Garbage analysis data appended for {len(analysis_data)} locations.")
        time.sleep(10) # Run every 1 minute

def play_alarm():
    """Plays an alarm sound."""
    try:
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play()
        print("🔊 Alarm sound played.")
    except Exception as e:
        print("❌ Error playing alarm:", str(e))

# impo
# rt pandas as pd
@app.route('/location_data')
def location_data():
    return jsonify(locations)

@app.route('/garbage_status')
def garbage_status():
    """Returns garbage percentage & marker colors dynamically."""
    garbage_data = {}

    for loc in locations:
        lat, lon = loc["lat"], loc["lon"]

        # Calculate garbage percentage & threshold
        garbage_percentage = calculate_garbage_percentage(lat, lon)
        threshold = get_threshold_for_location(lat, lon)

        # Determine marker color
        if garbage_percentage <= threshold:
            color = "green"
        elif garbage_percentage <= threshold * 1.5:
            color = "orange"
        else:
            color = "red"

        garbage_data[f"{lat},{lon}"] = {"garbage_percentage": garbage_percentage, "color": color}

    return jsonify(garbage_data)

@app.route('/map')
def map_page():
    """Renders the map with dynamic locations."""
    return render_template("map.html", locations=locations)

def calculate_garbage_percentage(lat, lon):
    """
    Calculates the percentage of the selected area covered with garbage.
    """
    # Load CSVs
    selected_areas = pd.read_csv("selected_areas.csv")
    garbage_data = pd.read_csv("realtime_garbage_data.csv")

    # Convert latitude & longitude to numeric (handling possible string format issues)
    selected_areas[["Latitude", "Longitude"]] = selected_areas[["Latitude", "Longitude"]].apply(pd.to_numeric)
    garbage_data[["Latitude", "Longitude"]] = garbage_data[["Latitude", "Longitude"]].apply(pd.to_numeric)

    # Get selected area for given lat & lon
    selected_row = selected_areas[(selected_areas["Latitude"] == lat) & (selected_areas["Longitude"] == lon)]
    if selected_row.empty:
        # print(f"⚠️ No selected area found for ({lat}, {lon})")
        return 0  # No selected area, no garbage coverage

    # Extract selected area coordinates
    x1_s, y1_s, x2_s, y2_s = selected_row.iloc[0][["x1", "y1", "x2", "y2"]]

    # Calculate selected area size
    selected_area = abs((x2_s - x1_s) * (y2_s - y1_s))
    if selected_area == 0:
        # print(f"⚠️ Invalid selected area for ({lat}, {lon})")
        return 0

    # Get garbage detections for the same location
    garbage_rows = garbage_data[(garbage_data["Latitude"] == lat) & (garbage_data["Longitude"] == lon)]
    if garbage_rows.empty:
        # print(f"⚠️ No garbage detected for ({lat}, {lon})")
        return 0  # No garbage, so percentage is 0%

    total_intersection_area = 0

    # Calculate intersection for each detected garbage area
    for _, row in garbage_rows.iterrows():
        x1_g, y1_g, x2_g, y2_g = row[["x1", "y1", "x2", "y2"]]

        # Calculate intersection coordinates
        x_overlap = max(0, min(x2_s, x2_g) - max(x1_s, x1_g))
        y_overlap = max(0, min(y2_s, y2_g) - max(y1_s, y1_g))

        # Calculate intersection area
        intersection_area = x_overlap * y_overlap
        total_intersection_area += intersection_area

    # Calculate garbage coverage percentage
    garbage_percentage = (total_intersection_area / selected_area) * 100
    # print(f"📊 Garbage covers {garbage_percentage:.2f}% of the selected area at ({lat}, {lon})")
    
    return min(garbage_percentage,100)

if __name__ == '__main__':
    detection_thread = threading.Thread(target=background_detection, daemon=True)
    detection_thread.start()  # Start background thread
    monitor_thread = threading.Thread(target=monitor_garbage_threshold, daemon=True)
    monitor_thread.start()  # Start background thread
    # monitor_garbage_threshold()
    app.run(debug=True, threaded=True)  # Flask app continues running
