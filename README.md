🚗 Smart Parking System

An AI-powered smart parking monitoring system that detects vehicles in real-time and tracks parking occupancy using Computer Vision.

This project combines YOLOv8 + ByteTrack for vehicle detection and tracking, a Flask backend API for real-time data processing, and a Next.js dashboard for live monitoring and analytics.

🌟 What It Does

Detects vehicles in real-time

Tracks parking occupancy and available slots

Streams live camera feed

Triggers an alert when the parking lot is full

Estimates wait time based on historical data

Displays everything in a clean, modern dashboard

🛠 Tech Stack

Backend

Python

Flask

YOLOv8

ByteTrack

OpenCV

Frontend

Next.js

TypeScript

Tailwind CSS

📊 How It Works

Video input is processed using YOLOv8 for vehicle detection.

ByteTrack tracks vehicles across frames.

The system calculates parking occupancy in real-time.

Data is exposed through a Flask API.

The Next.js dashboard displays live stats and predictions.

🚀 Key Highlights

Real-time AI inference

Live MJPEG video streaming

Parking statistics and event history

CSV-based prediction system

Full-lot alarm notification

📂 Project Structure

VechileParkingSystem → Backend (AI + API)

parking_ui → Frontend Dashboard

🔔 Note

GPU acceleration (CUDA) is optional for faster inference.
Alarm system works on Windows using system sound.

📜 License

MIT License
