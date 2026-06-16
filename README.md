# Automated Exam Management System

**AI2002 — Artificial Intelligence | Module 3: Unsupervised Learning**


---

## Description

An automated exam seating and faculty allocation system that uses **K-Means Clustering** to intelligently group students by domain and batch, then assigns them to exam halls across **3 shifts** (Morning, Evening, Night). Every student is guaranteed a seat — no waiting list.

Includes a full **Tkinter GUI dashboard** for visualization.

---

## AI Technique Used

| Technique | Description |
|-----------|-------------|
| K-Means Clustering | Groups students by domain, batch, and GPA similarity |
| Elbow Method | Automatically finds optimal number of clusters (K) |
| StandardScaler | Normalizes features before clustering |

---

## Features

- Generates **1,800+ student records** across 5 domains and 5 batches
- Determines optimal K using the Elbow Method
- Assigns all students to rooms across 3 shifts — no one left out
- Allocates faculty supervisors per shift per room
- Exports results to `seating_plan.csv` and `faculty_allocation.csv`
- Interactive **Tkinter dashboard** with:
  - Elbow curve chart
  - Cluster scatter plot
  - Shift distribution bar chart
  - Full searchable seating table

---

## Domains Covered

- Computer Science
- Artificial Intelligence
- Business Analytics
- Software Engineering
- Electrical Engineering

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python exam_management_system_M3.py
```

The program will:
1. Generate student data
2. Run K-Means clustering
3. Assign shifts and seats
4. Save CSV output files
5. Launch the GUI dashboard

---

## Tech Stack

- Python 3.x
- NumPy, Pandas
- scikit-learn (KMeans, StandardScaler)
- Matplotlib
- Tkinter (GUI)

---

## License

MIT License
