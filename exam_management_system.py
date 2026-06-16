"""
=====================================================================
AL-2002 PROJECT 2026 - MODULE 3: Unsupervised Learning
Automated Exam Management System using K-Means Clustering
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
CLR_BG        = "#0F1923"
CLR_PANEL     = "#1A2535"
CLR_ACCENT    = "#2979FF"
CLR_ACCENT2   = "#00E5FF"
CLR_TEXT      = "#E8EAF0"
CLR_SUBTEXT   = "#8B9AAD"
CLR_SUCCESS   = "#00C853"
CLR_WARN      = "#FFD600"
CLR_HEADER_BG = "#162032"
CLR_DANGER    = "#FF1744"

DOMAIN_COLORS = {
    "Computer Science":        "#2979FF",
    "Artificial Intelligence": "#00E5FF",
    "Business Analytics":      "#FFD600",
    "Software Engineering":    "#00C853",
    "Electrical Engineering":  "#FF6D00",
}

SHIFT_COLORS = {
    "Morning": "#2979FF",
    "Evening": "#00C853",
    "Night":   "#FFD600",
}

# ─────────────────────────────────────────────
# SECTION 1: DATA COLLECTION
# ─────────────────────────────────────────────

def collect_student_data():
    domains = ["Computer Science", "Artificial Intelligence",
               "Business Analytics", "Software Engineering",
               "Electrical Engineering"]
    batches = [19, 20, 21, 22, 23]

    domain_counts = {
        "Computer Science":        {19: 120, 20: 130, 21: 115, 22: 125, 23: 110},
        "Artificial Intelligence": {19:  60, 20:  70, 21:  75, 22:  80, 23:  85},
        "Business Analytics":      {19:  50, 20:  55, 21:  60, 22:  58, 23:  52},
        "Software Engineering":    {19: 100, 20: 105, 21: 110, 22: 115, 23: 108},
        "Electrical Engineering":  {19:  80, 20:  85, 21:  90, 22:  88, 23:  82},
    }

    records = []
    student_id = 1
    for domain in domains:
        for batch in batches:
            count = domain_counts[domain][batch]
            for _ in range(count):
                records.append({
                    "student_id":       f"{batch}F-{str(student_id).zfill(4)}",
                    "batch":            batch,
                    "domain":           domain,
                    "domain_code":      domains.index(domain),
                    "batch_normalized": batch - 19
                })
                student_id += 1

    df = pd.DataFrame(records)
    print(f"[Data Collection] Total students loaded: {len(df)}")
    print(df.groupby(["batch", "domain"]).size().unstack(fill_value=0).to_string())
    return df


def collect_room_data():
    rooms = []
    for i in range(1, 26):
        rooms.append({"room_id": f"R{str(i).zfill(2)}", "capacity": 30})
    for i in range(26, 31):
        rooms.append({"room_id": f"R{str(i).zfill(2)}", "capacity": 25})
    df_rooms = pd.DataFrame(rooms)
    total_seats = df_rooms["capacity"].sum()
    print(f"\n[Room Data] Total rooms: {len(df_rooms)} | Seats per shift: {total_seats}")
    return df_rooms


def collect_faculty_data():
    domains = ["Computer Science", "Artificial Intelligence",
               "Business Analytics", "Software Engineering",
               "Electrical Engineering"]
    faculty = []
    for i, domain in enumerate(domains):
        for j in range(8):
            faculty.append({
                "faculty_id": f"FAC-{i+1}{str(j+1).zfill(2)}",
                "name":       f"Prof_{domain.replace(' ', '_')}_{j+1}",
                "domain":     domain
            })
    df_faculty = pd.DataFrame(faculty)
    print(f"\n[Faculty Data] Total faculty: {len(df_faculty)}")
    return df_faculty


# ─────────────────────────────────────────────
# SECTION 2: PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_data(df_students):
    features = df_students[["domain_code", "batch_normalized"]].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print(f"\n[Preprocessing] Feature matrix shape: {scaled_features.shape}")
    return scaled_features, scaler


# ─────────────────────────────────────────────
# SECTION 3: K-MEANS CLUSTERING
# ─────────────────────────────────────────────

def determine_optimal_clusters(scaled_features, max_k=30):
    inertia_values = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia_values.append(kmeans.inertia_)

    second_deriv = np.diff(np.diff(inertia_values))
    optimal_k = k_range[np.argmax(second_deriv) + 2]
    print(f"\n[K-Means] Optimal k from Elbow Method: {optimal_k}")
    return optimal_k, list(k_range), inertia_values


def run_kmeans(df_students, scaled_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_students = df_students.copy()
    df_students["cluster"] = kmeans.fit_predict(scaled_features)
    print(f"\n[K-Means] Clustering complete.")
    print(df_students["cluster"].value_counts().head(10).to_string())
    return df_students, kmeans


# ─────────────────────────────────────────────
# SECTION 4: SHIFT-BASED SEATING PLAN  ← NEW
# ─────────────────────────────────────────────

SHIFTS = ["Morning", "Evening", "Night"]

def assign_shifts(df_students, df_rooms):
    """
    Divides all students into shifts so every student gets a seat.
    
    Logic:
    - Total seats per shift = sum of all room capacities (875)
    - Students sorted by cluster then domain then batch
    - First 875  → Morning shift
    - Next  875  → Evening shift
    - Remaining  → Night shift
    
    Returns df_students with 'shift' column added.
    """
    seats_per_shift = int(df_rooms["capacity"].sum())
    print(f"\n[Shift System] Seats per shift : {seats_per_shift}")

    # Sort students so same cluster/domain stay together in same shift
    df_sorted = df_students.sort_values(
        ["cluster", "domain_code", "batch"]
    ).reset_index(drop=True)

    shift_labels = []
    for i in range(len(df_sorted)):
        if i < seats_per_shift:
            shift_labels.append("Morning")
        elif i < seats_per_shift * 2:
            shift_labels.append("Evening")
        else:
            shift_labels.append("Night")

    df_sorted["shift"] = shift_labels

    counts = df_sorted["shift"].value_counts()
    for shift in SHIFTS:
        print(f"[Shift System] {shift:8s} shift : {counts.get(shift, 0)} students")

    return df_sorted


def generate_seating_plan_with_shifts(df_students_shifted, df_rooms):
    """
    Assigns rooms and seat numbers to each student within their shift.
    Every shift reuses all 30 rooms from scratch.
    
    Returns df_seating with columns:
        student_id, batch, domain, cluster, shift, room_id, seat_number
    """
    seating_records = []
    room_list     = df_rooms["room_id"].tolist()
    room_capacity = {row["room_id"]: row["capacity"] for _, row in df_rooms.iterrows()}

    for shift_name in SHIFTS:
        shift_students = df_students_shifted[
            df_students_shifted["shift"] == shift_name
        ].reset_index(drop=True)

        if len(shift_students) == 0:
            continue

        # Reset room usage for each shift
        room_usage       = {rid: 0 for rid in room_list}
        current_room_idx = 0
        current_seat     = 1

        for _, student in shift_students.iterrows():
            # Move to next room if current is full
            while current_room_idx < len(room_list):
                room_id = room_list[current_room_idx]
                if room_usage[room_id] < room_capacity[room_id]:
                    break
                current_room_idx += 1
                current_seat = 1

            if current_room_idx >= len(room_list):
                # Should never happen with correct shift sizing
                print(f"[WARNING] {shift_name} shift: rooms exhausted unexpectedly!")
                break

            room_id = room_list[current_room_idx]
            seating_records.append({
                "student_id":  student["student_id"],
                "batch":       student["batch"],
                "domain":      student["domain"],
                "cluster":     student["cluster"],
                "shift":       shift_name,
                "room_id":     room_id,
                "seat_number": current_seat
            })
            room_usage[room_id] += 1
            current_seat += 1

        print(f"[Seating] {shift_name:8s} shift → "
              f"{len(shift_students)} students seated across "
              f"{sum(1 for v in room_usage.values() if v > 0)} rooms")

    df_seating = pd.DataFrame(seating_records)

    total_seated = len(df_seating)
    total_students = len(df_students_shifted)

    print(f"\n[Seating Plan] Total students seated : {total_seated} / {total_students}")
    if total_seated == total_students:
        print("[Seating Plan] ✓ ALL students successfully seated — no waiting list!")
    else:
        print(f"[Seating Plan] ⚠ {total_students - total_seated} students unassigned")

    return df_seating


# ─────────────────────────────────────────────
# SECTION 5: FACULTY ALLOCATION
# ─────────────────────────────────────────────

def allocate_faculty(df_seating, df_faculty, df_rooms):
    """
    Allocates faculty per room per shift.
    Each shift × room combination gets one faculty member.
    """
    faculty_assignments = []

    for shift_name in SHIFTS:
        shift_seating = df_seating[df_seating["shift"] == shift_name]
        if len(shift_seating) == 0:
            continue

        faculty_pool = df_faculty.copy()
        faculty_pool["assigned"] = False
        rooms_used = shift_seating["room_id"].unique()

        for room_id in rooms_used:
            room_students   = shift_seating[shift_seating["room_id"] == room_id]
            dominant_domain = room_students["domain"].value_counts().idxmax()

            available = faculty_pool[
                (faculty_pool["domain"] == dominant_domain) &
                (~faculty_pool["assigned"])
            ]
            if len(available) == 0:
                available = faculty_pool[~faculty_pool["assigned"]]
            if len(available) == 0:
                # All assigned — reuse from domain
                available = faculty_pool[faculty_pool["domain"] == dominant_domain]

            chosen = available.iloc[0]
            faculty_pool.loc[chosen.name, "assigned"] = True

            faculty_assignments.append({
                "shift":                shift_name,
                "room_id":              room_id,
                "faculty_id":           chosen["faculty_id"],
                "faculty_name":         chosen["name"],
                "faculty_domain":       chosen["domain"],
                "room_dominant_domain": dominant_domain
            })

    df_allocation = pd.DataFrame(faculty_assignments)
    print(f"\n[Faculty Allocation] Total assignments: {len(df_allocation)}")
    return df_allocation


# ─────────────────────────────────────────────
# SECTION 6: CONSOLE REPORT
# ─────────────────────────────────────────────

def generate_report(df_students, df_seating, df_allocation, df_rooms, n_clusters):
    print("\n" + "=" * 60)
    print("        EXAM MANAGEMENT SYSTEM – FINAL REPORT")
    print("=" * 60)
    print(f"\n  Total Students   : {len(df_students)}")
    print(f"  Total Rooms      : {len(df_rooms)}")
    print(f"  Seats Per Shift  : {df_rooms['capacity'].sum()}")
    print(f"  Total Shifts     : {len(SHIFTS)}")
    print(f"  Clusters Formed  : {n_clusters}")
    print(f"  Students Seated  : {len(df_seating)} / {len(df_students)}")
    print(f"  Waiting List     : 0 (all seated!)")

    print("\n--- Students Per Shift ---")
    print(df_seating["shift"].value_counts().reindex(SHIFTS).to_string())

    print("\n--- Students Per Domain ---")
    print(df_students["domain"].value_counts().to_string())

    print("\n--- Students Per Batch ---")
    print(df_students["batch"].value_counts().sort_index().to_string())

    print("\n--- Faculty Allocation Sample (first 10) ---")
    print(df_allocation.head(10).to_string(index=False))

    print("\n--- Seating Plan Sample (first 10 entries) ---")
    print(df_seating.head(10).to_string(index=False))

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# SECTION 7: CHARTS
# ─────────────────────────────────────────────

def build_chart_elbow(k_range, inertia_values, optimal_k):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CLR_PANEL)
    ax.set_facecolor(CLR_BG)
    ax.plot(k_range, inertia_values, color=CLR_ACCENT2, linewidth=2,
            marker="o", markersize=4)
    ax.axvline(x=optimal_k, color=CLR_WARN, linestyle="--",
               linewidth=1.5, label=f"Optimal k = {optimal_k}")
    ax.fill_between(k_range, inertia_values, alpha=0.08, color=CLR_ACCENT)
    ax.set_xlabel("Number of Clusters (k)", color=CLR_SUBTEXT, fontsize=9)
    ax.set_ylabel("Inertia (WCSS)",         color=CLR_SUBTEXT, fontsize=9)
    ax.set_title("Elbow Method", color=CLR_TEXT, fontsize=11, fontweight="bold")
    ax.tick_params(colors=CLR_SUBTEXT)
    ax.spines[:].set_color("#2A3A50")
    ax.legend(facecolor=CLR_PANEL, edgecolor=CLR_ACCENT,
              labelcolor=CLR_TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def build_chart_domain_pie(df_students):
    domain_counts = df_students["domain"].value_counts()
    colors = [DOMAIN_COLORS[d] for d in domain_counts.index]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CLR_PANEL)
    ax.set_facecolor(CLR_PANEL)
    wedges, texts, autotexts = ax.pie(
        domain_counts.values, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(width=0.55, edgecolor=CLR_BG, linewidth=2),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color(CLR_BG); at.set_fontsize(8); at.set_fontweight("bold")
    ax.legend(wedges, [d.replace(" ", "\n") for d in domain_counts.index],
              loc="center left", bbox_to_anchor=(1, 0.5),
              facecolor=CLR_PANEL, edgecolor="none",
              labelcolor=CLR_TEXT, fontsize=7)
    ax.set_title("Students per Domain", color=CLR_TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def build_chart_batch_bar(df_students):
    batch_counts = df_students["batch"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CLR_PANEL)
    ax.set_facecolor(CLR_BG)
    bars = ax.barh([f"Batch {b}" for b in batch_counts.index],
                   batch_counts.values, color=CLR_ACCENT,
                   edgecolor="none", height=0.5)
    for bar, val in zip(bars, batch_counts.values):
        ax.text(val + 4, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=CLR_TEXT, fontsize=8)
    ax.set_xlabel("Number of Students", color=CLR_SUBTEXT, fontsize=9)
    ax.set_title("Students per Batch", color=CLR_TEXT, fontsize=11, fontweight="bold")
    ax.tick_params(colors=CLR_SUBTEXT)
    ax.spines[:].set_color("#2A3A50")
    ax.set_xlim(0, batch_counts.max() + 60)
    fig.tight_layout()
    return fig


def build_chart_cluster_scatter(df_students):
    cluster_colors = ["#2979FF", "#00E5FF", "#00C853", "#FFD600", "#FF6D00"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CLR_PANEL)
    ax.set_facecolor(CLR_BG)
    for i, cluster in enumerate(sorted(df_students["cluster"].unique())):
        subset = df_students[df_students["cluster"] == cluster]
        ax.scatter(subset["batch"], subset["domain_code"],
                   c=cluster_colors[i % len(cluster_colors)],
                   alpha=0.35, s=14, label=f"Cluster {cluster}")
    ax.set_xlabel("Batch",       color=CLR_SUBTEXT, fontsize=9)
    ax.set_ylabel("Domain Code", color=CLR_SUBTEXT, fontsize=9)
    ax.set_title("K-Means Cluster Scatter", color=CLR_TEXT,
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors=CLR_SUBTEXT)
    ax.spines[:].set_color("#2A3A50")
    ax.legend(facecolor=CLR_PANEL, edgecolor=CLR_ACCENT,
              labelcolor=CLR_TEXT, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def build_chart_room_occupancy(df_seating, df_rooms):
    """Shows room occupancy for each shift side by side."""
    fig, axes = plt.subplots(1, len(SHIFTS), figsize=(13, 3.2), sharey=True)
    fig.patch.set_facecolor(CLR_PANEL)

    for ax, shift_name in zip(axes, SHIFTS):
        shift_data = df_seating[df_seating["shift"] == shift_name]
        room_counts = shift_data.groupby("room_id").size().reset_index(name="students")
        all_rooms   = df_rooms["room_id"].tolist()
        room_counts = room_counts.set_index("room_id").reindex(all_rooms, fill_value=0)

        ax.set_facecolor(CLR_BG)
        color = SHIFT_COLORS.get(shift_name, CLR_ACCENT)
        ax.bar(room_counts.index, room_counts["students"],
               color=color, edgecolor="none", width=0.7, alpha=0.85)
        ax.axhline(y=30, color=CLR_WARN,  linestyle="--", linewidth=1, label="Cap 30")
        ax.axhline(y=25, color="#FF6D00", linestyle="--", linewidth=1, label="Cap 25")
        ax.set_title(f"{shift_name} Shift", color=CLR_TEXT, fontsize=10, fontweight="bold")
        ax.tick_params(colors=CLR_SUBTEXT, labelsize=6)
        ax.spines[:].set_color("#2A3A50")
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.legend(facecolor=CLR_PANEL, edgecolor="none",
                  labelcolor=CLR_TEXT, fontsize=7)

    axes[0].set_ylabel("Students", color=CLR_SUBTEXT, fontsize=9)
    fig.suptitle("Room Occupancy Per Shift", color=CLR_TEXT,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def build_chart_shift_distribution(df_seating):
    """Pie chart showing student distribution across shifts."""
    shift_counts = df_seating["shift"].value_counts().reindex(SHIFTS)
    colors = [SHIFT_COLORS[s] for s in SHIFTS]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CLR_PANEL)
    ax.set_facecolor(CLR_PANEL)
    wedges, texts, autotexts = ax.pie(
        shift_counts.values, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=CLR_BG, linewidth=2),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color(CLR_BG); at.set_fontsize(9); at.set_fontweight("bold")
    ax.legend(wedges,
              [f"{s} ({shift_counts[s]})" for s in SHIFTS],
              loc="center left", bbox_to_anchor=(1, 0.5),
              facecolor=CLR_PANEL, edgecolor="none",
              labelcolor=CLR_TEXT, fontsize=8)
    ax.set_title("Students per Shift", color=CLR_TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SECTION 8: GUI HELPERS
# ─────────────────────────────────────────────

def make_stat_card(parent, title, value, color, col):
    frame = tk.Frame(parent, bg=CLR_PANEL, bd=0,
                     highlightthickness=2, highlightbackground=color)
    frame.grid(row=0, column=col, padx=8, pady=8, sticky="nsew")
    tk.Label(frame, text=title,      bg=CLR_PANEL, fg=CLR_SUBTEXT,
             font=("Segoe UI", 9, "bold")).pack(pady=(10, 2))
    tk.Label(frame, text=str(value), bg=CLR_PANEL, fg=color,
             font=("Segoe UI", 20, "bold")).pack(pady=(0, 10))


def make_treeview(parent, columns, rows_data, col_widths):
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Custom.Treeview",
                    background=CLR_PANEL, foreground=CLR_TEXT,
                    fieldbackground=CLR_PANEL, rowheight=26,
                    font=("Segoe UI", 9))
    style.configure("Custom.Treeview.Heading",
                    background=CLR_HEADER_BG, foreground=CLR_ACCENT2,
                    font=("Segoe UI", 9, "bold"), relief="flat")
    style.map("Custom.Treeview",
              background=[("selected", CLR_ACCENT)],
              foreground=[("selected", "white")])

    frame = tk.Frame(parent, bg=CLR_PANEL)
    frame.pack(fill="both", expand=True, padx=4, pady=4)

    scrollbar_y = ttk.Scrollbar(frame, orient="vertical")
    scrollbar_y.pack(side="right", fill="y")

    tree = ttk.Treeview(frame, columns=columns, show="headings",
                        style="Custom.Treeview",
                        yscrollcommand=scrollbar_y.set)
    scrollbar_y.config(command=tree.yview)

    for col, width in zip(columns, col_widths):
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor="center")

    for i, row in enumerate(rows_data):
        tag = "odd" if i % 2 == 0 else "even"
        tree.insert("", "end", values=row, tags=(tag,))

    tree.tag_configure("odd",  background="#1E2D42")
    tree.tag_configure("even", background=CLR_PANEL)
    tree.pack(fill="both", expand=True)
    return tree


def embed_chart(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# ─────────────────────────────────────────────
# SECTION 9: GUI DASHBOARD
# ─────────────────────────────────────────────

def launch_dashboard(df_students, df_seating, df_allocation,
                     df_rooms, optimal_k, k_range, inertia_values):

    root = tk.Tk()
    root.title("AL-2002 Module 3 — Exam Management Dashboard")
    root.configure(bg=CLR_BG)
    root.state("zoomed")

    # Title bar
    title_bar = tk.Frame(root, bg=CLR_ACCENT, height=52)
    title_bar.pack(fill="x", side="top")
    tk.Label(title_bar,
             text="  FAST NUCES  |  AL-2002 Module 3  |  Exam Management — Shift System  |  K-Means",
             bg=CLR_ACCENT, fg="white",
             font=("Segoe UI", 12, "bold")).pack(side="left", pady=10)
    tk.Label(title_bar, text="24F-0806",
             bg=CLR_ACCENT, fg="white",
             font=("Segoe UI", 11)).pack(side="right", padx=20, pady=10)

    # Stat cards
    stats_frame = tk.Frame(root, bg=CLR_BG)
    stats_frame.pack(fill="x", padx=18, pady=(14, 4))

    shift_counts = df_seating["shift"].value_counts()
    stats = [
        ("Total Students",   len(df_students),                        CLR_ACCENT),
        ("Rooms Available",  len(df_rooms),                           CLR_ACCENT2),
        ("Seats Per Shift",  int(df_rooms["capacity"].sum()),          CLR_SUCCESS),
        ("Total Shifts",     len(SHIFTS),                             CLR_WARN),
        ("Morning Shift",    shift_counts.get("Morning", 0),          SHIFT_COLORS["Morning"]),
        ("Evening Shift",    shift_counts.get("Evening", 0),          SHIFT_COLORS["Evening"]),
        ("Night Shift",      shift_counts.get("Night",   0),          SHIFT_COLORS["Night"]),
        ("Clusters (k)",     optimal_k,                               "#FF6D00"),
        ("Waiting List",     0,                                       CLR_SUCCESS),
    ]
    for col, (title, value, color) in enumerate(stats):
        stats_frame.columnconfigure(col, weight=1)
        make_stat_card(stats_frame, title, value, color, col=col)

    # Notebook
    nb_style = ttk.Style()
    nb_style.configure("TNotebook",     background=CLR_BG, borderwidth=0)
    nb_style.configure("TNotebook.Tab", background=CLR_PANEL, foreground=CLR_SUBTEXT,
                       font=("Segoe UI", 10, "bold"), padding=[18, 6])
    nb_style.map("TNotebook.Tab",
                 background=[("selected", CLR_ACCENT)],
                 foreground=[("selected", "white")])

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=18, pady=(6, 14))

    # ── TAB 1: Analytics ──────────────────────────────────
    tab_analytics = tk.Frame(notebook, bg=CLR_BG)
    notebook.add(tab_analytics, text="  📊 Analytics  ")

    row1 = tk.Frame(tab_analytics, bg=CLR_BG)
    row1.pack(fill="both", expand=True, pady=(8, 4))

    for fig, title in [
        (build_chart_elbow(k_range, inertia_values, optimal_k), "Elbow Method"),
        (build_chart_domain_pie(df_students),                    "Domain Distribution"),
        (build_chart_shift_distribution(df_seating),             "Shift Distribution"),
    ]:
        card = tk.Frame(row1, bg=CLR_PANEL, bd=0,
                        highlightthickness=1, highlightbackground="#2A3A50")
        card.pack(side="left", fill="both", expand=True, padx=6)
        tk.Label(card, text=title, bg=CLR_PANEL, fg=CLR_ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(pady=(8, 0))
        embed_chart(card, fig)
        plt.close(fig)

    row2 = tk.Frame(tab_analytics, bg=CLR_BG)
    row2.pack(fill="both", expand=True, pady=(4, 8))

    for fig, title, exp in [
        (build_chart_cluster_scatter(df_students),         "K-Means Cluster Scatter", 1),
        (build_chart_batch_bar(df_students),               "Students per Batch",      1),
        (build_chart_room_occupancy(df_seating, df_rooms), "Room Occupancy Per Shift",2),
    ]:
        card = tk.Frame(row2, bg=CLR_PANEL, bd=0,
                        highlightthickness=1, highlightbackground="#2A3A50")
        card.pack(side="left", fill="both", expand=exp, padx=6)
        tk.Label(card, text=title, bg=CLR_PANEL, fg=CLR_ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(pady=(8, 0))
        embed_chart(card, fig)
        plt.close(fig)

    # ── TAB 2: Seating Plan ───────────────────────────────
    tab_seating = tk.Frame(notebook, bg=CLR_BG)
    notebook.add(tab_seating, text="  🪑 Seating Plan  ")

    # Filter by shift buttons
    filter_frame = tk.Frame(tab_seating, bg=CLR_BG)
    filter_frame.pack(fill="x", padx=10, pady=(8, 2))

    tk.Label(filter_frame, text="Filter by Shift:",
             bg=CLR_BG, fg=CLR_SUBTEXT,
             font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))

    tree_container = tk.Frame(tab_seating, bg=CLR_BG)
    tree_container.pack(fill="both", expand=True)

    current_tree = [None]

    def show_shift(shift_filter="All"):
        for widget in tree_container.winfo_children():
            widget.destroy()

        if shift_filter == "All":
            filtered = df_seating
        else:
            filtered = df_seating[df_seating["shift"] == shift_filter]

        seating_rows = [
            (r["student_id"], r["batch"], r["domain"],
             r["cluster"], r["shift"], r["room_id"], r["seat_number"])
            for _, r in filtered.iterrows()
        ]
        make_treeview(tree_container,
                      ["Student ID", "Batch", "Domain", "Cluster", "Shift", "Room", "Seat #"],
                      seating_rows,
                      col_widths=[110, 60, 190, 70, 80, 70, 70])

    for label, val in [("All", "All")] + [(s, s) for s in SHIFTS]:
        color = SHIFT_COLORS.get(val, CLR_ACCENT)
        btn = tk.Button(filter_frame, text=label,
                        bg=color, fg="white",
                        font=("Segoe UI", 9, "bold"),
                        relief="flat", padx=12, pady=4,
                        command=lambda v=val: show_shift(v))
        btn.pack(side="left", padx=4)

    show_shift("All")

    # ── TAB 3: Faculty Allocation ─────────────────────────
    tab_faculty = tk.Frame(notebook, bg=CLR_BG)
    notebook.add(tab_faculty, text="  👩‍🏫 Faculty Allocation  ")

    tk.Label(tab_faculty,
             text=f"  {len(df_allocation)} total assignments across {len(SHIFTS)} shifts",
             bg=CLR_BG, fg=CLR_SUBTEXT,
             font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 2))

    faculty_rows = [
        (r["shift"], r["room_id"], r["faculty_id"],
         r["faculty_name"], r["faculty_domain"], r["room_dominant_domain"])
        for _, r in df_allocation.iterrows()
    ]
    make_treeview(tab_faculty,
                  ["Shift", "Room", "Faculty ID", "Faculty Name",
                   "Faculty Domain", "Room Domain"],
                  faculty_rows,
                  col_widths=[80, 70, 90, 220, 190, 190])

    # ── TAB 4: Summary ────────────────────────────────────
    tab_summary = tk.Frame(notebook, bg=CLR_BG)
    notebook.add(tab_summary, text="  📋 Summary  ")

    shift_counts_full = df_seating["shift"].value_counts().reindex(SHIFTS, fill_value=0)

    summary_text = (
        f"{'─'*55}\n"
        f"  EXAM MANAGEMENT SYSTEM — SHIFT SUMMARY\n"
        f"{'─'*55}\n\n"
        f"  Total Students         :  {len(df_students)}\n"
        f"  Total Rooms            :  {len(df_rooms)}\n"
        f"  Seats Per Shift        :  {int(df_rooms['capacity'].sum())}\n"
        f"  Total Shifts           :  {len(SHIFTS)}\n"
        f"  Total Capacity         :  {int(df_rooms['capacity'].sum()) * len(SHIFTS)}\n"
        f"  Students Seated        :  {len(df_seating)} / {len(df_students)}\n"
        f"  Waiting List           :  0  ✓ All seated!\n"
        f"  Optimal Clusters (k)   :  {optimal_k}\n"
        f"  Faculty Assignments    :  {len(df_allocation)}\n\n"
        f"{'─'*55}\n"
        f"  STUDENTS PER SHIFT\n"
        f"{'─'*55}\n"
    )
    for shift in SHIFTS:
        summary_text += f"  {shift:10s} Shift  →  {shift_counts_full[shift]} students\n"

    summary_text += f"\n{'─'*55}\n  STUDENTS PER DOMAIN\n{'─'*55}\n"
    for domain, count in df_students["domain"].value_counts().items():
        summary_text += f"  {domain:<32} {count}\n"

    summary_text += f"\n{'─'*55}\n  STUDENTS PER BATCH\n{'─'*55}\n"
    for batch, count in df_students["batch"].value_counts().sort_index().items():
        summary_text += f"  Batch {batch}   →   {count} students\n"

    summary_text += (
        f"\n{'─'*55}\n  OUTPUT FILES SAVED\n{'─'*55}\n"
        f"  seating_plan.csv\n"
        f"  faculty_allocation.csv\n"
    )

    text_widget = tk.Text(tab_summary, bg=CLR_PANEL, fg=CLR_TEXT,
                          font=("Consolas", 11), relief="flat",
                          bd=0, padx=24, pady=20, wrap="word")
    text_widget.pack(fill="both", expand=True, padx=16, pady=16)
    text_widget.insert("1.0", summary_text)
    text_widget.config(state="disabled")

    root.mainloop()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AL-2002 PROJECT – MODULE 3: Exam Management System")
    print("         SHIFT-BASED SEATING (All Students Seated)")
    print("=" * 60)

    # Step 1: Data
    df_students = collect_student_data()
    df_rooms    = collect_room_data()
    df_faculty  = collect_faculty_data()

    # Step 2: Preprocess + Cluster
    scaled_features, scaler          = preprocess_data(df_students)
    optimal_k, k_range, inertia_vals = determine_optimal_clusters(scaled_features, max_k=30)
    df_students, kmeans_model        = run_kmeans(df_students, scaled_features, optimal_k)

    # Step 3: Assign shifts — NO waiting list
    df_students_shifted = assign_shifts(df_students, df_rooms)

    # Step 4: Generate seating with shifts
    df_seating = generate_seating_plan_with_shifts(df_students_shifted, df_rooms)

    # Step 5: Faculty allocation per shift
    df_allocation = allocate_faculty(df_seating, df_faculty, df_rooms)

    # Step 6: Console report
    generate_report(df_students, df_seating, df_allocation, df_rooms, optimal_k)

    # Step 7: Save CSVs
    df_seating.to_csv("seating_plan.csv",         index=False)
    df_allocation.to_csv("faculty_allocation.csv", index=False)
    print("\n[Output] seating_plan.csv saved.")
    print("[Output] faculty_allocation.csv saved.")
    print("[GUI]    Launching dashboard...")

    # Step 8: GUI
    launch_dashboard(df_students, df_seating, df_allocation,
                     df_rooms, optimal_k, k_range, inertia_vals)


if __name__ == "__main__":
    main()
