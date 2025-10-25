import tkinter as tk
from tkinter import ttk
import sqlite3

# --- Parameters ---
THRESHOLD = 60  # Score above this is considered 'attentive'
TIME_PER_ENTRY = 1  # Assume each row = 1 second

# --- Functions ---
def fetch_data():
    conn = sqlite3.connect("attentiveness.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM full_attentiveness ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def calculate_summary(data):
    if not data:
        return 0, 0, 0, 0.0
    total_entries = len(data)
    total_points = sum(row[-1] for row in data)  # final_score
    attentive_entries = sum(1 for row in data if row[-1] >= THRESHOLD)
    total_time = total_entries * TIME_PER_ENTRY
    attentive_time = attentive_entries * TIME_PER_ENTRY
    avg_score = round(total_points / total_entries, 2)
    return total_time, attentive_time, total_points, avg_score

def populate_tree(tree, data):
    for row in tree.get_children():
        tree.delete(row)
    for entry in data:
        tree.insert('', 'end', values=entry)

def refresh_data():
    data = fetch_data()
    populate_tree(tree, data)
    total_time, attentive_time, total_points, avg_score = calculate_summary(data)

    avg_label.config(text=f"Average Attentiveness: {avg_score}%")
    points_label.config(text=f"Attentive Points: {round(total_points, 2)}")
    time_label.config(text=f"Total Time: {total_time} sec | Attentive Time: {attentive_time} sec")

# --- GUI Setup ---
root = tk.Tk()
root.title("Attentiveness Logs Viewer")
root.geometry("1050x550")

columns = ('ID', 'Timestamp', 'Lip Status', 'Open Eyes', 'Total Eyes',
           'Lip Score', 'Eye Score', 'Final Score')
tree = ttk.Treeview(root, columns=columns, show='headings')

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor='center')
tree.pack(expand=True, fill='both', padx=10, pady=10)

avg_label = tk.Label(root, text="Average Attentiveness: --%", font=('Helvetica', 14, 'bold'))
avg_label.pack(pady=5)

points_label = tk.Label(root, text="Attentive Points: --", font=('Helvetica', 12))
points_label.pack(pady=2)

time_label = tk.Label(root, text="Total Time: -- sec | Attentive Time: -- sec", font=('Helvetica', 12))
time_label.pack(pady=2)

refresh_btn = tk.Button(root, text="Refresh Data", command=refresh_data, font=('Helvetica', 12))
refresh_btn.pack(pady=10)

# Load initially
refresh_data()
root.mainloop()
