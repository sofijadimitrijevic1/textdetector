import pandas as pd
import os

# CSV input path
csv_path = os.path.join(os.path.dirname(__file__), "raw", "essays.csv")
df = pd.read_csv(csv_path)


human_df = pd.DataFrame({
    "title": df["title"],
    "description": df["description"],
    "essay": df["essay"]
})

# Output path
out_dir = os.path.join(os.path.dirname(__file__), "raw")
out_csv = os.path.join(out_dir, "essays_sorted.csv")


os.makedirs(out_dir, exist_ok=True)

# Save
human_df.to_csv(out_csv, index=False)

print(f"Saved essays_sorted.csv to {out_csv}")