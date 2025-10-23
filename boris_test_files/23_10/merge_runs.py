import os, glob, pandas as pd
from run_config import RUN_ID

root = os.path.join("metrics_23_10", f"run_{RUN_ID}")

dfs = []
for path in glob.glob(os.path.join(root, "pid_*", "*.csv")):
    pid = os.path.basename(os.path.dirname(path))   # e.g., 'pid_46196'
    agent = os.path.splitext(os.path.basename(path))[0]  # 'KL001'
    df = pd.read_csv(path)
    df["pid"] = pid
    df["agent"] = agent
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
all_df.sort_values(["finished_at", "agent", "episode_index"], inplace=True)
out = os.path.join(root, "all_agents_merged_sorted.csv")
all_df.to_csv(out, index=False)
print(f"Wrote {out} with {len(all_df)} rows")


