import os
import glob
import pandas as pd

def analyze_latest_session():
    log_dir = "logs"
    
    # find all csv files in the logs directory
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    
    if not csv_files:
        print("No session log files found in the 'logs/' directory.")
        return
        
    # automatically grab the absolute latest log file sorted by creation time
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📊 Analyzing Most Recent Research Log: {latest_file}")
    print("-" * 50)
    
    # load data into pandas
    df = pd.read_csv(latest_file)
    
    # 1. calculate total data points and estimated time
    total_samples = len(df)
    # since our loop evaluates every 2 seconds:
    total_seconds = total_samples * 2 
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    # 2. calculate state distributions
    state_counts = df['Audio_State'].value_counts()
    focus_count = state_counts.get('focus', 0)
    ambient_count = state_counts.get('ambient', 0)
    
    focus_pct = (focus_count / total_samples) * 100 if total_samples > 0 else 0
    ambient_pct = (ambient_count / total_samples) * 100 if total_samples > 0 else 0
    
    # 3. calculate key metrics
    max_ratio = df['Raw_Ratio'].max()
    avg_smoothed = df['Smoothed_Ratio'].mean()
    
    # Print results to console
    print(f"Total Session Duration : {minutes}m {seconds}s ({total_samples} data chunks)")
    print(f"Time Spent Focusing    : {focus_count * 2}s ({focus_pct:.1f}%) [Threshold > 0.2]")
    print(f"Time Spent Relaxing    : {ambient_count * 2}s ({ambient_pct:.1f}%)")
    print(f"Peak Focus Ratio (Raw) : {max_ratio:.4f}")
    print(f"Average Smoothed Ratio : {avg_smoothed:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    analyze_latest_session()