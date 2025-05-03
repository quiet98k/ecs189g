import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('my_model/part2/full_results.csv')
filtered_df = df[df['test_acc'] > 0.9].sort_values(by='edit_acc', ascending=False)
filtered_df.to_csv("my_model/part2/filtered_results.csv", index=False)
print(f"{filtered_df.head(20)}")