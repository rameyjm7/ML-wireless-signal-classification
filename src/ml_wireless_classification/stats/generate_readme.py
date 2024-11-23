import os
import json
import pandas as pd
from tabulate import tabulate

def generate_readme(stats_folder='.', output_file='README.md'):
    """
    Generate a README.md file summarizing model statistics from JSON files.
    The summary table is sorted by 'Best Accuracy' in descending order, and
    models with zero accuracy are excluded.
    """
    # Initialize list to store data for each model
    model_data = []

    # Loop through each file in the stats folder
    for filename in os.listdir(stats_folder):
        if filename.endswith('.json'):
            # Construct file path
            filepath = os.path.join(stats_folder, filename)
            with open(filepath, 'r') as file:
                try:
                    # Load JSON data
                    stats = json.load(file)
                    best_accuracy = stats.get('best_accuracy', 0)
                    
                    # Skip models with zero accuracy
                    if best_accuracy > 0:
                        model_data.append({
                            'Model Name': filename.replace('.json', ''),
                            'Date Created': stats.get('date_created', 'N/A'),
                            'Epochs Trained': stats.get('epochs_trained', 'N/A'),
                            'Best Accuracy': best_accuracy,
                            'Current Accuracy': stats.get('current_accuracy', 'N/A'),
                            'Last Trained': stats.get('last_trained', 'N/A')
                        })
                except json.JSONDecodeError:
                    print(f"Could not decode JSON from file: {filename}")

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(model_data)

    # Sort the DataFrame by 'Best Accuracy' in descending order
    if not df.empty:
        df = df.sort_values(by='Best Accuracy', ascending=False)

    # Generate markdown table using tabulate and save it as README.md
    with open(output_file, 'w') as readme:
        readme.write("# Model Statistics Summary\n\n")
        readme.write("This table summarizes statistics for all models found, sorted by best accuracy.\n\n")
        if not df.empty:
            readme.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        else:
            readme.write("No valid models found with non-zero accuracy.\n")
        
    print(f"README.md file generated successfully with model statistics from {stats_folder}.")

# Run the script
generate_readme()
