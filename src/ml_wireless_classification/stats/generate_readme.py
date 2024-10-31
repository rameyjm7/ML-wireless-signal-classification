import os
import json
import pandas as pd
from tabulate import tabulate

def generate_readme(stats_folder='.', output_file='README.md'):
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
                    model_data.append({
                        'Model Name': filename.replace('.json', ''),
                        'Date Created': stats.get('date_created', 'N/A'),
                        'Epochs Trained': stats.get('epochs_trained', 'N/A'),
                        'Best Accuracy': stats.get('best_accuracy', 'N/A'),
                        'Current Accuracy': stats.get('current_accuracy', 'N/A'),
                        'Last Trained': stats.get('last_trained', 'N/A')
                    })
                except json.JSONDecodeError:
                    print(f"Could not decode JSON from file: {filename}")
    
    # Convert the collected data into a DataFrame
    df = pd.DataFrame(model_data)
    
    # Generate markdown table using tabulate and save it as README.md
    with open(output_file, 'w') as readme:
        readme.write("# Model Statistics Summary\n\n")
        readme.write("This table summarizes statistics for all models found\n\n")
        readme.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        
    print(f"README.md file generated successfully with model statistics from {stats_folder}.")

# Run the script
generate_readme()
