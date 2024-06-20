import os
import re

directory = '../ai_papers'

# Function to check if a filename matches the 4-digit format
def is_formatted(filename):
    return re.match(r'^\d{4}\.pdf$', filename) is not None

try:
    # Get a list of existing files and sort them
    files = os.listdir(directory)
    files.sort()

    # Determine the starting count based on existing files
    count = 1
    for filename in files:
        if filename == '.DS_Store':
            continue  # Skip .DS_Store file
        if is_formatted(filename):
            number = int(filename.split('.')[0])
            if number >= count:
                count = number + 1  # Start count after the highest existing number

    # Rename files that do not match the 4-digit format
    for filename in files:
        if filename == '.DS_Store':
            continue  # Skip .DS_Store file
        if not is_formatted(filename):
            new_name = f"{count:04}.pdf"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Renamed {filename} to {new_name}")
            count += 1

    print("Renaming complete.")

except Exception as e:
    print(f"Error: {e}")

