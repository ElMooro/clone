# Replace the old URL with the new working one
import fileinput
import sys

# Update the NY Fed URL in macro_signal_engine.py
with fileinput.FileInput('macro_signal_engine.py', inplace=True, backup='.bak') as file:
    for line in file:
        # Replace the old URL with the new one
        if 'https://www.newyorkfed.org/markets/primarydealer-fails-data' in line:
            line = line.replace(
                'https://www.newyorkfed.org/markets/primarydealer-fails-data',
                'https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics'
            )
        print(line, end='')

print("✅ Fixed NY Fed URL in macro_signal_engine.py")
