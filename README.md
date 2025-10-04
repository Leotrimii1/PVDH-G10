# PVDH-G10

## Project Setup

Follow these steps to run the project and the Jupyter Notebook.

```bash

# Reminder: Add Jupyter Extension on VS Code

# Clone the repository
git clone https://github.com/Leotrimii1/PVDH-G10.git
cd PVDH-G10

# Create a virtual environment
python3 -m venv ./venv

# Activate the virtual environment
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\activate

# Install project requirements
pip install -r requirements.txt

# Open your .ipynb notebook file. When you attempt to run a cell,
# VS Code will ask you to select a Python environment.
# Select the interpreter that points to your virtual environment:
# ./venv/bin/python
