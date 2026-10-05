from nbconvert import PythonExporter
import nbformat

input_file_name = input("input file name (with extension)\n")

# Load the notebook
with open(input_file_name) as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(notebook)

# Save to .py file
with open(input_file_name[:-6] + ".py", 'w') as f:
    f.write(source)
