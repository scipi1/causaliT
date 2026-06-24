# Evaluate Experiment Workflow

<task_objective>
Analyze the experimental results provided and generate a new, tailored Jupyter evaluation notebook (`ipynb`) for this specific experiment based on provided notebook examples for other similar experiments and further instruction for customization.
</task_objective>

<detailed_sequence_of_steps>
1. **Context Discovery**:
   - Use `list_files` or `search_files` to look at the current experiment folder contents (e.g., look for `.py`, `.json`, `.yaml`, or `.csv` files representing training parameters or logs).
   - Locate any existing `.ipynb` files in the same directory or adjacent parent directories to use as a "reference structure" or "template."

2. **Template and Structure Analysis**:
   - Read the reference `.ipynb` file to understand the baseline visualization libraries, standard evaluation metrics (e.g., loss curves, confusion matrices), and loading patterns used for prior experiments.
   - Understand optional experiment-specific information provided by the user. It will help you better tailoring the reference notebooks to the current use case.

3. **Code Adaptation & Variable Mapping**:
   - Parse the unique parameters, path strings, and file names specific to the *new* experiment.
   - Map those paths dynamically into the new Python code blocks for the notebook cells.

4. **Notebook Generation**:
   - Draft and create a new Jupyter notebook (e.g., `evaluate_experiment_run.ipynb`) using the VS Code and Cline native Jupyter support blocks.
   - Include distinct markdown documentation cells separating sections: Data Loading, Metric Analysis, and Visualizations.
   - Ensure all libraries (e.g., `matplotlib`, `pandas`, `seaborn`) map exactly to the logged data structures.

5. **Validation Verification**:
   - Inform the user that the notebook has been generated and is ready to be opened in the native VS Code Jupyter editor interface.
</detailed_sequence_of_steps>
