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
   - If the user has not specified any template, look for folders in the same level as the specified experiment and read `.ipynb` files. Then base the template on them.

3. **Code Adaptation & Variable Mapping**:
   - Parse the unique parameters, path strings, and file names specific to the *new* experiment.
   - Map those paths dynamically into the new Python code blocks for the notebook cells.

4. **Notebook Generation**:
   - Draft and create a new Jupyter notebook (e.g., `evaluate_experiment_run.ipynb`) using the VS Code and Cline native Jupyter support blocks.
   - Include distinct markdown documentation cells separating sections: Data Loading, Metric Analysis, and Visualizations.
   - Ensure all libraries (e.g., `matplotlib`, `pandas`, `seaborn`) map exactly to the logged data structures.

5. **Validation Verification**:
   - Inform the user that the notebook has been generated and is ready to be opened in the native VS Code Jupyter editor interface.


6. **General Evaluation Hygiene**
- If the experiment involves training a model and training metrics are available in the folder structure, make an initial cell evaluating the training itself. This is a pre-requisite before even looking at the experiment-specific results. 
- If metrics like R2, MAE are present, report them and suggest the user to carefully analyze them. The R2 score is the best indicator to quickly understand if the fit was successful, it should have the highest priority.
- If training and validation metrics logs are available, plot them against the epochs to visualize generalization, overfit and underfit.


7. **Notebook reusability**
- The notebook could be reused in another compatible experiment by simply copy-pasting it in the new experiment folder.
- Avoid long detailed particular description in the first creation phase
- Experiment-specific parameters can be extracted from the experiment configuration, so that it can be effortless adapted to other experiments with the same config structure.


</detailed_sequence_of_steps>