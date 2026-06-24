# Evaluate Experiment with Observations Workflow

<task_objective>
Collaborate with the user for improving the observations he wrote in the provided evaluation notebook (`.ipynb`). The notebook should have an "Observations" cell at the very bottom for manual/automated experiment insights.
</task_objective>

<detailed_sequence_of_steps>
1. **Target Identification**:
   - Use `ask_followup_question` to prompt the user for the exact experiment folder or file entry point if it was not explicitly provided next to the slash command.

2. **Understand the results**:
   - Read the notebook and understand what the results convey
   - Read how the user interpreted the results. His comments are generally in the last cell (e.g., look for markdown headings like `## Observations` or `### Insights`).
   - Understand the bigger picture. Make sure what the user wrote is clear, in the context of the general codebase, this particular experiment and similar experiments in the same parent folder.
   - Use `ask_followup_question` to prompt the user for unclear explanations.

3. **Critically assess user's understanding**:
    - Has the user identified problems supported by the data evaluated in the notebook?
    - Are there contradictions to pay attention to
    - Is the user overlooking important phenomena that the notebook highlights


3. **Engage**:
    - Reply to the user and engage in the conversation to either confirm or confute his observations
    - Use Use `ask_followup_question`to further deepen complicated concepts
    - Be clear and concise
    
4. **Update the notebook**:
    - Ask for permission to update the notebook with the result of this intellectual exchange.
    - Add your personal observation in a dedicated stage (under `## Cline`) as well as the next steps


</detailed_sequence_of_steps>
