Genie Limits


Genie – Known Limitations:

Token Capacity: Maximum support of 14,000 tokens - 
genie uses full metadata which leads to exhaustion of tokens faster. We can filter useful metadata in the start and then send to LLM.
in custom we will be able extending the number of follow up questions with the context. - genie may take 10 questions(MHR 20 questions, MBRDNA 10 to 15), custom may take ??
Visualizations:
When datasets exceed 100 rows, generating visualizations becomes challenging due to limitations in both the frontend and the LLM.

Throughput Constraints: Can handle no more than 10 users or 20 queries per minute at peak in a single workspace. And limit applies to all the genie spaces combined.
Custom build can scale up and down according to usage.

Query Complexity: Limited to the capabilities defined within the Genie environment. (Inter Genie answers not possible) -> Have a single metadata table

Integration of Emily’s Code
Emily’s TypeScript implementation is incompatible with our current Python-based backend. Converting the backend to TypeScript would be required to adopt her approach.
The existing Databricks (Flask) application lacks clean, easily readable function structures.


Custom Solution:
A custom-built solution could be completed within 2–3 sprints (including MLflow integration), but initial accuracy would be low.





Metadata  -> Table information and column information

1. Question asked by user -> Reframing of Question (Make it constant) -> LLM
2. Choose the correct tables.(Dont just pick the first right match) -> LLM
3. Choose the right columns. (Dont just pick the first right match) -> LLM
4. Send the info of right tables and columns(maybe multiple combo) -> select the right combo -> LLM
5. write smallest possible instruction for LLM with guardrails
6. Send the relevaent info to LLM for SQL generation
7. MLFlow logging in table
8. MLFlow testing, scoring and failsafe for quality of SQL generated.
9. Try to include human feedback function if possible, recommended better question, or choice of dataset from which data can be extracted to the user
10. Running generated SQL on SQL warehouse
11. Try to generate visuals. Send warning message if dataset is too big and dont run above a certain threshold for visual generation.
12. Insights from the data to the user



Tools:

1. Reframing question
2. Table and Col selection
3. SQL generator
4. SQL executor
5. SQL validator
6. Visual Tools
7. Insights Tools
