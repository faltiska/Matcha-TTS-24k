# Project location
IntelliJ and the AmazonQ plugin are running in Windows, but the project is in Ubuntu, in WSL.
You have access to the entire project, but you have to use UNC paths for WSL, like "\\wsl.localhost\Ubuntu\...".
You can execute any linux commands, but run them with wsl using a login shell: 
`wsl -d Ubuntu -- bash -lc "<command>"` 
so that PATH and environment variables from .profile/.bashrc are available.

# Coding Standards
1. Write self-explanatory code, as clear as for a person that never saw the code to be able to understand it.
2. Follow Extreme Programming principles.
   - Use expressive named boolean variables to clarify boolean conditions
   - Use classes, methods and variables with clear names that make inline comments unnecessary
   - Instead of explaining if statement's condition with a comment, introduce a variable whose name explains it.
   - Instead of explaining a code block with a comment, introduce a method whose name will make the comment unnecessary.
   - Appy the "exist fast" principle to avoid deep nesting and improve readability.
   - Never create a method that only holds a single call to another method.
3. Change only what is necessary. Do not rewrite existing code unless I asked for it.
4. NEVER delete my comments or docstrings. This is a hard rule with no exceptions, even when rewriting or replacing a block of code. If you are replacing a block that contains comments, you must carry those comments over to the new version.
5. Match the existing coding style as close as possible, unless it breaks the above rules.

6. Avoid using ternary expressions in assignments. Use classic `if/else` blocks instead.
Example:
   Instead of 
   ```
   rate = args.speaking_rate if args.speaking_rate is not None else voice["default_scale"]
   ```
   prefer 
   ```
    if args.speaking_rate is not None:
        length_scale = args.speaking_rate
    else:
        length_scale = voice["default_scale"]
   ```
You are allowed to use ternary expressions if where an `if/else` block would genuinely hurt readability. 
Use your judgment.
 
7. Do not nest multiple function calls or chain multiple values into a single expression when it hurts readability. 
Assign intermediate results to named variables instead.

# Keep performance in mind at all times
When writing code, think very carefully about performance:
- Think about alternate ways of writing code and choose the variant that executes faster
- Think about ways to avoid memory allocations that would not be strictly required
- Find the right balance between fast code and efficient memory allocations
- Avoid moving data between CPU and GPU

# General rules
1. Please do not answer using variable names, I am a human being, cannot remember the code by heart, I do not know what things like "logw_pred" represent.
2. Keep your answers short and focused.
3. Even if agentic coding is enabled, we plan first, you confirm the plan with me, and only then make changes.  
4. When executing grep commands, always exclude the "logs" and ".venv" folders  
