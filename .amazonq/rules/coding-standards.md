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
4. Do not delete my docstrings unless you had to delete the code it was documenting.
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
1. When writing code, think very carefully about performance:
   - Think about alternate ways of writing code and choose the variant that executes faster
   - Think about ways to avoid memory allocations that would not be strictly required
   - Find the right balance between fast code and efficient memory allocations
   - Avoid moving data between CPU and GPU

# Other rules
1. Be brief and stay focused.
2. Even if agentic coding is enabled, we plan first, you confirm the plan with me, and only then make changes.  