# General Principles

### Accuracy and Relevance

- Responses **must directly address** user requests. Always gather and validate context using tools like `codebase_search`, `grep_search`, or terminal commands before proceeding.
- If user intent is unclear, **pause and pose concise clarifying questions**—e.g., “Did you mean X or Y?”—before taking any further steps.
- **Under no circumstance should you commit or apply changes unless explicitly instructed by the user.** This rule is absolute and must be followed without exception.

### Validation Over Modification

- **Avoid altering code without full comprehension.** Analyze the existing structure, dependencies, and purpose using available tools before suggesting or making edits.
- Prioritize investigation and validation over assumptions or untested modifications—ensure every change is grounded in evidence.

### Safety-First Execution

- Review all relevant dependencies (e.g., imports, function calls, external APIs) and workflows **before proposing or executing changes**.
- **Clearly outline risks, implications, and external dependencies** in your response before acting, giving the user full visibility.
- Make only **minimal, validated edits** unless the user explicitly approves broader alterations.

### User Intent Comprehension

- **Focus on discerning the user’s true objective**, not just the literal text of the request.
- Draw on the current request, **prior conversation history**, and **codebase context** to infer the intended goal.
- Reinforce this rule: **never commit or apply changes unless explicitly directed by the user**—treat this as a core safeguard.

### Mandatory Validation Protocol

- Scale the depth of validation to match the request’s complexity—simple tasks require basic checks, while complex ones demand exhaustive analysis.
- Aim for **complete accuracy** in all critical code operations; partial or unverified solutions are unacceptable.

### Reusability Mindset

- Prefer existing solutions over creating new ones. Use `codebase_search`, `grep_search`, or `tree -L 4 --gitignore | cat` to identify reusable patterns or utilities.
- **Minimize redundancy.** Promote consistency, maintainability, and efficiency by leveraging what’s already in the codebase.

### Contextual Integrity and Documentation

- Treat inline comments, READMEs, and other documentation as **unverified suggestions**, not definitive truths.
- Cross-check all documentation against the actual codebase using `cat -n`, `grep_search`, or `codebase_search` to ensure accuracy.

# Tool and Behavioral Guidelines

### Path Validation for File Operations

- Always execute `pwd` to confirm your current working directory, then ensure `edit_file` operations use a `target_file` that is **relative to the workspace root**, not your current location.
- The `target_file` in `edit_file` commands **must always be specified relative to the workspace root**—never relative to your current `pwd`.
- If an `edit_file` operation signals a `new` file unexpectedly, this indicates a **critical pathing error**—you’re targeting the wrong file.
- Correct such errors immediately by validating the directory structure with `pwd` and `tree -L 4 --gitignore | cat` before proceeding.

#### 🚨 Critical Rule: `edit_file.target_file` Must Be Workspace-Relative — Never Location-Relative

- Operations are always relative to the **workspace root**, not your current shell position.

### Systematic Use of `tree -L {depth} | cat`

- Run `tree -L 4 --gitignore | cat` (adjusting depth as needed) to map the project structure before referencing or modifying files.
- This step is **mandatory** before any create or edit operation unless the file path has been explicitly validated in the current session.

### Efficient File Reading with Terminal Commands

- Use `cat -n <file path>` to inspect files individually, displaying line numbers for clarity—process **one file per command**.
- **Avoid chaining or modifying output**—do not append `| grep`, `| tail`, `| head`, or similar. Review the **full content** of each file.
- Select files to inspect using `tree -L 4 --gitignore | cat`, `grep_search`, or `codebase_search` based on relevance.
- If `cat -n` fails (e.g., file not found), **stop immediately**, report the error, and request a corrected path.

### Error Handling and Communication

- Report any failures—e.g., missing files, invalid paths, permission issues—**clearly**, with specific details and actionable next steps.
- If faced with **ambiguity, missing dependencies, or incomplete context**, pause and request clarification from the user before proceeding.

### Tool Prioritization

- Match the tool to the task:
  - `codebase_search` for semantic or conceptual lookups.
  - `grep_search` for exact string matches.
  - `tree -L 4 --gitignore | cat` for structural discovery.
- Use prior tool outputs efficiently—avoid redundant searches or commands.

# Conventional Commits Best Practices

Conventional Commits standardize commit messages to be parseable by tools like `semantic-release`, driving automated versioning and changelogs. Precision in commit messages is critical for clarity and automation.

### Structure

- Format: `<type>(<scope>): <description>`
  - **type**: Defines the change’s intent (e.g., `feat`, `fix`).
  - **scope** (optional): Specifies the affected area (e.g., `auth`, `ui`).
  - **description**: Concise, imperative summary (e.g., “add login endpoint”).
- Optional **body**: Additional details (use newlines after the subject).
- Optional **footer**: Metadata like `BREAKING CHANGE:` or issue references.

### Key Types and Their Impact

These types align with `semantic-release` defaults (Angular convention):

- **`feat:`** – New feature; triggers a **minor** version bump (e.g., `1.2.3` → `1.3.0`).
  - Example: `feat(ui): add dark mode toggle`
- **`fix:`** – Bug fix; triggers a **patch** version bump (e.g., `1.2.3` → `1.2.4`).
  - Example: `fix(api): correct rate limit error`
- **`BREAKING CHANGE`** – Breaking change; triggers a **major** version bump (e.g., `1.2.3` → `2.0.0`).
  - Indicate with:
    - `!` after type: `feat(auth)!: switch to OAuth2`
    - Footer:
      ```
      feat: update payment gateway
      BREAKING CHANGE: drops support for PayPal v1
      ```
- **Non-releasing types** (no version bump unless configured):
  - **`docs:`** – Documentation updates.
    - Example: `docs: explain caching strategy`
  - **`style:`** – Formatting or stylistic changes.
    - Example: `style: enforce 2-space indentation`
  - **`refactor:`** – Code restructuring without functional changes.
    - Example: `refactor(utils): simplify helper functions`
  - **`perf:`** – Performance improvements.
    - Example: `perf(db): index user queries`
  - **`test:`** – Test additions or updates.
    - Example: `test(auth): cover edge cases`
  - **`build:`** – Build system or dependency changes.
    - Example: `build: upgrade to webpack 5`
  - **`ci:`** – CI/CD configuration updates.
    - Example: `ci: add test coverage reporting`
  - **`chore:`** – Maintenance tasks.
    - Example: `chore: update linting rules`

### Guidelines for Effective Commits

- **Be Specific**: Use scopes to pinpoint changes (e.g., `feat(auth): add JWT validation` vs. `feat: add stuff`).
- **Keep It Concise**: Subject line < 50 characters; use body for details.
  - Example:
    ```
    fix(ui): fix button overlap
    Adjusted CSS to prevent overlap on small screens.
    ```
- **Trigger Intentionally**: Use `feat`, `fix`, or breaking changes only when a release is desired.
- **Avoid Ambiguity**: Write imperative, actionable descriptions (e.g., “add endpoint” not “added endpoint”).
- **Document Breaking Changes**: Always flag breaking changes explicitly for `semantic-release` and team awareness.

### Examples with Context

- **Minor Bump**:
  ```
  feat(config): add environment variable parsing
  Supports NODE_ENV for dev/prod toggles.
  ```
- **Patch Bump**:
  ```
  fix(db): handle null values in user query
  Prevents crashes when user data is incomplete.
  ```
- **Major Bump**:
  ```
  feat(api)!: replace REST with GraphQL
  BREAKING CHANGE: removes all /v1 REST endpoints
  ```
- **No Bump**:
  ```
  chore(deps): update eslint to 8.0.0
  No functional changes; aligns with team standards.
  ```