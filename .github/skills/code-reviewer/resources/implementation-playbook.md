# Code Review Playbook

Use this when a review needs a more explicit checklist or response format.

## Findings Order

1. Correctness and behavioral regressions
2. Security and data safety risks
3. Reliability and operability issues
4. Performance and scalability concerns
5. Maintainability and testing gaps

## Review Checklist

- Confirm the change matches the stated behavior or requirement
- Look for missing validation, unsafe assumptions, and broken edge cases
- Check whether tests cover the changed behavior and likely regressions
- Verify configuration and dependency changes are intentional and safe
- Prefer fixes at the controlling abstraction, not call-site patching

## Response Format

Each finding should include:

- Severity
- Impacted file or surface
- Concrete risk or regression
- Why it matters
- Smallest credible fix or follow-up

## Verification

- Run the narrowest relevant test, lint, or typecheck for touched code
- If no executable validation exists, note that explicitly and explain the residual risk
- Call out missing tests when behavior changed without coverage