# HPE network semantics

## Node types

- **Paper**: rounded rectangle. Use `Title\nshort role`.
- **Shared problem**: circle with a short problem statement.
- **Mechanism node**: normally omit. Use an ellipse only when multiple papers reuse the mechanism or it needs its own subnetwork.

## Edge types

- `inherit`: orange solid directed line for explicit inheritance, extension, or research evolution.
- `compare`: gray dashed line for different solutions, paradigms, or prediction formulations.
- `combine`: teal dotted directed line from a paper to a shared problem, labeled with the paper's mechanism.

## Relationship test

Complete: `Paper A and Paper B are related because both address ___; A uses ___, while B uses ___.`

If this cannot be supported, omit the edge. Add direct inheritance only when one method actually builds on, initializes from, cites as a baseline, or structurally extends the other.

## Layout

- Inspect current coordinates before choosing a new position.
- Place a new paper in the largest free sector around the anchor.
- Place a shared problem outside the paper ring, near all connected papers.
- Avoid moving existing nodes unless required for readability.
- Prefer 2–5 high-value new edges per paper. More requires unusually strong justification.

