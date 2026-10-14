---
name: add-hpe-paper-to-xmind
description: Analyze a newly supplied human-pose-estimation paper and incrementally add it to an existing XMind HPE paper relationship network. Use when the user provides an HPE paper as a PDF, URL, title, screenshot, or pasted text and asks to add it to their paper library, XMind network, knowledge graph, or TAR-ViTPose-centered literature map. Preserve existing nodes, discover links to every relevant paper rather than only the center paper, add shared-problem nodes when justified, prevent visual clutter, and return an updated verified .xmind file.
---

# Add an HPE Paper to an XMind Network

Extend an existing HPE literature network without turning it into a summary tree or a fully connected graph.

## Required inputs

Locate both the new paper (or an authoritative source) and the canonical library at:

`/Users/yrk/Desktop/xmind/TAR-ViTPose中心的HPE论文网.xmind`

Treat `/Users/yrk/Desktop/xmind` as the permanent storage directory for this user's HPE XMind library. Do not leave the authoritative or final updated `.xmind` in a workspace, Downloads, `/tmp`, or another directory. Temporary files may be created elsewhere only during validation and must not be presented as the final library. If the canonical file is missing, restore it from the bundled asset or ask the user before adopting a different library.

If the input is a PDF, read the complete paper using the PDF skill. If only a title or URL is supplied, retrieve the authoritative paper or project page and verify its exact identity. Never silently substitute a similarly named paper.

## Workflow

1. Run `scripts/inspect_xmind.py <library.xmind>` to inventory papers, shared problems, positions, and labeled relationships.
2. Analyze the new paper's task, inference paradigm, technical challenges, mechanisms, datasets, protocols, results, and limitations.
3. Compare it with **every existing paper node**, not only TAR-ViTPose. Test shared challenges, distinct mechanisms, inheritance, paradigm contrasts, and credible complementarity.
4. Keep only evidence-backed, useful edges. Do not make the graph complete merely because two papers concern HPE.
5. Read `references/graph-semantics.md` and draft a JSON update matching `references/update-spec.md`.
6. Run `scripts/update_xmind.py --library /Users/yrk/Desktop/xmind/TAR-ViTPose中心的HPE论文网.xmind --spec <spec.json> --output <temporary-updated.xmind>`.
7. Re-run `inspect_xmind.py` on the output and `unzip -t` it. Confirm the new paper occurs once, existing content remains, endpoints exist, and labels contain no emoji arrows.
8. Open the output in XMind when available. Fix overlaps, crossings through node text, tiny low-contrast labels, and excessive long arcs.
9. After all validation and visual fixes pass, atomically replace the canonical file in `/Users/yrk/Desktop/xmind`. Deliver that canonical `.xmind` as a clickable link and summarize only the new nodes and edges.

## Non-negotiable graph rules

- Keep TAR-ViTPose visually central when it is the anchor, but do not force every new paper to connect to it.
- Use rounded rectangles for papers and circles for shared research problems.
- Put a relationship description directly on a paper-to-paper line when it only describes those two papers.
- Put a mechanism directly on a paper-to-problem line when that mechanism belongs to one paper only.
- Promote a mechanism to an independent ellipse only when multiple papers reuse it or it becomes a research object itself.
- Add a shared-problem circle only when at least two papers genuinely address it.
- Prefer short labels such as `JTA 与 DAM：同一问题的不同机制` over paragraphs.
- Mark novel synthesis as `推断：` in the update specification; never present it as a paper claim.
- Preserve user-authored nodes and manual layout unless movement is necessary to prevent overlap.
- Always keep the final authoritative XMind at `/Users/yrk/Desktop/xmind/TAR-ViTPose中心的HPE论文网.xmind`.

## Quality bar

Revise the update when lines or labels cross node text, labels overlap, a relevant non-TAR link is missed, incompatible results are compared directly, a relation is duplicated, or the network becomes difficult to trace one paper at a time.
