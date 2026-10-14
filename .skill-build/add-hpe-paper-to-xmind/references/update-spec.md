# Update specification

Pass UTF-8 JSON to `scripts/update_xmind.py`:

```json
{
  "paper": {"title": "NewPaper\\nshort role", "x": 0, "y": 500, "color": "#B7D7F0"},
  "problems": [{"title": "shared problem", "x": 900, "y": 300}],
  "relationships": [
    {"from": "NewPaper\\nshort role", "to": "ExistingPaper\\nrole", "label": "short relationship", "type": "compare"},
    {"from": "NewPaper\\nshort role", "to": "shared problem", "label": "paper-specific mechanism", "type": "combine"}
  ]
}
```

`from` and `to` must exactly match topic titles. Types: `inherit`, `compare`, `combine`. Coordinates are relative to the central topic. Omit a problem already present and refer to its exact title.

