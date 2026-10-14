#!/usr/bin/env python3
import json, sys, zipfile
from pathlib import Path

def walk(topic):
    yield topic
    for group in topic.get("children", {}).values():
        for child in group:
            yield from walk(child)

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspect_xmind.py FILE.xmind")
    source = Path(sys.argv[1])
    with zipfile.ZipFile(source) as zf:
        sheets = json.loads(zf.read("content.json"))
    result = []
    for sheet in sheets:
        topics = list(walk(sheet["rootTopic"]))
        by_id = {t["id"]: t.get("title", "") for t in topics}
        result.append({
            "sheet": sheet.get("title", ""),
            "topics": [{"title": t.get("title", ""), "id": t["id"], "position": t.get("position")} for t in topics],
            "relationships": [{
                "from": by_id.get(r.get("end1Id"), r.get("end1Id")),
                "to": by_id.get(r.get("end2Id"), r.get("end2Id")),
                "label": r.get("title", "")
            } for r in sheet.get("relationships", [])]
        })
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

