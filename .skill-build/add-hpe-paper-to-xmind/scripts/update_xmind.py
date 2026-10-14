#!/usr/bin/env python3
import argparse, copy, json, tempfile, uuid, zipfile
from pathlib import Path

def uid(): return str(uuid.uuid4())

def walk(topic):
    yield topic
    for group in topic.get("children", {}).values():
        for child in group:
            yield from walk(child)

def node(title, x, y, fill, shape):
    return {
        "id": uid(), "class": "topic", "title": title, "titleUnedited": False,
        "position": {"x": x, "y": y},
        "style": {"id": uid(), "properties": {
            "svg:fill": fill, "fill-pattern": "solid", "fo:font-weight": "700",
            "fo:font-size": "16pt", "fo:color": "#20242A", "shape-class": shape,
            "border-line-width": "0pt"
        }}
    }

def relationship(a, b, label, kind):
    styles = {
        "inherit": ("#F28C28", "solid"),
        "compare": ("#8B95A1", "dash"),
        "combine": ("#24A69A", "dot")
    }
    color, pattern = styles[kind]
    rel = {"id": uid(), "class": "relationship", "end1Id": a, "end2Id": b,
           "controlPoints": {}, "style": {"id": uid(), "properties": {
               "line-color": color, "line-width": "2pt", "line-pattern": pattern,
               "arrow-begin-class": "org.xmind.arrowShape.none",
               "arrow-end-class": "org.xmind.arrowShape.triangle",
               "fo:font-size": "12pt", "fo:color": color
           }}}
    if label:
        rel.update(title=label, titleUnedited=False)
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source, output = Path(args.library), Path(args.output)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    with zipfile.ZipFile(source) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if not n.endswith("/")}
    sheets = json.loads(members["content.json"])
    sheet = sheets[0]
    root = sheet["rootTopic"]
    topics = list(walk(root))
    titles = {t.get("title", ""): t for t in topics}
    paper_spec = spec["paper"]
    if paper_spec["title"] in titles:
        raise SystemExit(f"paper already exists: {paper_spec['title']}")

    paper = node(paper_spec["title"], paper_spec["x"], paper_spec["y"],
                 paper_spec.get("color", "#B7D7F0"), "org.xmind.topicShape.roundedRect")
    root.setdefault("children", {}).setdefault("detached", []).append(paper)
    titles[paper["title"]] = paper

    for p in spec.get("problems", []):
        if p["title"] in titles:
            continue
        problem = node(p["title"], p["x"], p["y"], p.get("color", "#FFF2CC"),
                       "org.xmind.topicShape.circle")
        problem["style"]["properties"]["border-line-color"] = "#D6B656"
        problem["style"]["properties"]["border-line-width"] = "1.5pt"
        root["children"]["detached"].append(problem)
        titles[problem["title"]] = problem

    rels = sheet.setdefault("relationships", [])
    existing = {(r.get("end1Id"), r.get("end2Id"), r.get("title", "")) for r in rels}
    for r in spec.get("relationships", []):
        if r["from"] not in titles or r["to"] not in titles:
            raise SystemExit(f"unknown relationship endpoint: {r}")
        key = (titles[r["from"]]["id"], titles[r["to"]]["id"], r.get("label", ""))
        if key not in existing:
            rels.append(relationship(key[0], key[1], key[2], r["type"]))
            existing.add(key)

    members["content.json"] = json.dumps(sheets, ensure_ascii=False, separators=(",", ":")).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".xmind", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items(): zf.writestr(name, data)
    with zipfile.ZipFile(tmp_path) as zf:
        if zf.testzip() is not None: raise SystemExit("generated archive failed validation")
        json.loads(zf.read("content.json"))
    tmp_path.replace(output)
    print(output.resolve())

if __name__ == "__main__":
    main()

