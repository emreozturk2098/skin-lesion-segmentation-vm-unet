from pathlib import Path
import ast, json, hashlib, sys, warnings
ROOT = Path(__file__).resolve().parents[1]
issues = []
counts = {"python": 0, "notebooks": 0, "supplemental_files": 0}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in [".git", ".venv", "__pycache__"]):
            continue
        try:
            if p.suffix == ".py":
                ast.parse(p.read_text(encoding="utf-8-sig")); counts["python"] += 1
            elif p.suffix == ".ipynb":
                d = json.loads(p.read_text(encoding="utf-8-sig"))
                assert d.get("nbformat") == 4 and isinstance(d.get("cells"), list)
                counts["notebooks"] += 1
        except Exception as e:
            issues.append(str(p.relative_to(ROOT)) + ": " + str(e))
manifest = ROOT / "docs/SUPPLEMENTARY_MANIFEST.json"
if manifest.exists():
    for row in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
        p = ROOT / row["file"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            issues.append("Missing or changed supplemental source: " + row["file"])
        counts["supplemental_files"] += 1
print(json.dumps({"counts": counts, "issues": issues, "training_executed": False}, indent=2))
sys.exit(bool(issues))
