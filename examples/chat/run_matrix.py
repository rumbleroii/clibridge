import argparse
import asyncio
import json
import time

import capabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-code", help="claude-code, codex, or both")
    parser.add_argument("--only", default="", help="comma-separated capability ids")
    parser.add_argument("--out", default="matrix_results.json")
    args = parser.parse_args()

    models = ["claude-code", "codex"] if args.model == "both" else [args.model]
    selected = [m.strip() for m in args.only.split(",") if m.strip()] or [c[0] for c in capabilities.CAPABILITIES]

    results = []
    for model in models:
        for cap_id in selected:
            entry = capabilities.BY_ID.get(cap_id)
            if entry is None:
                print(f"unknown capability: {cap_id}")
                continue
            _id, label, _description, is_async, fn = entry
            start = time.time()
            try:
                result = asyncio.run(fn(model)) if is_async else fn(model)
                record = {"ok": True, "result": result}
            except Exception as exc:
                record = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1200]}
            record.update(
                {"capability": cap_id, "label": label, "model": model, "seconds": round(time.time() - start, 2)}
            )
            results.append(record)
            status = "PASS" if record["ok"] else "FAIL"
            print(f"{status:4} {model:12} {cap_id:20} {record['seconds']:6.1f}s")
            if not record["ok"]:
                print(f"     {record['error']}")

    capabilities.stop_server()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    passed = sum(1 for record in results if record["ok"])
    print(f"\n{passed}/{len(results)} passed -> {args.out}")


if __name__ == "__main__":
    main()
