import os
import sys
import json
import csv
import re
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from react_agent import run

# ── Paths ─────────────────────────────────────────────────────────────────────
QA_PATH       = os.path.join(os.path.dirname(__file__), "../../qa_dataset/gpt_qa_datasets_de.json")
OUTPUT_DIR    = os.path.dirname(__file__)

# ── Scoring ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation for loose matching."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s,.]", "", text)
    return text

def score_answer(predicted: str, expected) -> bool:
    """Return True if predicted answer matches expected (loose string match)."""
    pred = normalize(predicted)
    exp  = normalize(str(expected))
    return exp in pred or pred in exp

def parse_fetched_chunks(steps: list) -> list:
    """Parse chunk metadata from tool_result steps."""
    chunks = []
    for step in steps:
        if step.get("type") != "tool_result":
            continue
        content = step.get("content", "")
        for match in re.finditer(
            r"---\s*(.+?)\s*\[chunk_id=([^\]]+)\]\s*\[score=([^\]]+)\]\s*---",
            content
        ):
            chunks.append({
                "filename": match.group(1).strip(),
                "chunk_id": match.group(2).strip(),
                "score":    float(match.group(3).strip()),
            })
    return chunks

def parse_tool_calls(steps: list) -> list:
    """Extract all tool calls made during the agent run."""
    return [
        {
            "tool": step["tool"],
            "args": step.get("args", {}),
        }
        for step in steps
        if step.get("type") == "tool_call"
    ]

# ── Runner ────────────────────────────────────────────────────────────────────
def run_dataset(dataset_name: str, questions: list) -> tuple[list, dict]:
    results = []
    counts  = {"correct": 0, "total": 0, "by_level": {}}

    for i, item in enumerate(questions):
        question   = item["question"]
        expected   = item["answer"]
        level      = item.get("level", "unknown")
        source     = item.get("source", "")
        justification = item.get("justification", "")

        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Q{i+1}/{len(questions)} ({level}): {question}")
        print(f"  Expected: {expected}")

        agent_out = run(question)

        fetched_chunks = parse_fetched_chunks(agent_out["steps"])
        tool_calls     = parse_tool_calls(agent_out["steps"])
        num_tool_calls = len(tool_calls)

        is_correct = score_answer(agent_out["final"], expected)
        counts["total"] += 1
        if is_correct:
            counts["correct"] += 1
        counts["by_level"].setdefault(level, {"correct": 0, "total": 0})
        counts["by_level"][level]["total"] += 1
        if is_correct:
            counts["by_level"][level]["correct"] += 1

        print(f"  Final Answer : {agent_out['final']}")
        print(f"  Correct      : {is_correct}")

        results.append({
            "question_index": i + 1,
            "question":       question,
            "expected":       expected,
            "level":          level,
            "source":         source,
            "justification":  justification,
            "final_answer":   agent_out["final"],
            "summarized":     agent_out["summarized"],
            "raw":            agent_out["raw"],
            "steps":          agent_out["steps"],
            "fetched_chunks": fetched_chunks,
            "tool_calls":     tool_calls,
            "num_tool_calls": num_tool_calls,
            "correct":        is_correct,
        })

    return results, counts

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_datasets = json.load(f)

    dataset_items = list(qa_datasets.items())[:2]

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results  = []
    summary_rows = []

    for dataset_name, questions in dataset_items:
        print(f"\n{'#'*60}")
        print(f"# Running {dataset_name} ({len(questions)} questions)")
        print(f"{'#'*60}")

        results, counts = run_dataset(dataset_name, questions)

        # Tag each result with dataset name
        for r in results:
            r["dataset"] = dataset_name
        all_results.extend(results)

        # ── Save per-dataset JSON ──────────────────────────────────────────
        out_path = os.path.join(OUTPUT_DIR, f"results_{dataset_name}_{timestamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset":   dataset_name,
                "timestamp": timestamp,
                "counts":    counts,
                "results":   results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ Saved {out_path}")

        accuracy = counts["correct"] / counts["total"] if counts["total"] else 0
        summary_rows.append({
            "dataset":  dataset_name,
            "correct":  counts["correct"],
            "total":    counts["total"],
            "accuracy": round(accuracy, 4),
        })

    # ── Save single combined CSV ───────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f"results_combined_{timestamp}.csv")
    fieldnames = [
        "dataset", "question_index", "question", "expected", "level",
        "source", "justification", "final_answer", "summarized",
        "correct", "num_tool_calls", "tool_calls", "fetched_chunks", "steps",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow({
                **{k: row.get(k, "") for k in fieldnames},
                "fetched_chunks": json.dumps(row.get("fetched_chunks", []), ensure_ascii=False),
                "tool_calls":     json.dumps(row.get("tool_calls",     []), ensure_ascii=False),
                "steps":          json.dumps(row.get("steps",          []), ensure_ascii=False),
            })

    print(f"\n{'='*60}")
    print(f"  ✓ Combined CSV saved to {csv_path}")
    print(f"{'='*60}\n")
    for row in summary_rows:
        print(f"  {row['dataset']}: {row['correct']}/{row['total']} ({row['accuracy']*100:.1f}%)")
    print()

if __name__ == "__main__":
    main()
