import json
import csv
import re
from urllib.parse import urlparse

def extract_method_and_paths(oracle_string):
    """Extract method and path from a multiline Oracle string using space as separator."""
    if not oracle_string:
        return []
    
    cleaned = oracle_string.replace('"', '').replace('\\n', '\n').replace('\\r', '\r')
    lines = cleaned.strip().splitlines()
    
    result = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            method = parts[0].strip().upper()
            path = parts[1].strip().rstrip('/')
            result.append((method, path))

    return result

def extract_method_and_path_from_task(task):
    """Extract method and full path from a task (without hostname)."""
    method = task.get("operation")
    endpoint = task.get("endpoint", "")
    parsed = urlparse(endpoint)
    path = parsed.path.rstrip('/')
    return method, path

def normalize_question(q):
    return q.strip().lower()

def load_json_questions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_oracles(csv_path):
    oracle_map = {}
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_key = next((k for k in row if 'question' in k.lower()), None)
            oracle_key = next((k for k in row if 'oracle' in k.lower()), None)
            if question_key and oracle_key:
                question = normalize_question(row[question_key])
                oracle_raw = row.get(oracle_key)
                oracle_map[question] = extract_method_and_paths(oracle_raw)
    return oracle_map

def compare(json_data, oracle_map):
    results = []

    for item in json_data:
        question = normalize_question(item["question"])
        tasks = item.get("execution_plan", {}).get("tasks", [])
        oracle_steps = oracle_map.get(question, [])

        matches = []
        mismatches = []
        matched_indices = set()

        for task in tasks:
            task_method, task_path = extract_method_and_path_from_task(task)

            match_found = False
            for idx, (oracle_method, oracle_path) in enumerate(oracle_steps):
                if idx in matched_indices:
                    continue
                if task_method == oracle_method and task_path.endswith(oracle_path):
                    matches.append({
                        "task": (task_method, task_path),
                        "oracle": (oracle_method, oracle_path)
                    })
                    matched_indices.add(idx)
                    match_found = True
                    break

            if not match_found:
                mismatches.append({
                    "task": (task_method, task_path),
                    "oracle": None
                })

        status = "correct"
        if matches and mismatches:
            status = "partial"
        elif not matches:
            status = "incorrect"

        comparison = {
            "question_index": item["question_index"],
            "question": item["question"],
            "matches": matches,
            "mismatches": mismatches,
            "missing_in_oracle": len(tasks) > len(oracle_steps),
            "extra_in_oracle": len(oracle_steps) > len(tasks),
            "status": status,
            "total_tasks": len(tasks),
            "matched_tasks": len(matches)
        }

        results.append(comparison)

    return results

def write_detailed_output(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"--- Question #{result['question_index']} ---\n")
            f.write(f"Q: {result['question']}\n")
            if result["matches"]:
                f.write("Matches:\n")
                for match in result["matches"]:
                    task = match["task"]
                    oracle = match["oracle"]
                    f.write(f"  - Task:   {task[0]} {task[1]}\n")
                    f.write(f"    Oracle: {oracle[0]} {oracle[1]}\n")
            if result["mismatches"]:
                f.write("Mismatches:\n")
                for mm in result["mismatches"]:
                    task = mm["task"]
                    oracle = mm["oracle"]
                    f.write(f"  - Task:   {task[0]} {task[1]}\n")
                    if oracle:
                        f.write(f"    Oracle: {oracle[0]} {oracle[1]}\n")
                    else:
                        f.write("    Oracle: None\n")
            if result["missing_in_oracle"]:
                f.write("Oracle has fewer steps than tasks\n")
            if result["extra_in_oracle"]:
                f.write("Oracle has more steps than tasks\n")
            f.write(f"Status: {result['status'].upper()}\n")
            f.write("\n")

def write_summary_output(results, filename):
    total = len(results)
    correct = sum(1 for r in results if r['status'] == 'correct')
    partial = sum(1 for r in results if r['status'] == 'partial')
    incorrect = sum(1 for r in results if r['status'] == 'incorrect')
    total_endpoints = sum(r['total_tasks'] for r in results)
    matched_endpoints = sum(r['matched_tasks'] for r in results)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Summary Report\n")
        f.write("=================\n")
        f.write(f"Total execution plans: {total}\n")
        f.write(f"Correct plans: {correct}\n")
        f.write(f"Partially correct plans: {partial}\n")
        f.write(f"Incorrect plans: {incorrect}\n")
        f.write(f"\nTotal endpoints in all plans: {total_endpoints}\n")
        f.write(f"Matching endpoints: {matched_endpoints}\n")
        f.write(f"\nPercentages:\n")
        f.write(f"- Correct: {correct / total * 100:.2f}%\n")
        f.write(f"- Partial: {partial / total * 100:.2f}%\n")
        f.write(f"- Incorrect: {incorrect / total * 100:.2f}%\n")
        f.write(f"- Endpoint match rate: {matched_endpoints / total_endpoints * 100:.2f}%\n")

# === CONFIGURATION ===
json_file = "smart-city-results/test-no-roles/execution_plans.json"
csv_file = "smart-city-requests/requests_no_roles.csv"
detailed_output = "smart-city-results/request_oracle_details.txt"
summary_output = "smart-city-results/request_oracle_summary.txt"

# === EXECUTION ===
json_data = load_json_questions(json_file)
oracle_map = load_csv_oracles(csv_file)
results = compare(json_data, oracle_map)
write_detailed_output(results, detailed_output)
write_summary_output(results, summary_output)

print("Analysis completed. Output saved to:")
print(f"  - {detailed_output}")
print(f"  - {summary_output}")
