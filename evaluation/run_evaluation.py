import json
import csv
import sys
from pathlib import Path
import uuid

# --- ROBUST IMPORT FIX ---
backend_dir = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(backend_dir))

from app.services import (
    get_collection,
    get_generator,
    retrieve_context,
    build_prompt,
)

def evaluate_answer(question, expected, generated, generator):
    """LLM-as-a-judge to evaluate generated answer against the expected answer."""
    if generated is None:
        return {"score": None, "reason": "Generation failed."}

    prompt = f"""
Evaluate the generated answer against the expected answer on a scale of 0.0 to 1.0.
0.0 = completely incorrect or irrelevant
1.0 = completely correct, relevant, and accurate
Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}
Provide the response strictly in JSON format:
{{
    "score": 0.0,
    "reason": "Brief explanation"
}}
"""
    try:
        response = generator.generate(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            return {"score": None, "reason": "Evaluator did not return valid JSON."}
        result = json.loads(response[start:end])
        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"score": score, "reason": result.get("reason", "")}
    except Exception as e:
        return {"score": None, "reason": f"Evaluation failed: {str(e)}"}

def run_evaluation():
    eval_dir = Path(__file__).resolve().parent
    test_file = eval_dir / "test_questions.json"

    with open(test_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    collection = get_collection()
    generator = get_generator()
    results = []

    r1, r3, r5, total_with_gt = 0, 0, 0, 0
    successful_evals, failed_evals = 0, 0
    ood_count = 0
    unsupported_count = 0

    print(f"\nStarting evaluation of {len(questions)} questions...\n")

    for q in questions:
        print(f"Processing ID {q['id']}: {q['question'][:50]}...")

        # 1. RETRIEVAL
        try:
            docs, metas = retrieve_context(collection, q["question"], top_k=5)
            retrieved_filenames = [m.get("filename") for m in metas]
        except Exception:
            docs, metas = [], []
            retrieved_filenames = []

        ground_truth = q.get("ground_truth_filename")
        if ground_truth:
            total_with_gt += 1
            if ground_truth in retrieved_filenames[:1]: r1 += 1
            if ground_truth in retrieved_filenames[:3]: r3 += 1
            if ground_truth in retrieved_filenames[:5]: r5 += 1

        # 2. RAG GENERATION & EVAL
        rag_ans, rag_score, rag_reason = None, None, "Generation failed"
        try:
            prompt = build_prompt(q["question"], docs, metas)
            rag_ans = generator.generate(prompt)
            rag_eval = evaluate_answer(q["question"], q["expected_answer"], rag_ans, generator)
            rag_score = rag_eval["score"]
            rag_reason = rag_eval["reason"]
        except Exception as e:
            rag_reason = f"Generation exception: {str(e)}"

        # 3. NON-RAG GENERATION & EVAL
        non_rag_ans, non_rag_score, non_rag_reason = None, None, "Generation failed"
        try:
            non_rag_prompt = f"Answer concisely: {q['question']}"
            non_rag_ans = generator.generate(non_rag_prompt)
            non_rag_eval = evaluate_answer(q["question"], q["expected_answer"], non_rag_ans, generator)
            non_rag_score = non_rag_eval["score"]
            non_rag_reason = non_rag_eval["reason"]
        except Exception as e:
            non_rag_reason = f"Generation exception: {str(e)}"

        # 4. OUT-OF-DOMAIN
        if q.get("is_out_of_domain", False):
            ood_count += 1
            if rag_ans:
                rag_lower = rag_ans.lower()
                indicators = ["not found", "not available", "outside", "cannot answer", "no information"]
                if not any(i in rag_lower for i in indicators):
                    unsupported_count += 1

        if rag_score is not None and non_rag_score is not None:
            successful_evals += 1
        else:
            failed_evals += 1

        results.append({
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "rag_answer": rag_ans,
            "rag_score": rag_score,
            "rag_evaluation_reason": rag_reason,
            "non_rag_answer": non_rag_ans,
            "non_rag_score": non_rag_score,
            "non_rag_evaluation_reason": non_rag_reason,
            "retrieved_sources": json.dumps(retrieved_filenames),
            "retrieval_success": len(docs) > 0,
            "ground_truth_filename": ground_truth,
            "recall_at_1": 1 if (ground_truth and ground_truth in retrieved_filenames[:1]) else 0,
            "recall_at_3": 1 if (ground_truth and ground_truth in retrieved_filenames[:3]) else 0,
            "recall_at_5": 1 if (ground_truth and ground_truth in retrieved_filenames[:5]) else 0,
            "is_out_of_domain": q.get("is_out_of_domain", False),
            "evaluation_status": "success" if rag_score is not None else "failed"
        })

    # AGGREGATE
    valid_rag = [r["rag_score"] for r in results if r["rag_score"] is not None]
    valid_non_rag = [r["non_rag_score"] for r in results if r["non_rag_score"] is not None]

    avg_rag = sum(valid_rag) / len(valid_rag) if valid_rag else None
    avg_non_rag = sum(valid_non_rag) / len(valid_non_rag) if valid_non_rag else None
    rag_imp = (avg_rag - avg_non_rag) if (avg_rag is not None and avg_non_rag is not None) else None

    # CSV
    with open(eval_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # JSON REPORT
    report = {
        "total_questions": len(results),
        "questions_with_ground_truth": total_with_gt,
        "successful_evaluations": successful_evals,
        "failed_evaluations": failed_evals,
        "average_rag_relevance": round(avg_rag, 3) if avg_rag is not None else None,
        "average_non_rag_relevance": round(avg_non_rag, 3) if avg_non_rag is not None else None,
        "rag_improvement": round(rag_imp, 3) if rag_imp is not None else None,
        "rag_improvement_type": "absolute_difference",
        "recall_at_1": round(r1 / total_with_gt, 3) if total_with_gt > 0 else None,
        "recall_at_3": round(r3 / total_with_gt, 3) if total_with_gt > 0 else None,
        "recall_at_5": round(r5 / total_with_gt, 3) if total_with_gt > 0 else None,
        "out_of_domain_questions": ood_count,
        "unsupported_answer_count": unsupported_count
    }
    with open(eval_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "="*50)
    print("             RAG EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Questions:             {report['total_questions']}")
    print(f"Successful Evaluations:      {report['successful_evaluations']}")
    print(f"Failed Evaluations:          {report['failed_evaluations']}")
    print(f"Questions with Ground Truth: {report['questions_with_ground_truth']}")
    print(f"Average RAG Relevance:       {report['average_rag_relevance']}")
    print(f"Average Non-RAG Relevance:   {report['average_non_rag_relevance']}")
    print(f"RAG Improvement:             {report['rag_improvement']}")
    print(f"Recall@1:                    {report['recall_at_1']}")
    print(f"Recall@3:                    {report['recall_at_3']}")
    print(f"Recall@5:                    {report['recall_at_5']}")
    print(f"Out-of-Domain Questions:     {report['out_of_domain_questions']}")
    print(f"Unsupported Answers:         {report['unsupported_answer_count']}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
