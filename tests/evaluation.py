import json
import re
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.llm_answer import answer_query

# -------------------------
# Normalization utilities
# -------------------------

def normalize(text: str) -> str:
    return text.lower().strip()

# -------------------------
# Evaluation functions
# -------------------------

def evaluate_answers(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        pred_answer = normalize(pred['answer']['value'])
        gt_answer = normalize(gt['expected_answer']['value'])
        
        if pred_answer in gt_answer or gt_answer in pred_answer:
            correct += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total
    }

def evaluate_retrieval(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        pred_citations = [normalize(c['text']) for c in pred.get('citations', [])]
        gt_spans = [normalize(span) for span in gt.get('citation_spans', [])]
        
        if any(span in citation for citation in pred_citations for span in gt_spans):
            correct += 1
    
    return {
        "recall": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total
    }

# -------------------------
# Main evaluation
# -------------------------

def main():
    # Load ground truth
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'test_dataset.json')
    with open(dataset_path) as f:
        questions = json.load(f)["questions"]
    
    predictions = []
    
    print("Starting evaluation...")
    
    for i, q in enumerate(questions):
        print(f"Evaluating question {i+1}/{len(questions)}: {q['question']}")
        
        result = answer_query(q['question'], q['qid'])
        predictions.append(result)
    
    # Evaluate
    answer_metrics = evaluate_answers(predictions, questions)
    retrieval_metrics = evaluate_retrieval(predictions, questions)
    
    grounded_correct = sum(1 for p, q in zip(predictions, questions) 
                          if answer_metrics['correct'] and retrieval_metrics['correct'])
    
    grounded_accuracy = grounded_correct / len(questions) if questions else 0
    
    print("\nEvaluation Results:")
    print(f"Answer Accuracy: {answer_metrics['accuracy']:.2%}")
    print(f"Retrieval Recall: {retrieval_metrics['recall']:.2%}")
    print(f"Grounded Accuracy: {grounded_accuracy:.2%}")
    
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'evaluation_results.json')
    with open(results_path, "w") as f:
        json.dump({
            "summary": {
                "answer_accuracy": answer_metrics['accuracy'],
                "retrieval_recall": retrieval_metrics['recall'],
                "grounded_accuracy": grounded_accuracy
            },
            "results": predictions
        }, f, indent=2)
    
    print("Results saved to test/evaluation_results.json")

if __name__ == "__main__":
    main()
