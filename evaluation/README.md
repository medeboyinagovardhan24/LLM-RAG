# RAG Evaluation Methodology

## What is being evaluated?
We are evaluating the performance of a Retrieval-Augmented Generation (RAG) system in providing domain-specific answers.

## Test Dataset
The dataset consists of 20 questions, split between domain-specific questions (related to uploaded documents) and out-of-domain questions (general knowledge).

## Relevance Score (0-1)
The relevance score is a graded metric evaluating the quality of generated answers compared to a ground-truth "expected answer".
* 1.0 (Completely relevant/correct)
* ...
* 0.0 (Irrelevant/incorrect)

The score is calculated via an "LLM-as-a-judge" approach where a separate, powerful language model evaluates the answers.

## Recall@K
Measures if the correct document chunk is retrieved within the Top-K results.

## RAG vs. Non-RAG comparison
- **RAG**: Question -> Context Retrieval -> LLM Generation
- **Non-RAG**: Question -> LLM Generation (direct)

## How to run
1. Ensure your virtual environment is active.
2. From the project root, run:
   `python evaluation/run_evaluation.py`

## Interpretation
Compare the average relevance scores of RAG vs. Non-RAG to quantify the effectiveness of your retrieval pipeline.
