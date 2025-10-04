# RAGAS Evaluator V2 - Key Improvements

## Summary of Enhancements

### 1. **Faithfulness (Most Important Fix)**
**Problem**: Binary 0.5 threshold loses granularity and is too strict
**Solution**: 
- Uses continuous scores from reranker
- Applies sigmoid transformation for smooth 0-1 scaling
- Takes max similarity across all context sentences (allows paraphrasing)
- More forgiving of semantic equivalence

### 2. **Answer Relevance**
**Problem**: Weak prompt and fragile parsing
**Solution**:
- Improved prompt with clearer instructions
- Robust parsing that handles multiple formats (1., 1), -, *)
- Fallback parsing strategies
- Better error handling for edge cases

### 3. **Query-to-Context Relevance**
**Problem**: All passages weighted equally
**Solution**:
- Position-based weighting (earlier passages more important)
- Exponential decay: 1st passage = 1.0, 2nd = 0.5, 3rd = 0.33, etc.
- Better reflects retriever quality

### 4. **NEW: Answer Correctness**
**Addition**: Direct comparison with ground truth
- Semantic similarity (70%) + keyword overlap (30%)
- Handles multiple correct answers (takes max score)
- Provides objective quality measure

### 5. **RAGAS Score Rebalancing**
**Problem**: Simple averaging with binary metrics skews scores
**Solution**:
- Retrieval quality (precision + recall) = 30%
- Answer quality (faithfulness + relevance) = 70%
- If ground truth available: base score 60% + correctness 40%
- Score now ranges properly from 0 to 1

### 6. **Hallucination Detection**
**Improvement**: Better weighting
- Faithfulness: 60% (most important)
- Answer relevance: 25%
- Context relevance: 15%

## How to Use

```python
from evaluation.ragas_evaulator_v2 import RAGASEvaluatorV2

# Initialize (same as before)
ragas_v2 = RAGASEvaluatorV2(
    reranker_model_name="sdadas/polish-reranker-large-ranknet",
    cache=cache,
    generator_model_name="../../models/Bielik-11B-v2.2-Instruct-q4",
    vectorizer=vectorizer
)

# Evaluate WITH ground truth (recommended)
score = ragas_v2.ragas(
    retriever_result, 
    entry.passage_id, 
    answer,
    correct_answers=entry.answers  # NEW parameter
)

# Or without ground truth (like before)
score = ragas_v2.ragas(retriever_result, entry.passage_id, answer)

# Individual metrics still available
faithfulness = ragas_v2.faithfulness(result, answer)
answer_rel = ragas_v2.answer_relevance(query, answer)
correctness = ragas_v2.answer_correctness(answer, correct_answers)
```

## Expected Improvements

1. **Better discrimination**: Scores will spread more (not clustered around 0.9)
2. **More accurate**: Good answers with paraphrasing won't be penalized
3. **Ground truth aware**: When available, directly measures correctness
4. **Less random**: Continuous scores instead of binary thresholds
5. **Better weighting**: Answer quality weighted more than retrieval

## Migration Guide

Simply replace:
```python
from evaluation.ragas_evaulator import RAGASEvaluator
ragas = RAGASEvaluator(...)
```

With:
```python
from evaluation.ragas_evaulator_v2 import RAGASEvaluatorV2
ragas = RAGASEvaluatorV2(...)
```

And optionally add `correct_answers` parameter to get even better evaluation!
