# RAGAS V2 Re-evaluation - Complete Guide

## What This Does

The notebook `04_ragas_v2_reevaluation.ipynb` will:

### 1. **Regenerate ALL Answers** ✅
- **Cache Key Changed**: `instruction_v2` → `instruction_v3`
- **NOT using old cache** - will generate fresh answers for all questions
- **New answers saved** to cache under `instruction_v3` key
- Uses improved prompt that strictly forbids reasoning/explanations

### 2. **Improved Prompt** ✅
```python
"Odpowiedz TYLKO na pytanie użytkownika wykorzystując wyłącznie informacje z dokumentów. 
NIE dodawaj wyjaśnień, rozumowań ani dodatkowych komentarzy. 
Podaj TYLKO konkretną, zwięzłą odpowiedź.

### Odpowiedź (tylko sama odpowiedź bez wyjaśnień):"
```

Key changes:
- **"TYLKO"** in caps for emphasis
- **"NIE dodawaj wyjaśnień, rozumowań"** - explicitly forbids reasoning
- Cleaner formatting

### 3. **Evaluate with RAGAS V2** ✅
Uses the improved RAGAS V2 evaluator with:
- Continuous faithfulness scoring (not binary)
- Answer correctness vs ground truth
- Better metric weighting (70% answer quality, 30% retrieval)
- Position-weighted context relevance

### 4. **Generate Manual Evaluation Files** ✅
Creates **56 CSV files** (one per configuration) in format:
```
# RETRIEVER: <retriever_name>
# GENERATOR: <generator_name>
# TYPE: INST
# DATASET: <dataset_name>
# TOP_N: <n>
# CACHE_VERSION: instruction_v3
# RAGAS_VERSION: v2

question,question_id,hasCorrectPassages,answer,correct_answer,ragas_v2_score,manual_result
```

**New column**: `ragas_v2_score` - shows automated RAGAS V2 score for comparison with your manual evaluation

### 5. **Save Summary Statistics** ✅
- `ragas_v2_summary.csv` - Aggregate scores per configuration
- `ragas_v2_detailed.csv` - Per-question detailed results

## File Structure

```
output/
└── ragas_v2/
    ├── ragas_v2_summary.csv          # Summary scores by configuration
    ├── ragas_v2_detailed.csv         # Detailed per-question results
    └── manual_eval/                  # 56 CSV files for manual evaluation
        ├── ragas_v2_poquad_*.csv     # PoQuAD files
        └── ragas_v2_polqa_*.csv      # PolQA files
```

## Configuration Coverage

### PoQuAD (28 files)
- 3 retrievers (best, 50p, worst)
- 4 instruction models
- 2 OpenAI retrievers
- n = [1, 5]
- **= 3×4×2 + 2×1×2 = 24 + 4 = 28 files**

### PolQA (28 files)
- Same structure as PoQuAD
- **= 28 files**

### Total: **56 files**

## How It Works

1. **Retrieval**: Uses cached retrieval results (fast, no re-retrieval)
2. **Generation**: 
   - Checks cache with key: `hash(query + context + "instruction_v3" + model_name)`
   - If not found: generates new answer with improved prompt
   - Saves to cache under `instruction_v3` key
3. **Evaluation**: Runs RAGAS V2 on each answer
4. **File Creation**: Saves CSV with RAGAS scores for manual comparison

## Key Differences from Original

| Aspect | Original (notebook 03) | New (notebook 04) |
|--------|----------------------|-------------------|
| Cache Key | `instruction_v2` | `instruction_v3` |
| Answers | Uses old cache | **Regenerates all** |
| Prompt | Weaker | **Much stronger** |
| Evaluation | None | **RAGAS V2** |
| Files | Manual eval only | **Manual eval + RAGAS scores** |
| Reasoning | Some models add it | **Should be removed** |

## Expected Results

### Answer Quality
- ✅ Direct answers without reasoning
- ✅ No "wyjaśnienie z dokumentu wynika że..."
- ✅ Just the answer: e.g., "niemieckiej" instead of long explanation

### RAGAS V2 Scores
- Better score distribution (not clustered at 0.9)
- More correlation with actual answer quality
- Scores include ground truth correctness

## Usage

1. Open notebook: `src/notebooks/04_ragas_v2_reevaluation.ipynb`
2. Run all cells
3. Wait for completion (~15-30 mins depending on models)
4. Check results:
   - Sample answers in notebook output
   - CSV files in `output/ragas_v2/manual_eval/`
   - Summary in `output/ragas_v2/ragas_v2_summary.csv`

## Verification

After running, check a few manual evaluation files to confirm:
1. ✅ Answers are direct (no reasoning)
2. ✅ RAGAS V2 scores are in ragas_v2_score column
3. ✅ All 56 files created
4. ✅ Scores correlate better with answer quality

## Next Steps

1. Run the notebook to generate new files
2. Perform manual evaluation on subset of files
3. Compare your manual scores with RAGAS V2 scores
4. If RAGAS V2 correlates well, use it for future evaluations
5. If prompt still produces reasoning, we can make it even stronger

---

**Bottom Line**: This will regenerate everything with better prompts and better evaluation, giving you clean answers and more accurate automated scores! 🎯
