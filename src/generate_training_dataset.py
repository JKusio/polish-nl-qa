from dataset.poquad_dataset_getter import PoquadDatasetGetter
from dataset.polqa_dataset_getter import PolqaDatasetGetter
import random
from common.dataset_entry import DatasetEntry
import json


def main():
    poquad_dataset_getter = PoquadDatasetGetter()
    polqa_dataset_getter = PolqaDatasetGetter()

    poquad_dataset = poquad_dataset_getter.get_training_dataset()
    polqa_dataset = polqa_dataset_getter.get_training_dataset()

    print(polqa_dataset[0].answers)

    random_2k_poquad = random.sample(poquad_dataset, 5000)
    random_2k_polqa = random.sample(polqa_dataset, 5000)

    prompts = []

    for entry in random_2k_poquad:
        prompts.extend(
            get_prompt_for_training_dataset(entry, poquad_dataset, polqa_dataset)
        )

    for entry in random_2k_polqa:
        prompts.extend(
            get_prompt_for_training_dataset(entry, poquad_dataset, polqa_dataset)
        )

    shuffled_prompts = random.sample(prompts, len(prompts))

    split_point = int(len(prompts) * 0.8)
    training_split = shuffled_prompts[:split_point]
    remaining_prompts = shuffled_prompts[split_point:]
    midpoint = int(len(remaining_prompts) / 2)
    validation_split = remaining_prompts[:midpoint]
    test_split = remaining_prompts[midpoint:]

    with open("output/train.jsonl", "w", encoding="utf-8") as f:
        for prompt in training_split:
            f.write(
                json.dumps(
                    {"text": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open("output/valid.jsonl", "w", encoding="utf-8") as f:
        for prompt in validation_split:
            f.write(
                json.dumps(
                    {"text": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open("output/test.jsonl", "w", encoding="utf-8") as f:
        for prompt in test_split:
            f.write(
                json.dumps(
                    {"text": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_prompt_for_training_dataset(
    entry: DatasetEntry,
    poquad_dataset: list[DatasetEntry],
    polqa_dataset: list[DatasetEntry],
) -> list[str]:
    question = f"[Q] {entry.question} [/Q]"

    contexts = [entry.context]

    rand = random.randint(0, 2)
    for _ in range(0, rand):
        context = random.choice(poquad_dataset).context
        contexts.append(context)

    rand = random.randint(0, 2)
    for _ in range(0, rand):
        context = random.choice(polqa_dataset).context
        contexts.append(context)

    shuffled_contexts = random.sample(contexts, len(contexts))

    context = f"[C] {'\n'.join(shuffled_contexts)} [/C]"

    prompts = []

    for answer in entry.answers:
        answer_prompt = f"[A] {answer} [/A]"
        prompts.append(f"{question}\n{context}\n{answer_prompt}")

    return prompts


main()
