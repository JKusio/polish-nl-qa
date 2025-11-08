import json
import os
from cache.cache import Cache


def load_curated_batch_results():
    """
    Load GPT-4o-mini curated batch results to cache.
    Maps batch_*_output.jsonl files to proper curated dataset names.
    """
    cache = Cache()

    # Directory paths (both src/openai_batches/ for batch files)
    input_dir = "openai_batches"

    # Find all batch output files in src/openai_batches/
    output_files = []
    for filename in os.listdir(input_dir):
        if filename.startswith("batch_") and filename.endswith("_output.jsonl"):
            output_files.append(os.path.join(input_dir, filename))

    print(f"Found {len(output_files)} batch output files")

    # Find all curated input files
    input_files = []
    for filename in os.listdir(input_dir):
        if (
            "curated" in filename
            and filename.startswith("gpt-4o-mini_")
            and not filename.endswith("_output.jsonl")
        ):
            input_files.append(os.path.join(input_dir, filename))

    print(f"Found {len(input_files)} curated input files")

    # Build mapping: hash -> input filename
    hash_to_input = {}
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                custom_id = data["custom_id"]
                hash_to_input[custom_id] = os.path.basename(input_file)

    print(f"Loaded {len(hash_to_input)} unique hashes from input files")

    # Process each output file
    file_mapping = {}
    total_loaded = 0

    for output_file in output_files:
        # Read first line to get a custom_id
        with open(output_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            first_data = json.loads(first_line)
            sample_hash = first_data["custom_id"]

        # Find which input file this belongs to
        if sample_hash in hash_to_input:
            input_filename = hash_to_input[sample_hash]
            file_mapping[output_file] = input_filename

            # Load all answers from this file to cache
            count = 0
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    custom_id = data["custom_id"]

                    # Extract answer from response
                    try:
                        answer = data["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                        cache.set(custom_id, answer)
                        count += 1
                        total_loaded += 1
                    except (KeyError, IndexError) as e:
                        print(
                            f"Warning: Could not extract answer for {custom_id[:50]}..."
                        )

            print(
                f"✓ {os.path.basename(output_file)} -> {input_filename}: {count} answers"
            )
        else:
            print(
                f"✗ {os.path.basename(output_file)}: Could not map to input file (hash not found)"
            )

    print(f"\n✅ Loaded {total_loaded} GPT-4o-mini curated answers to cache")
    print(f"\nFile mapping:")
    for output_file, input_file in file_mapping.items():
        print(f"  {os.path.basename(output_file)} -> {input_file}")

    return total_loaded


if __name__ == "__main__":
    load_curated_batch_results()
