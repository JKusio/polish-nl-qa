import uuid

from common.passage import Passage

def get_passages_for_embedding(dataset):
    unique_contexts = set((row['title'], row['context']) for row in dataset)

    return list(map(lambda row: Passage(generate_id(), row[1], row[0]), unique_contexts))

def generate_id(): 
    return str(uuid.uuid4())