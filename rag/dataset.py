from llama_index.core.llama_dataset import download_llama_dataset

# a LabelledRagDataset and a list of source Document's
rag_dataset, documents = download_llama_dataset("PaulGrahamEssayDataset", "./data")
