from src.train import train
from src.rag import run_rag

if __name__ == "__main__":
    train()
    response = run_rag("What is RAG?")
    print("response is", response)
