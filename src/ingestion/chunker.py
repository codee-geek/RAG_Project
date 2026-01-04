from langchain_text_splitters import RecursiveCharacterTextSplitter


def hybrid_chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(docs)
