try:
    import RAG.llm
    import RAG.vision
    import RAG.embeddings
    print('RAG modules import OK')
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
