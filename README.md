# 🤖 Retrieval-Augmented Generation (RAG) AI for Codebase Understanding KnowledgeGraph

In this project, I developed a RAG-based AI system that enables deep semantic understanding of large Python codebases by combining static code analysis with language modeling. The system parses source code using Python's Abstract Syntax Tree (AST) module to extract structured data and builds a Knowledge Graph that serves as the retrieval backbone for the generative model.

🧠 Core Components:
AST Parsing & Knowledge Graph Generation
Utilized Python’s ast module to statically analyze code and extract entities like classes, functions, and imports. These were transformed into a structured Knowledge Graph representing the code's architecture and relationships.

Embedding & Vector Store
Each code component (function, class, module) was embedded using a code-aware model like CodeBERT or OpenAI embeddings, and stored in a vector database (e.g., FAISS or Chroma) for efficient retrieval.

RAG Pipeline
Integrated a generative language model (e.g., GPT-4 or LLaMA) with the vector store to build a retrieval-augmented QA system. This enabled the model to answer developer questions about the codebase with accurate, grounded responses.

Use Cases

Context-aware code search and summarization

Automated documentation generation

Developer onboarding assistant

Refactoring impact analysis

This project showcases how static program analysis and LLM-based reasoning can be combined to deliver intelligent development tools — bridging the gap between raw code and human-friendly understanding.
