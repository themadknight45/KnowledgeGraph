from azure.cosmos import CosmosClient
from openai import AzureOpenAI

# Azure Cosmos DB Connection
COSMOS_CONNECTION_STRING = "AccountEndpoint=https://knowledgebase.documents.azure.com:443/;AccountKey=ackey"
dbclient = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
database = dbclient.get_database_client("knowledgebase")
container = database.get_container_client("VectorStore")

# Azure OpenAI Configuration
embeddings_client = AzureOpenAI(
    api_key="key",
    azure_endpoint="azep",
    api_version="2024-10-21",
)

# Function to create embeddings for queries
def create_embeddings(text_chunk, model="text-embedding-ada-002"):
    response = embeddings_client.embeddings.create(input=text_chunk, model=model)
    return response.data[0].embedding

# Function to retrieve most relevant document snippets
def get_nearest_vectors(query_vector, k=5):
    query = '''
        SELECT TOP @k c.text
        FROM c 
        WHERE VectorDistance(c.embedding, @embedding) < 1
    '''
    parameters = [
        {"name": "@embedding", "value": query_vector},
        {"name": "@k", "value": k}
    ]

    results = container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)

    return [result["text"] for result in results]

# Chatbot loop
def chatbot():
    while True:
        user_input = input("\nAsk your query (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        query_vector = create_embeddings(user_input)

        # Retrieve relevant documents
        relevant_docs = get_nearest_vectors(query_vector, k=5)
        if not relevant_docs:
            print("I couldn't find relevant information in the knowledge base.")
            continue

        # Prepare context
        context = "\n".join(relevant_docs)
        messages = [
            {"role": "system", "content": "You are an AI assistant helping with AI-related queries."},
            {"role": "system", "content": f"Here are some relevant snippets from the knowledge base:\n{context}"},
            {"role": "user", "content": user_input}
        ]

        # Get response from OpenAI
        response = embeddings_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=800,
            messages=messages,
            stream=True
        )

        collected_response = []  # Store chunks before printing

        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                content = chunk.choices[0].delta.content  # Correct access method
                if content:
                    collected_response.append(content)

        # Print final response only if content exists
        if collected_response:
            print("\n🔹 Response:\n" + "".join(collected_response) + "\n")
        else:
            print("⚠️ No content received from API!")


# Run chatbot
chatbot()
