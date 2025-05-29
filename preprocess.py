from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import json

# Azure Cosmos DB Connection
COSMOS_ENDPOINT = "ep"
COSMOS_CONNECTION_STRING = "AccountEndpoint=ep/;AccountKey=ack"

dbclient = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
database = dbclient.create_database_if_not_exists(id="knowledgebase")

# Create container for storing embeddings
container = database.create_container_if_not_exists(
    id="VectorStore",
    partition_key=PartitionKey(path="/file")
)

# Azure OpenAI Configuration
embeddings_client = AzureOpenAI(
    api_key="apik",
    azure_endpoint="azep",
    api_version="2024-10-21",
)

# Function to split text into chunks
def split_text(text, max_length=1000, min_length=500):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) >= max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk and len(' '.join(current_chunk)) >= min_length:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to create embeddings
def create_embeddings(text_chunk, model="text-embedding-ada-002"):
    response = embeddings_client.embeddings.create(input=text_chunk, model=model)
    return response.data[0].embedding

# Read documentation file
with open(r"C:\Users\rajrishi\OneDrive - Microsoft\Desktop\documentation.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

# Split text into chunks
chunks = split_text(input_text)
print(f"✅ Number of chunks: {len(chunks)}")

# Generate embeddings and store them in Cosmos DB
for chunk in chunks:
    embedding = create_embeddings(chunk)
    item_id = str(hash(chunk))  # Unique ID

    try:
        container.create_item(
            body={
                'id': item_id,
                'embedding': embedding,
                'text': chunk,
                'file': "file1"  # Partition key value
            },
        )
    except Exception as e:
        print(f"Error storing item: {e}")

print("✅ All embeddings have been stored successfully.")
