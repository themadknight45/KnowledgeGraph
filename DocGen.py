import openai
import pickle
import time
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up your Azure OpenAI credentials
azure_endpoint = "azep"
api_key = "key"
api_version = "2024-10-21"
deployment_name = "gpt-4o"  # Your deployment name

# Create the OpenAI client
client = openai.AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)
# **Rate Limits**
MAX_REQUESTS_PER_MIN = 48
MAX_TOKENS_PER_MIN = 8000
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MIN  # ~1.25 sec delay per request

# **Token Estimation**
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def estimate_tokens(text):
    return len(tokenizer.encode(text))

def generate_documentation(gpickle_path, output_path="documentation.txt"):
    """Generates documentation while respecting API rate limits."""
    print(f"[INFO] Loading knowledge graph from: {gpickle_path}")

    # Load the graph
    with open(gpickle_path, "rb") as f:
        graph = pickle.load(f)

    print(f"[INFO] Graph loaded successfully! Total nodes: {len(graph.nodes)}")
    
    nodes = list(graph.nodes)
    total_methods = len(nodes)
    start_time = time.time()  # Define start_time here

    # **Track API usage**
    used_tokens = 0
    last_request_time = time.time()
    batch_size = max(1, min(10, MAX_TOKENS_PER_MIN // 100))  # Dynamic batch sizing

    def document_batch(batch_nodes, start_time):
        """Processes a batch while ensuring token & request limits."""
        nonlocal used_tokens, last_request_time
        
        batch_info = "\n\n".join([
            f"Node: {node}\nType: {graph.nodes[node].get('type', 'unknown')}\n"
            f"Metadata: {graph.nodes[node].get('metadata', 'No metadata')}\n"
            f"File: {graph.nodes[node].get('file', 'Unknown file')}\n"
            f"Children: {list(graph.successors(node))}\n"
            f"Parents: {list(graph.predecessors(node))}"
            for node in batch_nodes
        ])

        estimated_tokens = estimate_tokens(batch_info)
        prompt = f"Generate detailed documentation for the following code structure:\n\n{batch_info}"

        # **Wait if exceeding token limit**
        while used_tokens + estimated_tokens > MAX_TOKENS_PER_MIN:
            wait_time = max(0, 60 - (time.time() - start_time))
            print(f"[RATE LIMIT] Exceeded token limit. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            used_tokens = 0
            start_time = time.time()  # Reset start_time after waiting

        # **Wait if exceeding request limit**
        elapsed = time.time() - last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

        # **Make API call with retry mechanism**
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                used_tokens += estimated_tokens
                last_request_time = time.time()
                print(f"[SUCCESS] Processed batch of {len(batch_nodes)} nodes. Used tokens: {used_tokens}/{MAX_TOKENS_PER_MIN}")
                return response.choices[0].message.content
            except openai.RateLimitError:
                wait_time = (2 ** attempt)  # Exponential backoff
                print(f"[RATE LIMIT] Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return "[ERROR] Failed to get response"

    batches = [nodes[i:i + batch_size] for i in range(0, total_methods, batch_size)]
    completed_batches = 0

    print(f"[INFO] Processing {len(batches)} batches with batch size {batch_size}...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        with open(output_path, "w", encoding="utf-8") as f:
            futures = {executor.submit(document_batch, batch, start_time): batch for batch in batches}

            for future in as_completed(futures):
                result = future.result()
                f.write(result + "\n\n")

                # **Progress tracking**
                completed_batches += 1
                methods_left = total_methods - (completed_batches * batch_size)
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / completed_batches
                estimated_time_left = (len(batches) - completed_batches) * avg_time_per_batch

                print(f"[PROGRESS] {completed_batches}/{len(batches)} batches completed. "
                      f"Methods left: {methods_left}. "
                      f"Estimated time left: {estimated_time_left:.2f} seconds.")

    print(f"✅ Documentation saved as '{output_path}'")

# **Run the function**
gpickle_path = r"C:\Users\rajrishi\OneDrive - Microsoft\Desktop\knowledge_graph_large.gpickle"
generate_documentation(gpickle_path)
