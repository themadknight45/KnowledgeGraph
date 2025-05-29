import openai
import os
import ast
import networkx as nx
import openai
import pickle
import time
import json
from concurrent.futures import ThreadPoolExecutor

# Set up your Azure OpenAI credentials
azure_endpoint = "azep"
api_key = "apikey"
api_version = "2024-10-21"
deployment_name = "gpt-4o"  # Your deployment name

# Create the OpenAI client
client = openai.AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)


def extract_code_structure(file_path, G, current, total, method_count):
    """Parses a Python file and extracts classes, methods, and relationships into a DAG."""
    start_time = time.time()
    print(f"[INFO] Processing file {current}/{total}: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    parent_stack = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            code_snippet = ast.unparse(node)
            G.add_node(node.name, type="class", metadata=code_snippet, file=file_path)
            parent_stack.append(node.name)
            
            for base in node.bases:
                if isinstance(base, ast.Name):  
                    G.add_edge(base.id, node.name, relation="inherits")
        
        elif isinstance(node, ast.FunctionDef):
            if parent_stack:
                class_name = parent_stack[-1]
                code_snippet = ast.unparse(node)
                G.add_node(node.name, type="method", metadata=code_snippet, file=file_path)
                G.add_edge(class_name, node.name, relation="contains")
                method_count.append(node.name)
        
        elif isinstance(node, ast.Call):
            caller = parent_stack[-1] if parent_stack else None
            
            if hasattr(node.func, 'id'):
                callee = node.func.id
            elif hasattr(node.func, 'attr') and hasattr(node.func, 'value') and isinstance(node.func.value, ast.Name):
                callee = f"{node.func.value.id}.{node.func.attr}"
            else:
                callee = None

            if caller and callee:
                G.add_edge(caller, callee, relation="calls")
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] Completed {file_path} in {elapsed_time:.2f} seconds.")
    return G
def analyze_with_gpt4o(batch_methods):
    """Uses GPT-4o to analyze a batch of code snippets."""
    if not batch_methods:
        return {}

    print(f"[INFO] Sending batch of {len(batch_methods)} methods to GPT-4o for analysis...")
    start_time = time.time()

    batch_code_snippets = "\n\n".join([
        f"Method: {name}\nCode:\n{code}" for name, code in batch_methods.items()
    ])
    prompt = f"""
    Analyze the following batch of functions and summarize their purposes:
    {batch_code_snippets}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Updated for GPT-4o
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] GPT-4o analysis for batch completed in {elapsed_time:.2f} seconds.")

    response_text = response.choices[0].message.content.split("\n\n")
    analysis_results = {name: response_text[i] if i < len(response_text) else "No response"
                        for i, name in enumerate(batch_methods.keys())}
    
    return analysis_results

def process_graph_with_llm(G, batch_size=15, max_requests_per_minute=45):
    """Parallel processing for GPT-4o analysis with rate limiting and progress updates."""
    print("[INFO] Starting GPT-4o analysis for extracted code components...")
    start_time = time.time()

    # Extract method nodes
    methods = {node: G.nodes[node].get("metadata", "") for node in G.nodes if G.nodes[node].get("type") == "method"}
    total_methods = len(methods)
    method_names = list(methods.keys())

    print(f"[INFO] Total methods to analyze: {total_methods}")

    # Create batches
    batches = [method_names[i:i + batch_size] for i in range(0, len(method_names), batch_size)]
    total_batches = len(batches)

    # Rate limiting
    max_batches_per_minute = max_requests_per_minute // batch_size
    wait_time = 60 / max_batches_per_minute if max_batches_per_minute > 0 else 60

    remaining_methods = total_methods
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, batch in enumerate(batches):
            future = executor.submit(analyze_with_gpt4o, {name: methods[name] for name in batch})
            results = future.result()
            for name in batch:
                G.nodes[name]["metadata"] = results.get(name, "Analysis unavailable.")

            # Update progress
            remaining_methods -= len(batch)
            print(f"[INFO] Processed batch {i+1}/{total_batches}. Methods remaining: {remaining_methods}")

            # Rate limiting enforcement
            if (i + 1) % max_batches_per_minute == 0:
                print(f"[INFO] Rate limit reached. Pausing for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    elapsed_time = time.time() - start_time
    print(f"[INFO] GPT-4o analysis phase completed in {elapsed_time:.2f} seconds.")

def generate_documentation(graph, output_path="knowledge_graph_documentation.json"):
    """Generates a detailed documentation from the knowledge graph."""
    documentation = {
        "metadata": {
            "description": "Knowledge graph representation of a Python codebase.",
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
        },
        "nodes": [],
        "edges": []
    }

    for node, attributes in graph.nodes(data=True):
        documentation["nodes"].append({
            "name": node,
            "type": attributes.get("type", "unknown"),
            "metadata": attributes.get("metadata", "No metadata available"),
            "file": attributes.get("file", "Unknown file")
        })

    for source, target, attributes in graph.edges(data=True):
        documentation["edges"].append({
            "source": source,
            "target": target,
            "relation": attributes.get("relation", "unknown relation")
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documentation, f, indent=4)

    print(f"✅ Documentation saved as '{output_path}'")

def build_knowledge_graph(directory):
    """Process all Python files in a directory with progress tracking."""
    start_time = time.time()
    print("[INFO] Scanning directory for Python files...")
    
    G = nx.DiGraph()
    method_count = []
    
    python_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".py")]
    total_files = len(python_files)
    print(f"[INFO] Found {total_files} Python files.")
    
    for index, file_path in enumerate(python_files, start=1):
        G = extract_code_structure(file_path, G, index, total_files, method_count)
    
    print(f"[INFO] Total methods found: {len(method_count)}")
    print("[INFO] Starting GPT-4o-based processing of extracted code...")
    
    process_graph_with_llm(G)
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] Knowledge graph construction completed in {elapsed_time:.2f} seconds.")
    
    return G
# **Modify this path to point to your larger codebase**
codebase_path = r"C:\Users\rajrishi\OneDrive - Microsoft\Desktop\flask-main"

# **Generate Knowledge Graph**
print("[INFO] Starting knowledge graph generation...")
global_start_time = time.time()
graph = build_knowledge_graph(codebase_path)

# **Save Graph Using Pickle**
print("[INFO] Saving knowledge graph to file...")
with open("knowledge_graph_large.gpickle", "wb") as f:
    pickle.dump(graph, f)

# **Generate Documentation**
generate_documentation(graph)

total_execution_time = time.time() - global_start_time
print(f"✅ Knowledge graph saved successfully as 'knowledge_graph_large.gpickle'.")
print(f"⏱️ Total execution time: {total_execution_time:.2f} seconds.")