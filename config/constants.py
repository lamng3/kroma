# default models
DEFAULT_CHAT_BACKEND = "togetherai"
DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DEFAULT_EMBED_MODEL = "allenai/scibert_scivocab_uncased"

# total number of tokens
TOTAL_TOKENS_LIMIT = 8000

# maximum “token budget” for cost normalization
N_TOKENS = 1_000_000

# per‑model cost metadata (costs are in $ per token)
METADATA = {
    "huggingface": {
        "google/flan-t5-xxl": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        },
        "meta-llama/Llama-2-7b-hf": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        },
        "mosaicml/mpt-7b": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        },
    },
    "togetherai": {
        "google/gemma-2b-it": {
            "cost_per_input_token": 0.1 / N_TOKENS,
            "cost_per_output_token": 0.1 / N_TOKENS,
        },
        "meta-llama/Llama-3.2-3B-Instruct-Turbo": {
            "cost_per_input_token": 0.06 / N_TOKENS,
            "cost_per_output_token": 0.06 / N_TOKENS,
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "cost_per_input_token": 0.2 / N_TOKENS,
            "cost_per_output_token": 0.2 / N_TOKENS,
        },
        "meta-llama-llama-2-70b-hf": {
            "cost_per_input_token": 0.9 / N_TOKENS,
            "cost_per_output_token": 0.9 / N_TOKENS,
        },
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
            "cost_per_input_token": 0.88 / N_TOKENS,
            "cost_per_output_token": 0.88 / N_TOKENS,
        },
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
            "cost_per_input_token": 0.88 / N_TOKENS,
            "cost_per_output_token": 0.88 / N_TOKENS,
        },
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        },
        "Qwen/Qwen2.5-7B-Instruct-Turbo": {
            "cost_per_input_token": 0.3 / N_TOKENS,
            "cost_per_output_token": 0.3 / N_TOKENS,
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
            "cost_per_input_token": 0.18 / N_TOKENS,
            "cost_per_output_token": 0.18 / N_TOKENS,
        },
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        },
    },
    "openai": {
        "chatgpt-4o-latest": {
            "cost_per_input_token": 5.0 / N_TOKENS,
            "cost_per_output_token": 15.0 / N_TOKENS,
        },
        "o3-mini": {
            "cost_per_input_token": 1.10 / N_TOKENS,
            "cost_per_output_token": 4.40 / N_TOKENS,
        },
        "o1-mini": {
            "cost_per_input_token": 1.10 / N_TOKENS,
            "cost_per_output_token": 4.40 / N_TOKENS,
        },
        "gpt-4o": {
            "cost_per_input_token": 2.5 / N_TOKENS,
            "cost_per_output_token": 10.0 / N_TOKENS,
        },
        "gpt-4o-mini": {
            "cost_per_input_token": 0.15 / N_TOKENS,
            "cost_per_output_token": 0.6 / N_TOKENS,
        },
        "gpt-3.5-turbo-0125": {
            "cost_per_input_token": 0.5 / N_TOKENS,
            "cost_per_output_token": 1.5 / N_TOKENS,
        },
    },
}

SPARQLEndpoints = {
    "nell": "http://nell-ld.telecom-st-etienne.fr/sparql", # (not working) paper = Nell2RDF: Read the Web, and turn it into RDF
    "dbpedia": "http://dbpedia.org/sparql",
    "ncit": "https://shared.semantics.cancer.gov/sparql",
    "doid": "https://sparql.disease-ontology.org/",
    "snomed": "http://purl.bioontology.org/ontology/SNOMEDCT/",
    "fma": "http://bioportal.bioontology.org/ontologies/FMA_RadLex",
    "wikidata": "https://query.wikidata.org/bigdata/namespace/wdq/sparql?query={SPARQL}"
}
