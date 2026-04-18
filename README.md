# OCI AI Agents Workshop

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white) ![Black](https://img.shields.io/badge/code%20style-black-000000.svg) ![Pylint](https://img.shields.io/badge/linting-pylint-2A6AC7?logo=python&logoColor=white) ![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

Build and run practical AI agent workflows on **OCI Enterprise AI** with a
hands-on path from first API call to retrieval-augmented generation.

This workshop includes:
- quickstart programs for chat streaming, LangGraph orchestration, and embeddings;
- a simple RAG agent exposed via FastAPI, with support for fake KB or local PDF ingestion;
- shared runtime/model utilities to keep OCI configuration consistent across examples.

Special attention is given to **DAC (Dedicated AI Cluster)** deployments and
runtime configuration, so you can move from local experimentation to
enterprise-oriented serving patterns.

## Quickstart

### 1. Set required IAM policies

Before running the examples, create a dedicated compartment for this workshop and
configure IAM policies for your user group.
These policies must be set by a Tenant Admin.

Policy to set in your tenancy:

```text
allow group <your-group-name> to manage generative-ai-family in compartment <your-compartment-name>
```

If you use a Dedicated AI Cluster (DAC) in a different compartment, set the same
policy for the DAC compartment as well:

```text
allow group <your-group-name> to manage generative-ai-family in compartment <your-dac-compartment-name>
```

Reference:
[OCI Generative AI IAM Policies](https://docs.oracle.com/en-us/iaas/Content/generative-ai/iam-policies.htm)

### 2. Create and activate a Python environment

If you use **Anaconda/Miniconda**:

```bash
conda create -n oci_agents_workshop python=3.11 -y
conda activate oci_agents_workshop
```

If you prefer standard Python virtual environments (`venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install required libraries

For running the examples only:

```bash
pip install langchain langgraph langchain-oci langchain-text-splitters python-dotenv fastapi uvicorn pypdf tqdm numpy
```

If you are doing development in this repo and want to follow all repository conventions, also install:

```bash
pip install pytest black pylint
```

### 4. Configure OCI authentication

For simplicity, these examples use local OCI configuration with
`auth_type="API_KEY"` (USER_PRINCIPAL).
This means OCI credentials must be configured in:

```text
$HOME/.oci/config
```

Set up OCI API authentication by following the official guide:
[Oracle Cloud Infrastructure SDK and CLI Configuration File](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm).

### 5. Configure environment variables

Copy the sample file and set real values:

```bash
cp .env-sample .env
```

Then edit `.env` and set at least:

- `OCI_COMPARTMENT_ID` (required)
- `OCI_REGION` (for endpoint generation)
- `OCI_MODEL_ID`
- `OCI_EMBED_MODEL_ID` (embedding model for `quickstart/embedding01.py` and `simple_rag_agent`)
- `SIMPLE_RAG_TOP_K` (number of retrieved documents for `simple_rag_agent`, default: `4`)
- `OCI_PROVIDER`
- `OCI_AUTH_TYPE`
- `OCI_AUTH_PROFILE`

### 6. Run the first configuration test (`quickstart/test01.py`)

```bash
python -m quickstart.test01
```

This is the first test to run after environment setup. If it prints the runtime
configuration and returns a streamed response, your OCI configuration is working.

### 7. Run other examples

```bash
python -m quickstart.agent01 "Explain what a Dedicated AI Cluster is in OCI."
python -m quickstart.embedding01
```

## DAC Notes

When using DAC in these examples, the `OCI_MODEL_ID` value must be the OCID of the DAC endpoint.
The DAC endpoint must be public for these examples.

## Oracle Support Libraries

Integration with OCI Enterprise AI can be implemented in different ways.
One practical option, especially for LangChain/LangGraph development, is based on:

- `langchain-oci`
- `langchain-oracledb`

Reference repository:
[oracle/langchain-oracle](https://github.com/oracle/langchain-oracle)
