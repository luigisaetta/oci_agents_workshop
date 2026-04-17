# OCI AI Agents Workshop

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white) ![Black](https://img.shields.io/badge/code%20style-black-000000.svg) ![Pylint](https://img.shields.io/badge/linting-pylint-2A6AC7?logo=python&logoColor=white) ![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

This folder contains practical examples to help you start building agents with LangChain and LangGraph.
The examples are designed first and foremost to run in OCI Enterprise AI.
A specific focus is included for DAC (Dedicated AI Cluster) based deployments and runtime configuration.

## Quickstart

### 1. Create and activate the Conda environment

```bash
conda create -n agent_workshop python=3.11 -y
conda activate agent_workshop
```

### 2. Install required libraries

```bash
pip install langchain langgraph langchain-oci python-dotenv pytest black pylint
```

### 3. Configure OCI authentication

Set up OCI API authentication by following the official guide:
[Oracle Cloud Infrastructure SDK and CLI Configuration File](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm).

### 4. Configure environment variables

Copy the sample file and set real values:

```bash
cp .env-sample .env
```

Then edit `.env` and set at least:

- `OCI_COMPARTMENT_ID` (required)
- `OCI_REGION` (for endpoint generation)
- `OCI_MODEL_ID`
- `OCI_PROVIDER`
- `OCI_AUTH_TYPE`
- `OCI_AUTH_PROFILE`

### 5. Run an example

```bash
python agent01.py "Explain what a Dedicated AI Cluster is in OCI."
```

## DAC Notes

When using DAC in these examples, the `OCI_MODEL_ID` value must be the OCID of the DAC endpoint.
The DAC endpoint must be public for these examples.
