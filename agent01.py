"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Simple LangGraph agent with three sequential Runnable steps.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableSerializable
from langgraph.graph import END, StateGraph

from oci_models import build_llm
from utils import collect_oci_runtime_config, extract_text, print_oci_runtime_config


class AgentState(TypedDict, total=False):
    """State shared by all graph steps."""

    user_input: str
    runtime_config: Dict[str, str]
    model_response: Any
    output: str


# pylint: disable=too-few-public-methods
class Step1LogInput(RunnableSerializable[AgentState, AgentState]):
    """Step 1: log step name and input."""

    def invoke(
        self, state: AgentState, _config: Any = None, **_kwargs: Any
    ) -> AgentState:
        """Log the input and keep state unchanged."""
        logging.info("Running step: step1_log_input")
        logging.info("Input text: %s", state["user_input"])
        
        runtime_config = collect_oci_runtime_config()
        print_oci_runtime_config(runtime_config)

        updated_state: AgentState = dict(state)
        updated_state["runtime_config"] = runtime_config
        return updated_state


# pylint: disable=too-few-public-methods
class Step2InvokeModel(RunnableSerializable[AgentState, AgentState]):
    """Step 2: log step name and invoke the OCI model."""

    def __init__(
        self,
        llm_builder: Callable[[Dict[str, str]], Any] = build_llm,
    ) -> None:
        self._llm_builder = llm_builder

    def invoke(
        self, state: AgentState, _config: Any = None, **_kwargs: Any
    ) -> AgentState:
        """Invoke the model and store the raw response in state."""
        logging.info("Running step: step2_invoke_model")

        runtime_config = state.get("runtime_config", collect_oci_runtime_config())
        if not runtime_config["OCI_COMPARTMENT_ID"]:
            raise ValueError("Set OCI_COMPARTMENT_ID environment variable.")

        llm = self._llm_builder(runtime_config)
        response = llm.invoke(state["user_input"])

        updated_state: AgentState = dict(state)
        updated_state["model_response"] = response
        return updated_state


# pylint: disable=too-few-public-methods
class Step3BuildJsonOutput(RunnableSerializable[AgentState, AgentState]):
    """Step 3: log step name and extract text output."""

    def invoke(
        self, state: AgentState, _config: Any = None, **_kwargs: Any
    ) -> AgentState:
        """Extract model text and store it as final output."""
        logging.info("Running step: step3_build_json_output")
        output_text = extract_text(state["model_response"])

        updated_state: AgentState = dict(state)
        updated_state["output"] = output_text
        return updated_state


def build_agent_graph(
    llm_builder: Callable[[Dict[str, str]], Any] = build_llm,
):
    """Build and compile the three-step sequential graph."""
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("step1_log_input", Step1LogInput())
    graph_builder.add_node(
        "step2_invoke_model", Step2InvokeModel(llm_builder=llm_builder)
    )
    graph_builder.add_node("step3_build_json_output", Step3BuildJsonOutput())

    graph_builder.set_entry_point("step1_log_input")
    graph_builder.add_edge("step1_log_input", "step2_invoke_model")
    graph_builder.add_edge("step2_invoke_model", "step3_build_json_output")
    graph_builder.add_edge("step3_build_json_output", END)

    return graph_builder.compile()


def run_agent(
    user_input: str, llm_builder: Callable[[Dict[str, str]], Any] = build_llm
) -> Dict[str, str]:
    """Run the graph and return the final JSON-compatible output dictionary."""
    graph = build_agent_graph(llm_builder=llm_builder)

    result_state: AgentState = graph.invoke({"user_input": user_input})

    return {"output": result_state["output"]}


def main() -> None:
    """Read prompt from CLI, run the graph, and print JSON output."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(description="Simple 3-step LangGraph agent.")
    parser.add_argument("request", type=str, help="Prompt text to send to the model.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    # here we run the agent
    result = run_agent(args.request)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
