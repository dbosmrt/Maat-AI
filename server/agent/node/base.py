"""Base classes for LangGraph nodes.

Provides a common interface and utilities for all pipeline nodes
with dependency injection, structured logging, and error handling.
"""

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..utils.logger import get_logger, log_node_event, log_system_error
from ..state import AgentState
from ..model import ChatModels


class BaseNode(ABC):
    """Abstract base class for all LangGraph pipeline nodes.

    Provides:
    - Structured logging with timing
    - Standardized error handling
    - Execution time measurement
    - Node event logging for observability
    """

    def __init__(
        self,
        name: str,
        logger: Optional[Any] = None,
    ) -> None:
        """Initialize the node.

        Args:
            name: Unique identifier for this node (used in logging).
            logger: Optional custom logger instance.
        """
        self.name = name
        self._logger = logger or get_logger(f"node.{name}")

    def __call__(self, state: AgentState) -> dict:
        """Execute the node with timing and error handling.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary of state updates.

        Raises:
            Exception: Re-raised after logging for graph-level handling.
        """
        start_time = time.perf_counter()
        try:
            result = self.execute(state)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(
                "%s completed in %.1fms",
                self.name,
                duration_ms,
            )
            log_node_event(self.name, "SUCCESS", duration_ms=duration_ms)
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error("%s failed: %s", self.name, exc)
            log_system_error(traceback.format_exc())
            log_node_event(
                self.name,
                "FAILURE",
                duration_ms=duration_ms,
                error_payload=str(exc),
            )
            # Return safe fallback instead of crashing the graph
            return self._fallback(state, exc)

    @abstractmethod
    def execute(self, state: AgentState) -> dict:
        """Execute the node's main logic.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary of state updates to merge into the graph state.
        """
        pass

    def _fallback(self, state: AgentState, exc: Exception) -> dict:
        """Provide a safe fallback when node execution fails.

        Override in subclasses for node-specific fallback behavior.

        Args:
            state: Current LangGraph state.
            exc: The exception that was raised.

        Returns:
            Dictionary of state updates (typically empty or minimal).
        """
        return {}

    @property
    def logger(self):
        """Get the node's logger."""
        return self._logger


class LLMNode(BaseNode):
    """Base class for nodes that use LLM with prompt templates.

    Provides:
    - LLM model selection via ChatModels factory
    - Standardized prompt template handling
    - Prompt chain execution with structured output
    """

    def __init__(
        self,
        name: str,
        default_model: str = "sarvam_m",
        logger: Optional[Any] = None,
    ) -> None:
        """Initialize the LLM node.

        Args:
            name: Unique identifier for this node.
            default_model: Default model key from ChatModels factory.
            logger: Optional custom logger instance.
        """
        super().__init__(name, logger)
        self._default_model = default_model

    def _get_llm(self, state: AgentState):
        """Get LLM from state or use default model.

        Args:
            state: Current LangGraph state.

        Returns:
            LLM instance appropriate for the query.
        """
        model_name = state.get("user_chat_model")
        temperature = state.get("user_temperature")
        top_p = state.get("user_top_p")
        max_tokens = state.get("user_max_tokens")

        if model_name:
            return ChatModels.get_from_user_settings(
                preferred_model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        return ChatModels.get(self._default_model)

    @property
    @abstractmethod
    def prompt_template(self):
        """Get the prompt template for this node.

        Must be implemented by subclasses.
        """
        pass

    def _invoke_chain(self, state: AgentState, input_dict: dict, output_schema=None):
        """Execute the prompt chain with optional structured output.

        Args:
            state: Current LangGraph state for model selection.
            input_dict: Dictionary of variables for the prompt template.
            output_schema: Optional Pydantic model for structured output.

        Returns:
            Parsed result from the LLM call.
        """
        llm = self._get_llm(state)
        if output_schema:
            llm = llm.with_structured_output(output_schema)

        chain = self.prompt_template | llm
        return chain.invoke(input_dict)

    def _get_query(self, state: AgentState) -> str:
        """Extract the query from the state.

        Args:
            state: Current LangGraph state.

        Returns:
            The query string from state.
        """
        return state.get("query", "")


class PromptNode(LLMNode):
    """Base class for nodes that use a single prompt template with structured output.

    Provides a default prompt_template property that subclasses can override
    by implementing _get_prompt_template() or setting _prompt_template.
    """

    def __init__(
        self,
        name: str,
        default_model: str = "sarvam_m",
    ) -> None:
        """Initialize the prompt node.

        Args:
            name: Unique identifier for this node.
            default_model: Default model key from ChatModels factory.
        """
        super().__init__(name, default_model)
        self._prompt_template = None

    @property
    def prompt_template(self):
        """Get the prompt template, building it if necessary."""
        if self._prompt_template is None:
            self._prompt_template = self._get_prompt_template()
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, value):
        """Set the prompt template directly."""
        self._prompt_template = value

    def _get_prompt_template(self):
        """Get the prompt template. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _get_prompt_template()")