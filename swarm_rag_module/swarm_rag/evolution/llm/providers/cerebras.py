"""
Cerebras LLM Provider implementation.
"""
import json
import logging

from ..provider import BaseLLMProvider, LLMResponse
from ..utils import genome_to_json_context
from ...types.genome import Genome
from ...types.config import EvolutionContext

logger = logging.getLogger(__name__)


class CerebrasProvider(BaseLLMProvider):
    """
    LLM Provider using Cerebras API via keycycle.

    Requires the keycycle package with Cerebras API keys configured.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        env_path: str = ".env",
        **kwargs
    ):
        """
        Initialize Cerebras provider.

        Args:
            model: Model ID to use
            env_path: Path to .env file with API keys
            **kwargs: Additional arguments for BaseLLMProvider
        """
        super().__init__(**kwargs)

        try:
            from keycycle import MultiProviderWrapper
        except ImportError:
            raise ImportError(
                "keycycle is required for CerebrasProvider. "
                "Install it with: pip install keycycle"
            )

        from dotenv import load_dotenv
        load_dotenv()

        self.model = model
        self.wrapper = MultiProviderWrapper.from_env(
            provider="cerebras",
            default_model_id=model,
            env_file=env_path
        )

    def _call_llm(self, genome: Genome, context: EvolutionContext) -> LLMResponse:
        """
        Call Cerebras API to refine genome.

        Args:
            genome: Genome to refine
            context: Evolution context

        Returns:
            LLMResponse with results
        """
        context_data = genome_to_json_context(genome)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context_data)

        client = self.wrapper.get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Ensure required fields exist
        diagnosis = data.get("diagnosis", "No diagnosis provided")
        proposed_changes = data.get("proposed_changes", {})

        return LLMResponse(
            diagnosis=diagnosis,
            proposed_changes=proposed_changes,
            raw_response=content,
            success=True
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for genome refinement."""
        return (
            "You are an expert AI Geneticist. Your job is to optimize individual Retrieval Agents.\n"
            "You will be given an agent's **Code** (parameters & logic) and its **Report Card** (metrics).\n"
            "- **High Cost**: Reduce `steps`, `n_agents`, or make movement more focused (less random/exploration).\n"
            "- **Low Recall**: Increase exploration (`pheromone_repulsion`), add `n_agents`, or relax thresholds.\n"
            "You must output a valid JSON object representing the **refined** agent.\n"
            "The JSON must have keys: 'diagnosis' (string) and 'proposed_changes' (nested object with 'params' and 'strategies')."
        )

    def _build_user_prompt(self, context_data: dict) -> str:
        """Build the user prompt with genome context."""
        metrics_str = (
            f"- Quality: {context_data['performance']['quality_score']:.4f} (Target: 1.0)\n"
            f"- Cost: {context_data['performance']['cost_score']:.4f} (Target: 0.0 - Lower is better)\n"
            f"- Latency: {context_data['performance']['latency']:.4f}s"
        )

        return (
            f"**Agent ID**: {context_data['id']}\n"
            f"**Metrics**:\n{metrics_str}\n\n"
            f"**Current Logic**:\n{json.dumps(context_data['current_config'], indent=2)}\n\n"
            "**Task**: This agent is underperforming. Analyze the metrics above.\n"
            "1. Diagnosis: Why is the score low?\n"
            "2. Action: Edit the `params` or `strategies` to fix the specific weakness identified in the diagnosis.\n"
            "Return JSON format."
        )
