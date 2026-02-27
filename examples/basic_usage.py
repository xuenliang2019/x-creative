"""Basic usage example for X-Creative.

This example demonstrates how to use the Creativity Engine
to generate hypotheses for a research problem.

Before running, set your OpenRouter API key:
    export OPENROUTER_API_KEY=your-key-here
"""

import asyncio
import os

from x_creative.core.types import ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine


async def main() -> None:
    """Generate hypotheses for a sample problem."""

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable")
        print("Example: export OPENROUTER_API_KEY=sk-or-v1-...")
        return

    # Define the research problem
    problem = ProblemFrame(
        description="设计一个能实现病毒式传播的开源命令行工具",
        target_domain="open_source_development",
        constraints=[
            "Must be implementable as a single-binary CLI tool",
            "Should solve a real developer pain point",
        ],
    )

    # Configure the search
    config = SearchConfig(
        num_hypotheses=20,  # Generate top 20 hypotheses
        search_depth=2,      # 2 rounds of expansion
        search_breadth=3,    # Top 3 hypotheses expanded per round
        prune_threshold=5.0, # Filter out low-scoring hypotheses
    )

    print("=" * 60)
    print("X-Creative: Hypothesis Generation")
    print("=" * 60)
    print(f"\nProblem: {problem.description}")
    print(f"Target Domain: {problem.target_domain}")
    print()

    # Create engine and generate
    engine = CreativityEngine()

    try:
        print("Generating hypotheses...")
        print("(This may take a few minutes depending on model response times)")
        print()

        hypotheses = await engine.generate(problem, config)

        print(f"\nGenerated {len(hypotheses)} hypotheses\n")
        print("=" * 60)

        # Display top results
        for i, hyp in enumerate(hypotheses[:10], 1):
            score = hyp.composite_score()
            print(f"\n#{i} [Score: {score:.1f}] {hyp.description}")
            print(f"   Source: {hyp.source_domain}/{hyp.source_structure}")
            print(f"   Observable: {hyp.observable[:80]}...")
            if hyp.scores:
                print(f"   Scores: D={hyp.scores.divergence:.1f} "
                      f"T={hyp.scores.testability:.1f} "
                      f"R={hyp.scores.rationale:.1f} "
                      f"Rb={hyp.scores.robustness:.1f}")

    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
