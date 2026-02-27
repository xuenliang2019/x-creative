"""Example: Adding a custom domain.

This example shows how to add a custom domain to the library
for generating hypotheses from your own expertise area.
"""

from x_creative.core.types import Domain, DomainStructure, ProblemFrame, TargetMapping
from x_creative.core.domain_loader import DomainLibrary
from x_creative.creativity.biso import BISOModule
from x_creative.llm.router import ModelRouter

import asyncio
import os


def create_custom_domain() -> Domain:
    """Create a custom domain based on music theory."""
    return Domain(
        id="music_theory",
        name="音乐理论",
        name_en="Music Theory",
        description="研究音乐中的和声、节奏、旋律等结构",
        structures=[
            DomainStructure(
                id="harmonic_tension",
                name="和声张力",
                description="不协和音程产生紧张感，解决到协和音程产生放松",
                key_variables=["dissonance", "resolution", "tension_level"],
                dynamics="张力积累后需要释放，形成周期性的紧张-放松模式",
            ),
            DomainStructure(
                id="rhythmic_pattern",
                name="节奏模式",
                description="规律节拍中的重音和休止创造动态感",
                key_variables=["beat", "accent", "syncopation", "tempo"],
                dynamics="重音和休止打破均匀性，创造期望和意外",
            ),
            DomainStructure(
                id="theme_variation",
                name="主题变奏",
                description="核心主题通过变调、装饰、倒影等手法演变",
                key_variables=["theme", "variation", "transformation"],
                dynamics="主题反复出现但每次有所不同，形成熟悉与新奇的平衡",
            ),
        ],
        target_mappings=[
            TargetMapping(
                structure="harmonic_tension",
                target="社区参与度张力",
                observable="Issue 积压量累积偏离月均值的程度",
            ),
            TargetMapping(
                structure="rhythmic_pattern",
                target="贡献节奏异常",
                observable="PR 提交量在固定时间点的突变（周初、月末、发布前）",
            ),
            TargetMapping(
                structure="theme_variation",
                target="功能演化模式",
                observable="版本特性列表与历史版本的相似度（Jaccard距离）",
            ),
        ],
    )


async def main() -> None:
    """Generate hypotheses from a custom domain."""

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    # Create custom domain
    custom_domain = create_custom_domain()

    print("=" * 60)
    print("X-Creative: Custom Domain Example")
    print("=" * 60)
    print(f"\nDomain: {custom_domain.name} ({custom_domain.name_en})")
    print(f"Structures: {len(custom_domain.structures)}")
    print()

    # Define problem
    problem = ProblemFrame(
        description="设计一个能实现病毒式传播的开源命令行工具",
        target_domain="open_source_development",
    )

    # Generate analogies from the custom domain
    biso = BISOModule()

    try:
        print("Generating analogies from custom domain...")
        hypotheses = await biso.generate_analogies(
            domain=custom_domain,
            problem=problem,
            num_analogies=5,
        )

        print(f"\nGenerated {len(hypotheses)} hypotheses\n")

        for i, hyp in enumerate(hypotheses, 1):
            print(f"#{i} {hyp.description}")
            print(f"   Source structure: {hyp.source_structure}")
            print(f"   Observable: {hyp.observable}")
            print()

    finally:
        await biso._router.close()


if __name__ == "__main__":
    asyncio.run(main())
