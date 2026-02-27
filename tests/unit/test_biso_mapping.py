"""Tests for BISO mapping_table and failure_modes parsing."""

import pytest

from x_creative.core.types import Domain, DomainStructure, TargetMapping


class TestBISOParseMappingTable:
    """Test that BISO parses mapping_table and failure_modes from LLM output."""

    def test_parse_analogies_with_mapping_table(self) -> None:
        from x_creative.creativity.biso import BISOModule

        module = BISOModule()
        domain = Domain(
            id="test_domain",
            name="Test",
            description="Test domain",
            structures=[
                DomainStructure(
                    id="s1", name="S1", description="d", key_variables=[], dynamics="d"
                )
            ],
        )

        content = '''[
            {
                "structure_id": "s1",
                "analogy": "Test analogy",
                "explanation": "Test explanation",
                "observable": "test_metric = sum(x)",
                "confidence": 0.7,
                "mapping_table": [
                    {
                        "source_concept": "A",
                        "target_concept": "B",
                        "source_relation": "R1",
                        "target_relation": "R2",
                        "mapping_type": "relation",
                        "systematicity_group_id": "g1"
                    }
                ],
                "failure_modes": [
                    {
                        "scenario": "When X",
                        "why_breaks": "Because Y",
                        "detectable_signal": "Signal Z"
                    }
                ]
            }
        ]'''

        hypotheses = module._parse_analogies(content, domain)
        assert len(hypotheses) == 1
        h = hypotheses[0]
        assert len(h.mapping_table) == 1
        assert h.mapping_table[0].source_concept == "A"
        assert len(h.failure_modes) == 1
        assert h.failure_modes[0].scenario == "When X"

    def test_parse_analogies_without_mapping_table_is_rejected(self) -> None:
        """BISO should reject candidates missing required structural evidence."""
        from x_creative.creativity.biso import BISOModule

        module = BISOModule()
        domain = Domain(
            id="test_domain",
            name="Test",
            description="Test domain",
            structures=[
                DomainStructure(
                    id="s1", name="S1", description="d", key_variables=[], dynamics="d"
                )
            ],
        )

        content = '''[
            {
                "structure_id": "s1",
                "analogy": "Test analogy",
                "explanation": "Test explanation",
                "observable": "test_metric = sum(x)",
                "confidence": 0.7
            }
        ]'''

        hypotheses = module._parse_analogies(content, domain)
        assert hypotheses == []

    def test_parse_analogies_with_invalid_mapping_item_skips_gracefully(self) -> None:
        """Invalid mapping items should be skipped, not crash."""
        from x_creative.creativity.biso import BISOModule

        module = BISOModule()
        domain = Domain(
            id="test_domain",
            name="Test",
            description="Test domain",
            structures=[
                DomainStructure(
                    id="s1", name="S1", description="d", key_variables=[], dynamics="d"
                )
            ],
        )

        content = '''[
            {
                "structure_id": "s1",
                "analogy": "Test analogy",
                "explanation": "Test explanation",
                "observable": "test_metric = sum(x)",
                "mapping_table": [
                    {
                        "source_concept": "A",
                        "target_concept": "B",
                        "source_relation": "R1",
                        "target_relation": "R2",
                        "mapping_type": "relation",
                        "systematicity_group_id": "g1"
                    },
                    {
                        "bad_field": "should be skipped"
                    }
                ],
                "failure_modes": [
                    {
                        "scenario": "When X",
                        "why_breaks": "Because Y",
                        "detectable_signal": "Signal Z"
                    },
                    "not_a_dict_should_be_skipped"
                ]
            }
        ]'''

        hypotheses = module._parse_analogies(content, domain)
        assert len(hypotheses) == 1
        # Only valid items parsed
        assert len(hypotheses[0].mapping_table) == 1
        assert len(hypotheses[0].failure_modes) == 1
