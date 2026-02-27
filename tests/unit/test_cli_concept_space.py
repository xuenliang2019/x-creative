"""Tests for concept-space CLI commands."""

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestConceptSpaceCLI:
    def test_validate_command_exists(self, runner: CliRunner) -> None:
        from x_creative.cli.main import app
        result = runner.invoke(app, ["concept-space", "validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output.lower() or "Validate" in result.output

    def test_diff_command_exists(self, runner: CliRunner) -> None:
        from x_creative.cli.main import app
        result = runner.invoke(app, ["concept-space", "diff", "--help"])
        assert result.exit_code == 0
        assert "diff" in result.output.lower() or "Diff" in result.output

    def test_validate_valid_yaml(self, runner: CliRunner, tmp_path: "pytest.TempPathFactory") -> None:
        """Test validate with a valid ConceptSpace YAML."""
        from x_creative.cli.main import app

        yaml_content = """\
concept_space:
  version: "1.0.0"
  primitives:
    - id: p1
      name: Price
      description: Asset price
      type: variable
    - id: p2
      name: Volume
      description: Trading volume
      type: variable
  relations:
    - id: r1
      name: price_volume
      connects: [p1, p2]
      description: Price-volume relationship
  hard_constraints: []
  soft_preferences: []
  assumptions_fixed: []
  assumptions_mutable: []
  allowed_ops: []
  evaluation_criteria: []
"""
        yaml_path = tmp_path / "valid_cs.yaml"
        yaml_path.write_text(yaml_content)

        result = runner.invoke(app, ["concept-space", "validate", str(yaml_path), "--domain-id", "test"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_duplicate_ids(self, runner: CliRunner, tmp_path: "pytest.TempPathFactory") -> None:
        """Test validate catches duplicate IDs."""
        from x_creative.cli.main import app

        yaml_content = """\
concept_space:
  version: "1.0.0"
  primitives:
    - id: dup
      name: A
      description: First
      type: variable
    - id: dup
      name: B
      description: Second
      type: variable
  relations: []
  hard_constraints: []
  soft_preferences: []
  assumptions_fixed: []
  assumptions_mutable: []
  allowed_ops: []
  evaluation_criteria: []
"""
        yaml_path = tmp_path / "dup_cs.yaml"
        yaml_path.write_text(yaml_content)

        result = runner.invoke(app, ["concept-space", "validate", str(yaml_path)])
        assert result.exit_code == 1

    def test_diff_no_differences(self, runner: CliRunner, tmp_path: "pytest.TempPathFactory") -> None:
        """Test diff with identical files."""
        from x_creative.cli.main import app

        yaml_content = """\
concept_space:
  version: "1.0.0"
  primitives: []
  relations: []
  hard_constraints: []
  soft_preferences: []
  assumptions_fixed: []
  assumptions_mutable: []
  allowed_ops: []
  evaluation_criteria: []
"""
        path_a = tmp_path / "a.yaml"
        path_b = tmp_path / "b.yaml"
        path_a.write_text(yaml_content)
        path_b.write_text(yaml_content)

        result = runner.invoke(app, ["concept-space", "diff", str(path_a), str(path_b)])
        assert result.exit_code == 0
        assert "no differences" in result.output.lower()

    def test_diff_with_changes(self, runner: CliRunner, tmp_path: "pytest.TempPathFactory") -> None:
        """Test diff detects constraint changes."""
        from x_creative.cli.main import app

        old_yaml = """\
concept_space:
  version: "1.0.0"
  primitives: []
  relations: []
  hard_constraints:
    - id: c1
      text: "Must be positive"
      constraint_type: hard
      rationale: "Growth rate must be positive for the approach to be valid"
  soft_preferences: []
  assumptions_fixed: []
  assumptions_mutable: []
  allowed_ops: []
  evaluation_criteria: []
"""
        new_yaml = """\
concept_space:
  version: "1.1.0"
  primitives: []
  relations: []
  hard_constraints:
    - id: c1
      text: "Must be non-negative"
      constraint_type: hard
      rationale: "Growth rate must be non-negative for the approach to be valid"
    - id: c2
      text: "Max churn rate 10%"
      constraint_type: hard
      rationale: "Sustainability requires bounded user churn"
  soft_preferences: []
  assumptions_fixed: []
  assumptions_mutable: []
  allowed_ops: []
  evaluation_criteria: []
"""
        old_path = tmp_path / "old.yaml"
        new_path = tmp_path / "new.yaml"
        old_path.write_text(old_yaml)
        new_path.write_text(new_yaml)

        result = runner.invoke(app, ["concept-space", "diff", str(old_path), str(new_path)])
        assert result.exit_code == 0
        assert "modify_constraint" in result.output
        assert "add_constraint" in result.output
