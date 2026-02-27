"""CLI commands for ConceptSpace management."""

from pathlib import Path

import typer
from rich.console import Console

from x_creative.core.concept_space_compiler import ConceptSpaceCompiler

console = Console()
concept_space_app = typer.Typer(help="ConceptSpace management commands")


@concept_space_app.command()
def validate(
    yaml_path: Path = typer.Argument(..., help="Path to target domain YAML file"),
    domain_id: str = typer.Option("unknown", help="Domain identifier"),
) -> None:
    """Validate a ConceptSpace YAML definition."""
    compiler = ConceptSpaceCompiler()
    cs = compiler.compile_from_yaml(yaml_path, domain_id=domain_id)
    errors = compiler.validate(cs)
    if errors:
        for e in errors:
            typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"ConceptSpace '{cs.domain_id}' v{cs.version} is valid.")


@concept_space_app.command()
def diff(
    old_path: Path = typer.Argument(..., help="Path to old YAML"),
    new_path: Path = typer.Argument(..., help="Path to new YAML"),
    domain_id: str = typer.Option("unknown", help="Domain identifier"),
) -> None:
    """Show differences between two ConceptSpace versions."""
    compiler = ConceptSpaceCompiler()
    old_cs = compiler.compile_from_yaml(old_path, domain_id=domain_id)
    new_cs = compiler.compile_from_yaml(new_path, domain_id=domain_id)
    diffs = compiler.diff(old_cs, new_cs)
    if not diffs:
        typer.echo("No differences found.")
    else:
        for d in diffs:
            typer.echo(f"[{d.op_type}] {d.target_id}: {d.before_state} -> {d.after_state}")
