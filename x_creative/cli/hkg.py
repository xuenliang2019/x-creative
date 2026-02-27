"""CLI commands for HKG (Hypergraph Knowledge Grounding)."""
import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Hypergraph Knowledge Grounding (HKG) commands")


@app.command("ingest")
def ingest(
    source: Annotated[str, typer.Option("--source", "-s", help="Source type: yaml or jsonl")],
    path: Annotated[Path, typer.Option("--path", "-p", help="Path to source file")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output store path (JSON)")],
) -> None:
    """Ingest data into a HypergraphStore."""
    if source == "yaml":
        from x_creative.hkg.ingest import ingest_from_yaml
        store = ingest_from_yaml(path)
    elif source == "jsonl":
        from x_creative.hkg.ingest import ingest_from_jsonl
        store = ingest_from_jsonl(path)
    else:
        console.print(f"[red]Unknown source type: {source}. Use 'yaml' or 'jsonl'.[/red]")
        raise typer.Exit(1)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[red]Cannot create output directory {output.parent}: {e}[/red]")
        raise typer.Exit(1)
    store.save(output)
    stats = store.stats()
    console.print(f"[green]Store saved to {output}[/green]")
    console.print(f"  Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")


@app.command("build-index")
def build_index(
    store_path: Annotated[Path, typer.Option("--store", help="Path to store JSON")],
    embedding: Annotated[bool, typer.Option("--embedding", help="Build embedding index")] = False,
) -> None:
    """Build inverted index and optionally embedding index."""
    from x_creative.hkg.store import HypergraphStore

    store = HypergraphStore.load(store_path)
    console.print(f"Store loaded: {store.stats()}")
    console.print("[green]Inverted index ready (built on load).[/green]")

    if embedding:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[red]OPENROUTER_API_KEY is required to build embedding index.[/red]")
            raise typer.Exit(1)

        from x_creative.hkg.embeddings import EmbeddingClient, NodeEmbeddingIndex

        output_path = store_path.with_name(f"{store_path.stem}.embeddings.json")

        async def _build_and_save() -> int:
            client = EmbeddingClient(api_key=api_key)
            index = NodeEmbeddingIndex()
            try:
                await index.build(store.all_nodes, client)
            finally:
                await client.close()
            index.save(output_path)
            return len(store.all_nodes)

        node_count = asyncio.run(_build_and_save())
        console.print(f"[green]Embedding index built for {node_count} nodes.[/green]")
        console.print(f"[green]Saved to {output_path}[/green]")


@app.command("traverse")
def traverse(
    store_path: Annotated[Path, typer.Option("--store", help="Path to store JSON")],
    start: Annotated[str, typer.Option("--start", help="Comma-separated start terms")],
    end: Annotated[str, typer.Option("--end", help="Comma-separated end terms")],
    k: Annotated[int, typer.Option("--K", help="Number of shortest paths")] = 3,
    intersection_size: Annotated[int, typer.Option("--IS", help="Min shared nodes between edges")] = 1,
    max_len: Annotated[int, typer.Option("--max-len", help="Max path length")] = 6,
) -> None:
    """Find k shortest hyperpaths between start and end terms."""
    from x_creative.hkg.store import HypergraphStore
    from x_creative.hkg.traversal import k_shortest_hyperpaths

    store = HypergraphStore.load(store_path)

    start_nodes = []
    for term in start.split(","):
        ids = store.find_nodes_by_name(term.strip())
        start_nodes.extend(ids)

    end_nodes = []
    for term in end.split(","):
        ids = store.find_nodes_by_name(term.strip())
        end_nodes.extend(ids)

    if not start_nodes:
        console.print("[red]No start nodes matched.[/red]")
        raise typer.Exit(1)
    if not end_nodes:
        console.print("[red]No end nodes matched.[/red]")
        raise typer.Exit(1)

    paths = k_shortest_hyperpaths(store, start_nodes, end_nodes, K=k, IS=intersection_size, max_len=max_len)

    if not paths:
        console.print("[yellow]PATH_NOT_FOUND: No paths between given terms.[/yellow]")
        return

    for i, path in enumerate(paths, 1):
        console.print(f"\n[bold]Path {i}[/bold] (length={path.length}):")
        for eid in path.edges:
            edge = store.get_edge(eid)
            if edge:
                node_names = [store.get_node(n).name if store.get_node(n) else n for n in edge.nodes]
                console.print(f"  [{eid}] {' + '.join(node_names)} | {edge.relation}")
        if path.intermediate_nodes:
            names = [store.get_node(n).name if store.get_node(n) else n for n in path.intermediate_nodes]
            console.print(f"  Bridge: {', '.join(names)}")

    output: list[dict] = []
    for path in paths:
        hyperedges: list[dict] = []
        for eid in path.edges:
            edge = store.get_edge(eid)
            if edge is None:
                continue
            hyperedges.append(
                {
                    "edge_id": eid,
                    "nodes": edge.nodes,
                    "relation": edge.relation,
                    "provenance_refs": [f"{p.doc_id}/{p.chunk_id}" for p in edge.provenance],
                }
            )

        output.append(
            {
                "length": path.length,
                "intermediate_nodes": path.intermediate_nodes,
                "hyperedges": hyperedges,
            }
        )
    console.print(f"\n[dim]JSON output:[/dim]")
    console.print(json.dumps(output, indent=2, ensure_ascii=False))


@app.command("stats")
def stats(
    store_path: Annotated[Path, typer.Option("--store", help="Path to store JSON")],
) -> None:
    """Show statistics for a HypergraphStore."""
    from x_creative.hkg.store import HypergraphStore
    store = HypergraphStore.load(store_path)
    s = store.stats()

    table = Table(title="HKG Store Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key, value in s.items():
        table.add_row(key, str(value))
    console.print(table)
