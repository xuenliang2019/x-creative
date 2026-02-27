"""CLI commands for session management."""

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from x_creative.session import SessionManager, StageStatus

app = typer.Typer(help="Manage research sessions")
console = Console()


def get_data_dir() -> Path:
    """Get the data directory from environment or default."""
    env_dir = os.environ.get("X_CREATIVE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("local_data")


def get_manager() -> SessionManager:
    """Get the session manager."""
    return SessionManager(data_dir=get_data_dir())


@app.command("new")
def session_new(
    topic: Annotated[str, typer.Argument(help="Topic/description for the session")],
    session_id: Annotated[str | None, typer.Option("--id", help="Custom session ID")] = None,
) -> None:
    """Create a new research session."""
    manager = get_manager()
    session = manager.create_session(topic, session_id=session_id)

    console.print(Panel(
        f"[bold green]Created session:[/bold green] {session.id}\n"
        f"[dim]Topic:[/dim] {session.topic}\n"
        f"[dim]Directory:[/dim] {manager.data_dir / session.id}",
        title="New Session",
    ))


@app.command("list")
def session_list() -> None:
    """List all available sessions."""
    manager = get_manager()
    sessions = manager.list_sessions()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    current = manager.get_current_session()
    current_id = current.id if current else None

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Topic")
    table.add_column("Stage")
    table.add_column("Created")
    table.add_column("Current", justify="center")

    for s in sessions:
        is_current = "*" if s.id == current_id else ""
        table.add_row(
            s.id,
            s.topic[:40] + ("..." if len(s.topic) > 40 else ""),
            s.current_stage,
            s.created_at.strftime("%Y-%m-%d %H:%M"),
            is_current,
        )

    console.print(table)


@app.command("status")
def session_status() -> None:
    """Show current session status."""
    manager = get_manager()
    session = manager.get_current_session()

    if session is None:
        console.print("[yellow]No current session.[/yellow]")
        console.print("Use 'x-creative session new <topic>' to create one.")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]{session.topic}[/bold]",
        title=f"Session: {session.id}",
    ))

    table = Table(title="Pipeline Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Completed")

    status_styles = {
        StageStatus.PENDING: "[dim]pending[/dim]",
        StageStatus.RUNNING: "[yellow]running[/yellow]",
        StageStatus.COMPLETED: "[green]completed[/green]",
        StageStatus.FAILED: "[red]failed[/red]",
    }

    for stage_name, stage_info in session.stages.items():
        started = stage_info.started_at.strftime("%H:%M:%S") if stage_info.started_at else "-"
        completed = stage_info.completed_at.strftime("%H:%M:%S") if stage_info.completed_at else "-"
        table.add_row(
            stage_name,
            status_styles[stage_info.status],
            started,
            completed,
        )

    console.print(table)


@app.command("switch")
def session_switch(
    session_id: Annotated[str, typer.Argument(help="Session ID to switch to")],
) -> None:
    """Switch to a different session."""
    manager = get_manager()
    session = manager.switch_session(session_id)

    if session is None:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Switched to session:[/green] {session.id}")


@app.command("delete")
def session_delete(
    session_id: Annotated[str, typer.Argument(help="Session ID to delete")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Delete a session and all its data."""
    manager = get_manager()

    if not yes:
        confirm = typer.confirm(f"Delete session '{session_id}' and all its data?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    deleted = manager.delete_session(session_id)

    if deleted:
        console.print(f"[green]Deleted session:[/green] {session_id}")
    else:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)
