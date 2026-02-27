"""Session manager for creating, loading, and managing sessions."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from x_creative.session.types import STAGE_ORDER, Session, StageName, StageStatus


class SessionManager:
    """Manages session lifecycle and persistence."""

    CURRENT_SESSION_FILE = ".current_session"
    SESSION_FILE = "session.json"

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize the session manager.

        Args:
            data_dir: Base directory for session data. Defaults to ./local_data
        """
        if data_dir is None:
            data_dir = Path("local_data")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self._data_dir

    def _generate_session_id(self, topic: str) -> str:
        """Generate a session ID from date and topic."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        # Sanitize topic for use in directory name
        safe_topic = "".join(c if c.isalnum() or c in "-_" else "-" for c in topic.lower())
        safe_topic = safe_topic[:60].strip("-")
        return f"{date_str}-{safe_topic}"

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self._data_dir / session_id

    def _current_session_file(self) -> Path:
        """Get the path to the current session marker file."""
        return self._data_dir / self.CURRENT_SESSION_FILE

    def create_session(
        self,
        topic: str,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            topic: Description of the session topic.
            session_id: Optional custom session ID.

        Returns:
            The created Session object.
        """
        if session_id is None:
            session_id = self._generate_session_id(topic)

        session = Session(id=session_id, topic=topic)
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save session metadata
        self._save_session(session)

        # Set as current session
        self._set_current_session(session_id)

        return session

    def _save_session(self, session: Session) -> None:
        """Save session metadata to disk."""
        session_file = self._session_dir(session.id) / self.SESSION_FILE
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Session | None:
        """Load a session by ID.

        Args:
            session_id: The session ID to load.

        Returns:
            The Session object, or None if not found.
        """
        session_file = self._session_dir(session_id) / self.SESSION_FILE
        if not session_file.exists():
            return None

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Session.model_validate(data)

    def list_sessions(self) -> list[Session]:
        """List all available sessions.

        Returns:
            List of Session objects, sorted by creation time (newest first).
        """
        sessions = []
        for session_dir in self._data_dir.iterdir():
            if session_dir.is_dir() and not session_dir.name.startswith("."):
                session = self.load_session(session_dir.name)
                if session:
                    sessions.append(session)
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return False

        shutil.rmtree(session_dir)

        # If this was the current session, clear it
        if self._get_current_session_id() == session_id:
            self._current_session_file().unlink(missing_ok=True)

        return True

    def _set_current_session(self, session_id: str) -> None:
        """Set the current session."""
        with open(self._current_session_file(), "w") as f:
            f.write(session_id)

    def _get_current_session_id(self) -> str | None:
        """Get the current session ID."""
        current_file = self._current_session_file()
        if not current_file.exists():
            return None
        return current_file.read_text().strip()

    def get_current_session(self) -> Session | None:
        """Get the current session.

        Returns:
            The current Session, or None if no current session.
        """
        session_id = self._get_current_session_id()
        if session_id is None:
            return None
        return self.load_session(session_id)

    def switch_session(self, session_id: str) -> Session | None:
        """Switch to a different session.

        Args:
            session_id: The session ID to switch to.

        Returns:
            The Session if found, None otherwise.
        """
        session = self.load_session(session_id)
        if session:
            self._set_current_session(session_id)
        return session

    def save_stage_data(
        self,
        session_id: str,
        stage: StageName,
        data: dict[str, Any],
    ) -> Path:
        """Save stage output data.

        Args:
            session_id: The session ID.
            stage: The stage name.
            data: The data to save.

        Returns:
            Path to the saved JSON file.
        """
        session_dir = self._session_dir(session_id)
        data_file = session_dir / f"{stage}.json"

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data_file

    def load_stage_data(
        self,
        session_id: str,
        stage: StageName,
    ) -> dict[str, Any] | None:
        """Load stage output data.

        Args:
            session_id: The session ID.
            stage: The stage name.

        Returns:
            The loaded data, or None if not found.
        """
        data_file = self._session_dir(session_id) / f"{stage}.json"
        if not data_file.exists():
            return None

        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_stage_status(
        self,
        session_id: str,
        stage: StageName,
        status: StageStatus,
        error: str | None = None,
    ) -> Session | None:
        """Update the status of a stage.

        Args:
            session_id: The session ID.
            stage: The stage name.
            status: The new status.
            error: Optional error message if status is FAILED.

        Returns:
            The updated Session, or None if not found.
        """
        session = self.load_session(session_id)
        if session is None:
            return None

        stage_info = session.stages[stage]
        stage_info.status = status

        if status == StageStatus.RUNNING:
            stage_info.started_at = datetime.now()
        elif status in (StageStatus.COMPLETED, StageStatus.FAILED):
            stage_info.completed_at = datetime.now()
            if error:
                stage_info.error = error

        # Update current_stage to the next incomplete stage
        for s in STAGE_ORDER:
            if session.stages[s].status != StageStatus.COMPLETED:
                session.current_stage = s
                break

        self._save_session(session)
        return session
