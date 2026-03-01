"""Tests for recover_truncated_json_object in creativity.utils."""

import json

import pytest

from x_creative.creativity.utils import recover_truncated_json_object


class TestRecoverTruncatedJsonObject:
    """Tests for recover_truncated_json_object()."""

    def test_complete_object_parses_fine(self):
        """A fully valid JSON object should be returned as-is."""
        obj = {
            "id": "software_dev",
            "name": "Software Development",
            "source_domains": [
                {"domain": "biology", "structures": ["evolution", "ecosystem"]},
                {"domain": "architecture", "structures": ["blueprint", "foundation"]},
            ],
        }
        text = json.dumps(obj)
        result = recover_truncated_json_object(text)
        assert result == obj

    def test_truncated_mid_string(self):
        """Truncated in the middle of a string value — recovers earlier fields."""
        text = '{"id": "software_dev", "name": "Software Developm'
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("id") == "software_dev"
        # "name" was truncated mid-value, so it should not be present
        assert "name" not in result or result["name"] == "Software Developm"

    def test_truncated_mid_array(self):
        """Truncated inside a nested array — recovers fields before it."""
        text = (
            '{"id": "fresh", "name": "Fresh Domain", '
            '"constraints": [{"name": "c1", "description": "d1"}], '
            '"source_domains": [{"domain": "bio", "structures": ["a", "b"]}, '
            '{"domain": "arch", "struc'
        )
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("id") == "fresh"
        assert result.get("name") == "Fresh Domain"
        assert isinstance(result.get("constraints"), list)
        assert len(result["constraints"]) == 1

    def test_no_brace_returns_empty(self):
        """Input with no opening brace returns empty dict."""
        result = recover_truncated_json_object("no json here at all")
        assert result == {}

    def test_empty_string_returns_empty(self):
        """Empty string returns empty dict."""
        result = recover_truncated_json_object("")
        assert result == {}

    def test_truncated_after_comma(self):
        """Truncated right after a comma between top-level keys."""
        text = '{"id": "test", "name": "Test Domain", '
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("id") == "test"
        assert result.get("name") == "Test Domain"

    def test_object_with_surrounding_text(self):
        """JSON object embedded in surrounding text (e.g. markdown fences already stripped)."""
        text = 'Here is the JSON: {"id": "embedded", "value": 42} done'
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("id") == "embedded"
        assert result.get("value") == 42

    def test_nested_braces_in_strings(self):
        """Braces inside string values should not confuse the scanner."""
        text = '{"id": "test", "formula": "if x > 0 { return x }", "ok": true}'
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("ok") is True

    def test_truncated_after_scalar_string_no_comma(self):
        """EOF right after a complete string value with no trailing comma."""
        text = '{"id": "fresh"'
        result = recover_truncated_json_object(text)
        assert result == {"id": "fresh"}

    def test_truncated_after_scalar_number_no_comma(self):
        """EOF right after a complete number value with no trailing comma."""
        text = '{"id": "test", "count": 42'
        result = recover_truncated_json_object(text)
        assert result.get("id") == "test"
        assert result.get("count") == 42

    def test_truncated_after_scalar_boolean_no_comma(self):
        """EOF right after a complete boolean value with no trailing comma."""
        text = '{"id": "test", "active": true'
        result = recover_truncated_json_object(text)
        assert result.get("id") == "test"
        assert result.get("active") is True

    def test_truncated_after_complete_array_no_comma(self):
        """EOF after a complete array value (closed ]) with no trailing comma."""
        text = '{"id": "fresh", "source_domains": [{"domain": "bio"}]'
        result = recover_truncated_json_object(text)
        assert result.get("id") == "fresh"
        assert len(result.get("source_domains", [])) == 1

    def test_truncated_inside_array_object_recovers_earlier_fields(self):
        """Truncated mid-object inside an array — recovers top-level fields before it.

        Regression: open_arrays counted from the full scan was applied to a
        slice that predates the array opening, producing invalid JSON.
        """
        text = (
            '{"id": "fresh", "name": "Fresh Domain", '
            '"source_domains": [{"domain": "bio", "struc'
        )
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("id") == "fresh"
        assert result.get("name") == "Fresh Domain"

    def test_truncated_deep_inside_nested_array_recovers_scalar_fields(self):
        """Truncated deep inside nested structures — recovers earlier scalars."""
        text = (
            '{"id": "test", "count": 3, '
            '"items": [{"a": 1}, {"b": [1, 2, '
        )
        result = recover_truncated_json_object(text)
        assert isinstance(result, dict)
        assert result.get("id") == "test"
        assert result.get("count") == 3
