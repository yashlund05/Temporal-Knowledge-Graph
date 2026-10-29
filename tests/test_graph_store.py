"""
Unit tests for knowledge graph stores: InMemoryKGStore and TemporalKGStore.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.knowledge_graph.memory_store import InMemoryKGStore
from src.knowledge_graph.graph_store import TemporalKGStore


class TestInMemoryKGStore:
    """Test suite for the InMemoryKGStore class."""

    @pytest.fixture
    def store(self) -> InMemoryKGStore:
        kg = InMemoryKGStore()
        kg.add_entity("0", "United States")
        kg.add_entity("1", "Russia")
        kg.add_entity("2", "Ukraine")
        kg.add_event(
            subject_id="0",
            predicate="Make statement",
            object_id="1",
            timestamp="2014-03-15",
            source="icews14",
        )
        kg.add_event(
            subject_id="1",
            predicate="Host a visit",
            object_id="2",
            timestamp="2014-06-20",
            source="icews14",
        )
        return kg

    def test_entity_exists(self, store: InMemoryKGStore) -> None:
        assert store.entity_exists("United States") is True
        assert store.entity_exists("united states") is True  # case-insensitive
        assert store.entity_exists("NonExistentCountry") is False

    def test_get_entity_id(self, store: InMemoryKGStore) -> None:
        assert store.get_entity_id("United States") == "0"
        assert store.get_entity_id("russia") == "1"
        assert store.get_entity_id("Unknown") is None

    def test_find_entities_fuzzy(self, store: InMemoryKGStore) -> None:
        results = store.find_entities_fuzzy("States")
        assert len(results) == 1
        assert results[0]["name"] == "United States"

        results_multi = store.find_entities_fuzzy("u")
        assert len(results_multi) >= 2

    def test_get_events_by_entity(self, store: InMemoryKGStore) -> None:
        events = store.get_events_by_entity("United States")
        assert len(events) == 1
        assert events[0]["predicate"] == "Make statement"
        assert events[0]["object"] == "Russia"

        events_target = store.get_events_by_entity("Russia")
        assert len(events_target) == 2

    def test_get_events_by_entity_and_time(self, store: InMemoryKGStore) -> None:
        # Match within range
        events = store.get_events_by_entity_and_time("Russia", "2014-01-01", "2014-05-01")
        assert len(events) == 1
        assert events[0]["timestamp"] == "2014-03-15"

        # Out of range
        no_events = store.get_events_by_entity_and_time("Russia", "2015-01-01", "2015-12-31")
        assert len(no_events) == 0

    def test_get_events_between_entities(self, store: InMemoryKGStore) -> None:
        events = store.get_events_between_entities("United States", "Russia")
        assert len(events) == 1
        assert events[0]["predicate"] == "Make statement"

        reverse_events = store.get_events_between_entities("Russia", "United States")
        assert len(reverse_events) == 1

        unrelated = store.get_events_between_entities("United States", "Ukraine")
        assert len(unrelated) == 0

    def test_get_events_between_entities_in_range(self, store: InMemoryKGStore) -> None:
        events = store.get_events_between_entities_in_range(
            "United States", "Russia", "2014-03-01", "2014-03-31"
        )
        assert len(events) == 1

        events_out = store.get_events_between_entities_in_range(
            "United States", "Russia", "2014-04-01", "2014-04-30"
        )
        assert len(events_out) == 0

    def test_verify_fact(self, store: InMemoryKGStore) -> None:
        # Exact match
        res_exact = store.verify_fact("United States", "Make statement", "Russia", "2014-03-15")
        assert res_exact["subject_exists"] is True
        assert res_exact["object_exists"] is True
        assert res_exact["relation_exists"] is True
        assert res_exact["timestamp_match"] is True
        assert res_exact["exact_match"] is True

        # Timestamp mismatch
        res_wrong_time = store.verify_fact("United States", "Make statement", "Russia", "2014-05-15")
        assert res_wrong_time["relation_exists"] is True
        assert res_wrong_time["timestamp_match"] is False
        assert res_wrong_time["exact_match"] is False
        assert "2014-03-15" in res_wrong_time["closest_timestamps"]

        # Missing entity
        res_missing = store.verify_fact("Atlantis", "Make statement", "Russia", "2014-03-15")
        assert res_missing["subject_exists"] is False
        assert res_missing["exact_match"] is False

    def test_bulk_load_facts(self) -> None:
        store = InMemoryKGStore()
        store.add_entity("10", "A")
        store.add_entity("20", "B")
        facts = [
            {"subject_id": "10", "predicate": "Visit", "object_id": "20", "timestamp": "2014-01-01"},
            {"subject_id": "20", "predicate": "Praise", "object_id": "10", "timestamp": "2014-01-02"},
        ]
        count = store.bulk_load_facts(facts)
        assert count == 2
        assert len(store._events) == 2

    def test_statistics_and_getters(self, store: InMemoryKGStore) -> None:
        stats = store.get_statistics()
        assert stats["num_entities"] == 3
        assert stats["num_events"] == 2
        assert "United States" in store.get_all_entity_names()
        assert "Make statement" in store.get_all_relations()
        assert "2014-03-15" in store.get_all_timestamps()

    def test_save_and_load(self, store: InMemoryKGStore, tmp_path: Path) -> None:
        save_path = tmp_path / "kg_store.json"
        store.save(save_path)
        assert save_path.exists()

        new_store = InMemoryKGStore()
        new_store.load(save_path)
        assert new_store.get_statistics() == store.get_statistics()
        assert new_store.entity_exists("United States")
        res = new_store.verify_fact("United States", "Make statement", "Russia", "2014-03-15")
        assert res["exact_match"] is True


class TestTemporalKGStoreMocked:
    """Test suite for TemporalKGStore using mocked Neo4j driver."""

    @patch("src.knowledge_graph.graph_store.GraphDatabase.driver")
    def test_lifecycle_and_context_manager(self, mock_driver_fn: MagicMock) -> None:
        mock_driver = MagicMock()
        mock_driver_fn.return_value = mock_driver

        with TemporalKGStore(uri="bolt://localhost:7687", user="neo4j", password="pwd") as store:
            assert store is not None
            mock_driver.verify_connectivity.assert_called_once()

        mock_driver.close.assert_called_once()

    @patch("src.knowledge_graph.graph_store.GraphDatabase.driver")
    def test_batch_insert_empty(self, mock_driver_fn: MagicMock) -> None:
        mock_driver = MagicMock()
        mock_driver_fn.return_value = mock_driver

        store = TemporalKGStore()
        count = store.batch_insert_events([])
        assert count == 0

    @patch("src.knowledge_graph.graph_store.GraphDatabase.driver")
    def test_batch_insert_events(self, mock_driver_fn: MagicMock) -> None:
        mock_driver = MagicMock()
        mock_driver_fn.return_value = mock_driver

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 2
        mock_session.run.return_value.consume.return_value = mock_summary

        store = TemporalKGStore()
        events = [
            {"event_id": "e1", "actor": "A", "predicate": "P", "target": "B", "timestamp": "2014-01-01", "source": "test"},
            {"event_id": "e2", "actor": "B", "predicate": "Q", "target": "A", "timestamp": "2014-01-02", "source": "test"},
        ]
        inserted = store.batch_insert_events(events)
        assert inserted == 2
