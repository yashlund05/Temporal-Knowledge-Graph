"""
In-memory temporal knowledge graph store using NetworkX.

This module provides the same interface as TemporalKGStore but stores
everything in memory using NetworkX. No external database required.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)


def _normalize_iso_date(d: str | None, is_end: bool = False) -> str:
    """Normalize a date or year string into an ISO YYYY-MM-DD boundary string."""
    if not d or d.startswith("#"):
        return "9999-12-31" if is_end else "0000-01-01"
    d = str(d).strip()
    if len(d) == 4 and d.isdigit():
        return f"{d}-12-31" if is_end else f"{d}-01-01"
    if len(d) == 7 and d[:4].isdigit() and d[4] == "-":
        return f"{d}-31" if is_end else f"{d}-01"
    return d


class InMemoryKGStore:
    """
    In-memory temporal knowledge graph using NetworkX.

    Provides the same query interface as TemporalKGStore so it can be
    used as a drop-in replacement when Neo4j is not available.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._entities: dict[str, dict] = {}  # entity_id -> {name, ...}
        self._events: dict[str, dict] = {}    # event_id -> {predicate, timestamp, ...}
        self._entity_name_index: dict[str, str] = {}  # lower(name) -> entity_id
        self._subj_obj_events: dict[tuple[str, str], list[dict]] = {}  # (subj_id, obj_id) -> list of events

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def add_entity(self, entity_id: str, name: str, **attrs: Any) -> None:
        """Add an entity node."""
        self._entities[entity_id] = {"name": name, "entity_id": entity_id, **attrs}
        self._entity_name_index[name.lower()] = entity_id
        self.graph.add_node(f"entity:{entity_id}", type="entity", name=name, **attrs)

    def add_event(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        timestamp: str,
        source: str = "icews14",
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> str:
        """Add a temporal event (fact) linking subject -> event -> object."""
        event_id = str(uuid.uuid4())[:12]
        st = start_time or timestamp
        et = end_time or timestamp
        event_data = {
            "event_id": event_id,
            "predicate": predicate,
            "timestamp": timestamp,
            "source": source,
            "subject_id": subject_id,
            "object_id": object_id,
            "start_time": st,
            "end_time": et,
        }
        self._events[event_id] = event_data
        self._subj_obj_events.setdefault((subject_id, object_id), []).append(event_data)

        event_node = f"event:{event_id}"
        subj_node = f"entity:{subject_id}"
        obj_node = f"entity:{object_id}"

        self.graph.add_node(event_node, type="event", **event_data)

        # Ensure entity nodes exist
        if not self.graph.has_node(subj_node):
            name = self._entities.get(subject_id, {}).get("name", subject_id)
            self.graph.add_node(subj_node, type="entity", name=name)
        if not self.graph.has_node(obj_node):
            name = self._entities.get(object_id, {}).get("name", object_id)
            self.graph.add_node(obj_node, type="entity", name=name)

        self.graph.add_edge(subj_node, event_node, relation="ACTOR")
        self.graph.add_edge(event_node, obj_node, relation="TARGET")

        return event_id

    def bulk_load_facts(self, facts: list[dict]) -> int:
        """Load a batch of facts. Each fact: {subject_id, predicate, object_id, timestamp}."""
        count = 0
        for fact in facts:
            ts = fact.get("timestamp") or fact.get("start_time") or ""
            st = fact.get("start_time") or ts
            et = fact.get("end_time") or ts
            self.add_event(
                subject_id=str(fact["subject_id"]),
                predicate=fact.get("predicate") or fact.get("relation", ""),
                object_id=str(fact["object_id"]),
                timestamp=ts,
                source=fact.get("source", "icews14"),
                start_time=st,
                end_time=et,
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Queries (same interface as TemporalKGStore)
    # ------------------------------------------------------------------

    def entity_exists(self, entity_name: str) -> bool:
        """Check if an entity exists by name (case-insensitive)."""
        return entity_name.lower() in self._entity_name_index

    def get_entity_id(self, entity_name: str) -> Optional[str]:
        """Get entity ID from name."""
        return self._entity_name_index.get(entity_name.lower())

    def find_entities_fuzzy(self, name: str, limit: int = 5) -> list[dict]:
        """Find entities whose name contains the given substring."""
        name_lower = name.lower()
        results = []
        for eid, edata in self._entities.items():
            if name_lower in edata["name"].lower():
                results.append(edata)
                if len(results) >= limit:
                    break
        return results

    def get_events_by_entity(
        self, entity_name: str, limit: int = 50
    ) -> list[dict]:
        """Get all events involving an entity (as subject or object)."""
        entity_id = self.get_entity_id(entity_name)
        if not entity_id:
            return []

        entity_node = f"entity:{entity_id}"
        results = []

        # Events where entity is ACTOR (subject)
        if self.graph.has_node(entity_node):
            for _, event_node in self.graph.out_edges(entity_node):
                if self.graph.nodes[event_node].get("type") == "event":
                    event_data = dict(self.graph.nodes[event_node])
                    # Find target
                    for _, target_node in self.graph.out_edges(event_node):
                        if self.graph.nodes[target_node].get("type") == "entity":
                            event_data["subject"] = entity_name
                            event_data["object"] = self.graph.nodes[target_node].get("name", "")
                            results.append(event_data)
                            break

            # Events where entity is TARGET (object)
            for source_node, _ in self.graph.in_edges(entity_node):
                if self.graph.nodes[source_node].get("type") == "event":
                    event_data = dict(self.graph.nodes[source_node])
                    # Find subject
                    for actor_node, _ in self.graph.in_edges(source_node):
                        if self.graph.nodes[actor_node].get("type") == "entity":
                            event_data["subject"] = self.graph.nodes[actor_node].get("name", "")
                            event_data["object"] = entity_name
                            results.append(event_data)
                            break

        return results[:limit]

    def get_events_by_entity_and_time(
        self,
        entity_name: str,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get events involving entity within a time range."""
        all_events = self.get_events_by_entity(entity_name, limit=9999)
        norm_start = _normalize_iso_date(start_date, is_end=False)
        norm_end = _normalize_iso_date(end_date, is_end=True)
        filtered = []
        for ev in all_events:
            ev_start = _normalize_iso_date(ev.get("start_time") or ev.get("timestamp", ""), is_end=False)
            ev_end = _normalize_iso_date(ev.get("end_time") or ev.get("timestamp", ""), is_end=True)
            if ev_start <= norm_end and ev_end >= norm_start:
                filtered.append(ev)
        return filtered[:limit]

    def get_events_between_entities(
        self,
        entity1_name: str,
        entity2_name: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get all events between two entities."""
        id1 = self.get_entity_id(entity1_name)
        id2 = self.get_entity_id(entity2_name)
        if not id1 or not id2:
            return []

        results = []
        for eid, edata in self._events.items():
            if (edata["subject_id"] == id1 and edata["object_id"] == id2) or \
               (edata["subject_id"] == id2 and edata["object_id"] == id1):
                # Attach readable names
                subj_name = self._entities.get(edata["subject_id"], {}).get("name", "")
                obj_name = self._entities.get(edata["object_id"], {}).get("name", "")
                result = {**edata, "subject": subj_name, "object": obj_name}
                results.append(result)

        return results[:limit]

    def get_events_between_entities_in_range(
        self,
        entity1_name: str,
        entity2_name: str,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get events between two entities within a time range."""
        all_events = self.get_events_between_entities(entity1_name, entity2_name, limit=9999)
        norm_start = _normalize_iso_date(start_date, is_end=False)
        norm_end = _normalize_iso_date(end_date, is_end=True)
        filtered = []
        for ev in all_events:
            ev_start = _normalize_iso_date(ev.get("start_time") or ev.get("timestamp", ""), is_end=False)
            ev_end = _normalize_iso_date(ev.get("end_time") or ev.get("timestamp", ""), is_end=True)
            if ev_start <= norm_end and ev_end >= norm_start:
                filtered.append(ev)
        return filtered[:limit]

    def verify_fact(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        timestamp: str,
    ) -> dict:
        """
        Verify a specific temporal fact against the knowledge graph.

        Returns a dict with verification details:
        - entity_exists: bool
        - relation_exists: bool
        - timestamp_match: bool
        - matching_facts: list of supporting facts
        """
        subj_id = self.get_entity_id(subject_name)
        obj_id = self.get_entity_id(object_name)

        result = {
            "subject_exists": subj_id is not None,
            "object_exists": obj_id is not None,
            "relation_exists": False,
            "timestamp_match": False,
            "exact_match": False,
            "matching_facts": [],
            "closest_timestamps": [],
        }

        if not subj_id or not obj_id:
            return result

        candidate_events = (
            self._subj_obj_events.get((subj_id, obj_id), []) +
            self._subj_obj_events.get((obj_id, subj_id), [])
        )
        norm_target_ts = _normalize_iso_date(timestamp, is_end=False)

        for edata in candidate_events:
            if edata["predicate"].lower() == predicate.lower():
                result["relation_exists"] = True
                result["closest_timestamps"].append(edata["timestamp"])
                ev_start = _normalize_iso_date(edata.get("start_time") or edata["timestamp"], is_end=False)
                ev_end = _normalize_iso_date(edata.get("end_time") or edata["timestamp"], is_end=True)
                if edata["timestamp"] == timestamp or (ev_start <= norm_target_ts <= ev_end):
                    result["timestamp_match"] = True
                    result["exact_match"] = True
                    subj_name = self._entities.get(edata["subject_id"], {}).get("name", subject_name)
                    obj_name = self._entities.get(edata["object_id"], {}).get("name", object_name)
                    result["matching_facts"].append({
                        "subject": subj_name,
                        "predicate": predicate,
                        "object": obj_name,
                        "timestamp": edata["timestamp"],
                    })

        result["exists"] = result["exact_match"]
        return result

    def query_entity_events(
        self,
        entity_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find events involving an entity (as actor or target), matching TemporalKGStore format."""
        if not self.entity_exists(entity_name):
            return []
        if limit == 1 and start_date is None and end_date is None:
            return [{"actor": entity_name, "predicate": "", "target": "", "timestamp": "", "event_id": "", "source": "memory"}]
        events = self.get_events_by_entity(entity_name, limit=limit if not (start_date or end_date) else 500)
        norm_start = _normalize_iso_date(start_date, is_end=False) if start_date else None
        norm_end = _normalize_iso_date(end_date, is_end=True) if end_date else None
        filtered = []
        for ev in events:
            ev_start = _normalize_iso_date(ev.get("start_time") or ev.get("timestamp", ""), is_end=False)
            ev_end = _normalize_iso_date(ev.get("end_time") or ev.get("timestamp", ""), is_end=True)
            if norm_start and ev_end < norm_start:
                continue
            if norm_end and ev_start > norm_end:
                continue
            filtered.append({
                "actor": ev.get("subject", ""),
                "predicate": ev.get("predicate", ""),
                "target": ev.get("object", ""),
                "timestamp": ev.get("timestamp", ""),
                "event_id": ev.get("event_id", ""),
                "source": ev.get("source", "memory"),
            })
            if len(filtered) >= limit:
                break
        filtered.sort(key=lambda x: x["timestamp"])
        return filtered[:limit]

    def verify_entity_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> dict[str, Any]:
        """Check whether a subject-predicate-object triple exists at any timestamp."""
        subj_id = self.get_entity_id(subject)
        obj_id = self.get_entity_id(obj)
        if not subj_id or not obj_id:
            return {"exists": False, "timestamps": []}

        timestamps = set()
        candidate_events = (
            self._subj_obj_events.get((subj_id, obj_id), []) +
            self._subj_obj_events.get((obj_id, subj_id), [])
        )
        for edata in candidate_events:
            if edata["predicate"].lower() == predicate.lower():
                timestamps.add(edata["timestamp"])

        return {"exists": len(timestamps) > 0, "timestamps": sorted(timestamps)}

    def query_events_by_predicate(
        self,
        predicate: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find events with a specific predicate."""
        filtered = []
        norm_start = _normalize_iso_date(start_date, is_end=False) if start_date else None
        norm_end = _normalize_iso_date(end_date, is_end=True) if end_date else None
        for edata in self._events.values():
            if edata["predicate"].lower() == predicate.lower():
                ev_start = _normalize_iso_date(edata.get("start_time") or edata.get("timestamp", ""), is_end=False)
                ev_end = _normalize_iso_date(edata.get("end_time") or edata.get("timestamp", ""), is_end=True)
                if norm_start and ev_end < norm_start:
                    continue
                if norm_end and ev_start > norm_end:
                    continue
                subj_name = self._entities.get(edata["subject_id"], {}).get("name", "")
                obj_name = self._entities.get(edata["object_id"], {}).get("name", "")
                filtered.append({
                    "subject": subj_name,
                    "predicate": edata["predicate"],
                    "object": obj_name,
                    "timestamp": edata.get("timestamp", ""),
                    "event_id": edata.get("event_id", ""),
                })
        filtered.sort(key=lambda x: x["timestamp"])
        return filtered[:limit]

    def search_entities_by_name(self, name: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search events for entities whose name matches or contains 'name'."""
        matching = self.find_entities_fuzzy(name, limit=5)
        facts: list[dict[str, Any]] = []
        for ent in matching:
            evs = self.get_events_by_entity(ent["name"], limit=limit)
            facts.extend(evs)
        return facts[:limit]

    def get_entity_events(self, entity_name: str, limit: int = 50) -> list[dict[str, Any]]:
        """Alias for query_entity_events / get_events_by_entity matching BaselineRunner expectations."""
        return self.get_events_by_entity(entity_name, limit=limit)

    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Dispatch query according to parameters to match TemporalRetriever expectations."""
        params = parameters or {}
        limit = params.get("limit", 50)
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        ref_date = params.get("reference_date")

        norm_start = _normalize_iso_date(start_date, is_end=False) if start_date else None
        norm_end = _normalize_iso_date(end_date, is_end=True) if end_date else None
        norm_ref = _normalize_iso_date(ref_date, is_end=False) if ref_date else None

        if "entity_a" in params and "entity_b" in params:
            raw = self.get_events_between_entities(params["entity_a"], params["entity_b"], limit=9999)
            filtered = []
            for ev in raw:
                ev_start = _normalize_iso_date(ev.get("start_time") or ev.get("timestamp", ""), is_end=False)
                ev_end = _normalize_iso_date(ev.get("end_time") or ev.get("timestamp", ""), is_end=True)
                if norm_start and ev_end < norm_start:
                    continue
                if norm_end and ev_start > norm_end:
                    continue
                filtered.append({
                    "subject": ev.get("subject", ""),
                    "predicate": ev.get("predicate", ""),
                    "object": ev.get("object", ""),
                    "timestamp": ev.get("timestamp", ""),
                    "event_id": ev.get("event_id", ""),
                })
            filtered.sort(key=lambda x: x["timestamp"])
            return filtered[:limit]

        elif "entity_name" in params:
            ent = params["entity_name"]
            raw = self.get_events_by_entity(ent, limit=9999)
            filtered = []
            is_before = "< date($reference_date)" in cypher or (ref_date and "before" in cypher.lower())
            is_after = "> date($reference_date)" in cypher or (ref_date and "after" in cypher.lower())
            for ev in raw:
                ev_start = _normalize_iso_date(ev.get("start_time") or ev.get("timestamp", ""), is_end=False)
                ev_end = _normalize_iso_date(ev.get("end_time") or ev.get("timestamp", ""), is_end=True)
                if norm_start and ev_end < norm_start:
                    continue
                if norm_end and ev_start > norm_end:
                    continue
                if is_before and norm_ref and ev_start >= norm_ref:
                    continue
                if is_after and norm_ref and ev_end <= norm_ref:
                    continue
                filtered.append({
                    "subject": ev.get("subject", ""),
                    "predicate": ev.get("predicate", ""),
                    "object": ev.get("object", ""),
                    "timestamp": ev.get("timestamp", ""),
                    "event_id": ev.get("event_id", ""),
                })
            if is_before:
                filtered.sort(key=lambda x: x["timestamp"], reverse=True)
            else:
                filtered.sort(key=lambda x: x["timestamp"])
            return filtered[:limit]

        elif "predicate" in params:
            return self.query_events_by_predicate(params["predicate"], start_date, end_date, limit)

        return []

    def get_entity_count(self) -> int:
        return len(self._entities)

    def get_event_count(self) -> int:
        return len(self._events)

    def get_all_predicates(self) -> list[str]:
        return sorted(self.get_all_relations())

    def find_temporal_neighbours(
        self,
        entity: str,
        timestamp: str,
        window_days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find events involving entity within window_days of timestamp."""
        try:
            from datetime import datetime, timedelta
            base_dt = datetime.strptime(timestamp[:10], "%Y-%m-%d")
            min_dt = (base_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
            max_dt = (base_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
        except Exception:
            min_dt = ""
            max_dt = "9999-99-99"

        events = self.query_entity_events(entity, start_date=min_dt, end_date=max_dt, limit=limit)
        return events

    def get_all_entity_names(self) -> list[str]:
        """Return all entity names."""
        return [e["name"] for e in self._entities.values()]

    def get_all_relations(self) -> list[str]:
        """Return all unique relation/predicate names."""
        return list({e["predicate"] for e in self._events.values()})

    def get_all_timestamps(self) -> list[str]:
        """Return all unique timestamps."""
        return sorted({e["timestamp"] for e in self._events.values()})

    def get_statistics(self) -> dict:
        """Return graph statistics."""
        return {
            "num_entities": len(self._entities),
            "num_events": len(self._events),
            "num_relations": len(self.get_all_relations()),
            "num_timestamps": len(self.get_all_timestamps()),
        }

    # ------------------------------------------------------------------
    # Persistence (save/load to disk)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the graph to a JSON file for reloading later."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": self._entities,
            "events": self._events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved graph to %s (%d entities, %d events)",
                     path, len(self._entities), len(self._events))

    def load(self, path: Path) -> None:
        """Load a previously saved graph from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._entities = data["entities"]
        self._events = data["events"]

        # Rebuild indexes
        self._entity_name_index = {}
        for eid, edata in self._entities.items():
            self._entity_name_index[edata["name"].lower()] = eid

        self._subj_obj_events = {}
        for evid, evdata in self._events.items():
            self._subj_obj_events.setdefault((evdata["subject_id"], evdata["object_id"]), []).append(evdata)

        # Rebuild NetworkX graph
        self.graph = nx.DiGraph()
        for eid, edata in self._entities.items():
            self.graph.add_node(f"entity:{eid}", type="entity", **edata)
        for evid, evdata in self._events.items():
            event_node = f"event:{evid}"
            subj_node = f"entity:{evdata['subject_id']}"
            obj_node = f"entity:{evdata['object_id']}"
            self.graph.add_node(event_node, type="event", **evdata)
            self.graph.add_edge(subj_node, event_node, relation="ACTOR")
            self.graph.add_edge(event_node, obj_node, relation="TARGET")

        logger.info("Loaded graph from %s (%d entities, %d events)",
                     path, len(self._entities), len(self._events))

    def close(self) -> None:
        """No-op for compatibility with TemporalKGStore."""
        pass
