"""Concurrency regression tests for POST /events (single-transaction submit).

The endpoint used to run its idempotency check, cross-channel conflict
check, event insert and task inserts as separate transactions, so two
concurrent submissions of one event_id could both pass the checks and the
loser died on the events-table primary key (500), and a concurrent topic
publication could interleave so one event ended up with tasks from both
channels. The checks and writes now commit in one atomic SQLite
transaction (Database.submit_event); these tests pin that behavior:

- N concurrent identical submissions (same event_id, payload and
  subscriptions) create exactly one set of tasks: one 201, the rest 200,
  every response naming the same task ids, and never a 500;
- N concurrent submissions with the same event_id but different payloads
  have exactly one winner (201); every loser gets 409 and stores nothing;
- a direct-vs-topic race on one event_id is won by exactly one channel:
  the losing side gets 409 and no task for the other channel's recipients
  is ever created;
- a replay with a different deliver_at never rewrites the first
  submission's schedule, also under concurrency;
- a rejected submission (unknown subscription id, conflicting payload)
  leaves neither an event row nor partial tasks, even when racing a valid
  submission.

Pure stdlib + the app's own modules; run with either of

    python3 -m unittest tests.test_events_concurrency -v
    python3 -m pytest tests/test_events_concurrency.py
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main

# Threads released together per scenario; large enough that a
# check-then-act implementation interleaves, small enough to stay fast.
FANOUT = 8


class _RunningApp:
    """Entered TestClient against a fresh temp database, workers disabled.

    WORKER_COUNT=0 keeps every submitted task 'pending' so final-state
    assertions are deterministic; the races under test are between API
    requests, not between requests and deliveries.
    """

    def __init__(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="wr-events-")
        self.db_path = os.path.join(self.tmp, "test.db")
        self._stack = ExitStack()

    def __enter__(self) -> TestClient:
        self._stack.enter_context(patch.object(main.config, "DB_PATH", self.db_path))
        self._stack.enter_context(patch.object(main.config, "WORKER_COUNT", 0))
        client = TestClient(main.app)
        self._stack.enter_context(client)
        self._stack.callback(self._cleanup)
        return client

    def __exit__(self, *exc) -> None:
        self._stack.close()

    def _cleanup(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self.db_path + suffix)
            except FileNotFoundError:
                pass
        os.rmdir(self.tmp)


def _post_concurrently(client: TestClient, bodies: list[dict], url: str = "/events") -> tuple[list, list]:
    """POST one body per thread, all released on one barrier.

    Returns the responses in the same order as ``bodies``. A shared
    TestClient is safe here: httpx.Client is thread-safe and the portal
    serializes entry into the app's event loop, while FastAPI runs these
    sync endpoints concurrently in its threadpool -- exactly the
    interleaving the race needed.
    """
    barrier = threading.Barrier(len(bodies))
    responses: list = [None] * len(bodies)
    errors: list[BaseException] = []

    def fire(i: int, body: dict) -> None:
        try:
            barrier.wait(timeout=10)
            responses[i] = client.post(url, json=body)
        except BaseException as exc:  # noqa: BLE001 - report any thread error
            errors.append(exc)

    threads = [threading.Thread(target=fire, args=(i, b)) for i, b in enumerate(bodies)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return responses, errors


class ConcurrentEventSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = _RunningApp()
        self.client = self._app.__enter__()
        self.addCleanup(self._app.__exit__, None, None, None)

    def _make_subscription(self, tag: str) -> str:
        resp = self.client.post(
            "/subscriptions",
            json={"target_url": f"http://example.test/{tag}", "secret": "s3cr3t"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _post_race(self, bodies: list[dict], url: str = "/events") -> list:
        responses, errors = _post_concurrently(self.client, bodies, url)
        self.assertEqual(errors, [])
        self.assertTrue(all(r is not None for r in responses))
        return responses

    def _db_rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self.client.app.state.db._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # identical replays: one task set, stable task ids, no 500s
    # ------------------------------------------------------------------
    def test_concurrent_identical_submissions_create_one_task_set(self) -> None:
        subs = [self._make_subscription("a"), self._make_subscription("b")]
        body = {
            "event_id": "evt-race",
            "subscription_ids": subs,
            "payload": {"msg": "hello", "n": 1},
        }
        responses = self._post_race([body] * FANOUT)

        statuses = sorted(r.status_code for r in responses)
        self.assertEqual(statuses, [200] * (FANOUT - 1) + [201])

        # Every response names the same task ids; the single 201 reports
        # them as created, the 200s as duplicates.
        first = next(r for r in responses if r.status_code == 201).json()
        self.assertEqual(len(first["created"]), 2)
        self.assertEqual(first["duplicates"], [])
        expected_ids = sorted(t["task_id"] for t in first["created"])
        for r in responses:
            data = r.json()
            self.assertEqual(data["event_id"], "evt-race")
            ids = sorted(t["task_id"] for t in data["created"] + data["duplicates"])
            self.assertEqual(ids, expected_ids)
            if r.status_code == 200:
                self.assertEqual(data["created"], [])
                self.assertEqual(len(data["duplicates"]), 2)

        # Final state: exactly one event with exactly one task per
        # subscription -- no double-created tasks from the race.
        status = self.client.get("/events/evt-race").json()
        self.assertEqual(len(status["targets"]), 2)
        self.assertEqual(status["counts"]["pending"], 2)
        self.assertEqual(
            sorted(t["task_id"] for t in status["targets"]), expected_ids
        )
        self.assertEqual(
            self._db_rows("SELECT COUNT(*) FROM events WHERE id = 'evt-race'")[0][0],
            1,
        )

    def test_concurrent_replays_with_extra_subscription_extend_once(self) -> None:
        # Same id + payload but a wider subscription set: the new
        # subscription's task is created exactly once even when the wider
        # requests race each other.
        subs = [self._make_subscription("a"), self._make_subscription("b")]
        narrow = {
            "event_id": "evt-extend",
            "subscription_ids": subs[:1],
            "payload": {"n": 1},
        }
        wide = {
            "event_id": "evt-extend",
            "subscription_ids": subs,
            "payload": {"n": 1},
        }
        responses = self._post_race([narrow, wide, wide, wide])

        statuses = sorted(r.status_code for r in responses)
        # Narrow first: it 201s, then the first wide request 201s the new
        # subscription's task. Wide first: it 201s both tasks and the
        # narrow request is then a pure replay (200). Either way exactly
        # one task per subscription is ever created.
        self.assertTrue(set(statuses) <= {200, 201})
        self.assertIn(statuses.count(201), (1, 2))
        # Whatever the interleaving, the event ends with exactly one task
        # per subscription and every response references only those ids.
        status = self.client.get("/events/evt-extend").json()
        self.assertEqual(len(status["targets"]), 2)
        all_ids = sorted(t["task_id"] for t in status["targets"])
        for r in responses:
            ids = sorted(t["task_id"] for t in r.json()["created"] + r.json()["duplicates"])
            self.assertTrue(set(ids) <= set(all_ids))

    # ------------------------------------------------------------------
    # conflicting payloads: exactly one winner, losers get a stable 409
    # ------------------------------------------------------------------
    def test_concurrent_conflicting_payloads_have_exactly_one_winner(self) -> None:
        sub = self._make_subscription("a")
        bodies = [
            {
                "event_id": "evt-conflict",
                "subscription_ids": [sub],
                "payload": {"n": i},
            }
            for i in range(FANOUT)
        ]
        responses = self._post_race(bodies)

        statuses = sorted(r.status_code for r in responses)
        self.assertEqual(statuses, [201] + [409] * (FANOUT - 1))
        for r in responses:
            if r.status_code == 409:
                self.assertIn("different payload", r.json()["detail"])

        # The stored event is the winner's payload, with exactly the
        # winner's single task; the losers left nothing behind.
        winner_idx = next(i for i, r in enumerate(responses) if r.status_code == 201)
        rows = self._db_rows("SELECT payload FROM events WHERE id = 'evt-conflict'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0][0]), {"n": winner_idx})
        status = self.client.get("/events/evt-conflict").json()
        self.assertEqual(len(status["targets"]), 1)
        self.assertEqual(status["targets"][0]["subscription_id"], sub)

    def test_sequential_conflict_leaves_first_event_untouched(self) -> None:
        sub = self._make_subscription("a")
        first = self.client.post(
            "/events",
            json={"event_id": "evt-seq", "subscription_ids": [sub], "payload": {"v": 1}},
        )
        self.assertEqual(first.status_code, 201, first.text)

        conflict = self.client.post(
            "/events",
            json={"event_id": "evt-seq", "subscription_ids": [sub], "payload": {"v": 2}},
        )
        self.assertEqual(conflict.status_code, 409)

        rows = self._db_rows("SELECT payload FROM events WHERE id = 'evt-seq'")
        self.assertEqual(json.loads(rows[0][0]), {"v": 1})
        status = self.client.get("/events/evt-seq").json()
        self.assertEqual(len(status["targets"]), 1)
        self.assertEqual(status["counts"]["pending"], 1)

    # ------------------------------------------------------------------
    # cross-channel race: direct event vs topic publication on one id
    # ------------------------------------------------------------------
    def test_concurrent_direct_and_topic_publish_never_mix_channels(self) -> None:
        direct_subs = [self._make_subscription("d1"), self._make_subscription("d2")]
        topic_subs = [self._make_subscription("t1"), self._make_subscription("t2")]
        for sid in topic_subs:
            resp = self.client.put(f"/subscriptions/{sid}/topics", json={"topics": ["news"]})
            self.assertEqual(resp.status_code, 200, resp.text)

        direct_body = {
            "event_id": "evt-xchannel",
            "subscription_ids": direct_subs,
            "payload": {"origin": "direct"},
        }
        topic_body = {
            "event_id": "evt-xchannel",
            "topic": "news",
            "payload": {"origin": "topic"},
        }
        pairs = FANOUT // 2
        barrier = threading.Barrier(pairs)
        direct_responses: list = [None] * pairs
        topic_responses: list = [None] * pairs
        errors: list[BaseException] = []

        def fire(i: int) -> None:
            try:
                barrier.wait(timeout=10)
                direct_responses[i] = self.client.post("/events", json=direct_body)
                topic_responses[i] = self.client.post("/topic-events", json=topic_body)
            except BaseException as exc:  # noqa: BLE001 - report any thread error
                errors.append(exc)

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(pairs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])

        direct_codes = sorted(r.status_code for r in direct_responses)
        topic_codes = sorted(r.status_code for r in topic_responses)
        direct_won = all(c in (200, 201) for c in direct_codes)
        topic_won = all(c in (200, 201) for c in topic_codes)
        # Exactly one channel commits; the other is a stable 409 wall.
        self.assertNotEqual(direct_won, topic_won)
        if direct_won:
            self.assertEqual(direct_codes, [200] * (pairs - 1) + [201])
            self.assertEqual(topic_codes, [409] * pairs)
            expected_subs, expected_origin = set(direct_subs), "direct"
        else:
            self.assertEqual(topic_codes, [200] * (pairs - 1) + [201])
            self.assertEqual(direct_codes, [409] * pairs)
            expected_subs, expected_origin = set(topic_subs), "topic"

        # The committed event's tasks belong exclusively to the winning
        # channel's recipients -- no recipient from the loser mixed in.
        status = self.client.get("/events/evt-xchannel").json()
        self.assertEqual(
            {t["subscription_id"] for t in status["targets"]}, expected_subs
        )
        self.assertEqual(len(status["targets"]), 2)
        rows = self._db_rows("SELECT payload FROM events WHERE id = 'evt-xchannel'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0][0]), {"origin": expected_origin})
        topic_rows = self._db_rows("SELECT COUNT(*) FROM topic_events WHERE id = 'evt-xchannel'")
        self.assertEqual(topic_rows[0][0], 0 if direct_won else 1)

    def test_cross_channel_conflict_is_symmetric_sequentially(self) -> None:
        direct_sub = self._make_subscription("d")
        topic_sub = self._make_subscription("t")
        self.client.put(f"/subscriptions/{topic_sub}/topics", json={"topics": ["news"]})

        # Direct first, then topic on the same id: 409, nothing stored.
        resp = self.client.post(
            "/events",
            json={"event_id": "evt-ns-1", "subscription_ids": [direct_sub], "payload": {"a": 1}},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        conflict = self.client.post(
            "/topic-events",
            json={"event_id": "evt-ns-1", "topic": "news", "payload": {"a": 1}},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            self._db_rows("SELECT COUNT(*) FROM topic_events WHERE id = 'evt-ns-1'")[0][0],
            0,
        )

        # Topic first, then direct on the same id: 409, no extra tasks.
        resp = self.client.post(
            "/topic-events",
            json={"event_id": "evt-ns-2", "topic": "news", "payload": {"b": 2}},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        conflict = self.client.post(
            "/events",
            json={"event_id": "evt-ns-2", "subscription_ids": [direct_sub], "payload": {"b": 2}},
        )
        self.assertEqual(conflict.status_code, 409)
        status = self.client.get("/events/evt-ns-2").json()
        self.assertEqual({t["subscription_id"] for t in status["targets"]}, {topic_sub})

    # ------------------------------------------------------------------
    # scheduled delivery: the first deliver_at survives concurrent replays
    # ------------------------------------------------------------------
    def test_replay_never_rewrites_first_deliver_at(self) -> None:
        sub = self._make_subscription("a")
        first = self.client.post(
            "/events",
            json={
                "event_id": "evt-sched",
                "subscription_ids": [sub],
                "payload": {"n": 1},
                "deliver_at": "2027-01-01T00:00:00Z",
            },
        )
        self.assertEqual(first.status_code, 201, first.text)
        first_deliver_at = first.json()["created"][0]["deliver_at"]
        self.assertIsNotNone(first_deliver_at)

        # Concurrent replays with a *different* schedule: all 200, all
        # reporting the first submission's deliver_at, none rewriting it.
        replay = {
            "event_id": "evt-sched",
            "subscription_ids": [sub],
            "payload": {"n": 1},
            "deliver_at": "2027-06-01T00:00:00Z",
        }
        responses = self._post_race([replay] * FANOUT)
        for r in responses:
            self.assertEqual(r.status_code, 200, r.text)
            dup = r.json()["duplicates"]
            self.assertEqual(len(dup), 1)
            self.assertEqual(dup[0]["deliver_at"], first_deliver_at)

        status = self.client.get("/events/evt-sched").json()
        self.assertEqual(len(status["targets"]), 1)
        self.assertEqual(status["targets"][0]["deliver_at"], first_deliver_at)

    def test_concurrent_scheduled_submissions_agree_on_one_schedule(self) -> None:
        sub = self._make_subscription("a")
        base = {"event_id": "evt-sched-race", "subscription_ids": [sub], "payload": {"n": 1}}
        early = {**base, "deliver_at": "2027-01-01T00:00:00Z"}
        late = {**base, "deliver_at": "2027-06-01T00:00:00Z"}
        bodies = [early, late] * (FANOUT // 2)
        responses = self._post_race(bodies)

        statuses = sorted(r.status_code for r in responses)
        self.assertEqual(statuses, [200] * (FANOUT - 1) + [201])

        # Whichever request committed first set the schedule; every other
        # response and the stored task agree with it.
        winner = next(r for r in responses if r.status_code == 201).json()
        scheduled = winner["created"][0]["deliver_at"]
        self.assertIsNotNone(scheduled)
        for r in responses:
            for ref in r.json()["created"] + r.json()["duplicates"]:
                self.assertEqual(ref["deliver_at"], scheduled)
        status = self.client.get("/events/evt-sched-race").json()
        self.assertEqual(len(status["targets"]), 1)
        self.assertEqual(status["targets"][0]["deliver_at"], scheduled)

    # ------------------------------------------------------------------
    # rejected submissions leave no event row and no partial tasks
    # ------------------------------------------------------------------
    def test_unknown_subscription_leaves_nothing_behind(self) -> None:
        good = self._make_subscription("a")
        resp = self.client.post(
            "/events",
            json={
                "event_id": "evt-unknown",
                "subscription_ids": [good, "missing-sub"],
                "payload": {"n": 1},
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("missing-sub", resp.json()["detail"])

        self.assertEqual(self.client.get("/events/evt-unknown").status_code, 404)
        self.assertEqual(
            self._db_rows("SELECT COUNT(*) FROM events WHERE id = 'evt-unknown'")[0][0],
            0,
        )
        self.assertEqual(
            self._db_rows("SELECT COUNT(*) FROM tasks WHERE event_id = 'evt-unknown'")[0][0],
            0,
        )

    def test_unknown_subscription_race_leaves_no_partial_state(self) -> None:
        good = self._make_subscription("a")
        valid = {
            "event_id": "evt-partial",
            "subscription_ids": [good],
            "payload": {"ok": True},
        }
        invalid = {
            "event_id": "evt-partial",
            "subscription_ids": [good, "missing-sub"],
            "payload": {"ok": True},
        }
        bodies = [valid, invalid] * (FANOUT // 2)
        responses = self._post_race(bodies)

        valid_codes = sorted(r.status_code for i, r in enumerate(responses) if i % 2 == 0)
        invalid_codes = sorted(r.status_code for i, r in enumerate(responses) if i % 2 == 1)
        self.assertEqual(valid_codes, [200] * (FANOUT // 2 - 1) + [201])
        self.assertEqual(invalid_codes, [400] * (FANOUT // 2))

        # The rejected requests left nothing: one event, exactly one task
        # (the valid request's only subscription).
        status = self.client.get("/events/evt-partial").json()
        self.assertEqual(len(status["targets"]), 1)
        self.assertEqual(status["targets"][0]["subscription_id"], good)
        self.assertEqual(
            self._db_rows("SELECT COUNT(*) FROM tasks WHERE event_id = 'evt-partial'")[0][0],
            1,
        )


if __name__ == "__main__":
    unittest.main()
