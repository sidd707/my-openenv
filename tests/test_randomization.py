"""
Tests for seeded randomization of task initial states.
Verifies reproducibility, variety, backward compat, and invariants.
"""

import pytest

from server.tasks.easy import FileCleanupTask
from server.tasks.medium import DatabaseMaintenanceTask
from server.tasks.hard import ServerMigrationTask
from server.environment import IrreversibleActionEnv


# ── Easy Task ────────────────────────────────────────────────────────────────


class TestEasyRandomization:
    def test_same_seed_produces_same_state(self):
        task = FileCleanupTask()
        s1 = task.get_initial_state(seed=42)
        s2 = task.get_initial_state(seed=42)
        assert s1 == s2

    def test_different_seeds_produce_different_states(self):
        task = FileCleanupTask()
        s1 = task.get_initial_state(seed=42)
        s2 = task.get_initial_state(seed=99)
        assert s1 != s2

    def test_no_seed_returns_deterministic_state(self):
        task = FileCleanupTask()
        s1 = task.get_initial_state()
        s2 = task.get_initial_state()
        assert s1 == s2

    def test_fixed_files_always_present(self):
        task = FileCleanupTask()
        required = [
            "report_2024.pdf", "config.yaml",
            "backup_data.zip", "system_cache_cleanup.tmp",
        ]
        for seed in [1, 42, 99, 200]:
            state = task.get_initial_state(seed=seed)
            for name in required:
                assert name in state["files"], f"{name} missing with seed={seed}"

    def test_total_freeable_mb_matches_files(self):
        task = FileCleanupTask()
        for seed in [None, 42, 99]:
            state = task.get_initial_state(seed=seed)
            expected = sum(
                f["size_mb"] for f in state["files"].values()
                if not f["important"]
            )
            assert state["total_freeable_mb"] == expected


# ── Medium Task ──────────────────────────────────────────────────────────────


class TestMediumRandomization:
    def test_same_seed_produces_same_state(self):
        task = DatabaseMaintenanceTask()
        s1 = task.get_initial_state(seed=42)
        s2 = task.get_initial_state(seed=42)
        assert s1 == s2

    def test_different_seeds_produce_different_states(self):
        task = DatabaseMaintenanceTask()
        s1 = task.get_initial_state(seed=42)
        s2 = task.get_initial_state(seed=99)
        assert s1 != s2

    def test_live_tables_always_present(self):
        task = DatabaseMaintenanceTask()
        for seed in [1, 42, 99, 200]:
            state = task.get_initial_state(seed=seed)
            for name in ["users", "sessions", "products", "orders"]:
                assert name in state["tables"], f"{name} missing with seed={seed}"

    def test_stale_tables_stay_stale(self):
        task = DatabaseMaintenanceTask()
        stale_pool = {"audit_log", "temp_imports", "old_reports", "cache_data"}
        for seed in [1, 42, 99, 200]:
            state = task.get_initial_state(seed=seed)
            for name, info in state["tables"].items():
                if name in stale_pool:
                    assert info["last_accessed_days_ago"] > 30, (
                        f"{name} has last_accessed={info['last_accessed_days_ago']} with seed={seed}"
                    )


# ── Hard Task ────────────────────────────────────────────────────────────────


class TestHardRandomization:
    def test_same_seed_produces_same_state(self):
        task = ServerMigrationTask()
        s1 = task.get_initial_state(seed=42)
        s2 = task.get_initial_state(seed=42)
        assert s1 == s2

    def test_no_seed_returns_deterministic_state(self):
        task = ServerMigrationTask()
        s1 = task.get_initial_state()
        s2 = task.get_initial_state()
        assert s1 == s2

    def test_metadata_present_with_seed(self):
        task = ServerMigrationTask()
        state = task.get_initial_state(seed=42)
        assert "metadata" in state
        assert state["metadata"]["traffic_distribution"] in [
            "80/20", "60/40", "50/50",
        ]

    def test_no_metadata_without_seed(self):
        task = ServerMigrationTask()
        state = task.get_initial_state()
        assert "metadata" not in state


# ── Environment seed passthrough ─────────────────────────────────────────────


class TestEnvironmentSeedPassthrough:
    def test_seed_produces_reproducible_state(self):
        env = IrreversibleActionEnv()
        obs1 = env.reset(seed=42, task_name="easy")
        state1 = obs1.current_state
        obs2 = env.reset(seed=42, task_name="easy")
        state2 = obs2.current_state
        assert state1 == state2

    def test_different_seeds_different_state_via_env(self):
        env = IrreversibleActionEnv()
        obs1 = env.reset(seed=42, task_name="easy")
        obs2 = env.reset(seed=99, task_name="easy")
        assert obs1.current_state != obs2.current_state
