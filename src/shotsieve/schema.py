SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    path_key TEXT NOT NULL,
    size_bytes INTEGER,
    modified_time REAL,
    file_hash TEXT,
    format TEXT,
    width INTEGER,
    height INTEGER,
    capture_time TEXT,
    preview_path TEXT,
    preview_status TEXT,
    last_scan_time TEXT,
    last_error TEXT,
    scan_status TEXT NOT NULL DEFAULT 'new'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_files_path_key
ON files(path_key);

CREATE INDEX IF NOT EXISTS idx_files_last_scan_time
ON files(last_scan_time);

CREATE INDEX IF NOT EXISTS idx_files_preview_path
ON files(preview_path);

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    file_id INTEGER PRIMARY KEY,
    overall_score REAL,
    learned_backend TEXT,
    learned_raw_score REAL,
    learned_score_normalized REAL,
    learned_confidence REAL,
    source_modified_time REAL,
    source_size_bytes INTEGER,
    preset_name TEXT,
    model_version TEXT,
    computed_time TEXT,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scores_overall_score
ON scores(overall_score);

CREATE INDEX IF NOT EXISTS idx_scores_learned_normalized
ON scores(learned_score_normalized);

CREATE TABLE IF NOT EXISTS review_state (
    file_id INTEGER PRIMARY KEY,
    decision_state TEXT DEFAULT 'pending',
    delete_marked BOOLEAN DEFAULT 0,
    export_marked BOOLEAN DEFAULT 0,
    updated_time TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_review_delete_marked
ON review_state(delete_marked);

CREATE INDEX IF NOT EXISTS idx_review_export_marked
ON review_state(export_marked);

CREATE INDEX IF NOT EXISTS idx_review_decision_state
ON review_state(decision_state);

CREATE TRIGGER IF NOT EXISTS trg_review_state_no_conflict_insert
BEFORE INSERT ON review_state
FOR EACH ROW
WHEN COALESCE(NEW.delete_marked, 0) = 1 AND COALESCE(NEW.export_marked, 0) = 1
BEGIN
    SELECT RAISE(ABORT, 'delete_marked and export_marked cannot both be true');
END;

CREATE TRIGGER IF NOT EXISTS trg_review_state_no_conflict_update
BEFORE UPDATE ON review_state
FOR EACH ROW
WHEN COALESCE(NEW.delete_marked, 0) = 1 AND COALESCE(NEW.export_marked, 0) = 1
BEGIN
    SELECT RAISE(ABORT, 'delete_marked and export_marked cannot both be true');
END;

CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY,
    started_time TEXT NOT NULL,
    completed_time TEXT,
    root_path TEXT NOT NULL,
    preset_name TEXT,
    files_seen INTEGER NOT NULL DEFAULT 0,
    files_added INTEGER NOT NULL DEFAULT 0,
    files_updated INTEGER NOT NULL DEFAULT 0,
    files_unchanged INTEGER NOT NULL DEFAULT 0,
    files_removed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    error_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_scan_runs_started_time
ON scan_runs(started_time);
"""

SCHEMA_MIGRATIONS = {
    "scores": {
        "learned_confidence": "ALTER TABLE scores ADD COLUMN learned_confidence REAL",
        "source_modified_time": "ALTER TABLE scores ADD COLUMN source_modified_time REAL",
        "source_size_bytes": "ALTER TABLE scores ADD COLUMN source_size_bytes INTEGER",
    },
    "scan_runs": {
        "files_removed": "ALTER TABLE scan_runs ADD COLUMN files_removed INTEGER NOT NULL DEFAULT 0",
    },
    "review_state": {},
}