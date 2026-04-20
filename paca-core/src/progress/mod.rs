/// Per-file download progress reporter. CLI crates provide the concrete
/// rendering; paca-core depends only on this abstraction so nothing in
/// the core speaks indicatif (or any other UI library).
pub trait FileProgress: Send + Sync {
    /// Transition from pending to active state with an initial byte
    /// position. Called once before any `inc` calls.
    fn start(&self, initial_position: u64);

    /// Add `delta` bytes to the current position.
    fn inc(&self, delta: u64);

    /// Emit a log line alongside the progress display. Used for retry
    /// notifications and transient errors.
    fn println(&self, msg: &str);

    /// Mark the download complete.
    fn finish(&self);
}
