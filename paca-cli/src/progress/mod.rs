use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use paca::progress::FileProgress;

pub struct IndicatifProgress {
    bar: ProgressBar,
}

impl IndicatifProgress {
    pub fn new(bar: ProgressBar) -> Self {
        Self { bar }
    }
}

impl FileProgress for IndicatifProgress {
    fn start(&self, initial_position: u64) {
        self.bar.set_style(download_style());
        self.bar.set_position(initial_position);
    }

    fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    fn println(&self, msg: &str) {
        self.bar.println(msg);
    }

    fn finish(&self) {
        self.bar.finish();
    }
}

/// Builds one progress reporter per `(filename, size)` tuple, attached
/// to a shared [`MultiProgress`] so bars render together in the terminal.
pub fn build_progress<'a, I>(iter: I) -> (MultiProgress, Vec<Arc<dyn FileProgress>>)
where
    I: IntoIterator<Item = (&'a str, u64)>,
{
    let multi = MultiProgress::new();
    let reporters = iter
        .into_iter()
        .map(|(filename, size)| {
            let bar = multi.add(ProgressBar::new(size));
            bar.set_style(pending_style());
            bar.set_message(filename.to_string());
            Arc::new(IndicatifProgress::new(bar)) as Arc<dyn FileProgress>
        })
        .collect();
    (multi, reporters)
}

fn pending_style() -> ProgressStyle {
    ProgressStyle::default_bar().template("{msg}").unwrap()
}

fn download_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
        .unwrap()
        .progress_chars("#>-")
}
