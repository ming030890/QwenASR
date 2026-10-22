//! One cancellable scoring session for a joined keyboard candidate request.
//! The initial distribution stays pinned; other distributions use a bounded cache.
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

struct Session {
    cancelled: Arc<AtomicBool>,
    distributions: HashMap<Vec<usize>, Vec<f32>>,
    first: Option<Vec<usize>>,
    bytes: usize,
    forwards: usize,
    hits: usize,
}
thread_local! { static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) }; }

pub fn check() -> Result<(), String> {
    SESSION.with(|s| {
        if s.borrow()
            .as_ref()
            .is_some_and(|s| s.cancelled.load(Ordering::Acquire))
        {
            Err("Suggestion request cancelled".into())
        } else {
            Ok(())
        }
    })
}
pub fn cached(ids: &[usize]) -> Option<Vec<f32>> {
    SESSION.with(|s| {
        let mut slot = s.borrow_mut();
        let s = slot.as_mut()?;
        let result = s.distributions.get(ids)?.clone();
        s.hits += 1;
        Some(result)
    })
}
pub fn remember(ids: &[usize], values: &[f32]) {
    SESSION.with(|s| {
        let mut slot = s.borrow_mut();
        let Some(s) = slot.as_mut() else {
            return;
        };
        s.forwards += 1;
        let bytes = values.len() * 4;
        // At most 8 MiB per active request, released on success/error/cancellation.
        // Preserve the common left-context distribution even for large phrase pools.
        if s.bytes + bytes > 8 * 1024 * 1024 {
            let first = s.first.clone();
            s.distributions.retain(|k, _| Some(k) == first.as_ref());
            s.bytes = s.distributions.values().map(|v| v.len() * 4).sum();
        }
        if s.first.is_none() {
            s.first = Some(ids.to_vec());
        }
        s.distributions.insert(ids.to_vec(), values.to_vec());
        s.bytes += bytes;
    });
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ScoringStats {
    pub forwards: usize,
    pub cache_hits: usize,
    pub cached_bytes: usize,
}
pub fn stats() -> ScoringStats {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|s| ScoringStats {
                forwards: s.forwards,
                cache_hits: s.hits,
                cached_bytes: s.bytes,
            })
            .unwrap_or_default()
    })
}

pub fn run<T>(
    cancelled: Arc<AtomicBool>,
    work: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            SESSION.with(|s| *s.borrow_mut() = None);
        }
    }
    SESSION.with(|s| {
        assert!(s.borrow().is_none(), "Nested suggestion request");
        *s.borrow_mut() = Some(Session {
            cancelled,
            distributions: HashMap::new(),
            first: None,
            bytes: 0,
            forwards: 0,
            hits: 0,
        });
    });
    let _guard = Guard;
    check()?;
    let result = work()?;
    check()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_interrupts_running_work_and_does_not_poison_next_request() {
        let token = Arc::new(AtomicBool::new(false));
        assert!(run(token.clone(), || {
            remember(&[1], &[-1.]);
            assert_eq!(cached(&[1]), Some(vec![-1.]));
            token.store(true, Ordering::Release);
            check()
        })
        .is_err());
        run(Arc::new(AtomicBool::new(false)), || {
            assert!(cached(&[1]).is_none());
            check()
        })
        .unwrap();
    }
}
