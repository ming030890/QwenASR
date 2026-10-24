//! One cancellable scoring session for a joined keyboard candidate request.
//! The initial distribution stays pinned; other distributions use a bounded cache.
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

struct Session {
    cancelled: Arc<AtomicBool>,
    distributions: HashMap<Vec<usize>, Vec<f32>>,
    first: Option<Vec<usize>>,
    projections: HashMap<Vec<usize>, (Vec<f32>, f32)>,
    projection_order: VecDeque<Vec<usize>>,
    projection_hits: usize,
    bytes: usize,
    forwards: usize,
    hits: usize,
    generation_forwards: usize,
    generation_complete: bool,
    output_batches: usize,
    batched_contexts: usize,
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
pub fn generation_complete() {
    SESSION.with(|s| {
        if let Some(s) = s.borrow_mut().as_mut() { s.generation_forwards = s.forwards; s.generation_complete = true; }
    });
}
pub fn record_output_batch(contexts: usize) {
    SESSION.with(|s| {
        if let Some(s)=s.borrow_mut().as_mut() { s.output_batches+=1;s.batched_contexts+=contexts; }
    });
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
/// Read only requested entries without cloning a full 256 KiB distribution.
pub fn cached_targets(ids: &[usize], targets: &[usize]) -> Option<Vec<f32>> {
    SESSION.with(|s| {
        let mut slot = s.borrow_mut();
        let s = slot.as_mut()?;
        let values = s.distributions.get(ids)?;
        let result = targets.iter().map(|&id| values[id]).collect();
        s.hits += 1;
        Some(result)
    })
}

// A full distribution can be evicted while its exact 128-float CNN projection
// and softmax normalizer remain inexpensive to retain. Later small target sets
// then need only those output rows, with identical float arithmetic.
pub fn projection(ids: &[usize]) -> Option<(Vec<f32>, f32)> {
    SESSION.with(|s| {
        let mut slot = s.borrow_mut();
        let s = slot.as_mut()?;
        let result = s.projections.get(ids)?.clone();
        s.projection_hits += 1;
        Some(result)
    })
}
pub fn remember_projection(ids: &[usize], projected: &[f32], normalizer: f32) {
    SESSION.with(|s| {
        let mut slot = s.borrow_mut();
        let Some(s) = slot.as_mut() else { return; };
        if !s.projections.contains_key(ids) {
            // About 1 MiB of numeric payload for the supported 32-token CNN.
            if s.projections.len() == 1024 {
                if let Some(oldest) = s.projection_order.pop_front() { s.projections.remove(&oldest); }
            }
            s.projection_order.push_back(ids.to_vec());
        }
        s.projections.insert(ids.to_vec(), (projected.to_vec(), normalizer));
    });
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
    pub generation_forwards: usize,
    pub generation_complete: bool,
    pub output_batches: usize,
    pub batched_contexts: usize,
    pub cache_hits: usize,
    pub projection_hits: usize,
    pub cached_bytes: usize,
}
pub fn stats() -> ScoringStats {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|s| ScoringStats {
                forwards: s.forwards,
                generation_forwards: s.generation_forwards,
                generation_complete: s.generation_complete,
                output_batches: s.output_batches,
                batched_contexts: s.batched_contexts,
                cache_hits: s.hits,
                projection_hits: s.projection_hits,
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
            projections: HashMap::new(), projection_order: VecDeque::new(), projection_hits: 0,
            bytes: 0,
            forwards: 0,
            hits: 0,
            generation_forwards: 0, generation_complete: false, output_batches: 0, batched_contexts: 0,
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
