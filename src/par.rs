//! Fork-join parallelism on a small persistent thread pool (std only).
//!
//! Kernels split their output into a few contiguous parts; `Par` runs the
//! parts on the pool's workers plus the calling thread and returns when
//! all are done. Workers are created once, because starting OS threads for
//! every kernel call costs more than many of the calls themselves.

use std::cell::Cell;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Smallest kernel call, in multiply-adds, worth splitting across threads.
pub const PARALLEL_MIN_WORK: usize = 1 << 18;

#[derive(Clone)]
pub struct Par {
    pub threads: usize,
    pool: Option<Arc<Pool>>,
}

impl std::fmt::Debug for Par {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Par {{ threads: {} }}", self.threads)
    }
}

impl Par {
    /// `threads` = 0 uses every available core.
    pub fn new(threads: usize) -> Self {
        let t = if threads == 0 {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            threads
        }
        .max(1);
        Par {
            threads: t,
            pool: (t > 1).then(|| Arc::new(Pool::new(t - 1))),
        }
    }

    pub fn serial() -> Self {
        Par { threads: 1, pool: None }
    }

    /// How many parts to split `units` work items into, given the total
    /// amount of work.
    pub fn parts(&self, units: usize, work: usize) -> usize {
        if work < PARALLEL_MIN_WORK {
            1
        } else {
            self.threads.min(units).max(1)
        }
    }

    /// Runs `f(i)` for `i in 0..tasks`, in parallel when possible.
    fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        match &self.pool {
            Some(pool) if tasks > 1 && !IN_WORKER.with(Cell::get) => pool.run(tasks, f),
            _ => (0..tasks).for_each(f),
        }
    }

    /// Calls `f(begin, end)` on `parts` contiguous ranges covering
    /// `0..units`, each range a multiple of `quantum` except the last.
    pub fn for_ranges<F>(&self, units: usize, quantum: usize, parts: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let ranges = split(units, quantum, parts);
        self.run(ranges.len(), &|i| f(ranges[i].0, ranges[i].1));
    }

    /// Runs `f(first_row, chunk)` on contiguous chunks of `out`, where each
    /// row is `row_len` elements and chunks hold a multiple of `quantum`
    /// rows. The last chunk also gets a trailing partial row, if any.
    pub fn for_row_chunks<F>(&self, out: &mut [f32], row_len: usize, quantum: usize, work: usize, f: F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let rows = if row_len == 0 { 0 } else { out.len() / row_len };
        let parts = self.parts(rows.div_ceil(quantum.max(1)), work);
        let ranges = split(rows, quantum, parts);
        if ranges.len() <= 1 {
            f(0, out);
            return;
        }
        let len = out.len();
        let base = SendPtr(out.as_mut_ptr());
        let last = ranges.len() - 1;
        self.run(ranges.len(), &|i| {
            let (b, e) = ranges[i];
            let end = if i == last { len } else { e * row_len };
            // SAFETY: the ranges are disjoint and inside `out`, which stays
            // mutably borrowed until every task has finished.
            let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(b * row_len), end - b * row_len) };
            f(b, chunk);
        });
    }
}

fn split(units: usize, quantum: usize, parts: usize) -> Vec<(usize, usize)> {
    let q = quantum.max(1);
    let blocks = units.div_ceil(q);
    let parts = parts.min(blocks).max(1);
    let mut out = Vec::with_capacity(parts);
    let mut b = 0;
    for i in 0..parts {
        let nb = blocks * (i + 1) / parts;
        let e = (nb * q).min(units);
        if e > b {
            out.push((b, e));
        }
        b = e;
    }
    out
}

/// A raw pointer shared by the tasks of one fork-join call, each writing
/// a disjoint part of the buffer.
#[derive(Clone, Copy)]
pub struct SendPtr(pub *mut f32);

unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    pub fn get(self) -> *mut f32 {
        self.0
    }
}

// ------------------------------------------------------------------ pool

thread_local! {
    /// Set on pool workers: a kernel called from a task runs serially
    /// instead of waiting on the pool it is part of.
    static IN_WORKER: Cell<bool> = const { Cell::new(false) };
}

/// The current job, with the lifetime of its closure erased. `Pool::run`
/// does not return before every task of the job has finished, so the
/// closure outlives all uses.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync + 'static));

unsafe impl Send for Job {}

struct State {
    job: Option<Job>,
    next: usize,
    total: usize,
    remaining: usize,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
}

struct Pool {
    shared: Arc<Shared>,
    /// One job at a time when several threads share the executor.
    run_lock: Mutex<()>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(workers: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, next: 0, total: 0, remaining: 0, shutdown: false }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        let workers = (0..workers)
            .map(|i| {
                let s = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("mobileclip-{}", i))
                    .spawn(move || worker(&s))
                    .expect("spawn worker thread")
            })
            .collect();
        Pool { shared, run_lock: Mutex::new(()), workers }
    }

    fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        let _one_job = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: only the lifetime is changed; see `Job`.
        let job = Job(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(f)
        });
        {
            let mut st = self.shared.state.lock().unwrap();
            st.job = Some(job);
            st.next = 0;
            st.total = tasks;
            st.remaining = tasks;
        }
        self.shared.work.notify_all();

        // The calling thread takes tasks too.
        let mut st = self.shared.state.lock().unwrap();
        while st.next < st.total {
            let i = st.next;
            st.next += 1;
            drop(st);
            f(i);
            st = self.shared.state.lock().unwrap();
            st.remaining -= 1;
        }
        while st.remaining > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
        st.job = None;
    }
}

fn worker(s: &Shared) {
    IN_WORKER.with(|w| w.set(true));
    let mut st = s.state.lock().unwrap();
    loop {
        if st.shutdown {
            return;
        }
        match st.job {
            Some(job) if st.next < st.total => {
                let i = st.next;
                st.next += 1;
                drop(st);
                // SAFETY: the job is alive until `remaining` reaches 0,
                // which needs this task's decrement below.
                unsafe { (*job.0)(i) };
                st = s.state.lock().unwrap();
                st.remaining -= 1;
                if st.remaining == 0 {
                    s.done.notify_all();
                }
            }
            _ => st = s.work.wait(st).unwrap(),
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn split_covers_everything() {
        for units in [0, 1, 5, 16, 17, 100] {
            for q in [1, 4, 16] {
                for p in [1, 2, 3, 8] {
                    let r = split(units, q, p);
                    let mut next = 0;
                    for &(b, e) in &r {
                        assert_eq!(b, next);
                        assert!(e > b);
                        next = e;
                    }
                    assert_eq!(next, units);
                }
            }
        }
    }

    #[test]
    fn pool_runs_every_task_once() {
        let par = Par::new(4);
        for tasks in [1, 2, 3, 4, 9] {
            let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            for _ in 0..200 {
                par.for_ranges(tasks, 1, tasks, |b, e| {
                    for h in &hits[b..e] {
                        h.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 200));
        }
    }

    #[test]
    fn row_chunks_cover_the_tail() {
        let par = Par::new(3);
        let mut v = vec![0.0f32; 1000 * 7 + 3];
        par.for_row_chunks(&mut v, 7, 1, usize::MAX, |_, c| c.iter_mut().for_each(|x| *x += 1.0));
        assert!(v.iter().all(|&x| x == 1.0));
    }
}
