use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

pub struct ComputeThreadPool {
    handles: Vec<JoinHandle<()>>,
    thread_data: Vec<Arc<ThreadData>>,

    num_threads: usize,
    shutdown: Arc<AtomicBool>,
}

struct ThreadData {
    busy: AtomicBool,

    task: Mutex<Option<Task>>,
    condvar: Condvar,

    done: AtomicBool,
    done_mutex: Mutex<()>,
    done_condvar: Condvar,
}

type Task = Box<dyn FnOnce(usize) + Send + 'static>;

impl ComputeThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let num_threads = if num_threads == 0 {
            num_cpus::get()
        } else {
            num_threads
        };

        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::with_capacity(num_threads);
        let mut thread_datas = Vec::with_capacity(num_threads);

        for id in 0..num_threads {
            let thread_barrier = Arc::clone(&barrier);
            let thread_shutdown = Arc::clone(&shutdown);

            let data = Arc::new(ThreadData {
                busy: AtomicBool::new(false),
                task: Mutex::new(None),
                condvar: Condvar::new(),
                done: AtomicBool::new(true),
                done_mutex: Mutex::new(()),
                done_condvar: Condvar::new(),
            });

            let thread_data = Arc::clone(&data);
            thread_datas.push(Arc::clone(&data));

            let handle = thread::spawn(move || {
                thread_barrier.wait();

                loop {
                    if thread_shutdown.load(Ordering::Acquire) {
                        break;
                    }

                    let task = {
                        let mut task_guard = thread_data.task.lock().unwrap();

                        if task_guard.is_none() {
                            while task_guard.is_none() && !thread_shutdown.load(Ordering::Acquire) {
                                task_guard = thread_data.condvar.wait(task_guard).unwrap();
                            }

                            if task_guard.is_none() && thread_shutdown.load(Ordering::Acquire) {
                                break;
                            }
                        }

                        thread_data.busy.store(true, Ordering::Release);
                        thread_data.done.store(false, Ordering::Release);
                        task_guard.take()
                    };

                    if let Some(task) = task {
                        task(id);
                    }

                    thread_data.busy.store(false, Ordering::Release);
                    thread_data.done.store(true, Ordering::Release);
                    let _lock = thread_data.done_mutex.lock().unwrap();

                    thread_data.done_condvar.notify_all();
                }
            });

            handles.push(handle);
        }

        barrier.wait();

        ComputeThreadPool {
            handles,
            thread_data: thread_datas,
            num_threads,
            shutdown,
        }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn parallel_for<I, T, F>(&self, iter: I, f: F)
    where
        I: IntoIterator<Item = T>,
        T: Send + Sync + 'static,
        F: Fn(T) + Send + Sync + 'static,
    {
        let f = Arc::new(f);

        let mut threads = Vec::new();
        for (thread, item) in self.thread_data.iter().zip(iter.into_iter()) {
            threads.push(thread);
            let f = Arc::clone(&f);

            let task: Task = Box::new(move |_| {
                f(item);
            });

            let mut task_guard = thread.task.lock().unwrap();
            *task_guard = Some(task);

            let _lock = thread.done_mutex.lock().unwrap();

            thread.busy.store(true, Ordering::Relaxed);
            thread.done.store(false, Ordering::Relaxed);
            thread.condvar.notify_one();
        }

        self.wait_specific(&threads);
    }

    fn wait_specific(&self, threads: &[&Arc<ThreadData>]) {
        let count_busy_threds = || {
            threads
                .iter()
                .filter(|t| t.busy.load(Ordering::Relaxed))
                .count()
        };
        let mut remaining = count_busy_threds();
        if remaining == 0 {
            return;
        }

        let spin_start = Instant::now();
        let spin_duration = std::time::Duration::from_micros(50);

        while Instant::now().duration_since(spin_start) < spin_duration {
            remaining = count_busy_threds();
            if remaining == 0 {
                return;
            }
            std::hint::spin_loop();
        }

        let yield_start = Instant::now();
        let yield_duration = std::time::Duration::from_millis(1);

        while Instant::now().duration_since(yield_start) < yield_duration {
            remaining = count_busy_threds();
            if remaining == 0 {
                return;
            }
            std::thread::yield_now();
        }

        for thread in threads {
            if thread.done.load(Ordering::Acquire) {
                continue;
            }

            let guard = thread.done_mutex.lock().unwrap();
            if thread.done.load(Ordering::Acquire) {
                continue;
            }

            let _unused = thread
                .done_condvar
                .wait_while(guard, |_| !thread.done.load(Ordering::SeqCst))
                .unwrap();
        }
    }

    pub fn shutdown(&mut self) -> std::thread::Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);

        for thread in &self.thread_data {
            let mut task_guard = thread.task.lock().unwrap();
            *task_guard = None;
            thread.condvar.notify_one();
        }
        let handles = std::mem::take(&mut self.handles);

        for handle in handles {
            handle.join()?;
        }
        Ok(())
    }

    pub fn resize(&mut self, num_threads: usize) -> std::thread::Result<()> {
        let num_threads = if num_threads == 0 {
            num_cpus::get()
        } else {
            num_threads
        };
        if num_threads == self.num_threads {
            return Ok(());
        }
        self.shutdown()?;
        let new_pool = ComputeThreadPool::new(num_threads);
        *self = new_pool;
        Ok(())
    }
}

impl Drop for ComputeThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);

        for thread in &self.thread_data {
            let mut task_guard = thread.task.lock().unwrap();
            *task_guard = None;
            thread.condvar.notify_one();
        }
    }
}

impl Default for ComputeThreadPool {
    fn default() -> Self {
        Self::new(0)
    }
}
