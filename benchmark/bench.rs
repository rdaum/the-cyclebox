#[macro_use]
extern crate criterion;

use std::borrow::BorrowMut;
use std::collections::HashSet;

use std::sync::{Arc, Once};
use std::thread::sleep;
use std::time::Duration;

use criterion::Criterion;
use cyclebox::CollectorOp::{DnCount, UpCount};
use cyclebox::{CollectorOp, Handle, MemMutResult, CycleCollector, ObjectMemory, Collector};
use indexmap::IndexMap;
use parking_lot::Mutex;
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;
use threadpool::ThreadPool;

struct TestObj {
    children: HashSet<Handle>,
}

struct MockDB {
    objects: IndexMap<Handle, TestObj>,
    collected: Vec<Handle>,
}

static mut COLLECTOR_GLOBAL: Option<CycleCollector<MockDB>> = None;
static INIT: Once = Once::new();

fn get_collector<'a>() -> &'a CycleCollector<MockDB> {
    unsafe { COLLECTOR_GLOBAL.as_ref().unwrap() }
}

fn get_collector_mut<'a>() -> &'a mut CycleCollector<MockDB> {
    unsafe { COLLECTOR_GLOBAL.as_mut().unwrap() }
}

extern "C" fn gc_create() {
    let objs = MockDB {
        objects: Default::default(),
        collected: vec![],
    };
    let collector = CycleCollector::new(objs);
    INIT.call_once(|| unsafe {
        *COLLECTOR_GLOBAL.borrow_mut() = Some(collector);
    });
}

impl MockDB {
    fn append_child(&mut self, parent: Handle, child: Handle) {
        self.objects
            .get_mut(&parent)
            .unwrap()
            .children
            .insert(child);
    }
    fn remove_child(&mut self, parent: Handle, child: Handle) {
        self.objects
            .get_mut(&parent)
            .unwrap()
            .children
            .remove(&child);
    }
}

impl ObjectMemory for MockDB {
    fn finalize(&mut self, handle: Handle) {
        self.objects
            .remove(&handle)
            .expect("Invalid object reference in finalize");
        self.collected.push(handle);
    }
    fn created(&mut self, handle: Handle) -> Option<Vec<CollectorOp>> {
        self.objects.insert(
            handle,
            TestObj {
                children: HashSet::new(),
            },
        );
        None
    }
    fn children_of(&self, handle: Handle) -> Vec<Handle> {
        match self.objects.get(&handle) {
            None => panic!("Could not find children for object"),
            Some(o) => o.children.iter().copied().collect(),
        }
    }
}

unsafe fn concurrent_produce_bench_object_graph(
    num_objects: usize,
    max_child_width: usize,
    num_mutate_cycles: usize,
) {
    for _onum in 0..num_objects {
        let handle = get_collector_mut().make();
        get_collector_mut().increment(handle);
    }

    let n_workers = 4;
    let pool = ThreadPool::new(n_workers);

    let still_running = Arc::new(Mutex::new(true));
    let still_running_loop = still_running.clone();

    pool.execute(move || {
        loop {
            {
                if !(*still_running_loop.clone().lock()) {
                    break;
                }
            }

            // and collect any cycles
            get_collector_mut().process_cycles();
            sleep(Duration::from_millis(500));
        }
    });

    for i in 0..num_mutate_cycles {
        let num_objects = num_objects;
        let max_child_width = max_child_width;

        let still_running_cycle = still_running.clone();

        pool.execute(move || {
            // create random references between them, some of which could (probably) be cycles
            let parent_child_pairs = get_collector().with_memory(|objs| {
                let mut rng = rand::thread_rng();
                let mut parent_child_pairs = vec![];

                for o in objs.objects.iter() {
                    let num_children = rng.gen_range(0..max_child_width);
                    for _i in 0..num_children {
                        let child = objs.objects.get_index(rng.gen_range(0..num_objects));
                        if let Some(child) = child {
                            parent_child_pairs.push((*o.0, *child.0));
                        }
                    }
                }
                parent_child_pairs
            });

            for pair in parent_child_pairs {
                get_collector_mut().with_memory_mut(|m| {
                    m.append_child(pair.0, pair.1);
                    MemMutResult::with_operations(vec![UpCount(pair.1)])
                })
            }
            let objs_copy: Vec<Handle> =
                get_collector().with_memory(|objs| objs.objects.keys().copied().collect());

            let mut rng = rand::thread_rng();
            for o in objs_copy.iter() {
                if get_collector().with_memory(|objs| objs.objects.contains_key(o)) {
                    continue;
                }

                let rand_max_removes = rng.gen_range(0..max_child_width);
                for (i, c) in get_collector().children(*o).iter().enumerate() {
                    if i >= rand_max_removes {
                        break;
                    };
                    let valid_child = get_collector().with_memory(|objs| {
                        objs.objects.contains_key(o) && objs.objects.contains_key(c)
                    });

                    if valid_child {
                        get_collector_mut().with_memory_mut(|m| {
                            m.remove_child(*o, *c);
                            MemMutResult::with_operations(vec![DnCount(*c)])
                        })
                    }
                }
            }
            if i == num_mutate_cycles - 1 {
                (*still_running_cycle.lock()) = false;
            }
        });
    }
    pool.join();

    // And then collect last garbage.
    get_collector_mut().process_cycles();
}

unsafe fn produce_bench_object_graph(
    num_objects: usize,
    max_child_width: usize,
    num_mutate_cycles: usize,
) {
    for _onum in 0..num_objects {
        let handle = get_collector_mut().make();
        get_collector_mut().increment(handle);
    }

    for _i in 0..num_mutate_cycles {
        let num_objects = num_objects;
        let max_child_width = max_child_width;

        // create random references between them, some of which could (probably) be cycles
        let parent_child_pairs = get_collector().with_memory(|objs| {
            let mut rng = rand::thread_rng();
            let mut parent_child_pairs = vec![];

            for o in objs.objects.iter() {
                let num_children = rng.gen_range(0..max_child_width);
                for _i in 0..num_children {
                    let child = objs.objects.get_index(rng.gen_range(0..num_objects));
                    if let Some(child) = child {
                        parent_child_pairs.push((*o.0, *child.0));
                    }
                }
            }
            parent_child_pairs
        });

        for pair in parent_child_pairs {
            get_collector_mut().with_memory_mut(|m| {
                m.append_child(pair.0, pair.1);
                MemMutResult::with_operations(vec![UpCount(pair.1)])
            })
        }
        let objs_copy: Vec<Handle> =
            get_collector().with_memory(|objs| objs.objects.keys().copied().collect());

        let mut rng = rand::thread_rng();
        for o in objs_copy.iter() {
            if get_collector().with_memory(|objs| objs.objects.contains_key(o)) {
                continue;
            }

            let rand_max_removes = rng.gen_range(0..max_child_width);
            for (i, c) in get_collector().children(*o).iter().enumerate() {
                if i >= rand_max_removes {
                    break;
                };
                let valid_child = get_collector().with_memory(|objs| {
                    objs.objects.contains_key(o) && objs.objects.contains_key(c)
                });

                if valid_child {
                    get_collector_mut().with_memory_mut(|m| {
                        m.remove_child(*o, *c);
                        MemMutResult::with_operations(vec![DnCount(*c)])
                    })
                }
            }
        }
    }

    get_collector_mut().process_cycles();

    // And then collect last garbage.
    get_collector_mut().process_cycles();
}

fn bench_group(c: &mut Criterion) {
    gc_create();
    let mut group = c.benchmark_group("object_rand_graph");
    group.bench_function("concurrent", |b| {
        b.iter(|| unsafe { concurrent_produce_bench_object_graph(100, 10, 10) });
    });
    group.bench_function("single", |b| {
        b.iter(|| unsafe { produce_bench_object_graph(100, 10, 10) });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_group
}
criterion_main!(benches);
