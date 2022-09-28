#[macro_use]
extern crate criterion;

use std::borrow::{BorrowMut};
use std::collections::HashSet;
use std::sync::Once;

use criterion::{BenchmarkId, Criterion};
use cyclebox::{Handle, NodeCollector, ObjectMemory};
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;
use threadpool::ThreadPool;

static mut COLLECTOR_GLOBAL: Option<NodeCollector<MockDB>> = None;
static INIT: Once = Once::new();

fn get_collector<'a>() -> &'a NodeCollector<MockDB> {
    unsafe { COLLECTOR_GLOBAL.as_ref().unwrap() }
}

fn get_collector_mut<'a>() -> &'a mut NodeCollector<MockDB> {
    unsafe { COLLECTOR_GLOBAL.as_mut().unwrap() }
}

extern "C" fn gc_create() {
    let objs = MockDB {
        objects: Default::default(),
    };
    let collector = NodeCollector::new(objs);
    INIT.call_once(|| unsafe {
        *COLLECTOR_GLOBAL.borrow_mut() = Some(collector);
    });
}

pub struct MockDB {
    pub objects: HashSet<Handle>,
}

impl ObjectMemory for MockDB {
    fn finalize(&mut self, handle: Handle) {
        self.objects.remove(&handle);
    }
    fn created(&mut self, handle: Handle) {
        self.objects.insert(handle);
    }
}

unsafe fn produce_bench_object_graph(num_objects: usize, max_child_width: usize, num_mutate_cycles: usize) {


    // Create more objects,
    for _onum in 0..num_objects {
        let handle = get_collector_mut().make();
        get_collector_mut().increment(handle);
    }

    let n_workers = 16;
    let pool = ThreadPool::new(n_workers);

    for _i in 0..num_mutate_cycles {
        let num_objects = num_objects;
        let max_child_width = max_child_width;

        pool.execute(move || {

            // create random references between them, some of which could (probably) be cycles
            let parent_child_pairs =
                get_collector().with_memory(|objs| {
                    let mut rng = rand::thread_rng();
                    let mut parent_child_pairs = vec![];

                    for o in objs.objects.iter() {
                        let num_children = rng.gen_range(0..max_child_width);
                        for _i in 0..num_children {
                            let child = objs.objects.iter().nth(rng.gen_range(0..num_objects));
                            if let Some(child) = child {
                                parent_child_pairs.push((*o, *child));
                            }
                        }
                    }
                    parent_child_pairs
                });

            for pair in parent_child_pairs {
                get_collector_mut().append_child(pair.0, pair.1);
            }
            let objs_copy: Vec<Handle>  = get_collector().with_memory(|objs| {
                objs.objects.iter().copied().collect()
            });

            let mut rng = rand::thread_rng();
            for o in objs_copy.iter() {
                if get_collector().with_memory(|objs| objs.objects.contains(o)) {
                    continue;
                }

                let rand_max_removes = rng.gen_range(0..max_child_width);
                for (i, c) in get_collector().children(*o).iter().enumerate() {
                    if i >= rand_max_removes {
                        break;
                    };
                    let valid_child = get_collector().with_memory(|objs| {
                        objs.objects.contains(o)
                            && objs.objects.contains(c)
                    });

                    if valid_child {
                        get_collector_mut().remove_child(*o, *c);
                    }
                }
            }
        });

        {
            // and collect any cycles
            get_collector_mut().process_cycles();
        }
    }
}

fn bench_group(c: &mut Criterion) {
    gc_create();
    let mut group = c.benchmark_group("object_rand_graph");
    for s in &[1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(s), s, |b, s| {
            b.iter(|| unsafe { produce_bench_object_graph(*s, 10, 10) });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_group
}
criterion_main!(benches);
