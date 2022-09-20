#[macro_use]
extern crate criterion;

use std::collections::HashSet;

use criterion::{BenchmarkId, Criterion};
use cyclebox::{Handle, NodeCollector, ObjectMemory};
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;


pub struct MockDB {
    pub objects: HashSet<Handle>
}

impl ObjectMemory for MockDB {
    fn finalize(&mut self, handle: Handle) {
        self.objects.remove(&handle);
    }
    fn created(&mut self, handle: Handle) {
        self.objects.insert(handle);
    }
}

fn produce_bench_object_graph(num_objects: usize, max_child_width: usize, num_mutate_cycles: usize) {
    let objs = MockDB {
        objects: Default::default(),
    };
    let mut collector = NodeCollector::new(objs);

    // Create more objects,
    for _onum in 0..num_objects {
        let handle = collector.make();
        collector.increment(handle);
    }

    for _i in 0..num_mutate_cycles {
        let mut rng = rand::thread_rng();

        // create random references between them, some of which could (probably) be cycles
        let mut parent_child_pairs = vec![];
        {
            let objs = collector.memory_mut();

            for o in objs.objects.iter() {
                let num_children = rng.gen_range(0..max_child_width);
                for _i in 0..num_children {
                    let child = objs.objects.iter().nth(rng.gen_range(0..num_objects));
                    if let Some(child) = child {
                        parent_child_pairs.push((*o, *child));
                    }
                }
            }
        }
        for pair in parent_child_pairs {
            collector.append_child(pair.0, pair.1);
        }
        let objs_copy: Vec<Handle> = {
            let objs = collector.memory_mut();

            objs.objects.iter().copied().collect()
        };

        for o in objs_copy.iter() {
            if !collector.memory().objects.contains(o) {
                continue;
            }

            let rand_max_removes = rng.gen_range(0..max_child_width);
            for (i, c) in collector.children(*o).iter().enumerate() {
                if i >= rand_max_removes {
                    break;
                };
                let valid_child = collector.memory().objects.contains(o)
                    && collector.memory().objects.contains(c);

                if valid_child {
                    collector.remove_child(*o, *c);
                }
            }
        }

        // and collect any cycles
        collector.process_cycles();
    }
}

fn bench_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("object_rand_graph");
    for s in &[1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(s), s, |b, s| {
            b.iter(|| produce_bench_object_graph(*s, 10, 10));
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_group
}
criterion_main!(benches);
