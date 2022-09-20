#[macro_use]
extern crate criterion;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use criterion::{BenchmarkId, Criterion};
use cyclebox::{Handle, NodeCollector};
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;


fn produce_object_graph(num_objects: usize, max_child_width: usize, num_mutate_cycles: usize) {
    let objs = Arc::new(Mutex::new(HashSet::new()));
    let mut collector = NodeCollector::new(|o| {
        objs.lock().unwrap().remove(&o);
    });


    // Create more objects,
    for _onum in 0..num_objects {
        let handle = collector.make();
        collector.increment(handle);
        objs.lock().unwrap().insert(handle);
    }

    for _i in 0..num_mutate_cycles {
        let mut rng = rand::thread_rng();

        // create random references between them, some of which could (probably) be cycles
        {
            let objs = objs.lock().unwrap();
            for o in objs.iter() {
                let num_children = rng.gen_range(0..max_child_width);
                for _i in 0..num_children {
                    let child = objs.iter().nth(rng.gen_range(0..num_objects));
                    if child.is_some() {
                        collector.append_child(*o, *child.unwrap());
                    }
                }
            }
        }

        let objs_copy: Vec<Handle> = {
            objs.lock().unwrap().iter().map(|h| {
                *h
            }).collect()
        };

        for o in objs_copy.iter() {

            let valid_h = {
                let x = objs.lock().unwrap();
                x.contains(o)
            };
            if !valid_h { continue; }

            let rand_max_removes = rng.gen_range(0..max_child_width);
            for (i, c) in collector.children(*o).iter().enumerate() {
                if i >= rand_max_removes { break; };
                let valid_child = {
                    let x = objs.lock().unwrap();
                    x.contains(o) && x.contains(c)
                };

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
            b.iter(|| produce_object_graph(*s, 10, 10));
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_group
}
criterion_main!(benches);
