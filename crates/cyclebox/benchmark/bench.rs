#[macro_use]
extern crate criterion;
use criterion::{BenchmarkId, Criterion};
use cyclebox::{Handle, NodeCollector};
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;

struct AnObject {
    handle: Handle,
}
fn produce_object_graph(num_objects: usize, max_child_width: usize, num_mutate_cycles : usize) {
    let mut collector = NodeCollector::new(|_o| {});
    let mut objs = vec![];
    // Create objects,
    for _onum in 0..num_objects {
        let handle = collector.make();
        objs.push(AnObject { handle });
    }

    for _i in 0..num_mutate_cycles {
        let mut rng = rand::thread_rng();

        // then create random references between them, some of which could (probably) be cycles
        for o in objs.iter() {
            let num_children = rng.gen_range(0..max_child_width);
            for _i in 0..num_children {
                let child = objs[rng.gen_range(0..num_objects)].handle;
                collector.append_child(o.handle, child);
            }
        }

        // then remove random references between them
        for o in objs.iter() {
            let rand_max_removes = rng.gen_range(0..max_child_width);
            for (i, c) in collector.children(o.handle).iter().enumerate() {
                if i >= rand_max_removes { break };
                collector.remove_child(o.handle, *c);
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
