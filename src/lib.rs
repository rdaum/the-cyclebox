extern crate alloc;
extern crate core;

use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;

use std::ops::DerefMut;

use atomic_enum::atomic_enum;
use parking_lot::{RwLock};
use sharded_slab::{Slab};

pub type Handle = usize;

pub trait Collector<O: ObjectMemory> {
    /// Construct an object inside the collector's owned ObjectMemory and register it into the
    /// collector.
    /// The collector's ObjectMemory will be called with information about the newly created object
    fn make(&mut self) -> Handle;

    /// Perform some function with the object memory owned by the collector.
    /// Function is given an immutable reference to the memory.
    fn with_memory<F: Fn(&O) -> R, R>(&self, f: F) -> R;

    /// Perform some mutating function with the object memory owned by the collector.
    /// Function is given an immutable reference to the memory, and should return a series
    /// of collector operations (e.g. upcount, downcounts) that should be applied by the collector
    /// after the execution.
    fn with_memory_mut<F: Fn(&mut O) -> MemMutResult<R>, R>(&mut self, f: F) -> R;

    /// Ask the collector's ObjectMemory for information about the children of a given object.
    fn children(&self, of: Handle) -> Vec<Handle>;

    /// Increase the reference count of a given object known to the collector.
    fn increment(&mut self, sn: Handle);

    /// Decrement the reference count of a given object known to the collector.
    fn decrement(&mut self, n: Handle);

    /// Go through, find any objects being kept alive due to cycles, and clean them up.
    fn process_cycles(&mut self);
}

/// Concurrent, reference counting garbage collector that can collect cycles.
/// Based on https://pages.cs.wisc.edu/~cymen/misc/interests/Bacon01Concurrent.pdf
pub struct CycleCollector<O: ObjectMemory> {
    nodes: Slab<RwLock<NodeHeader>>,
    roots: RwLock<Vec<Handle>>,
    cycle_buffer: RwLock<Vec<Vec<Handle>>>,
    object_memory: RwLock<O>,
}

pub trait ObjectMemory {
    fn finalize(&mut self, handle: Handle);
    fn created(&mut self, handle: Handle) -> Option<Vec<CollectorOp>>;
    fn children_of(&self, handle: Handle) -> Vec<Handle>;
}

pub enum CollectorOp {
    UpCount(Handle),
    DnCount(Handle),
}

pub struct MemMutResult<R> {
    result: R,
    operations: Option<Vec<CollectorOp>>,
}

impl<R> MemMutResult<R> {
    pub fn with_result_and_operations(result: R, operations: Vec<CollectorOp>) -> Self {
        MemMutResult {
            result,
            operations: Some(operations),
        }
    }
    pub fn with_result(result: R) -> Self {
        MemMutResult {
            result,
            operations: None,
        }
    }
}

impl MemMutResult<()> {
    pub fn with_operations(operations: Vec<CollectorOp>) -> MemMutResult<()> {
        MemMutResult::<()>::with_result_and_operations((), operations)
    }

    pub fn unit() -> MemMutResult<()> {
        MemMutResult::<()>::with_result(())
    }
}

#[atomic_enum]
#[derive(Eq, PartialEq)]
enum Color {
    White,
    Gray,
    Black,
    Orange,
    Purple,
    Red,
}

// The header is kept separate from the object so that they can be kept in a contiguous fashion in
// the arena so that garbage collection can hopefully be kept in cache, or at least parts of it.
struct NodeHeader {
    rc: u32,
    crc: u32,
    color: Color,
    buffered: bool,
}

impl NodeHeader {
    pub fn new() -> Self {
        Self {
            rc: 0,
            crc: 0,
            color: Color::Black,
            buffered: false,
        }
    }
}


/// We make CycleCollector Send&Sync-able because the locks inside are taking care of making this
/// (in theory) safe.
/// This allows for global shared reference to the collector itself without having to stash it
/// behind an Arc or Mutex, which would defeat the purpose of the whole exercise.
unsafe impl<O: ObjectMemory> Send for CycleCollector<O> {}

unsafe impl<O: ObjectMemory> Sync for CycleCollector<O> {}

impl<O: ObjectMemory> CycleCollector<O> {
    pub fn new(object_memory: O) -> Self {
        Self {
            nodes: Slab::new(),
            roots: RwLock::new(Vec::with_capacity(8)),
            cycle_buffer: RwLock::new(Vec::with_capacity(8)),
            object_memory: RwLock::new(object_memory),
        }
    }

    fn apply_operations(&mut self, operations: Vec<CollectorOp>) {
        for op in operations {
            match op {
                CollectorOp::UpCount(handle) => self.increment(handle),
                CollectorOp::DnCount(handle) => self.decrement(handle),
            }
        }
    }

    fn scan_black(&self, sn: Handle) {
        if let Some(s) = self.nodes.get(sn) {
            let mut s = s.write();
            if s.color != Color::Black {
                s.color = Color::Black;
                drop(s);
                let children = self.object_memory.read().children_of(sn);
                for t in children {
                    self.scan_black(t);
                }
            }
        }
    }

    fn possible_root(&mut self, sn: Handle) {
        self.scan_black(sn);
        if let Some(s) = self.nodes.get(sn) {
            let mut s = s.write();
            s.color = Color::Purple;
            if !s.buffered {
                s.buffered = true;
                self.roots.write().push(sn);
            }
        }
    }

    pub fn release(&mut self, sn: Handle) {
        let children = { self.object_memory.read().children_of(sn) };
        for t in children {
            self.decrement(t);
        }
        let should_free = if let Some(s) = self.nodes.get(sn) {
            let mut s = s.write();
            s.color = Color::Black;
            !s.buffered
        } else {
            false
        };
        if should_free {
            self.free(sn);
        }
    }

    fn collect_roots(&mut self) {
        let roots = {
            let mut roots = self.roots.write();
            std::mem::take(roots.deref_mut())
        };
        for sn in roots {
            let is_white = {
                if let Some(s) = self.nodes.get(sn) {
                    let s = s.write();
                    s.color == Color::White
                } else {
                    false
                }
            };
            if is_white {
                let mut current_cycle = vec![];
                // cheat code
                self.collect_white(sn, &mut current_cycle);
                self.cycle_buffer.write().push(current_cycle.to_owned());
            } else if let Some(s) = self.nodes.get(sn) {
                let mut s = s.write();
                s.buffered = false;
            }
        }
    }

    fn collect_white(&mut self, sn: Handle, current_cycle: &mut Vec<Handle>) {
        let made_orange = {
            let s = self.nodes.get(sn).unwrap();
            let mut s = s.write();
            if s.color == Color::White {
                s.color = Color::Orange;
                s.buffered = true;
                true
            } else {
                false
            }
        };
        if made_orange {
            current_cycle.push(sn);
            let children: Vec<Handle> = { self.object_memory.read().children_of(sn) };
            for handle in children {
                self.collect_white(handle, current_cycle);
            }
        }
    }

    fn sigma_preparation(&mut self) {
        let cb = { self.cycle_buffer.read().to_owned() };
        for c in &cb {
            for n in c {
                let n = self.nodes.get(*n).unwrap();
                let mut n = n.write();
                n.color = Color::Red;
                n.crc = n.rc;
            }
            for n in c {
                let children = { self.object_memory.read().children_of(*n) };
                for m in children {
                    let m = self.nodes.get(m).unwrap();
                    let mut m = m.write();
                    if m.color == Color::Red && m.crc > 0 {
                        m.crc -= 1;
                    }
                }
            }
            for n in c {
                let n = self.nodes.get(*n).unwrap();
                let mut n = n.write();
                n.color = Color::Orange;
            }
        }
    }

    fn free_cycles(&mut self) {
        let (mut to_free, mut to_refurbish) = (vec![], vec![]);

        {
            let cb = {
                let mut cb = self.cycle_buffer.write();
                let new_cb = std::mem::take(cb.deref_mut());
                new_cb
            };
            for c in cb.iter().rev() {
                if self.delta_test(c) && self.sigma_test(c) {
                    to_free.extend(c);
                } else {
                    to_refurbish.extend(c);
                }
            }
        }
        self.free_cycle(to_free);
        self.refurbish(to_refurbish);
    }

    fn delta_test(&self, c: &Vec<Handle>) -> bool {
        for n in c {
            let n = self.nodes.get(*n).unwrap();
            let n = n.read();
            if n.color != Color::Orange {
                return false;
            }
        }
        true
    }

    fn sigma_test(&self, c: &Vec<Handle>) -> bool {
        let mut extern_rc: u32 = 0;
        for n in c {
            let n = self.nodes.get(*n).unwrap();
            let n = n.read();

            extern_rc += n.crc;
        }
        extern_rc == 0
    }

    fn refurbish(&mut self, c: Vec<Handle>) {
        let mut first = false;
        for nn in c {
            {
                let n = self.nodes.get(nn).unwrap();
                let mut n = n.write();
                if (first && n.color == Color::Orange) || n.color == Color::Purple {
                    n.color = Color::Purple;
                    self.roots.write().push(nn);
                } else {
                    n.color = Color::Black;
                    n.buffered = false;
                }
            }
            first = false;
        }
    }

    fn free_cycle(&mut self, c: Vec<Handle>) {
        for n in &c {
            let n = self.nodes.get(*n).unwrap();
            let mut n = n.write();
            n.color = Color::Red;
        }
        for n in &c {
            let children = { self.object_memory.read().children_of(*n) };
            for m in children {
                self.cyclic_decrement(m);
            }
        }
        for n in c {
            self.free(n);
        }
    }

    fn cyclic_decrement(&mut self, mn: Handle) {
        let do_decrement = {
            let n = self.nodes.get(mn).unwrap();
            let mut m = n.write();
            if m.color != Color::Red {
                if m.color != Color::Orange {
                    m.rc -= 1;
                    m.crc -= 1;
                    false
                } else {
                    true
                }
            } else {
                false
            }
        };
        if do_decrement {
            self.decrement(mn);
        }
    }

    fn mark_roots(&mut self) {
        let (to_rm_root, to_gray) = {
            let roots = self.roots.read();
            let (to_gray, to_rm_root): (Vec<Handle>, Vec<Handle>) = roots.iter().partition(|sn| {
                let s = self.nodes.get(**sn).unwrap();
                let s = s.read();
                s.color == Color::Purple && s.rc > 0
            });

            (to_rm_root, to_gray)
        };
        {
            let mut roots = self.roots.write();
            *roots = to_gray.clone();
        }
        for s in to_gray.iter() {
            self.mark_gray(*s);
        }
        let to_free = to_rm_root.into_iter().filter(|sn| {
            let s = self.nodes.get(*sn).unwrap();
            let mut s = s.write();
            s.buffered = false;
            s.rc == 0
        });
        let to_free: Vec<Handle> = to_free.collect();
        for sn in to_free {
            self.free(sn);
        }
    }

    fn mark_gray(&mut self, sn: Handle) {
        let mut to_gray = vec![];
        {
            let s = self.nodes.get(sn).unwrap();
            let mut s = s.write();

            if s.color != Color::Gray {
                s.color = Color::Gray;
                s.crc = s.rc;
                let children = { self.object_memory.read().children_of(sn) };
                for t in children {
                    to_gray.push(t);
                }
            } else {
                s.crc -= 1;
            }
        }
        for t in to_gray {
            self.mark_gray(t);
        }
    }

    fn scan_roots(&mut self) {
        let roots = {
            self.roots.read().to_owned()
        };
        for s in roots {
            self.scan(s);
        }
    }

    fn scan(&mut self, sn: Handle) {
        let (mut scan_black, mut scan) = (vec![], vec![]);
        {
            let s = self.nodes.get(sn).unwrap();
            let mut s = s.write();
            if s.color == Color::Gray && s.crc == 0 {
                s.color = Color::White;
                let children = { self.object_memory.read().children_of(sn) };
                for t in children {
                    scan.push(t);
                }
            } else {
                scan_black.push(sn);
            }
        }
        for t in scan {
            self.scan(t);
        }
        for sn in scan_black {
            self.scan_black(sn);
        }
    }

    fn collect_cycles(&mut self) {
        self.mark_roots();
        self.scan_roots();
        self.collect_roots();
    }

    fn free(&mut self, handle: Handle) {
        self.nodes.remove(handle);
        self.object_memory.write().finalize(handle);
    }
}

impl<O: ObjectMemory> Collector<O> for CycleCollector<O> {
    fn make(&mut self) -> Handle {
        let handle = { self.nodes.insert(RwLock::new(NodeHeader::new())).unwrap() };
        let result = { self.object_memory.write().created(handle) };
        if let Some(operations) = result {
            self.apply_operations(operations);
        }
        handle
    }

    /// Perform some function with the object memory owned by the collector.
    /// Function is given an immutable reference to the memory.
    fn with_memory<F, R>(&self, f: F) -> R
        where
            F: Fn(&O) -> R,
    {
        let m = self.object_memory.read();
        f(&m)
    }

    /// Perform some mutating function with the object memory owned by the collector.
    /// Function is given an immutable reference to the memory, and should return a series
    /// of collector operations (e.g. upcount, downcounts) that should be applied by the collector
    /// after the execution.
    fn with_memory_mut<F, R>(&mut self, f: F) -> R
        where
            F: Fn(&mut O) -> MemMutResult<R>,
    {
        let result = {
            let mut m = self.object_memory.write();

            f(&mut m)
        };
        if let Some(operations) = result.operations {
            self.apply_operations(operations);
        }

        result.result
    }

    fn children(&self, of: Handle) -> Vec<Handle> {
        self.object_memory.read().children_of(of)
    }

    /// Phase 1 (pre epoch boundary)
    fn increment(&mut self, sn: Handle) {
        {
            let s = self.nodes.get(sn).unwrap();
            let mut s = s.write();

            s.rc += 1;
        }
        self.scan_black(sn);
    }

    fn decrement(&mut self, n: Handle) {
        let res = {
            let s = self.nodes.get(n).unwrap();
            let mut s = s.write();
            s.rc -= 1;
            s.rc == 0
        };
        if res {
            // See comments in Arc about ordering and fence here.
            self.release(n);
        } else {
            self.possible_root(n);
        }
    }
    // Phase 2 functions, spawned by epoch boundary
    /// Invoked once per epoch after increment/decrements have been collected.
    fn process_cycles(&mut self) {
        self.free_cycles();
        self.collect_cycles();
        self.sigma_preparation();
    }
}


#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::borrow::BorrowMut;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex, Once};
    use std::thread::sleep;
    use std::time::Duration;

    use indexmap::IndexMap;
    use rand::Rng;
    use threadpool::ThreadPool;

    use crate::{Collector, CollectorOp, CycleCollector, Handle, MemMutResult, ObjectMemory};
    use crate::CollectorOp::{DnCount, UpCount};

    struct TestObj {
        pub children: HashSet<Handle>,
    }

    // A very simple mock object memory, that just tracks the list of what's been garbage collected.
    struct TestMem {
        objects: IndexMap<Handle, TestObj>,
        collected: Vec<Handle>,
    }

    impl TestMem {
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

    impl ObjectMemory for TestMem {
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

    #[test]
    fn test_basic() {
        let mem = TestMem {
            objects: Default::default(),
            collected: vec![],
        };
        let mut collector = CycleCollector::new(mem);
        let a = collector.make();
        collector.increment(a);
        let b = collector.make();
        collector.with_memory_mut(|m| {
            m.append_child(a, b);
            MemMutResult::with_operations(vec![UpCount(b)])
        });
        collector.process_cycles();

        // Should trigger finalization of 'b' as there are no longer any live references.
        collector.with_memory_mut(|m| {
            m.remove_child(a, b);
            MemMutResult::with_operations(vec![DnCount(b)])
        });
        collector.process_cycles();

        assert!(collector.with_memory(|m| m.collected.contains(&b)));
        assert!(!collector.with_memory(|m| m.collected.contains(&a)));
    }

    #[test]
    fn test_simple_cycle() {
        let mem = TestMem {
            objects: Default::default(),
            collected: vec![],
        };
        let mut collector = CycleCollector::new(mem);
        let a = collector.make();
        collector.increment(a);
        let b = collector.make();
        collector.with_memory_mut(|m| {
            m.append_child(a, b);
            MemMutResult::with_operations(vec![UpCount(b)])
        });

        // Form a cycle.
        collector.with_memory_mut(|m| {
            m.append_child(b, a);
            MemMutResult::with_operations(vec![UpCount(a)])
        });

        // Should trigger finalization of 'b' as there are no longer any live references.
        collector.with_memory_mut(|m| {
            m.remove_child(a, b);
            MemMutResult::with_operations(vec![DnCount(b)])
        });
        collector.process_cycles();

        assert!(collector.with_memory(|m| m.collected.contains(&b)));
        assert!(!collector.with_memory(|m| m.collected.contains(&a)));
    }

    static mut COLLECTOR_GLOBAL: Option<CycleCollector<TestMem>> = None;
    static INIT: Once = Once::new();

    fn get_collector<'a>() -> &'a CycleCollector<TestMem> {
        unsafe { COLLECTOR_GLOBAL.as_ref().unwrap() }
    }

    fn get_collector_mut<'a>() -> &'a mut CycleCollector<TestMem> {
        unsafe { COLLECTOR_GLOBAL.as_mut().unwrap() }
    }

    extern "C" fn gc_create() {
        let objs = TestMem {
            objects: Default::default(),
            collected: vec![],
        };

        let collector = CycleCollector::new(objs);
        INIT.call_once(|| unsafe {
            *COLLECTOR_GLOBAL.borrow_mut() = Some(collector);
        });
    }

    unsafe fn produce_bench_object_graph(
        num_objects: usize,
        max_child_width: usize,
        num_mutate_cycles: usize,
    ) {
        // Create more objects,
        for _onum in 0..num_objects {
            let handle = get_collector_mut().make();
            get_collector_mut().increment(handle);
        }

        let n_workers = 4;
        let pool = ThreadPool::new(n_workers);

        let cycle_num = Arc::new(Mutex::new(0));

        let collector_cycle_num = cycle_num;
        pool.execute(move || {
            loop {
                {
                    let cycle_num = collector_cycle_num.clone();
                    let mut cycles = cycle_num.lock().unwrap();
                    if *cycles >= 4 {
                        break;
                    }
                    println!("Executing cycle {}", *cycles);
                    *cycles += 1;
                }

                // and collect any cycles
                get_collector_mut().process_cycles();
                sleep(Duration::from_secs(10));
            }
        });

        for _i in 0..num_mutate_cycles {
            let num_objects = num_objects;
            let max_child_width = max_child_width;

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
            });
        }
        pool.join();

        // And then collect last garbage.
        get_collector_mut().process_cycles();
    }

    #[test]
    fn test_mock_obj_graph() {
        println!("Entering test");
        gc_create();
        unsafe { produce_bench_object_graph(10000, 10, 50) };
    }
}
