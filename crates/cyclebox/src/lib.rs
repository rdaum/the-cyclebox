extern crate alloc;
extern crate core;

use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;

use core::ops::{Index, IndexMut};

use std::hash::Hash;



use spin::RwLock;
use atomic_enum::atomic_enum;

use slotmap::{new_key_type, SlotMap};
use spin::rwlock::RwLockWriteGuard;

#[atomic_enum]
#[derive(Eq, PartialEq)]
pub enum Color {
    White,
    Gray,
    Black,
    Orange,
    Purple,
    Red,
}

new_key_type! { pub struct Handle; }
impl Handle {
    pub fn id(&self) -> u64 {
        self.0.as_ffi()
    }
}

// The header is kept separate from the object so that they can be kept in a contiguous fashion in
// the arena so that garbage collection can hopefully be kept in cache, or at least parts of it.
struct NodeHeader {
    rc: u32,
    crc: u32,
    color: Color,
    buffered: bool,
    children: Vec<Handle>,
}

impl NodeHeader {
    pub fn new() -> Self {
        Self {
            rc: 0,
            crc: 0,
            color: Color::Black,
            buffered: false,
            children: Vec::new(),
        }
    }
}

pub trait ObjectMemory {
    fn finalize(&mut self, handle: Handle);
    fn created(&mut self, handle: Handle);
}

// Persistent garbage collected storage for slots.
pub struct NodeCollector<O: ObjectMemory> {
    nodes: RwLock<SlotMap<Handle, NodeHeader>>,
    roots: RwLock<Vec<Handle>>,
    cycle_buffer: RwLock<Vec<Vec<Handle>>>,
    object_memory: RwLock<O>,
}

unsafe impl<O: ObjectMemory> Send for NodeCollector<O> {}
unsafe impl<O: ObjectMemory> Sync for NodeCollector<O> {}

impl<O: ObjectMemory> NodeCollector<O> {
    pub fn new(object_memory: O) -> Self {
        Self {
            nodes: RwLock::new(SlotMap::with_key()),
            roots: RwLock::new(Vec::with_capacity(8)),
            cycle_buffer: RwLock::new(Vec::with_capacity(8)),
            object_memory: RwLock::new(object_memory),
        }
    }

    pub fn make(&mut self) -> Handle {
        let handle = {
            self.nodes.write().insert(NodeHeader::new())
        };
        self.object_memory.write().created(handle);
        handle
    }

    pub fn with_memory<F, R>(&self, f: F) -> R where F: Fn(&O) -> R {
        let m = self.object_memory.read();
        f(&m)
    }

    pub fn with_memory_mut<F, R>(&self, f: F) -> R where F: Fn(&mut O) -> R {
        let mut m = self.object_memory.write();

        f(&mut m)
    }

    pub fn children(&self, h: Handle) -> Vec<Handle> {
        let nodes = self.nodes.read();
        let n = nodes.index(h);
        n.children.to_vec()
    }

    pub fn append_child(&mut self, parent: Handle, child: Handle) {
        {
            let mut nodes = self.nodes.write();
            let parent = nodes.index_mut(parent);
            parent.children.push(child);
        }
        self.increment(child);
    }

    pub fn remove_child(&mut self, parent: Handle, child: Handle) {
        {
            let mut nodes = self.nodes.write();
            let parent = nodes.index_mut(parent);
            parent.children.swap_remove(parent.children.iter().position(|x| *x == child).unwrap());
        }
        self.decrement(child);
    }

    /// Phase 1 (pre epoch boundary)
    pub fn increment(&mut self, sn: Handle) {
        {
            let mut nodes = self.nodes.write();
            let s = nodes.index_mut(sn);

            s.rc += 1;
        }
        self.scan_black(sn);
    }

    pub fn decrement(&mut self, n: Handle) {
        let res = {
            let mut nodes = self.nodes.write();
            let mut s = nodes.index_mut(n);
            s.rc -= 1;
            s.rc == 0
        }; if res {
            // See comments in Arc about ordering and fence here.
            self.release(n);
        } else {
            self.possible_root(n);
        }
    }

    fn scan_black(&self, sn: Handle) {
        let mut nodes = self.nodes.write();
        // Recursive function that will just share the same lock guard  as it goes deeper.
        self.scan_black_inner(&mut nodes, sn);
    }


    fn scan_black_inner(&self, nodes: &mut RwLockWriteGuard<SlotMap<Handle, NodeHeader>>, sn : Handle) {
        let s = nodes.index_mut(sn);
        if s.color != Color::Black {
            s.color = Color::Black;
            let children = nodes.get(sn).unwrap().children.clone();
            for t in children {
                self.scan_black_inner(nodes,t);
            }
        }
    }

    fn possible_root(&mut self, sn: Handle) {
        self.scan_black(sn);
        let mut nodes = self.nodes.write();
        let s = nodes.index_mut(sn);
        s.color = Color::Purple;
        if !s.buffered {
            s.buffered = true;
            self.roots.write().push(sn);
        }
    }

    pub fn release(&mut self, sn: Handle) {
        let children = {
            self.nodes.read().get(sn).unwrap().children.clone()
        };
        for t in children {
            assert!(self.nodes.read().contains_key(t));
            self.decrement(t);
        }
        let is_buffered = {
            let mut nodes = self.nodes.write();
            let s = nodes.index_mut(sn);
            {
                s.color = Color::Black;
            }
            s.buffered
        };
        if !is_buffered {
            self.free(sn);
        }
    }

    // Phase 2 functions, spawned by epoch boundary
    /// Invoked once per epoch after increment/decrements have been collected.
    pub fn process_cycles(&mut self) {
        self.free_cycles();
        self.collect_cycles();
        self.sigma_preparation();
    }

    fn collect_roots(&mut self) {
        let roots = {
            let mut roots = self.roots.write();
            let roots_copy = roots.to_owned();
            (*roots).clear();
            roots_copy
        };
        for sn in roots {
            let is_white = {
                let nodes = self.nodes.read();
                let s = nodes.index(sn);
                s.color == Color::White
            };
            if is_white {
                let mut current_cycle = vec![];
                // cheat code
                self.collect_white(sn, &mut current_cycle);
                self.cycle_buffer.write().push(current_cycle.to_owned());
            } else {
                let mut nodes = self.nodes.write();
                let s = nodes.index_mut(sn);
                s.buffered = false;
            }
        }
    }

    fn collect_white(&mut self, sn: Handle, current_cycle: &mut Vec<Handle>) {

        let made_orange = {
            let mut nodes = self.nodes.write();
            let s = nodes.index_mut(sn);
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
            let children: Vec<Handle> = {
                self.nodes.read().get(sn).unwrap().children.to_vec()
            };
            for handle in children {
                self.collect_white(handle, current_cycle);
            }
        }
    }

    fn sigma_preparation(&mut self) {
        let cb = {
            self.cycle_buffer.read().to_owned()
        };
        for c in &cb {
            for n in c {
                let mut nodes = self.nodes.write();
                let n = nodes.index_mut(*n);
                    n.color = Color::Red;

                // TODO: correct ordering?
                n.crc = n.rc;
            }
            for n in c {
                let children = {
                    self.nodes.read().get(*n).unwrap().children.clone()
                };
                for m in children {
                    let mut nodes = self
                        .nodes
                        .write();
                    let m = nodes.index_mut(m);
                    // TODO: check ordering
                    if m.color == Color::Red && m.crc > 0 {
                        m.crc -= 1;
                    }
                }
            }
            for n in c {
                let mut nodes = self.nodes.write();
                let n = nodes.index_mut(*n);
                {
                    n.color = Color::Orange;
                }
            }
        }
    }

    fn free_cycles(&mut self) {
        let (mut to_free, mut to_refurbish) = (vec![], vec![]);

        // TODO: We're holding a write lock for the whole duration here. This may or may not be necessary
        {
            let mut cb = self.cycle_buffer.write();
            for c in cb.iter().rev() {
                if self.delta_test(c) && self.sigma_test(c) {
                    to_free.extend(c);
                } else {
                    to_refurbish.extend(c);
                }
            }
            *cb = vec![];
        }
        self.free_cycle(&to_free);
        self.refurbish(&to_refurbish);
    }

    fn delta_test(&self, c: &Vec<Handle>) -> bool {
        for n in c {
            let nodes = self.nodes.read();
            let n = nodes.index(*n);
            if n.color != Color::Orange {
                return false;
            }
        }
        true
    }

    fn sigma_test(&self, c: &Vec<Handle>) -> bool {
        let mut extern_rc: u32 = 0;
        for n in c {
            let nodes = self.nodes.read();
            let n = nodes.index(*n);

            extern_rc += n.crc;
        }
        extern_rc == 0
    }

    fn refurbish(&mut self, c: &Vec<Handle>) {
        let mut first = false;
        for nn in c {
            let mut nodes = self.nodes.write();
            {
                let n = nodes.index_mut(*nn);
                if (first && n.color == Color::Orange) || n.color == Color::Purple {
                    n.color = Color::Purple;
                    self.roots.write().push(*nn);
                } else {
                    n.color = Color::Black;
                    n.buffered = false;
                }
            }
            first = false;
        }
    }

    fn free_cycle(&mut self, c: &Vec<Handle>) {
        for n in c {
            let mut nodes = self.nodes.write();
            let n = nodes.index_mut(*n);
            n.color = Color::Red;
        }
        for n in c {
            let children = {
                self.nodes.read().get(*n).unwrap().children.clone()
            };
            for m in children {
                self.cyclic_decrement(m);
            }
        }
        for n in c {
            self.free(*n);
        }
    }

    fn cyclic_decrement(&mut self, mn: Handle) {
        let do_decrement = {
            let mut nodes = self.nodes.write();
            let m = nodes.index_mut(mn);
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
            let nodes = self.nodes.read();
            let (to_gray, to_rm_root): (Vec<Handle>, Vec<Handle>) =
                roots.iter().partition(|sn| {
                    let s = nodes.index(**sn);
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
        let to_free = to_rm_root.iter().filter(|sn| {
            let mut nodes = self.nodes.write();
            let s = nodes.index_mut(**sn);
            s.buffered = false ;
            s.rc == 0
        });
        let to_free = to_free.copied();
        let to_free : Vec<Handle> = to_free.collect();
        for sn in to_free {
            self.free(sn);
        }
    }

    fn mark_gray(&mut self, sn: Handle) {
        let mut to_gray = vec![];
        {
            let mut nodes = self.nodes.write();
            let s = nodes.index_mut(sn);

            if s.color != Color::Gray {
                s.color = Color::Gray;
                s.crc = s.rc;
                let children = {
                    nodes.get(sn).unwrap().children.clone()
                };
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
        let roots = self.roots.read().to_owned();
        for s in roots {
            self.scan(s);
        }
    }

    fn scan(&mut self, sn: Handle) {
        let (mut scan_black, mut scan) = (vec![], vec![]);
        {
            let mut nodes = self.nodes.write();
            let s = nodes.index_mut(sn);
            if s.color == Color::Gray && s.crc == 0 {
                s.color = Color::White;
                let children = self.nodes.read().get(sn).unwrap().children.clone();
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
        self.object_memory.write().finalize(handle);
        self.nodes.write().remove(handle).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use std::collections::{HashMap};

    

    use crate::{Handle, NodeCollector, ObjectMemory};

    struct TestMem {
        collected: Vec<Handle>,
    }

    impl ObjectMemory for TestMem {
        fn finalize(&mut self, handle: Handle) {
            self.collected.push(handle);
        }
        fn created(&mut self, _handle: Handle) {}
    }

    #[test]
    fn test_basic() {
        let mem = TestMem { collected: vec![] };
        let mut collector = NodeCollector::new(mem);
        let a = collector.make();
        let b = collector.make();
        collector.append_child(a, b);
        collector.increment(a); // A is a root, so starts with an RC of 1
        collector.process_cycles();

        // Should trigger finalization of 'b' as there are no longer any live references.
        collector.remove_child(a, b);
        collector.process_cycles();

        assert!(collector.with_memory(|m| m.collected.contains(&b)));
        assert!(!collector.with_memory(|m| m.collected.contains(&a)));
    }

    #[test]
    fn test_simple_cycle() {
        let mem = TestMem { collected: vec![] };
        let mut collector = NodeCollector::new(mem);
        let a = collector.make();
        let b = collector.make();
        collector.append_child(a, b);
        collector.increment(a); // A is a root, so starts with an RC of 1

        // Form a cycle.
        collector.append_child(b, a);

        // Should trigger finalization of 'b' as there are no longer any live references.
        collector.remove_child(a, b);
        collector.process_cycles();

        assert!(collector.with_memory(|m| m.collected.contains(&b)));
        assert!(!collector.with_memory(|m| m.collected.contains(&a)));
    }

    pub struct MockDB {
        pub index : HashMap<Handle, usize>,
        pub objects: Vec<Option<Handle>>,
    }

    impl ObjectMemory for MockDB {
        fn finalize(&mut self, handle: Handle) {
            if let Some(idx) = self.index.get(&handle) {
                self.objects[*idx] = None;
            }
        }
        fn created(&mut self, handle: Handle) {
            let idx = self.objects.len();
            self.objects.push(Some(handle));
            self.index.insert(handle, idx);
        }

    }
}
