extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use slotmap::{new_key_type, SlotMap};
use std::cell::{Ref, RefCell, RefMut};
use std::hash::Hash;

#[derive(Clone, Eq, PartialEq)]
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
            children: Vec::new()
        }
    }
}

pub trait ObjectMemory {
    fn finalize(&mut self, handle: Handle);
    fn created(&mut self, handle: Handle);
}

// Persistent garbage collected storage for slots.
pub struct NodeCollector<O: ObjectMemory> {
    nodes: SlotMap<Handle, NodeHeader>,
    roots: Vec<Handle>,
    cycle_buffer: Vec<Vec<Handle>>,
    object_memory: RefCell<Box<O>>,
}

impl<O: ObjectMemory> NodeCollector<O> {
    pub fn new(object_memory: O) -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: vec![],
            cycle_buffer: vec![],
            object_memory: RefCell::new(Box::new(object_memory)),
        }
    }

    pub fn make(&mut self) -> Handle {
        let handle = self.nodes.insert(NodeHeader::new());
        self.object_memory.borrow_mut().created(handle);
        handle
    }

    pub fn memory(&self) -> Ref<Box<O>> {
        self.object_memory.borrow()
    }

    pub fn memory_mut(&self) -> RefMut<Box<O>> {
        self.object_memory.borrow_mut()
    }

    pub fn children(&self, h: Handle) -> Vec<Handle> {
        let n = self.nodes.index(h);
        n.children.to_vec()
    }

    pub fn append_child(&mut self, parent: Handle, child: Handle) {
        let parent = self.nodes.index_mut(parent);
        parent.children.push(child);
        self.increment(child);
    }

    pub fn remove_child(&mut self, parent: Handle, child: Handle) {
        assert!(self.nodes.contains_key(parent));
        let parent = self.nodes.index_mut(parent);
        assert!(parent.children.contains(&child));
        parent.children.swap_remove(parent.children.iter().position(|x| *x == child).unwrap());
        self.decrement(child);
    }

    fn free(&mut self, handle: Handle) {
        assert!(self.nodes.contains_key(handle));
        self.object_memory.borrow_mut().finalize(handle);
        self.nodes.remove(handle).unwrap();
    }

    fn possible_root(&mut self, sn: Handle) {
        assert!(self.nodes.contains_key(sn));
        self.scan_black(sn);
        let s = self.nodes.index_mut(sn);
        s.color = Color::Purple;
        if !s.buffered {
            s.buffered = true;
            self.roots.push(sn);
        }
    }

    fn collect_roots(&mut self) {
        let roots = self.roots.to_owned();
        self.roots.clear();
        for sn in roots {
            let mut s = self.nodes.index_mut(sn);
            if s.color == Color::White {
                let mut current_cycle = vec![];
                self.collect_white(sn, &mut current_cycle);
                self.cycle_buffer.push(current_cycle.to_owned());
            } else {
                s.buffered = false;
            }
        }
    }

    fn collect_white(&mut self, sn: Handle, current_cycle: &mut Vec<Handle>) {
        let s = self.nodes.index_mut(sn);
        if s.color == Color::White {
            s.color = Color::Orange;
            s.buffered = true;
            current_cycle.push(sn);
            let children: Vec<Handle> = self.nodes.get(sn).unwrap().children.to_vec();
            for handle in children {
                assert!(self.nodes.contains_key(handle));
                self.collect_white(handle, current_cycle);
            }
        }
    }

    fn sigma_preparation(&mut self) {
        for c in &self.cycle_buffer {
            for n in c {
                let n = self.nodes.index_mut(*n);
                n.color = Color::Red;
                n.crc = n.rc;
            }
            for n in c {
                let children = self.nodes.get(*n).unwrap().children.clone();
                for m in children {
                    let m = self
                        .nodes
                        .index_mut(m);
                    if m.color == Color::Red && m.crc > 0 {
                        m.crc -= 1;
                    }
                }
            }
            for n in c {
                let n = self.nodes.index_mut(*n);
                n.color = Color::Orange;
            }
        }
    }

    fn free_cycles(&mut self) {
        let (mut to_free, mut to_refurbish) = (vec![], vec![]);
        for c in self.cycle_buffer.iter().rev() {
            if self.delta_test(c) && self.sigma_test(c) {
                to_free.extend(c);
            } else {
                to_refurbish.extend(c);
            }
        }
        self.free_cycle(&to_free);
        self.refurbish(&to_refurbish);

        self.cycle_buffer = vec![];
    }

    fn delta_test(&self, c: &Vec<Handle>) -> bool {
        for n in c {
            assert!(self.nodes.contains_key(*n));
            let n = self.nodes.index(*n);
            if n.color != Color::Orange {
                return false;
            }
        }
        true
    }

    fn sigma_test(&self, c: &Vec<Handle>) -> bool {
        let mut extern_rc: u32 = 0;
        for n in c {
            assert!(self.nodes.contains_key(*n));
            let n = self.nodes.index(*n);
            extern_rc += n.crc;
        }
        extern_rc == 0
    }

    fn refurbish(&mut self, c: &Vec<Handle>) {
        let mut first = false;
        for nn in c {
            assert!(self.nodes.contains_key(*nn));

            let n = self.nodes.index_mut(*nn);
            if (first && n.color == Color::Orange) || n.color == Color::Purple {
                n.color = Color::Purple;
                self.roots.push(*nn);
            } else {
                n.color = Color::Black;
                n.buffered = false;
            }
            first = false;
        }
    }

    fn free_cycle(&mut self, c: &Vec<Handle>) {
        for n in c {
            let n = self.nodes.index_mut(*n);
            n.color = Color::Red;
        }
        for n in c {
            let children = self.nodes.get(*n).unwrap().children.clone();
            for m in children {
                self.cyclic_decrement(m);
            }
        }
        for n in c {
            self.free(*n);
        }
    }

    fn cyclic_decrement(&mut self, mn: Handle) {
        let m = self.nodes.index_mut(mn);
        if m.color != Color::Red {
            if m.color != Color::Orange {
                m.rc -= 1;
                m.crc -= 1;
            } else {
                self.decrement(mn);
            }
        }
    }

    fn mark_roots(&mut self) {
        let (to_gray, to_rm_root) : (Vec<Handle>, Vec<Handle> )=
            self.roots.iter().partition(|sn| {
                assert!(self.nodes.contains_key(**sn));
                let s = self.nodes.index(**sn);
                s.color == Color::Purple && s.rc > 0
            });
        for s in to_gray.iter() {
            self.mark_gray(*s);
        }
        self.roots = to_gray;

        for sn in to_rm_root {
            let s = self.nodes.index_mut(sn);
            s.buffered = false;
            if s.rc == 0 {
                self.free(sn);
            }
        }
    }

    fn mark_gray(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);
        if s.color != Color::Gray {
            s.color = Color::Gray;
            s.crc = s.rc;
            let children = self.nodes.get(sn).unwrap().children.clone();
            for t in children {
                assert!(self.nodes.contains_key(t));
                self.mark_gray(t);
            }
        } else {
            s.crc = s.crc.saturating_sub(1);
        }
    }

    fn scan_roots(&mut self) {
        let roots = self.roots.to_owned();
        for s in roots {
            assert!(self.nodes.contains_key(s));
            self.scan(s);
        }
    }

    fn scan(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);

        if s.color == Color::Gray && s.crc == 0 {
            s.color = Color::White;
            let children = self.nodes.get(sn).unwrap().children.clone();
            for t in children {
                assert!(self.nodes.contains_key(t));
                self.scan(t);
            }
        } else {
            self.scan_black(sn);
        }
    }

    fn scan_black(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);
        if s.color != Color::Black {
            s.color = Color::Black;
            let children = &self.nodes.get(sn).unwrap().children.clone();
            for t in children {
                assert!(self.nodes.contains_key(*t));
                self.scan_black(*t);
            }
        }
    }

    pub fn increment(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);
        s.rc += 1;
        self.scan_black(sn);
    }

    pub fn decrement(&mut self, n: Handle) {
        let s = self.nodes.index_mut(n);
        // s.rc = s.rc.saturating_sub(1);
        s.rc -= 1;
        if s.rc == 0 {
            self.release(n);
        } else {
            self.possible_root(n);
        }
    }

    pub fn release(&mut self, sn: Handle) {
        let children = self.nodes.get(sn).unwrap().children.clone();
        for t in children {
            assert!(self.nodes.contains_key(t));
            self.decrement(t);
        }
        let s = self.nodes.index_mut(sn);
        s.color = Color::Black;
        if !s.buffered {
            self.free(sn);
        }
    }

    fn collect_cycles(&mut self) {
        self.mark_roots();
        self.scan_roots();
        self.collect_roots();
    }

    pub fn process_cycles(&mut self) {
        self.free_cycles();
        self.collect_cycles();
        self.sigma_preparation();
    }
}

#[cfg(test)]
mod tests {
    use crate::{Handle, NodeCollector, ObjectMemory};
    use alloc::vec;
    use rand::Rng;
    
    use std::collections::HashSet;

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

        assert!(collector.memory().collected.contains(&b));
        assert!(!collector.memory().collected.contains(&a));
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

        assert!(collector.memory().collected.contains(&b));
        assert!(!collector.memory().collected.contains(&a));
    }

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

    fn produce_mock_object_graph(num_objects: usize, max_child_width: usize, num_mutate_cycles: usize) {
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
    #[test]
    fn test_mock_obj_graph() {
        println!("Entering test");
        produce_mock_object_graph(1000, 10, 40);
    }
}
