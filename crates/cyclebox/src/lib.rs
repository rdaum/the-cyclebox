#![no_std]

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};
use roaring::RoaringBitmap;
use slotmap::{new_key_type, KeyData, SlotMap};

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

// The header is kept separate from the object so that they can be kept in a contiguous fashion in
// the arena so that garbage collection can hopefully be kept in cache, or at least parts of it.
struct NodeHeader
{
    rc: u32,
    crc: u32,
    color: Color,
    buffered: bool,
    children: RoaringBitmap,
}

impl NodeHeader
{
    pub fn new() -> Self {
        Self {
            rc: 0,
            crc: 0,
            color: Color::White,
            buffered: false,
            children: RoaringBitmap::new(), }
    }
}

// Persistent garbage collected storage for slots.
pub struct NodeCollector<Finalizer: FnMut(Handle)> {
    nodes: SlotMap<Handle, NodeHeader>,
    roots: Vec<Handle>,
    current_cycle: Vec<Handle>,
    cycle_buffer: Vec<Vec<Handle>>,
    finalizer: Option<Finalizer>,
}

impl<Finalizer : FnMut(Handle)>  NodeCollector<Finalizer> {
    pub fn new(finalizer: Option<Finalizer>) -> Self {
        NodeCollector {
            nodes: SlotMap::with_key(),
            roots: vec![],
            current_cycle: vec![],
            cycle_buffer: vec![],
            finalizer,
        }
    }

    pub fn make(&mut self) -> Handle {
        
        self.nodes.insert(NodeHeader::new())
    }

    pub fn append_child(&mut self, parent: Handle, child: Handle) {
        let parent = self.nodes.index_mut(parent);
        parent.children.insert(child.0.as_ffi() as u32);
        self.increment(child);
    }

    pub fn remove_child(&mut self, parent: Handle, child: Handle) {
        let parent = self.nodes.index_mut(parent);
        parent.children.remove(child.0.as_ffi() as u32);
        self.decrement(child);
    }

    fn free(&mut self, handle: Handle) {
        match &mut self.finalizer {
            None => {}
            Some(f) => f(handle)
        }
        self.nodes.remove(handle).unwrap();
    }

    fn possible_root(&mut self, sn: Handle) {
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
        for sn in roots {
            let mut s = self.nodes.index_mut(sn);
            if s.color == Color::White {
                self.current_cycle = vec![];
                self.collect_white(sn);
                self.cycle_buffer.push(self.current_cycle.to_owned());
            } else {
                s.buffered = false;
            }
        }
    }

    fn collect_white(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);
        if s.color == Color::White {
            s.color = Color::Orange;
            s.buffered = true;
            self.current_cycle.push(sn);
            let children = self.nodes.get(sn).unwrap().children.clone();
            for t in children {
                self.collect_white(Handle::from(KeyData::from_ffi(t as u64)));
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
                        .index_mut(Handle::from(KeyData::from_ffi(m as u64)));
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
        while !self.cycle_buffer.is_empty() {
            let c = &self.cycle_buffer.pop().unwrap();
            if self.delta_test(c) && self.sigma_test(c) {
                self.free_cycle(c);
            } else {
                self.refurbish(c);
            }
        }
    }

    fn delta_test(&mut self, c: &Vec<Handle>) -> bool {
        for n in c {
            let n = self.nodes.index(*n);
            if n.color != Color::Orange {
                return false;
            }
        }
        true
    }

    fn sigma_test(&mut self, c: &Vec<Handle>) -> bool {
        let mut extern_rc: u32 = 0;
        for n in c {
            let n = self.nodes.index(*n);
            extern_rc += n.crc;
        }
        extern_rc == 0
    }

    fn refurbish(&mut self, c: &Vec<Handle>) {
        let mut first = false;
        for nn in c {
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
                self.cyclic_decrement(Handle::from(KeyData::from_ffi(m as u64)));
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
        let (to_gray, to_rm_root) = self.roots.iter().partition(|sn| {
            let s = self.nodes.index(**sn);
            s.color == Color::Purple && s.rc > 0
        });
        self.roots = to_gray;
        let roots = self.roots.to_owned();
        for s in roots {
            self.mark_gray(s);
        }
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
                self.mark_gray(Handle::from(KeyData::from_ffi(t as u64)));
            }
        } else {
            s.crc -= 1;
        }
    }

    fn scan_roots(&mut self) {
        let roots = self.roots.to_owned();
        for s in roots {
            self.scan(s);
        }
    }

    fn scan(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);

        if s.color == Color::Gray && s.crc == 0 {
            s.color = Color::White;
            let children = self.nodes.get(sn).unwrap().children.clone();
            for t in children {
                self.scan(Handle::from(KeyData::from_ffi(t as u64)));
            }
        }
    }

    fn scan_black(&mut self, sn: Handle) {
        let s = self.nodes.index_mut(sn);
        if s.color != Color::Black {
            s.color = Color::Black;
            let children = self.nodes.get(sn).unwrap().children.clone();
            for t in children {
                self.scan_black(Handle::from(KeyData::from_ffi(t as u64)));
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
        s.rc -= 1;
        if s.rc == 0 {
            self.free(n);
        } else {
            self.possible_root(n);
        }
    }

    pub fn release(&mut self, sn: Handle) {
        let children = self.nodes.get(sn).unwrap().children.clone();
        for t in children {
            self.decrement(Handle::from(KeyData::from_ffi(t as u64)));
        }
        let s = self.nodes.index_mut(sn);
        s.color = Color::Black;
        if !s.buffered {
            self.nodes.remove(sn);
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
    use alloc::vec;
    use crate::NodeCollector;

    #[test]
    fn test_basic() {
        let mut collected = vec![];
        let finalizer_closure = |h| {
            collected.push(h);
        };
        let mut collector = NodeCollector::new(Some(finalizer_closure));
        let a = collector.make();
        let b = collector.make();
        collector.append_child(a, b);
        collector.increment(a); // A is a root, so starts with an RC of 1
        collector.process_cycles();

        // Should trigger finalization of 'b' as there are no longer any live references.
        collector.remove_child(a, b);
        collector.process_cycles();

        assert!(collected.contains(&b));
        assert!(!collected.contains(&a));
    }

    #[test]
    fn test_simple_cycle() {
        let mut collected = vec![];
        let finalizer_closure = |h| {
            collected.push(h);
        };
        let mut collector = NodeCollector::new(Some(finalizer_closure));
        let a = collector.make();
        let b = collector.make();
        collector.append_child(a, b);
        collector.increment(a); // A is a root, so starts with an RC of 1

        // Form a cycle.
        collector.append_child(b, a);

        // Should trigger finalization of 'b' as there are no longer any live references.
        collector.remove_child(a, b);
        collector.process_cycles();

        assert!(collected.contains(&b));
        assert!(!collected.contains(&a));
    }
}
