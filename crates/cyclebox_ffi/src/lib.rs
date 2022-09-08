/// An attempt at "extern C" implementation meant for exposing the GC via FFI to
/// non-Rust programs or to e.g. a WebAssembly runtime.
/// TODO(ryan): finalizer callback support.

use cyclebox::{Handle, NodeCollector};
use slotmap::{Key, KeyData};
use std::borrow::BorrowMut;
use std::sync::{Mutex, Once};


type Finalizer = fn(Handle) -> ();
static mut COLLECTOR_GLOBAL: Option<Mutex<NodeCollector<Finalizer>>> = None;
static INIT: Once = Once::new();

#[no_mangle]
extern "C" fn gc_cycle_collect() {
    unsafe {
        let mut cc = COLLECTOR_GLOBAL.as_ref().unwrap().lock().unwrap();
        cc.process_cycles();
    }
}

#[no_mangle]
extern "C" fn gc_init_storage() {
    INIT.call_once(|| unsafe {
        *COLLECTOR_GLOBAL.borrow_mut() = Some(Mutex::new(NodeCollector::new(None)));
    });
}


#[no_mangle]
extern "C" fn gc_make_node() -> u64 {
    unsafe {
        let mut cc = COLLECTOR_GLOBAL.as_ref().unwrap().lock().unwrap();
        let handle = cc.make();
        handle.data().as_ffi()
    }
}

#[no_mangle]
extern "C" fn gc_append_child(parent: u64, child: u64) {
    let (parent, child) = (KeyData::from_ffi(parent), KeyData::from_ffi(child));
    let (parent, child) = (Handle::from(parent), Handle::from(child));
    unsafe {
        let mut cc = COLLECTOR_GLOBAL.as_ref().unwrap().lock().unwrap();
        cc.append_child(parent, child);
    }
}

#[no_mangle]
extern "C" fn gc_remove_child(parent: u64, child: u64) {
    let (parent, child) = (KeyData::from_ffi(parent), KeyData::from_ffi(child));
    let (parent, child) = (Handle::from(parent), Handle::from(child));
    unsafe {
        let mut cc = COLLECTOR_GLOBAL.as_ref().unwrap().lock().unwrap();
        cc.remove_child(parent, child);
    }
}

#[no_mangle]
extern "C" fn gc_release(node: u64) {
    let node = KeyData::from_ffi(node);
    let node = Handle::from(node);
    unsafe {
        let mut cc = COLLECTOR_GLOBAL.as_ref().unwrap().lock().unwrap();
        cc.release(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::{gc_append_child, gc_cycle_collect, gc_init_storage, gc_make_node, gc_remove_child};

    #[test]
    fn test_basic() {
        gc_init_storage();

        // Make a simple parent child relationship and test that the child is collected when the
        // reference from the parent to it is removed.
        let parent = gc_make_node();
        let child = gc_make_node();
        gc_append_child(parent, child);

        gc_cycle_collect();

        gc_remove_child(parent, child);
        gc_cycle_collect();
    }
}
