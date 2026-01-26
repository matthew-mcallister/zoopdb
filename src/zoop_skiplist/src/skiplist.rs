use std::alloc::Layout;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering as AtomicOrdering};

use rand::Rng;

use crate::rng::rng;
use crate::spinlock::{SpinLock, SpinLockGuard};

const MAX_HEIGHT: usize = 32;

fn random_height() -> usize {
    // This produces an exponential distribution
    let val = (1 << 31) | rng().random::<u32>();
    val.trailing_zeros() as usize + 1
}

/// Zero-sized trailing slice. The reason we don't use an unsized trailing
/// slice is because it would require fat pointers to track the length.
#[derive(Default)]
struct Trailing<T> {
    _data: [T; 0],
}

impl<T> Trailing<T> {
    unsafe fn get(&self, index: usize) -> &T {
        let ptr = self as *const Self as *const T;
        unsafe { &*ptr.add(index) }
    }

    unsafe fn get_mut(&mut self, index: usize) -> &mut T {
        let ptr = self as *mut Self as *mut T;
        unsafe { &mut *ptr.add(index) }
    }
}

/// All node data except for the key/value.
// repr(C) since the successor list must be at the end
#[repr(C)]
struct NodeMeta<T> {
    lock: SpinLock,
    /// Marked as removed from list; will be collected after epoch ends. No
    /// live node ever points to a marked node.
    marked: AtomicBool,
    height: u8,
    pointers: Trailing<AtomicPtr<Node<T>>>,
}

impl<T> std::ops::Index<usize> for NodeMeta<T> {
    type Output = AtomicPtr<Node<T>>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.height as usize);
        unsafe { self.pointers.get(index) }
    }
}

impl<T> std::ops::IndexMut<usize> for NodeMeta<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.height as usize);
        unsafe { self.pointers.get_mut(index) }
    }
}

impl<T> NodeMeta<T> {
    fn is_marked(&self) -> bool {
        self.marked.load(AtomicOrdering::Relaxed)
    }

    fn mark(&self) {
        debug_assert!(self.lock.is_locked());
        self.marked.store(true, AtomicOrdering::Relaxed)
    }

    fn link(&self, level: usize, target: *mut Node<T>) {
        debug_assert!(self.lock.is_locked());
        self[level].store(target, AtomicOrdering::Relaxed);
    }
}

fn as_ptr<T>(reference: Option<&T>) -> *const T {
    match reference {
        Some(r) => r as *const T,
        None => ptr::null(),
    }
}

/// Locks all predecessors, handling reentrancy and aborting if any of the
/// nodes were invalidated by concurrent mutation.
fn lock_all<'pr, T>(
    preds: &'pr [&NodeMeta<T>],
    succs: &[Option<&Node<T>>],
) -> Option<[Option<SpinLockGuard<'pr>>; MAX_HEIGHT]> {
    debug_assert_eq!(preds.len(), succs.len());
    let height = preds.len();

    let mut guards: [Option<SpinLockGuard>; MAX_HEIGHT] = Default::default();
    for level in (0..height).rev() {
        let pred = preds[level];

        // Check for reentrancy
        let already_locked = (level + 1..height)
            .any(|l| preds[l] as *const _ == pred as *const _);
        if !already_locked {
            let guard = pred.lock.lock();
            guards[level] = Some(guard);
        }

        let s = pred[level].load(AtomicOrdering::Relaxed);
        if pred.is_marked() || s as *const _ != as_ptr(succs[level]) {
            // Predecessor was mutated, need to refresh lists
            return None;
        }
    }

    Some(guards)
}

// Fixed-size pointer list
#[repr(C)]
struct Head<T> {
    lock: SpinLock,
    marked: AtomicBool,
    height: u8,
    pointers: [AtomicPtr<Node<T>>; MAX_HEIGHT],
}

impl<T> Default for Head<T> {
    fn default() -> Self {
        Self {
            lock: Default::default(),
            marked: AtomicBool::new(false),
            height: MAX_HEIGHT as u8,
            pointers: Default::default(),
        }
    }
}

impl<T> AsRef<NodeMeta<T>> for Head<T> {
    fn as_ref(&self) -> &NodeMeta<T> {
        unsafe { &*(self as *const Self as *const NodeMeta<T>) }
    }
}

/// A node in the skiplist, with element data.
#[repr(C)]
struct Node<T> {
    element: T,
    inner: NodeMeta<T>,
}

impl<T> Node<T> {
    const _ASSERT_LAYOUT: () = {
        let layout1 = Self::layout(0);
        let layout2 = Layout::new::<Self>();
        assert!(layout1.size() == layout2.size());
        assert!(layout1.align() == layout2.align());
    };

    const fn layout(height: usize) -> Layout {
        let elem = Layout::new::<T>();

        // Compute the layout as if NodeMeta contained a field of type [AtomicPtr<T>; height]
        let inner = Layout::new::<NodeMeta<T>>();
        let Ok(array) = Layout::array::<AtomicPtr<Node<T>>>(height) else { panic!() };
        let Ok((inner, _)) = inner.extend(array) else { panic!() };

        let Ok((layout, _)) = elem.extend(inner) else { panic!() };
        layout.pad_to_align()
    }

    fn alloc<'a>(node: Self, pointers: &[*mut Node<T>]) -> *mut Self {
        let height = node.inner.height as usize;
        assert!(height >= 1 && height <= MAX_HEIGHT);
        assert_eq!(pointers.len(), height);

        let layout = Self::layout(height);

        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr.write(node);

            let n = &mut *ptr;
            for level in 0..height {
                *n.inner[level].get_mut() = pointers[level];
            }

            ptr
        }
    }

    /// Deallocate a node. If `ptr` is null, does nothing.
    #[allow(dead_code)]
    unsafe fn dealloc(ptr: *mut Self) {
        if ptr.is_null() {
            return;
        }

        unsafe {
            let height = (*ptr).inner.height as usize;
            let layout = Self::layout(height);
            std::ptr::drop_in_place(ptr);
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

/// A reference to an entry in the skiplist.
pub struct Ref<'a, T: Ord> {
    node: &'a Node<T>,
    // TODO: Epoch guard
}

impl<'a, T: Ord> std::ops::Deref for Ref<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.node.element
    }
}

/// A concurrent skiplist based on epoch-based reclamation (EBR).
///
/// Read/write operations are not guaranteed to be serializable without
/// external synchronization. If thread 1 writes AB, thread 2 may read ABAB,
/// for example.
///
/// # Example
///
/// ```ignore
/// let skiplist = SkipList::new();
/// skiplist.insert(1);
/// skiplist.insert(2);
///
/// assert!(skiplist.contains(&1));
/// assert!(skiplist.contains(&2));
/// assert!(!skiplist.contains(&3));
///
/// skiplist.remove(&1);
/// assert!(!skiplist.contains(&1));
/// ```
// TODO: Comparator
pub struct SkipList<T: Ord> {
    head: Head<T>,
    len: AtomicUsize,
}

unsafe impl<T: Ord + Send> Send for SkipList<T> {}
unsafe impl<T: Ord + Send + Sync> Sync for SkipList<T> {}

impl<T: Ord> SkipList<T> {
    /// Creates a new empty skiplist.
    pub fn new() -> Self {
        Self {
            head: Default::default(),
            len: AtomicUsize::new(0),
        }
    }

    /// Returns the approximate number of elements in the skiplist. This is
    /// exact if there are no concurrent writers.
    pub fn len(&self) -> usize {
        self.len.load(AtomicOrdering::Relaxed)
    }

    /// Returns true if the skiplist is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Searches for a element and returns a reference to its entry if found.
    ///
    /// It is possible for the returned element to be removed from the list
    /// after, or even before, this method returns. It is up to the caller to
    /// perform synchronization if desired to prevent this effect.
    pub fn get<'a, Q>(&'a self, element: &Q) -> Option<Ref<'a, T>>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // TODO: Pin epoch here when EBR is implemented
        // let _guard = self.epoch.pin();

        let node = self.find_node(element)?;
        Some(Ref { node })
    }

    /// Returns true if the skiplist contains the given element.
    pub fn contains<Q>(&self, element: &T) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(element.borrow()).is_some()
    }

    /// Inserts a new element and returns a reference to it.
    ///
    /// If the list already contains an existing element that compares equal to
    /// the new element, the old element will be replaced.
    ///
    /// Note that concurrent readers may not uniformly observe the newly
    /// inserted element unless external synchronization is imposed.
    pub fn insert(&self, element: T) -> Ref<'_, T> where T: std::fmt::Debug {
        // TODO: Pin epoch here when EBR is implemented
        // let _guard = self.epoch.pin();

        let mut preds: [&NodeMeta<T>; MAX_HEIGHT] = [self.head.as_ref(); MAX_HEIGHT];
        let mut succs: [Option<&Node<T>>; MAX_HEIGHT] = [None; MAX_HEIGHT];

        let (height, existing, _guards) = loop {
            // Find insertion point
            let existing = self.find(&element, &mut preds, &mut succs);

            // New height must be >= old height or else pointers to old node will persist
            let height = if let Some(e) = existing { e.inner.height as usize } else { random_height() };

            // Attempt to lock all predecessors
            let Some(guards) = lock_all(&preds[..height], &succs[..height]) else { continue };
            break (height, existing, guards);
        };
        let _guard = if let Some(existing) = existing {
            let guard = existing.inner.lock.lock();
            debug_assert!(!existing.inner.is_marked(), "node not fully unlinked");
            Some(guard)
        } else {
            None
        };

        // Build new node
        let node = Node {
            element,
            inner: NodeMeta {
                height: height as _,
                marked: AtomicBool::new(false),
                lock: Default::default(),
                pointers: Default::default(),
            },
        };
        let mut pointers: Vec<*mut Node<T>> = (0..height)
            .map(|_| Default::default())
            .collect();
        for level in (0..height).rev() {
            pointers[level] = as_ptr(succs[level]) as *mut _;
        }
        if let Some(existing) = existing {
            for level in 0..std::cmp::min(height, existing.inner.height as usize) {
                pointers[level] = existing.inner[level].load(AtomicOrdering::Relaxed);
            }
        }

        // Allocate and link new node
        let node_ptr = Node::alloc(node, &pointers);
        let node = unsafe { &*node_ptr };
        for level in (0..height).rev() {
            preds[level].link(level, node_ptr);
        }

        if let Some(existing) = existing {
            existing.inner.mark();
        } else {
            self.len.fetch_add(1, AtomicOrdering::Relaxed);
        }

        Ref { node }
    }

    /// Removes an entry by key, returning a reference to the existing entry
    /// if found.
    pub fn remove<'a, Q>(&'a self, key: &Q) -> Option<Ref<'a, T>>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // TODO: Pin epoch here when EBR is implemented
        // let _guard = self.epoch.pin();

        let mut preds: [&NodeMeta<T>; MAX_HEIGHT] = [self.head.as_ref(); MAX_HEIGHT];
        let mut succs: [Option<&Node<T>>; MAX_HEIGHT] = [None; MAX_HEIGHT];

        let (node, _guards) = loop {
            let node = self.find(&key, &mut preds, &mut succs)?;
            if node.inner.is_marked() {
                return None;
            }

            let height = node.inner.height as usize;
            let Some(guards) = lock_all(&preds[..height], &succs[..height]) else { continue };
            break (node, guards);
        };
        let _guard = node.inner.lock.lock();

        debug_assert!(!node.inner.is_marked(), "node not fully unlinked");

        // Update pointers of predecessors
        for level in (0..node.inner.height as usize).rev() {
            let p = node.inner[level].load(AtomicOrdering::Relaxed);
            preds[level].link(level, p);
        }

        node.inner.mark();
        self.len.fetch_sub(1, AtomicOrdering::Relaxed);

        Some(Ref { node })
    }

    /// Finds the position for a key, filling in predecessors and successors.
    ///
    /// - `preds[0]` points to the largest element less than the key
    /// - `succs[0]` points to the smallest element greater than or equal
    ///   to the key
    ///
    /// If a node already exists with a matching key, returns a reference to
    /// that node.
    fn find<'a, Q>(
        &'a self,
        key: &Q,
        preds: &mut [&'a NodeMeta<T>; MAX_HEIGHT],
        succs: &mut [Option<&'a Node<T>>; MAX_HEIGHT],
    ) -> Option<&'a Node<T>>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut existing = None;

        let mut pred: &NodeMeta<T> = self.head.as_ref();
        for level in (0..MAX_HEIGHT).rev() {
            let mut curr_opt = unsafe {
                pred[level].load(AtomicOrdering::Relaxed).as_ref()
            };

            // Traverse forward while curr < key
            while let Some(curr) = curr_opt {
                match key.cmp(curr.element.borrow()) {
                    Ordering::Greater => {
                        pred = &curr.inner;
                        curr_opt = unsafe { curr.inner[level].load(AtomicOrdering::Relaxed).as_ref() };
                    }
                    Ordering::Equal => {
                        existing = Some(curr);
                        break;
                    }
                    Ordering::Less => {
                        break;
                    }
                }
            }

            preds[level] = pred;
            succs[level] = curr_opt;
        }

        existing
    }

    /// Finds the node that matches the given key, if it exists. Unlike
    /// `find()`, does not construct predecessor/successor lists and terminates
    /// early when a matching node is found.
    fn find_node<'a, Q>(&'a self, key: &Q) -> Option<&'a Node<T>>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut pred: &NodeMeta<T> = self.head.as_ref();
        for level in (0..MAX_HEIGHT).rev() {
            let mut curr_opt = unsafe {
                pred[level].load(AtomicOrdering::Relaxed).as_ref()
            };
            while let Some(curr) = curr_opt {
                match key.cmp(curr.element.borrow()) {
                    Ordering::Greater => {
                        pred = &curr.inner;
                        curr_opt = unsafe { curr.inner[level].load(AtomicOrdering::Relaxed).as_ref() };
                    }
                    Ordering::Equal => {
                        return Some(curr)
                    }
                    Ordering::Less => {
                        break;
                    }
                }
            }
        }
        None
    }

    fn iter_nodes(&self) -> impl Iterator<Item = &Node<T>> {
        struct Iter<'a, T: Ord> {
            current: Option<&'a Node<T>>,
        }

        impl<'a, T: Ord> Iterator for Iter<'a, T> {
            type Item = &'a Node<T>;

            fn next(&mut self) -> Option<Self::Item> {
                let curr = self.current?;
                let next_ptr = curr.inner[0].load(AtomicOrdering::Relaxed);
                let next = unsafe { next_ptr.as_ref() };
                self.current = next;
                Some(curr)
            }
        }

        let first_ptr = self.head.pointers[0].load(AtomicOrdering::Relaxed);
        let first = unsafe { first_ptr.as_ref() };
        Iter { current: first }
    }

    /// Returns an iterator over the nodes of the list. Note that this iterator
    /// may iterate over deleted values if the element it points to is removed.
    pub fn iter(&self) -> impl Iterator<Item = Ref<'_, T>> {
        self.iter_nodes().map(|node| Ref { node })
    }
}

impl<T: Ord> Default for SkipList<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn smoke_test() {
        let skiplist = SkipList::new();
        skiplist.insert(2);
        skiplist.insert(1);
        assert_eq!(skiplist.len(), 2);

        assert!(skiplist.contains(&1));
        assert_eq!(*skiplist.get(&1).unwrap(), 1);
        assert!(skiplist.contains(&2));
        assert!(!skiplist.contains(&3));

        skiplist.remove(&1);
        assert!(!skiplist.contains(&1));
        assert_eq!(skiplist.len(), 1);

        skiplist.insert(4);
        assert_eq!(skiplist.len(), 2);

        assert!(!skiplist.contains(&1));
        assert!(skiplist.contains(&2));
        assert!(!skiplist.contains(&3));
        assert!(skiplist.contains(&4));
    }

    #[test]
    fn reference_test() {
        // Do a bunch of random inserts/deletes on a SkipList and a BTree map
        // and make sure the map iterators compare equal at each step
        use std::collections::BTreeSet;
        let skiplist = SkipList::new();
        let mut btree = BTreeSet::new();
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let op: u8 = rng.random_range(0..3);
            let val: i32 = rng.random_range(0..100);
            match op {
                0 => {
                    skiplist.insert(val);
                    btree.insert(val);
                }
                1 => {
                    skiplist.remove(&val);
                    btree.remove(&val);
                }
                2 => {
                    let contains_skiplist = skiplist.contains(&val);
                    let contains_btree = btree.contains(&val);
                    assert_eq!(contains_skiplist, contains_btree);
                }
                _ => unreachable!(),
            }

            let skiplist_elems: Vec<i32> = skiplist.iter().map(|r| *r).collect();
            let btree_elems: Vec<i32> = btree.iter().cloned().collect();
            assert_eq!(skiplist_elems, btree_elems);
            assert_eq!(skiplist.len(), skiplist_elems.len());
        }
    }

    // TODO: Implement SkipListMap and test properly
    #[test]
    fn test_insert_overwrite() {
        #[derive(Debug)]
        struct Pair(i32, i32);

        impl PartialEq for Pair {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl Eq for Pair {}

        impl Ord for Pair {
            fn cmp(&self, other: &Self) -> Ordering {
                Ord::cmp(&self.0, &other.0)
            }
        }

        impl PartialOrd for Pair {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl std::borrow::Borrow<i32> for Pair {
            fn borrow(&self) -> &i32 {
                &self.0
            }
        }

        // Test that insert() overwrites the existing element when it compares
        // equal to the new element
        let skiplist = SkipList::new();
        skiplist.insert(Pair(1, 10));
        assert_eq!(skiplist.get(&1).unwrap().1, 10);
        skiplist.insert(Pair(1, 20));
        assert_eq!(skiplist.get(&1).unwrap().1, 20);
    }

    #[test]
    fn test_concurrent_insert() {
        // Insert numbers from 0 to 7999 split across 8 threads.
        let skiplist = Arc::new(SkipList::new());
        let n_threads = 8;
        let n_items = 1000;
        let barrier = Arc::new(Barrier::new(n_threads));

        let mut handles = Vec::new();
        for i in 0..n_threads {
            let skiplist = skiplist.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let start = i * n_items;
                for j in 0..n_items {
                    skiplist.insert(start + j);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(skiplist.len(), n_threads * n_items);
        for i in 0..n_threads * n_items {
            assert!(skiplist.contains(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_concurrent_insert_remove() {
        let skiplist = Arc::new(SkipList::new());
        let n_threads = 8;
        let n_ops = 2000;

        let barrier = Arc::new(Barrier::new(n_threads));
        let mut handles = Vec::new();
        for _ in 0..n_threads {
            let skiplist = skiplist.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let mut rng = rand::rng();
                for _ in 0..n_ops {
                    let key = rng.random_range(0..100);
                    if rng.random_bool(0.5) {
                        skiplist.insert(key);
                    } else {
                        skiplist.remove(&key);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let elems: Vec<u32> = skiplist.iter().map(|r| *r).collect();
        // Elems has the correct len
        assert_eq!(skiplist.len(), elems.len());
        // Elems is ordered
        for w in elems.windows(2) {
            assert!(w[0] <= w[1]);
        }
        // Each elem is < 100
        assert!(elems.last().map_or(true, |&n| n < 100));
    }
}