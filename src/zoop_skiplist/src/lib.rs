//! A lazy concurrent skiplist based on the design of Heller, Herlihy,
//! Luchangco, and Moir.
//!
//! This implementation uses fine-grained locking with optimistic lock-free
//! traversal, validation after acquiring locks, and logical deletion (marking)
//! before physical removal.
//!
//! Removed nodes are freed using epoch-based reclamation (WIP).

mod ebr;
mod rng;
pub mod list;
mod spinlock;

pub use crate::list::SkipList;
