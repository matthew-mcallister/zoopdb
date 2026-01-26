use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// This trait provides an equivalence relation between two types. It follows
/// the same contract as the `Eq` trait, except that the equality function
/// used may vary at runtime.
///
/// Generally `Equivalence` and `Eq` will disagree,
///
/// See also `Comparison`, `HashKey`.
pub trait Equivalence<L: ?Sized, R: ?Sized = L> {
    fn eq(&self, lhs: &L, rhs: &R) -> bool;
}

/// This trait provides a comparison function between two types. It follows the
/// same contract as the `Ord` trait, except that the comparison function used
/// may vary at runtime.
///
/// Implementations of `Equivalence` and `Comparison` must agree on equality.
///
/// See also `Equivalence`, `HashKey`.
pub trait Comparison<L: ?Sized, R: ?Sized = L>: Equivalence<L, R> {
    fn cmp(&self, lhs: &L, rhs: &R) -> Ordering;
}

/// This trait provides an alternate hash extraction function for a type or
/// types. It follows the same contract as the `Hash` trait, except that the
/// function used may vary at runtime.
///
/// Two objects that compare equal according to the `Equivalence`
/// implementation must have the same hash.
///
/// See also `Comparison`.
pub trait HashKey<T: ?Sized>: Equivalence<T> {
    fn hash<H: Hasher>(&self, state: &mut H, obj: &T);
}

/// A trivial comparison function that uses the standard library `Eq`, `Ord`,
/// `Hash`, and `Borrow` traits to do comparisons/hashing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BasicComparison;

impl<L, R> Equivalence<L, R> for BasicComparison
where
    L: ?Sized + Borrow<R>,
    R: ?Sized + Eq,
{
    fn eq(&self, lhs: &L, rhs: &R) -> bool {
        lhs.borrow() == rhs
    }
}

impl<L, R> Comparison<L, R> for BasicComparison
where
    L: ?Sized + Borrow<R>,
    R: ?Sized + Ord,
{
    fn cmp(&self, lhs: &L, rhs: &R) -> Ordering {
        lhs.borrow().cmp(rhs)
    }
}

impl<T> HashKey<T> for BasicComparison
where
    T: ?Sized + Eq + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H, obj: &T) {
        obj.hash(state);
    }
}
