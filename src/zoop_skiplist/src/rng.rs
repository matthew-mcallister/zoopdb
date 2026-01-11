use std::cell::UnsafeCell;
use std::rc::Rc;

use rand::{RngCore, SeedableRng};
use rand::rngs::SmallRng;

thread_local!(
    static THREAD_RNG_KEY: Rc<UnsafeCell<SmallRng>> = {
        let rng = SmallRng::from_os_rng();
        Rc::new(UnsafeCell::new(rng))
    }
);

/// Clone of ThreadRng but uses SmallRng under the hood.
pub(crate) struct ThreadSmallRng {
    inner: Rc<UnsafeCell<SmallRng>>,
}

pub(crate) fn rng() -> ThreadSmallRng {
    let rc = THREAD_RNG_KEY.with(|r| Rc::clone(&r));
    ThreadSmallRng { inner: rc }
}

impl Default for ThreadSmallRng {
    fn default() -> Self {
        rng()
    }
}

impl RngCore for ThreadSmallRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        let rng = unsafe { &mut *self.inner.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let rng = unsafe { &mut *self.inner.get() };
        rng.next_u64()
    }

    #[inline(always)]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let rng = unsafe { &mut *self.inner.get() };
        rng.fill_bytes(dest)
    }}

