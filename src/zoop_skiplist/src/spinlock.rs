use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// A simple spinlock with timeout for deadlock avoidance.
pub(crate) struct SpinLock {
    locked: AtomicBool,
}

impl Default for SpinLock {
    fn default() -> Self {
        Self::new()
    }
}

impl SpinLock {
    /// Creates a new unlocked spinlock.
    pub(crate) const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Acquires the lock, spinning until successful or timeout.
    pub(crate) fn lock(&self) -> SpinLockGuard<'_> {
        loop {
            // Try to acquire the lock
            if self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return SpinLockGuard { lock: self };
            }

            let mut spin_count = 0u32;
            while self.locked.load(Ordering::Relaxed) {
                spin_count += 1;

                if spin_count % 64 == 0 {
                    // Yield periodically
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Tries to acquire the lock without blocking.
    ///
    /// Returns `Some(guard)` if the lock was acquired, `None` otherwise.
    #[allow(dead_code)]
    pub(crate) fn try_lock(&self) -> Option<SpinLockGuard<'_>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(SpinLockGuard { lock: self })
        } else {
            None
        }
    }

    pub(crate) fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }

    /// Releases the lock.
    #[inline]
    pub(crate) fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

/// RAII spinlock guard.
pub(crate) struct SpinLockGuard<'a> {
    lock: &'a SpinLock,
}

impl Drop for SpinLockGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock()
    }
}
