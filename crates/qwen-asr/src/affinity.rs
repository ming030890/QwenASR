//! CPU core affinity detection and pinning for Linux / Android.

use std::sync::OnceLock;

static PERF_CORES: OnceLock<Vec<usize>> = OnceLock::new();

#[cfg(any(target_os = "android", target_os = "linux"))]
pub fn get_performance_cores() -> &'static [usize] {
    PERF_CORES.get_or_init(|| {
        let mut core_freqs: Vec<(usize, u64)> = Vec::new();
        let num_cpus = crate::kernels::get_num_cpus();

        for cpu in 0..num_cpus {
            let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", cpu);
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(freq) = content.trim().parse::<u64>() {
                    core_freqs.push((cpu, freq));
                }
            }
        }

        if core_freqs.len() >= 4 {
            let min_freq = core_freqs.iter().map(|(_, f)| *f).min().unwrap_or(0);
            let max_freq = core_freqs.iter().map(|(_, f)| *f).max().unwrap_or(0);

            // If heterogeneous (big.LITTLE / DynamIQ cluster):
            if max_freq > min_freq {
                let perf_cores: Vec<usize> = core_freqs
                    .iter()
                    .filter(|(_, f)| *f > min_freq)
                    .map(|(c, _)| *c)
                    .collect();
                if !perf_cores.is_empty() {
                    return perf_cores;
                }
            }
        }

        // Fallback heuristic for 8-core ARM SoCs if sysfs is inaccessible:
        // Cores 0..2 are typically little cores; cores 2..8 are performance cores.
        if num_cpus >= 8 {
            return (2..num_cpus).collect();
        }

        (0..num_cpus).collect()
    })
}

#[cfg(any(target_os = "android", target_os = "linux"))]
pub fn pin_current_thread_to_cores(cores: &[usize]) -> bool {
    if cores.is_empty() {
        return false;
    }

    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cores {
            if cpu < 1024 {
                libc::CPU_SET(cpu, &mut set);
            }
        }
        let ret = libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set as *const libc::cpu_set_t,
        );
        ret == 0
    }
}

#[cfg(not(any(target_os = "android", target_os = "linux")))]
pub fn get_performance_cores() -> &'static [usize] {
    PERF_CORES.get_or_init(|| (0..crate::kernels::get_num_cpus()).collect())
}

#[cfg(not(any(target_os = "android", target_os = "linux")))]
pub fn pin_current_thread_to_cores(_cores: &[usize]) -> bool {
    true
}

thread_local! {
    static IS_PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Ensures the calling thread is pinned away from efficiency (LITTLE) cores.
pub fn ensure_current_thread_pinned() {
    #[cfg(any(target_os = "android", target_os = "linux"))]
    {
        IS_PINNED.with(|pinned| {
            if !pinned.get() {
                let cores = get_performance_cores();
                pin_current_thread_to_cores(cores);
                pinned.set(true);
            }
        });
    }
}
