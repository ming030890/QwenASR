use super::*;
use std::ffi::c_void;

#[repr(C)]
struct RhsParams {
    lhs_zero_point: i32,
    scale_multiplier: f32,
}

#[test]
fn single_row_matches_matrix_kernel_with_same_packed_weights() {
    unsafe extern "C" {
        fn kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
            num_groups: usize,
            n: usize,
            k: usize,
            nr: usize,
            kr: usize,
            sr: usize,
            rhs: *const i8,
            bias: *const f32,
            scale: *const f32,
            rhs_packed: *mut c_void,
            extra_bytes: usize,
            params: *const RhsParams,
        );
    }
    set_threads(1);
    for (k, n) in [(32, 5), (64, 16), (2048, 1025), (6144, 512)] {
        let weights: Vec<i8> = (0..n * k)
            .map(|i| ((i * 31 + 17) % 127) as i8 - 63)
            .collect();
        let scales: Vec<f32> = (0..n).map(|i| 0.01 + (i % 7) as f32 * 0.002).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.5).collect();
        let mut rhs = vec![0i8; kleidiai_packed_rhs_size(k, n)];
        unsafe {
            kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                1,
                n,
                k,
                4,
                8,
                1,
                weights.as_ptr(),
                bias.as_ptr(),
                scales.as_ptr(),
                rhs.as_mut_ptr().cast(),
                0,
                &RhsParams {
                    lhs_zero_point: 1,
                    scale_multiplier: 1.0,
                },
            );
        }
        for positive in [false, true] {
            let input: Vec<f32> = (0..k)
                .map(|i| ((i * 7 + 11) % 101) as f32 / 31.0 - if positive { 0.0 } else { 1.5 })
                .collect();
            let mut expected = vec![0.0f32; n];
            let mut actual = vec![0.0f32; n];
            linear_int8_kleidiai_impl(&mut [(&mut expected, &rhs, n)], &input, 1, k, false);
            for threads in [1, 4, 2, 1] {
                set_threads(threads);
                linear_int8_kleidiai_impl(&mut [(&mut actual, &rhs, n)], &input, 1, k, true);
                for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
                    let tolerance = 1e-4 * (1.0 + expected.abs());
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "k={k} n={n} positive={positive} row={index}: {actual} != {expected}"
                    );
                }
            }
        }
    }
    set_threads(1);
}
