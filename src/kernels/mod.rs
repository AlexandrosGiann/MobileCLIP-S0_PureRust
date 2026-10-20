//! CPU kernels.

pub mod conv;
pub mod gemm;
pub mod isa;
pub mod math;
pub mod ops;

pub use isa::kernel_set;
