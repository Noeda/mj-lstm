#![allow(clippy::needless_range_loop)]
#![feature(stdsimd)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
pub mod simd_tests;

pub mod lstm;
pub mod rnn;
#[cfg(target_arch = "aarch64")]
pub mod simd_aarch64;
#[cfg(target_arch = "x86_64")]
pub mod simd_amd64;
pub mod simd_common;

pub use crate::lstm::*;
