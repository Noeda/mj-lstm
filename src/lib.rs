#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]
#![feature(stdsimd)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
pub mod simd_tests;

pub mod gru;
pub mod lstm;
pub mod rnn;
#[cfg(target_arch = "aarch64")]
pub mod simd_aarch64;
#[cfg(target_arch = "x86_64")]
pub mod simd_amd64;
pub mod simd_common;
pub mod simple;
pub mod stop_condition;
pub mod unpackable;

pub use crate::gru::*;
pub use crate::lstm::*;
pub use crate::simple::*;
pub use crate::stop_condition::*;
