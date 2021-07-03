#![allow(clippy::needless_range_loop)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub mod lstm;
pub mod rnn;
pub mod simd;

pub use crate::lstm::*;
