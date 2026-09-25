//! Shared deterministic test inputs for the `TensorQuay` engine.
//!
//! This crate is test infrastructure, not the production token sampler. It has no dependencies,
//! allocates nothing and contains no `unsafe` code, so it builds in a `no_std` environment.

#![no_std]

pub mod philox;
