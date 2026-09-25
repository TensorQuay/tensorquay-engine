//! Pure host logic for the `TensorQuay` engine.
//!
//! This crate validates metadata. It allocates no device memory, dereferences no pointer and
//! reads back nothing from a GPU, so it builds and tests anywhere.

#![no_std]

pub mod contracts;
