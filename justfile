# List the just recipe list
list:
    just --list

format:
    cargo fmt

build:
    cargo build

build-wasm:
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' cargo build --target wasm32-unknown-unknown

clippy:
    cargo clippy

test:
    cargo test --release

checks:
    just format
    just build
    just build-wasm
    just clippy
    just test

clean:
    find . -name target -type d -exec rm -r {} +
    just remove-lockfiles

remove-lockfiles:
    find . -name Cargo.lock -type f -exec rm {} +

list-outdated:
    cargo outdated -R -w

update:
    cargo update --aggressive

publish:
    cargo publish --no-verify
