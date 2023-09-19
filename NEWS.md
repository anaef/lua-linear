# Lua Linear Release Notes


## Release 1.0.1

- Fix potential infinity values returned from the normal function.

- Change vector and matrix index access to return nil for out-of-bound indexes. This is required
for ipairs iteration as of Lua 5.4.

- Deprecate axpy, which is superseded by axpby.


## Release 1.0.0 (2023-09-17)

- Initial release.
