{
  description = "Universal LLM Mock Server Dev Environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { self, nixpkgs, rust-overlay, ... }@inputs:
    let
      # Pinned rather than `stable.latest`, so every developer and CI run uses
      # the same compiler. `latest` moved to 1.99.0 on its own and nobody chose
      # it.
      rustVersion = "1.99.0";

      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; overlays = [ (import rust-overlay) ]; };
    in {
      devShells = {
        x86_64-linux = {
          default = pkgs.mkShell {
            buildInputs = [
              pkgs.rust-bin.stable.${rustVersion}.default
              pkgs.openssl
              pkgs.pkg-config
              pkgs.cacert
            ];
            RUST_BACKTRACE = "1";
            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH
            '';
          };
        };
      };
    };
}
