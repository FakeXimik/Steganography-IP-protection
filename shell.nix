let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-25.11") {}; 

in pkgs.mkShell {
  packages = [
    vi

    (pkgs.python313.withPackages (python-pkgs: with python-pkgs; [
      torch
      torchvision
      opencv4
      numpy
      cryptography
      psycopg2
    ]))
  ];
}
