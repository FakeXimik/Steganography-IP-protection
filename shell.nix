let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-25.11") {}; 

in pkgs.mkShell {
  packages = [
    (pkgs.python313.withPackages (python-pkgs: with python-pkgs; [
      torch
      opencv4
      numpy
      pytest
    ]))
  ];
}
