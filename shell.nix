let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-25.11") {}; 

in pkgs.mkShell {
  packages = [
    (pkgs.python313.withPackages (python-pkgs: with python-pkgs; [
      torch
      torchvision
      opencv4
      numpy
      pytest
      cryptography
      psycopg2
      python-dotenv
      reedsolo
      kornia
    ]))
  ];
}
