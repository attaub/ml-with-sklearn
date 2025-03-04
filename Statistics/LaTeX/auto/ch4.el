;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "ch4"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "") ("amsfonts" "") ("amssymb" "") ("geometry" "")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "amsfonts"
    "amssymb"
    "geometry"))
 :latex)

