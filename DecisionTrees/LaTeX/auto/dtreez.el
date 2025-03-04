;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "dtreez"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "") ("amsthm" "") ("graphicx" "") ("hyperref" "") ("geometry" "margin=0.5in") ("algorithm" "") ("algpseudocode" "")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "amsthm"
    "graphicx"
    "hyperref"
    "geometry"
    "algorithm"
    "algpseudocode")
   (LaTeX-add-labels
    "cart_algorithm"))
 :latex)

