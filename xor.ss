;;; XOR with ANN

(load "ann.ss")

(define ann (new-ann '(2 3 1)))

(do ((i 0 (+ i 1)))
    ((> i 100000) #t)
  (train-ann ann '(0 0) '(0))
  (train-ann ann '(0 1) '(1))
  (train-ann ann '(1 0) '(1))
  (train-ann ann '(1 1) '(0)))

(round-output (run-ann ann '(0 0)))
(round-output (run-ann ann '(0 1)))
(round-output (run-ann ann '(1 0)))
(round-output (run-ann ann '(1 1)))
