;;; Addition with ANN

(load "ann.ss")

(define ann (new-ann '(4 6 6 3)))
(define (run-add ann a b)
  (blist->int (round-output
               (run-ann ann
                        (append (int->blist a 2)
                                (int->blist b 2))))))
(do ((i 0 (+ i 1)))
    ((> i 100000) #t)
  ;(printf "~a\n" i)
  (let ((a (random 4))
        (b (random 4)))
    (train-ann ann
               (append (int->blist a 2) (int->blist b 2))
               (int->blist (+ a b) 3))))
(run-add ann 0 0)
(run-add ann 0 1)
(run-add ann 2 1)
(run-add ann 3 2)
(run-add ann 3 1)
(run-add ann 3 0)
