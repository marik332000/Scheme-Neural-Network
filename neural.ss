;;; Scheme Artificial Neural Network

;; Stack push macro
(define-syntax push!
  (syntax-rules ()
    ((push item place)
     (set! place (cons item place)))))

;; Sum weights in given list
(define (sum-weight lst)
  (if (empty? lst) 0
      (+ (* ((caar lst)) (cadar lst))
         (sum-weight (cdr lst)))))

;; Sigmoid learning function
(define (sigmoid x)
  (/ (+ 1.0 (exp (- x)))))

;; Generate a new random weight
(define (rand-weight)
  (- (random) 0.5))

;; Generate a new random threshold
(define (rand-theta)
  (- (* (random) 4) 2))

;; Create a new neuron
(define (new-neuron)
  (let ((theta (rand-theta))
        (backward '())
        (forward '())
        (input #f))
    ;; Neuron function with closure
    (lambda ([method 'activate] [arg '()])
      (cond
       ((eq? method 'forward)
        (push! (list arg (rand-weight)) forward))
       ((eq? method 'backward)
        (push! (list arg (rand-weight)) backward))
       ((eq? method 'set)
        (set! input arg))
       ((eq? method 'activate)
        (if input
            input
            (sigmoid (sum-weight backward))))))))

;; Create a new neuron layer
(define (new-layer n)
  (if (= n 0) '()
      (cons (new-neuron) (new-layer (- n 1)))))

;; Link two layers together
(define (link-layers left right)
  (if (or (empty? left) (empty? right))
      '()
      (begin
        ((car right) 'backward (car left))
        (link-layers (cdr left) right)
        (link-layers left (cdr right)))))

;; Link up layers in an unlinked ann
(define (link-ann ann)
  (if (= (length ann) 1) '()
      (begin
        (link-layers (car ann) (cadr ann))
        (link-ann (cdr ann)))))

;; Create new ann based on specification
(define (new-ann spec)
  (let ((ann (map new-layer spec)))
    (link-ann ann)
    ann))

;; Hard set a layer of neurons
(define (set-layer layer in)
  (if (empty? layer) '()
      (begin
        ((car layer) 'set (car in))
        (set-layer (cdr layer) (cdr in)))))

;; Activate a layer, which activates all layers behind it
(define (run-layer layer)
  (if (empty? layer) '()
      (cons ((car layer)) (run-layer (cdr layer)))))

;; Get output of ann
(define (run-ann ann in)
  (set-layer (car ann) in)
  (run-layer (last ann)))
