;;; Scheme Artificial Neural Network

;;; Supporting functions

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

;;; Neuron functions

;; Train neurons in weight list
(define (train lst err)
  (if (empty? lst)
      '()
      (let ((n (caar lst))
            (w (cadar lst)))
        (n 'train err)
        (cons (list n (+ w (* (n) err)))
              (train (cdr lst) err)))))

;; Reset neurons in weight list
(define (reset lst)
  (if (empty? lst)
      '()
      (begin
        ((caar lst) 'reset)
        (reset (cdr lst)))))

;; Create a new neuron
(define (new-neuron)
  (let ((theta (rand-theta))
        (backward '())
        (cache #f)
        (trained #f))
    ;; Neuron function with closure
    (lambda ([method 'activate] [arg '()])
      (cond
       ((eq? method 'backward)
        (push! (list arg (rand-weight)) backward))
       ((eq? method 'set)
        (set! cache arg))
       ((eq? method 'reset)
        (set! cache #f)
        (set! trained #f)
        (reset backward))
       ((eq? method 'train)
        (if (not trained)
            (set! backward (train backward arg))
            (set! trained #t)))
       ((eq? method 'activate)
        (if cache
            cache
            (begin
              (set! cache (sigmoid (sum-weight backward)))
              cache)))))))

;;; Layer functions

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

;; Reset layer, which back-propagates
(define (reset-layer layer)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'reset)
        (reset-layer (cdr layer)))))

;; Train a layer, which back-propagates
(define (train-layer layer out desired a)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'train (* a (- (car desired) (car out))))
        (cons (car out)
              (train-layer (cdr layer)
                           (cdr out)
                           (cdr desired)
                           a)))))

;;; ANN functions

;; Create new ann based on specification
(define (new-ann spec)
  (let ((ann (map new-layer spec)))
    (link-ann ann)
    ann))

;; Get output of ann
(define (run-ann ann in)
  (set-layer (car ann) in)
  (let ((out (run-layer (last ann))))
    (reset-layer (last ann))
    out))

;; Train the ann
(define (train-ann ann in desired [a 1])
  (set-layer (car ann) in)
  (let ((out (run-layer (last ann))))
    (train-layer (last ann) out desired a)
    (reset-layer (last ann))
    out))
