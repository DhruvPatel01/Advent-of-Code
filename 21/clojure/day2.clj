(defn parse-command [c]
  (let [[command x]
        (clojure.string/split c #" ")
        x (Integer. x)]
    (cond (= command "up") {:y (- x) :x 0}
          (= command "down") {:y x :x 0}
          (= command "forward") {:x x :y 0})))

(defn reducer [a b]
  {:x (+ (:x a) (:x b))
   :y (+ (:y a) (:y b))})


(defn reducer2 [a b]
  {:aim (+ (:aim a) (:y b))
   :x (+ (:x a) (:x b))
   :y (+ (:y a) (* (:aim a) (:x b)))
   })

(let [aseq  (map parse-command
                 (clojure.string/split (slurp *in*) #"\n"))
      part1 (reduce reducer {:x 0 :y 0} aseq)
      part2 (reduce reducer2 {:aim 0 :x 0 :y 0} aseq)]
  (println (* (:x part1) (:y part1)))
  (println (* (:x part2) (:y part2)))
  )