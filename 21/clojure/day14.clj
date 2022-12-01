(def inp (clojure.string/split (slurp *in*) #"\n"))

(def pattern (seq (first inp)))
(def rules (map #(clojure.string/split % #" -> ") (rest (rest inp))))
(def rules (reduce (fn [r [k v]]
                     (assoc r (seq k) (first v))) {} rules ))

(defn step [template]
  (loop [new [(first template)]
         [head & rest] (partition 2 1 template)]
    (if (nil? head) new
        (recur (into new [(rules head) (nth head 1)]) rest)
      )))

(defn steps [i n pattern]
  (if (= i n) pattern (recur (+ i 1) n (step pattern))))

(def part1 (frequencies (steps 0 10 pattern)))
(println (- (apply max (vals part1)) (apply min (vals part1))))

(def part2 (frequencies (steps 0 40 pattern)))
(println (- (apply max (vals part1)) (apply min (vals part1))))
